#!/usr/bin/env python3
"""
Train MSAF-DenseNet201 on a folder of class sub-folders.

This is a consolidated, single-file version of the training configuration used
for the reported results (the original runs were done block-by-block in Google
Colab on an A100). It writes a checkpoint in exactly the format that
src/model.py:load_model() and the web app expect.

    python src/train.py --data /path/to/Raw_Data --out runs/msaf

The data folder must contain one sub-folder per class, e.g.

    Raw_Data/
      early_leaf_spot/   healthy leaf/   late leaf spot/
      nutrition deficiency/   rust/

A stratified 70/30 train/test split is made with a fixed seed, and 15 % of the
training part is held out as a validation set for checkpoint selection. The test
split is only touched once, at the very end.
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import IMG_SIZE, MEAN, STD, MSAF_DenseNet201  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------------------------------------------------------- arguments
def build_args():
    ap = argparse.ArgumentParser(description="Train MSAF-DenseNet201")
    ap.add_argument("--data", required=True, help="folder with one sub-folder per class")
    ap.add_argument("--out", default="runs/msaf", help="output folder")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--val-frac", type=float, default=0.15, help="fraction of the training part")
    ap.add_argument("--freeze-epochs", type=int, default=8, help="epochs with the backbone frozen")
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1.5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--min-lr-factor", type=float, default=1e-3, help="final lr = base lr x this")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--mixup-p", type=float, default=0.30)
    ap.add_argument("--mixup-alpha", type=float, default=0.2)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (testing only)")
    return ap.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- data
def scan(root):
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    items = []
    for ci, c in enumerate(classes):
        for f in sorted(os.listdir(os.path.join(root, c))):
            if os.path.splitext(f)[1].lower() in IMG_EXT:
                items.append((os.path.join(root, c, f), ci))
    if not items:
        sys.exit(f"No images found under {root}")
    return classes, items


def stratified_split(items, frac, rng):
    """Split so that every class keeps (almost exactly) the same proportion."""
    by_cls = {}
    for it in items:
        by_cls.setdefault(it[1], []).append(it)
    keep, held = [], []
    for c in sorted(by_cls):
        group = by_cls[c][:]
        rng.shuffle(group)
        n = int(round(len(group) * frac))
        held += group[:n]
        keep += group[n:]
    return keep, held


class ImageList(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), y


def build_transforms(size):
    train = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        # mild on purpose: lesion colour is the diagnostic signal
        transforms.ColorJitter(0.15, 0.15, 0.10, 0.02),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.10)),
    ])
    evaluate = transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train, evaluate


# ---------------------------------------------------------------- training pieces
def backbone_params(model):
    mods = [model.stem, model.block2, model.trans2, model.block3, model.trans3, model.block4]
    return [p for m in mods for p in m.parameters()]


def set_backbone_trainable(model, flag):
    for p in backbone_params(model):
        p.requires_grad = flag


class EMA:
    """Exponential moving average of weights (buffers are copied)."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


def lr_factor(epoch, args):
    """epoch is 1-based. Linear warm-up, then cosine decay to min_lr_factor."""
    if epoch <= args.warmup_epochs:
        return epoch / args.warmup_epochs
    t = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
    return args.min_lr_factor + (1 - args.min_lr_factor) * 0.5 * (1 + math.cos(math.pi * t))


def class_weights(items, ncls):
    """Inverse square-root frequency, normalised to mean 1."""
    cnt = Counter(y for _, y in items)
    w = torch.tensor([1.0 / math.sqrt(cnt.get(c, 1)) for c in range(ncls)])
    return w * ncls / w.sum()


@torch.no_grad()
def predict(model, loader, device, tta=False):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        x = x.to(device)
        if tta:   # original + h-flip + v-flip + both
            p = sum(F.softmax(model(v), 1) for v in
                    (x, x.flip(3), x.flip(2), x.flip(2).flip(3))) / 4
        else:
            p = F.softmax(model(x), 1)
        probs.append(p.cpu())
        ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def metrics(probs, y, classes):
    from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                                 precision_recall_fscore_support)
    pred = probs.argmax(1)
    pw, rw, fw, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    pc, rc, fc, sc = precision_recall_fscore_support(y, pred, zero_division=0)
    summary = {
        "test_images": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision_w": float(pw), "recall_w": float(rw), "f1_w": float(fw),
        "precision_macro": float(pm), "recall_macro": float(rm), "f1_macro": float(fm),
        "cohen_kappa": float(cohen_kappa_score(y, pred)),
    }
    per_class = [(classes[i], float(pc[i]), float(rc[i]), float(fc[i]), int(sc[i]))
                 for i in range(len(classes))]
    cm = confusion_matrix(y, pred, labels=list(range(len(classes))))
    return summary, per_class, cm


# ---------------------------------------------------------------- main
def main():
    args = build_args()
    seed_everything(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    use_amp = device.type == "cuda"

    # ---- data and splits
    classes, items = scan(args.data)
    rng = random.Random(args.seed)
    trainval, test = stratified_split(items, args.test_frac, rng)
    train, val = stratified_split(trainval, args.val_frac, rng)
    print(f"classes: {classes}")
    print(f"images: {len(items)}  train {len(train)}  val {len(val)}  test {len(test)}")
    with open(os.path.join(args.out, "split.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "class", "split"])
        for name, part in (("train", train), ("val", val), ("test", test)):
            for p, y in part:
                w.writerow([p, classes[y], name])

    tf_train, tf_eval = build_transforms(args.img_size)
    dl = dict(num_workers=args.workers, pin_memory=device.type == "cuda")
    train_dl = DataLoader(ImageList(train, tf_train), args.batch_size, shuffle=True, drop_last=True, **dl)
    val_dl = DataLoader(ImageList(val, tf_eval), args.batch_size * 2, **dl)
    test_dl = DataLoader(ImageList(test, tf_eval), args.batch_size * 2, **dl)

    # ---- model, loss, optimiser
    model = MSAF_DenseNet201(len(classes), pretrained=not args.no_pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params / 1e6:.2f} M")

    bb_ids = {id(p) for p in backbone_params(model)}
    head_params = [p for p in model.parameters() if id(p) not in bb_ids]
    opt = torch.optim.AdamW([
        {"params": backbone_params(model), "lr": args.lr_backbone, "base_lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head, "base_lr": args.lr_head},
    ], weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss(weight=class_weights(train, len(classes)).to(device),
                                    label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(model, args.ema_decay)

    history, best_acc, best_epoch = [], -1.0, -1
    ckpt_path = os.path.join(args.out, "msaf_best.pt")

    # ---- training loop (all epochs are run: no early stopping)
    for epoch in range(1, args.epochs + 1):
        set_backbone_trainable(model, epoch > args.freeze_epochs)
        f = lr_factor(epoch, args)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * f

        model.train()
        t0, run_loss, correct, seen = time.time(), 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            mixed = random.random() < args.mixup_p
            if mixed:
                lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                idx = torch.randperm(x.size(0), device=device)
                x = lam * x + (1 - lam) * x[idx]
                y2 = y[idx]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
                loss = (lam * criterion(out, y) + (1 - lam) * criterion(out, y2)) if mixed \
                    else criterion(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ema.update(model)

            run_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            seen += x.size(0)

        # ---- validation: raw weights and EMA weights
        pv, yv = predict(model, val_dl, device)
        val_acc = float((pv.argmax(1) == yv).mean())
        raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.shadow)
        pe, _ = predict(model, val_dl, device)
        val_acc_ema = float((pe.argmax(1) == yv).mean())
        model.load_state_dict(raw_state)

        which, acc = ("ema", val_acc_ema) if val_acc_ema > val_acc else ("model", val_acc)
        history.append({"epoch": epoch, "loss": run_loss / max(seen, 1),
                        "train_acc": correct / max(seen, 1), "val_acc": val_acc,
                        "val_acc_ema": val_acc_ema, "lr": opt.param_groups[1]["lr"]})
        flag = ""
        if acc > best_acc:
            best_acc, best_epoch, flag = acc, epoch, "  * best"
            torch.save({"model": raw_state, "ema": ema.shadow, "which": which,
                        "classes": classes, "acc": acc, "epoch": epoch - 1,
                        "img_size": args.img_size}, ckpt_path)
        print(f"epoch {epoch:3d}/{args.epochs}  loss {history[-1]['loss']:.4f}  "
              f"train {history[-1]['train_acc']:.4f}  val {val_acc:.4f}  ema {val_acc_ema:.4f}  "
              f"lr {history[-1]['lr']:.2e}  {time.time() - t0:.0f}s{flag}")
        with open(os.path.join(args.out, "history.json"), "w") as fh:
            json.dump(history, fh, indent=2)

    # ---- final, one-time test evaluation of the best checkpoint (4-way flip TTA)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck[ck["which"]])
    probs, y = predict(model, test_dl, device, tta=True)
    summary, per_class, cm = metrics(probs, y, classes)
    summary.update({"model": "MSAF-DenseNet201", "best_epoch": best_epoch,
                    "weights": "raw" if ck["which"] == "model" else "ema", "val_accuracy": best_acc})

    np.save(os.path.join(args.out, "test_probs.npy"), probs)
    np.save(os.path.join(args.out, "test_true.npy"), y)
    with open(os.path.join(args.out, "summary_metrics.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(args.out, "per_class_metrics.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Class", "Precision", "Recall", "F1_Score", "Support"])
        w.writerows(per_class)
    with open(os.path.join(args.out, "confusion_matrix.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([""] + classes)
        for c, row in zip(classes, cm):
            w.writerow([c] + list(map(int, row)))

    print(f"\nbest epoch {best_epoch}  val {best_acc:.4f}")
    print(f"TEST  acc {summary['accuracy']:.4f}  f1_w {summary['f1_w']:.4f}  "
          f"kappa {summary['cohen_kappa']:.4f}")
    print(f"outputs in {args.out}/")


if __name__ == "__main__":
    main()
