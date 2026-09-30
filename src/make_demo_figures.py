"""
Generates the "model works on real images" figures for the presentation.

For each class it picks correctly-classified test images and renders
[ input | Grad-CAM | class confidences ] panels, plus one combined grid
that drops straight onto a slide.
"""
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec
from PIL import Image
from torchvision import transforms

from model import IMG_SIZE, MEAN, STD, load_model

CKPT   = os.environ.get("MSAF_CKPT", "msaf_best.pt")
OUTDIR = os.environ.get("RESULTS_DIR", "results")
SEED   = 42
os.makedirs(OUTDIR, exist_ok=True)

GREEN, GREY = "#1b5e20", "#c8d6c8"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net, meta = load_model(CKPT, device)
CLASSES = meta["classes"]
PRETTY  = {c: c.replace("_", " ").title() for c in CLASSES}

tf = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)), transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def predict(x):
    with torch.no_grad():
        p = (torch.softmax(net(x).float(), 1)
             + torch.softmax(net(torch.flip(x, [3])).float(), 1)
             + torch.softmax(net(torch.flip(x, [2])).float(), 1)
             + torch.softmax(net(torch.flip(x, [2, 3])).float(), 1)) / 4
    return p[0].cpu().numpy()


def gradcam(x, idx):
    net.zero_grad(set_to_none=True)
    c3, c4, c5 = net.features(x)
    c5.retain_grad()
    net.classify(c3, c4, c5)[0, idx].backward()
    w = c5.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * c5).sum(1, keepdim=True))
    cam = F.interpolate(cam, (IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().float().cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def analyse(path):
    img = Image.open(path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    probs = predict(x)
    top = int(probs.argmax())
    return img.resize((IMG_SIZE, IMG_SIZE)), probs, top, gradcam(x, top)


def draw_row(axes, img, probs, top, cam, true_label=None):
    ax0, ax1, ax2 = axes
    ax0.imshow(img); ax0.axis("off")
    ax0.set_title(f"Input — {PRETTY.get(true_label, true_label)}" if true_label else "Input",
                  fontsize=10, color="#16261a", pad=6)

    ax1.imshow(img); ax1.imshow(cam, cmap="jet", alpha=0.42); ax1.axis("off")
    ax1.set_title("Grad-CAM", fontsize=10, color="#16261a", pad=6)

    order = np.argsort(-probs)
    names = [PRETTY[CLASSES[i]] for i in order][::-1]
    vals = [probs[i] * 100 for i in order][::-1]
    colors = [GREEN if i == len(vals) - 1 else GREY for i in range(len(vals))]
    ax2.barh(names, vals, color=colors, height=0.62)
    ax2.set_xlim(0, 108)
    for j, v in enumerate(vals):
        ax2.text(v + 2, j, f"{v:.1f}%", va="center", fontsize=8.5,
                 color="#16261a", fontweight="bold" if j == len(vals) - 1 else "normal")
    ax2.tick_params(labelsize=8.5, length=0)
    ax2.set_xticks([])
    for sp in ax2.spines.values():
        sp.set_visible(False)
    correct = (true_label is None) or (CLASSES[top] == true_label)
    ax2.set_title(f"{PRETTY[CLASSES[top]]} — {probs[top]*100:.1f}%"
                  + ("" if correct else "  (misclassified)"),
                  fontsize=10, pad=6,
                  color=GREEN if correct else "#c62828", fontweight="bold")


def build(test_items, per_class=1):
    """test_items: list of (path, class_index) from the notebook's test split."""
    rng = random.Random(SEED)
    picked = []
    for ci, cname in enumerate(CLASSES):
        pool = [p for p, y in test_items if y == ci]
        rng.shuffle(pool)
        chosen, tried = [], 0
        for p in pool:
            if len(chosen) >= per_class or tried > 25:
                break
            tried += 1
            img, probs, top, cam = analyse(p)
            if CLASSES[top] == cname:                     # prefer correct predictions
                chosen.append((p, img, probs, top, cam))
        if not chosen and pool:                            # fall back to any image
            p = pool[0]
            chosen.append((p, *analyse(p)))
        picked += [(cname, *c[1:]) for c in chosen]

    # ---- combined grid for the slide ----
    n = len(picked)
    fig = plt.figure(figsize=(9.0, 2.05 * n), facecolor="white")
    gs = GridSpec(n, 3, figure=fig, width_ratios=[1, 1, 1.75],
                  hspace=0.45, wspace=0.55)
    for r, (cname, img, probs, top, cam) in enumerate(picked):
        axes = [fig.add_subplot(gs[r, c]) for c in range(3)]
        draw_row(axes, img, probs, top, cam, true_label=cname)
    fig.suptitle("MSAF-DenseNet201 — predictions on held-out test images",
                 fontsize=13, fontweight="bold", color="#1b5e20", y=0.995)
    fig.savefig(f"{OUTDIR}/demo_grid.png", dpi=170, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

    # ---- one panel per class, for individual slides ----
    for cname, img, probs, top, cam in picked:
        fig = plt.figure(figsize=(11.5, 2.15), facecolor="white")
        gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 2.1], wspace=0.42)
        draw_row([fig.add_subplot(gs[0, c]) for c in range(3)],
                 img, probs, top, cam, true_label=cname)
        safe = cname.replace(" ", "_")
        fig.savefig(f"{OUTDIR}/demo_{safe}.png", dpi=170,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)

    print(f"wrote {OUTDIR}/demo_grid.png and {len(picked)} per-class panels")
    return picked
