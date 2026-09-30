#!/usr/bin/env python3
"""
Command-line inference for MSAF-DenseNet201.

    python src/predict.py leaf.jpg
    python src/predict.py leaf.jpg --cam out.png
    python src/predict.py folder/ --json results.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import IMG_SIZE, MEAN, STD, load_model

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def build_args():
    ap = argparse.ArgumentParser(description="MSAF-DenseNet201 leaf disease inference")
    ap.add_argument("path", help="image file or a directory of images")
    ap.add_argument("--ckpt", default=os.environ.get("MSAF_CKPT", "msaf_best.pt"))
    ap.add_argument("--cam", metavar="OUT.png", help="save a Grad-CAM overlay (single image only)")
    ap.add_argument("--json", metavar="OUT.json", help="write predictions as JSON")
    ap.add_argument("--no-tta", action="store_true", help="disable 4-way flip TTA (faster)")
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    return ap.parse_args()


def pick_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    a = build_args()
    device = pick_device(a.device)
    if not os.path.exists(a.ckpt):
        sys.exit(f"Checkpoint not found: {a.ckpt}\nRun scripts/download_model.sh first.")

    net, meta = load_model(a.ckpt, device)
    classes = meta["classes"]

    tf = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)), transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    if os.path.isdir(a.path):
        files = sorted(os.path.join(a.path, f) for f in os.listdir(a.path)
                       if os.path.splitext(f)[1].lower() in IMG_EXT)
        if not files:
            sys.exit(f"No images found in {a.path}")
    else:
        files = [a.path]

    print(f"model: MSAF-DenseNet201 | device: {device} | "
          f"val_acc: {meta['val_accuracy']:.4f} | TTA: {not a.no_tta}\n")

    out = []
    for path in files:
        img = Image.open(path).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        t0 = time.time()
        with torch.no_grad():
            p = torch.softmax(net(x).float(), 1)
            if not a.no_tta:
                p = (p + torch.softmax(net(torch.flip(x, [3])).float(), 1)
                       + torch.softmax(net(torch.flip(x, [2])).float(), 1)
                       + torch.softmax(net(torch.flip(x, [2, 3])).float(), 1)) / 4
        probs = p[0].cpu().numpy()
        top = int(probs.argmax())
        ms = (time.time() - t0) * 1000

        print(f"{os.path.basename(path)}")
        print(f"  -> {classes[top]}  ({probs[top]*100:.2f}%)   [{ms:.0f} ms]")
        for i in np.argsort(-probs)[1:]:
            print(f"     {classes[i]:<24} {probs[i]*100:6.2f}%")
        print()

        out.append({"file": path, "prediction": classes[top],
                    "confidence": float(probs[top]),
                    "probabilities": {classes[i]: float(probs[i]) for i in range(len(classes))},
                    "inference_ms": round(ms, 1)})

        if a.cam and len(files) == 1:
            net.zero_grad(set_to_none=True)
            c3, c4, c5 = net.features(x)
            c5.retain_grad()
            net.classify(c3, c4, c5)[0, top].backward()
            w = c5.grad.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((w * c5).sum(1, keepdim=True))
            cam = F.interpolate(cam, (IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
            cam = cam[0, 0].detach().float().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            base = np.asarray(img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
            c = np.clip(cam, 0, 1)
            heat = np.stack([np.clip(1.5 - np.abs(4 * c - 3), 0, 1),
                             np.clip(1.5 - np.abs(4 * c - 2), 0, 1),
                             np.clip(1.5 - np.abs(4 * c - 1), 0, 1)], -1)
            al = (0.65 * cam)[..., None]
            mix = np.clip((1 - al) * base + al * heat, 0, 1)
            Image.fromarray((mix * 255).astype(np.uint8)).save(a.cam)
            print(f"Grad-CAM written to {a.cam}\n")

    if a.json:
        json.dump(out, open(a.json, "w"), indent=2)
        print(f"JSON written to {a.json}")


if __name__ == "__main__":
    main()
