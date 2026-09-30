"""
MSAF-DenseNet201 — the proposed CNN architecture.

DenseNet201 backbone + multi-scale feature extraction (dense blocks 2-4)
+ CBAM attention at each scale + learned attention pooling.

This file MUST match the architecture used at training time, otherwise
load_state_dict() will fail.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

IMG_SIZE = 256
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class ChannelAttention(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(ch, max(ch // r, 8)), nn.ReLU(inplace=False),
            nn.Linear(max(ch // r, 8), ch))

    def forward(self, x):
        b, c, _, _ = x.shape
        a = self.mlp(F.adaptive_avg_pool2d(x, 1).flatten(1)) + \
            self.mlp(F.adaptive_max_pool2d(x, 1).flatten(1))
        return x * torch.sigmoid(a).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        s = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], 1)
        return x * torch.sigmoid(self.conv(s))


class CBAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ca, self.sa = ChannelAttention(ch), SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


class AttnPool(nn.Module):
    """Learned spatial pooling - stops small lesions being averaged away by GAP."""

    def __init__(self, ch):
        super().__init__()
        self.score = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        a = torch.softmax(self.score(x).view(b, 1, h * w), dim=-1)
        att = (x.view(b, c, h * w) * a).sum(-1)
        gap = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return torch.cat([att, gap], 1)


class ScaleBranch(nn.Module):
    def __init__(self, in_ch, proj=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, proj, 1, bias=False),
            nn.BatchNorm2d(proj), nn.SiLU(inplace=True))
        self.cbam = CBAM(proj)
        self.pool = AttnPool(proj)
        self.out_dim = proj * 2

    def forward(self, x):
        return self.pool(self.cbam(self.proj(x)))


class MSAF_DenseNet201(nn.Module):
    """Proposed CNN: multi-scale attention fusion over a DenseNet201 backbone."""

    def __init__(self, ncls, proj=256, p=0.4, pretrained=False):
        super().__init__()
        weights = models.DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
        f = models.densenet201(weights=weights).features
        self.stem = nn.Sequential(f.conv0, f.norm0, f.relu0, f.pool0,
                                  f.denseblock1, f.transition1)
        self.block2 = f.denseblock2
        self.trans2 = f.transition2
        self.block3 = f.denseblock3
        self.trans3 = f.transition3
        self.block4 = nn.Sequential(f.denseblock4, f.norm5)

        self.s3 = ScaleBranch(512, proj)
        self.s4 = ScaleBranch(1792, proj)
        self.s5 = ScaleBranch(1920, proj)
        d = self.s3.out_dim + self.s4.out_dim + self.s5.out_dim

        self.head = nn.Sequential(
            nn.BatchNorm1d(d), nn.Dropout(p),
            nn.Linear(d, 512), nn.BatchNorm1d(512), nn.SiLU(inplace=True),
            nn.Dropout(p * 0.5), nn.Linear(512, ncls))

    def features(self, x):
        """Returns the three scale feature maps (needed for Grad-CAM)."""
        x = self.stem(x)
        c3 = F.relu(self.block2(x), inplace=False)
        c4 = F.relu(self.block3(self.trans2(c3)), inplace=False)
        c5 = F.relu(self.block4(self.trans3(c4)), inplace=False)
        return c3, c4, c5

    def classify(self, c3, c4, c5):
        return self.head(torch.cat([self.s3(c3), self.s4(c4), self.s5(c5)], 1))

    def forward(self, x):
        c3, c4, c5 = self.features(x)
        return self.classify(c3, c4, c5)


def load_model(ckpt_path, device, prefer="auto"):
    """
    Load a trained checkpoint saved by the training notebook.
    prefer: 'auto' uses whichever weights scored best at training time,
            or force 'model' / 'ema'.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classes = ck.get("classes")
    if classes is None:
        raise ValueError("Checkpoint has no 'classes' key - retrain or pass CLASS_NAMES manually.")

    key = ck.get("which", "model") if prefer == "auto" else prefer
    state = ck[key] if key in ck else ck["model"]

    model = MSAF_DenseNet201(len(classes), pretrained=False)
    model.load_state_dict(state)
    model.to(device).eval()

    meta = {
        "classes": classes,
        "weights": key,
        "val_accuracy": float(ck.get("acc", 0.0)),
        "epoch": int(ck.get("epoch", -1)) + 1,
    }
    return model, meta
