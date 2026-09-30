"""Export MSAF-DenseNet201 to ONNX for the in-browser demo (docs/index.html).

    python scripts/export_onnx.py [msaf_best.pt]

Writes docs/demo/model/msaf_web_fp16.onnx: input 1x3x256x256 (normalised),
outputs probs (1x5, 4-way flip TTA averaged) and attention (1x32x32).
"""
import os, sys, torch, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from model import load_model

CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "msaf_best.pt")
OUT = os.path.join(ROOT, "docs", "demo", "model")
os.makedirs(OUT, exist_ok=True)
net, meta = load_model(CKPT, "cpu")
print(meta)

def attn_map(branch, feat):
    """softmax attention weights of a ScaleBranch's AttnPool, as B x 1 x H x W"""
    x = branch.cbam(branch.proj(feat))
    b, c, h, w = x.shape
    return torch.softmax(branch.pool.score(x).view(b, 1, h * w), -1).view(b, 1, h, w)

class Web(torch.nn.Module):
    """input: 1x3x256x256 normalised. outputs: probs (1x5, 4-way flip TTA), attention (1x32x32)"""
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x):
        xs = torch.cat([x, x.flip(3), x.flip(2), x.flip(2).flip(3)], 0)
        c3, c4, c5 = self.m.features(xs)
        logits = self.m.classify(c3, c4, c5)
        probs = torch.softmax(logits, 1).mean(0, keepdim=True)
        # attention maps of the un-flipped view, all three scales brought to 32x32
        a3 = attn_map(self.m.s3, c3[:1])
        a4 = F.interpolate(attn_map(self.m.s4, c4[:1]), size=a3.shape[-2:], mode="bilinear", align_corners=False)
        a5 = F.interpolate(attn_map(self.m.s5, c5[:1]), size=a3.shape[-2:], mode="bilinear", align_corners=False)
        def nrm(a):
            a = a - a.amin(dim=(2, 3), keepdim=True)
            return a / (a.amax(dim=(2, 3), keepdim=True) + 1e-8)
        att = (nrm(a3) + nrm(a4) + nrm(a5)) / 3
        return probs, att[:, 0]

w = Web(net).eval()
x = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    p, a = w(x)
print(p.shape, a.shape, p.sum().item())
tmp = os.path.join(OUT, "_fp32.onnx")
torch.onnx.export(w, x, tmp, input_names=["image"], output_names=["probs", "attention"],
                  opset_version=17, dynamo=False, do_constant_folding=True)
import onnx
m = onnx.load(tmp); onnx.checker.check_model(m)
from onnxconverter_common import float16
m16 = float16.convert_float_to_float16(m, keep_io_types=True)
dst = os.path.join(OUT, "msaf_web_fp16.onnx")
onnx.save(m16, dst); os.remove(tmp)
print("wrote", dst, round(os.path.getsize(dst) / 1e6, 1), "MB")
