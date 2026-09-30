"""
Groundnut Leaf Disease Detection — Flask web app
Serves the proposed MSAF-DenseNet201 model.

Run:  python app.py       then open http://127.0.0.1:5000
"""
import base64
import io
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import transforms

# model.py lives in ../src ; make it importable no matter where this is run from
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "..", "src"), HERE):
    if os.path.isfile(os.path.join(p, "model.py")):
        sys.path.insert(0, p)
        break

from model import IMG_SIZE, MEAN, STD, load_model

# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
def find_checkpoint():
    """Look for msaf_best.pt next to app.py, at the repo root, or in ./models."""
    env = os.environ.get("MSAF_CKPT")
    if env:
        return env
    for cand in (os.path.join(HERE, "msaf_best.pt"),
                 os.path.join(HERE, "..", "msaf_best.pt"),
                 os.path.join(HERE, "..", "models", "msaf_best.pt"),
                 "msaf_best.pt"):
        if os.path.isfile(cand):
            return cand
    sys.exit(
        "\nERROR: msaf_best.pt not found.\n"
        "Download the trained weights and place the file next to app.py,\n"
        "or set the path explicitly:  export MSAF_CKPT=/path/to/msaf_best.pt\n")


CKPT_PATH = find_checkpoint()
USE_TTA = True                 # 4-way flip TTA, same as the reported evaluation
MAX_UPLOAD_MB = 16

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"}


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():      # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device()
print(f"[init] device: {DEVICE}")
print(f"[init] loading checkpoint: {CKPT_PATH}")
MODEL, META = load_model(CKPT_PATH, DEVICE)
CLASSES = META["classes"]
print(f"[init] ready — {len(CLASSES)} classes: {CLASSES}")
print(f"[init] weights={META['weights']} val_acc={META['val_accuracy']:.4f} epoch={META['epoch']}")

preprocess = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# --------------------------------------------------------------------------
# agronomy notes shown alongside the prediction
# --------------------------------------------------------------------------
DISEASE_INFO = {
    "early_leaf_spot": {
        "label": "Early Leaf Spot",
        "pathogen": "Passalora arachidicola (syn. Cercospora arachidicola)",
        "symptoms": "Small brown-to-black circular lesions on the upper leaf surface, "
                    "usually surrounded by a yellow halo. Appears first on older, "
                    "lower leaves roughly 30–45 days after sowing.",
        "management": "Rotate out of groundnut for at least two seasons, remove and destroy "
                      "crop residue, avoid dense canopies that hold leaf wetness, and use "
                      "resistant varieties where available. Foliar fungicides are effective "
                      "when applied early — consult your local agricultural extension office "
                      "for products and schedules approved in your region.",
        "severity": "moderate",
    },
    "late leaf spot": {
        "label": "Late Leaf Spot",
        "pathogen": "Nothopassalora personata (syn. Cercosporidium personatum)",
        "symptoms": "Darker, near-black lesions, typically without the yellow halo seen in "
                    "early leaf spot, and often more visible on the lower leaf surface. "
                    "Heavy infection causes severe defoliation late in the season.",
        "management": "Same cultural controls as early leaf spot. Because it strikes later, "
                      "protecting the canopy through pod-fill matters most for yield. "
                      "Check with local extension services for a spray programme.",
        "severity": "high",
    },
    "rust": {
        "label": "Groundnut Rust",
        "pathogen": "Puccinia arachidis",
        "symptoms": "Orange-brown raised pustules, mainly on the underside of leaflets, "
                    "which rupture and release rusty spores. Leaves dry out and stay "
                    "attached to the plant rather than dropping.",
        "management": "Rust spreads on wind-borne spores, so scout neighbouring fields early. "
                      "Destroy volunteer groundnut plants that carry the pathogen between "
                      "seasons, and plant resistant cultivars where available.",
        "severity": "high",
    },
    "nutrition deficiency": {
        "label": "Nutrition Deficiency",
        "pathogen": "Abiotic — not an infectious disease",
        "symptoms": "Yellowing (chlorosis) between leaf veins while the veins stay green, "
                    "or general pale colour and stunted growth. Iron deficiency in calcareous "
                    "soils is a common cause in groundnut.",
        "management": "Confirm with a soil test before treating — the visual symptoms of "
                      "iron, manganese and sulphur deficiency overlap heavily. Correct soil "
                      "pH where it is the underlying cause; foliar micronutrient sprays give "
                      "a faster but temporary response.",
        "severity": "low",
    },
    "healthy leaf": {
        "label": "Healthy",
        "pathogen": "—",
        "symptoms": "Uniform green colour, no lesions, pustules or chlorosis.",
        "management": "No action needed. Keep scouting weekly — early leaf spot is easiest "
                      "to control before lesions become widespread.",
        "severity": "none",
    },
}


def info_for(cls_name):
    key = cls_name.strip().lower()
    for k, v in DISEASE_INFO.items():
        if k.strip().lower() == key:
            return v
    return {"label": cls_name, "pathogen": "—", "symptoms": "No reference notes available "
            "for this class.", "management": "—", "severity": "unknown"}


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------
@torch.no_grad()
def predict(tensor):
    """tensor: 1x3xHxW on DEVICE. Returns probability vector (numpy)."""
    p = torch.softmax(MODEL(tensor).float(), 1)
    if USE_TTA:
        p = p + torch.softmax(MODEL(torch.flip(tensor, [3])).float(), 1)
        p = p + torch.softmax(MODEL(torch.flip(tensor, [2])).float(), 1)
        p = p + torch.softmax(MODEL(torch.flip(tensor, [2, 3])).float(), 1)
        p = p / 4
    return p[0].cpu().numpy()


def grad_cam(tensor, class_idx):
    """
    Grad-CAM over the deepest scale (dense block 4). Needs gradients,
    so this deliberately runs outside torch.no_grad().
    """
    MODEL.zero_grad(set_to_none=True)
    c3, c4, c5 = MODEL.features(tensor)
    c5.retain_grad()
    out = MODEL.classify(c3, c4, c5)
    out[0, class_idx].backward()

    grads = c5.grad                                  # 1,C,H,W
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * c5).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE),
                        mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    lo, hi = cam.min(), cam.max()
    return (cam - lo) / (hi - lo + 1e-8)


def jet(x):
    """Minimal jet-style colormap, avoids a matplotlib dependency."""
    x = np.clip(x, 0, 1)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], -1)


def overlay_cam(pil_img, cam, alpha=0.65):
    """Blend the heatmap in proportional to activation, so quiet regions of the
    leaf stay visible instead of being washed out to dark blue."""
    base = pil_img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    arr = np.asarray(base).astype(np.float32) / 255.0
    heat = jet(cam)
    a = (alpha * cam)[..., None]
    mixed = (1 - a) * arr + a * heat
    return Image.fromarray((np.clip(mixed, 0, 1) * 255).astype(np.uint8))


def to_data_uri(pil_img, fmt="JPEG", quality=88):
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format=fmt, quality=quality)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def run_inference(pil_img, want_cam=True):
    t0 = time.time()
    tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)

    probs = predict(tensor)
    order = np.argsort(-probs)
    top = int(order[0])

    cam_uri = None
    if want_cam:
        try:
            cam = grad_cam(tensor.clone().requires_grad_(False), top)
            cam_uri = to_data_uri(overlay_cam(pil_img, cam))
        except Exception as e:            # never let the visual break the prediction
            print("[warn] grad-cam failed:", e)

    ranked = [{"class": CLASSES[i],
               "label": info_for(CLASSES[i])["label"],
               "probability": float(probs[i])} for i in order]

    return {
        "prediction": CLASSES[top],
        "label": info_for(CLASSES[top])["label"],
        "confidence": float(probs[top]),
        "ranked": ranked,
        "info": info_for(CLASSES[top]),
        "gradcam": cam_uri,
        "original": to_data_uri(pil_img.resize((IMG_SIZE, IMG_SIZE))),
        "inference_ms": round((time.time() - t0) * 1000, 1),
        "device": str(DEVICE),
        "tta": USE_TTA,
    }


def read_upload(file_storage):
    name = (file_storage.filename or "").lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else ""
    if ext not in ALLOWED:
        raise ValueError(f"Unsupported file type '.{ext}'. Use: {', '.join(sorted(ALLOWED))}")
    img = Image.open(file_storage.stream)
    img.load()
    return img.convert("RGB")


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", classes=CLASSES, meta=META,
                           device=str(DEVICE), n_classes=len(CLASSES))


@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"error": "No image uploaded."}), 400
    try:
        img = read_upload(request.files["image"])
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({"error": "Could not read that image — it may be corrupt."}), 400

    want_cam = request.form.get("gradcam", "1") != "0"
    try:
        return jsonify(run_inference(img, want_cam))
    except Exception as e:
        print("[error]", e)
        return jsonify({"error": f"Inference failed: {e}"}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Same as /predict but never returns images — for scripting."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400
    try:
        img = read_upload(request.files["image"])
        r = run_inference(img, want_cam=False)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    r.pop("original", None)
    r.pop("gradcam", None)
    return jsonify(r)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": str(DEVICE),
                    "classes": CLASSES, "model": "MSAF-DenseNet201", **META})


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"File too large. Limit is {MAX_UPLOAD_MB} MB."}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
