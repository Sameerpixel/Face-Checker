"""
FGAN Face Check: local web app for the trained detector.

Run (Windows, from the project folder, with .venv activated):
    pip install fastapi uvicorn python-multipart
    python app.py
Open:  http://127.0.0.1:8000

Expects the trained weights at outputs/detector.pt (saved by fgan_tonight.py).
If the file is missing or does not load, the app starts in DEMO MODE with placeholder
numbers and says so clearly in the UI.
"""
import io
import base64
import hashlib
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms as T
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import uvicorn

# Optional face check. If OpenCV is not installed the app still works, but it cannot tell
# a face from a blank screen. To turn it on:  pip install "opencv-python-headless<5"
try:
    import cv2
    import numpy as np
    _cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    FACE_CHECK = not _cascade.empty()
except Exception:
    FACE_CHECK = False

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "outputs" / "detector.pt"
INDEX_PATH = BASE / "static" / "index.html"
MAX_BYTES = 15 * 1024 * 1024
UNSURE_BAND = (0.35, 0.65)   # AI-generated score inside this range = "inconclusive"

dev = "cuda" if torch.cuda.is_available() else "cpu"


# Same architecture as the training script (layer order must match the saved weights).
def blk(i, o, bn=True):
    return [nn.Conv2d(i, o, 4, 2, 1, bias=not bn)] + ([nn.BatchNorm2d(o)] if bn else []) + [nn.LeakyReLU(.2, True)]


D = nn.Sequential(*blk(3, 64, False), *blk(64, 128), *blk(128, 256), *blk(256, 512),
                  nn.Flatten(), nn.Dropout(.3), nn.Linear(512 * 4 * 4, 1)).to(dev)

DEMO, DEMO_REASON = False, ""
if not MODEL_PATH.exists():
    DEMO, DEMO_REASON = True, "No trained model found at outputs/detector.pt."
else:
    try:
        D.load_state_dict(torch.load(MODEL_PATH, map_location=dev, weights_only=True))
        D.eval()
        print("Loaded", MODEL_PATH, "on", dev)
    except Exception as e:  # architecture mismatch, corrupt file, ...
        DEMO, DEMO_REASON = True, f"The model file could not be loaded ({type(e).__name__})."
if DEMO:
    print("WARNING: DEMO MODE.", DEMO_REASON)

# Measured results from the final run (README): 2,000-image validation subset
# (1,007 AI-generated, 993 real), 5 GAN epochs + 15 detector epochs, one seed.
MODEL_CARD = {
    "n_valid": 2000, "n_fake": 1007, "n_real": 993,
    "acc_clean": 0.875, "acc_jpeg30": 0.868,
    "fake_recall": 0.914,                   # 920 / 1007 AI-generated images caught
    "real_false_alarm": round(163 / 993, 3),  # 163 / 993 real faces flagged as AI
}

# Matches training preprocessing: resize shorter side to 64, centre crop, scale to [-1, 1].
crop_tf = T.Compose([T.Resize(64), T.CenterCrop(64)])
tf = T.Compose([crop_tf, T.ToTensor(), T.Normalize((.5,) * 3, (.5,) * 3)])


def crop_png(img: Image.Image) -> str:
    """The exact 64x64 crop the model is given, as a data URL for the page to display."""
    b = io.BytesIO()
    crop_tf(img).save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def score_ai(img: Image.Image, salt: bytes = b"") -> float:
    """AI-generated score in 0..1. The network outputs a logit for REAL (label 1)."""
    if DEMO:  # placeholder derived from the pixels, NOT a real prediction
        h = hashlib.md5(img.tobytes()[:4096] + salt).digest()[0] / 255
        return round(0.05 + 0.9 * h, 4)
    with torch.no_grad():
        logit = D(tf(img).unsqueeze(0).to(dev)).item()
    return float(1 - torch.sigmoid(torch.tensor(logit)).item())


def jpeg(img: Image.Image, q: int) -> Image.Image:
    b = io.BytesIO()
    img.save(b, "JPEG", quality=q)
    b.seek(0)
    return Image.open(b).convert("RGB")


def face_status(img: Image.Image) -> str:
    """Is there a face in the centre square the model looks at?
    'found' | 'none' | 'outside_crop' | 'unchecked' (OpenCV not installed).
    This is a simple Haar-cascade detector: in a quick test it found a face in about 93% of
    dataset faces and in none of the blank/noise/gradient images, so it can miss real faces."""
    if not FACE_CHECK:
        return "unchecked"
    g = np.array(img.convert("L"))
    h, w = g.shape
    sc = 640 / max(h, w) if max(h, w) > 640 else 1.0       # shrink big images for speed
    if sc != 1.0:
        g = cv2.resize(g, (int(w * sc), int(h * sc)))
    g = cv2.equalizeHist(g)
    m = max(24, int(0.1 * min(g.shape)))                     # ignore tiny detections
    boxes = _cascade.detectMultiScale(g, 1.1, 4, minSize=(m, m))
    if len(boxes) == 0:
        return "none"
    s = min(w, h)
    x0, y0 = (w - s) / 2, (h - s) / 2                        # the square the model sees
    for x, y, bw, bh in boxes:
        cx, cy = (x + bw / 2) / sc, (y + bh / 2) / sc
        if x0 <= cx <= x0 + s and y0 <= cy <= y0 + s:
            return "found"
    return "outside_crop"


def verdict_of(p: float) -> str:
    if UNSURE_BAND[0] <= p <= UNSURE_BAND[1]:
        return "inconclusive"
    return "fake" if p > 0.5 else "real"


VERDICT_WORDS = {"fake": "AI-generated", "real": "real", "inconclusive": "inconclusive"}

app = FastAPI()


@app.get("/")
def index():
    return FileResponse(INDEX_PATH)


@app.get("/status")
def status():
    return {"demo": DEMO, "reason": DEMO_REASON, "device": dev,
            "band": UNSURE_BAND, "model_card": MODEL_CARD, "face_check": FACE_CHECK}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "That file is empty. Choose a different image.")
    if len(raw) > MAX_BYTES:
        raise HTTPException(400, "That image is larger than 15 MB. Choose a smaller one.")
    try:
        src = Image.open(io.BytesIO(raw))
        fmt = src.format or "unknown"
        img = ImageOps.exif_transpose(src).convert("RGB")   # honour phone-photo rotation
    except Exception:
        raise HTTPException(400, "That file isn't an image this app can read. Try a JPG, PNG or WebP.")

    w, h = img.size
    p = score_ai(img)
    p_jpg = score_ai(jpeg(img, 30), b"jpg")
    v, v_jpg = verdict_of(p), verdict_of(p_jpg)
    face = face_status(img)

    notes = []
    if face == "unchecked":
        notes.append("The face check is off, so images without a face are scored too. "
                     "To turn it on, run: pip install \"opencv-python-headless<5\" and restart the app.")
    if min(w, h) < 64:
        notes.append("This image is smaller than 64 pixels on one side, so the result is less reliable.")
    if abs(w - h) > 0.2 * max(w, h):
        notes.append("The image isn't square. The model only looks at a square crop from the centre.")
    if v_jpg != v:
        notes.append(f"After heavy JPEG compression the verdict flips to {VERDICT_WORDS[v_jpg]}. "
                     "Treat this result with extra caution.")
    if fmt.upper() not in ("JPEG", "JPG"):
        notes.append(f"This is a {fmt} file. The training images were JPEGs.")

    return {
        "demo": DEMO,
        "verdict": v,
        "p_fake": round(p, 4),
        "p_fake_jpeg30": round(p_jpg, 4),
        "verdict_jpeg30": v_jpg,
        "stable_under_jpeg": v_jpg == v,
        "width": w, "height": h, "format": fmt,
        "size_kb": round(len(raw) / 1024, 1),
        "crop_png": crop_png(img),
        "face": face,
        "notes": notes,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
