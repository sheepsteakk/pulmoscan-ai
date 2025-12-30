import os

# Reduce TF log noise on Render
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import io
import base64
import json
import numpy as np
from PIL import Image, ImageFilter

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

import matplotlib.cm as cm

IMG_SIZE = 224

# Toggle heavy explanation (occlusion heatmap)
# Render free tier (512MB) can OOM if occlusion grid is too fine.
ENABLE_HEATMAP = os.getenv("ENABLE_HEATMAP", "0").strip() == "1"

app = FastAPI()

# CORS: allow your GitHub Pages site + local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://sheepsteakk.github.io",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Load model + label map
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # backend/
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "pneumonia_model.keras")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")

print(f"[Backend] Loading model from {MODEL_PATH}")
model = load_model(MODEL_PATH)

if os.path.exists(LABEL_MAP_PATH):
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
else:
    label_map = {0: "Normal", 1: "Pneumonia"}

print("[Backend] Label map:", label_map)
print("[Backend] ENABLE_HEATMAP:", ENABLE_HEATMAP)

# --------------------------------------------------
# Utility helpers
# --------------------------------------------------
def read_imagefile(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def prepare_array_for_model(img_np: np.ndarray) -> np.ndarray:
    arr = img_np.astype("float32")
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr


def predict_prob_pneumonia(img_np: np.ndarray) -> float:
    batch = prepare_array_for_model(img_np)
    prob = float(model.predict(batch, verbose=0)[0][0])
    return prob


def make_occlusion_heatmap_colab_style(
    img_np_224: np.ndarray,
    patch_size: int = 48,
    stride: int = 24,
    thr: float = 0.10,
    gamma: float = 0.85,
    blur_radius: float = 2.0,
) -> np.ndarray:
    """
    Occlusion sensitivity map:
      heat = base_pred - occluded_pred, clamped to >=0 and normalized.

    patch_size + stride control smoothness:
      - smaller stride = less blocky but more compute/memory
      - Render free tier may OOM if too fine, so keep it moderate
    """
    base_pred = predict_prob_pneumonia(img_np_224)

    H, W, _ = img_np_224.shape
    out_h = (H - patch_size) // stride + 1
    out_w = (W - patch_size) // stride + 1
    heatmap = np.zeros((out_h, out_w), dtype=np.float32)

    for i in range(out_h):
        for j in range(out_w):
            y = i * stride
            x = j * stride

            occluded = img_np_224.copy()
            occluded[y : y + patch_size, x : x + patch_size, :] = 0.0

            occl_pred = predict_prob_pneumonia(occluded)
            heatmap[i, j] = base_pred - occl_pred

    heatmap = np.maximum(heatmap, 0.0)
    maxv = float(heatmap.max())
    if maxv > 0:
        heatmap = heatmap / maxv

    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        (IMG_SIZE, IMG_SIZE),
        method="bilinear",
    ).numpy().squeeze()

    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)

    # threshold weak noise
    if thr is not None and thr > 0:
        heatmap_resized = np.where(heatmap_resized >= thr, heatmap_resized, 0.0)
        m = float(heatmap_resized.max())
        if m > 0:
            heatmap_resized = heatmap_resized / m

    # gamma shaping
    if gamma is not None and gamma > 0:
        heatmap_resized = np.power(heatmap_resized, gamma)

    # blur for smoother look
    if blur_radius is not None and blur_radius > 0:
        hm_img = Image.fromarray((heatmap_resized * 255).astype(np.uint8))
        hm_img = hm_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        heatmap_resized = np.array(hm_img).astype(np.float32) / 255.0

    return np.clip(heatmap_resized, 0.0, 1.0)


def overlay_heatmap_jet_like_colab(
    orig_img: Image.Image,
    heatmap_224: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    base_gray = orig_img.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    base_gray = base_gray.convert("RGB")
    base = np.array(base_gray).astype(np.float32) / 255.0

    hm = np.clip(heatmap_224, 0.0, 1.0)
    jet = cm.get_cmap("jet")(hm)[..., :3]

    blended = (1.0 - alpha) * base + alpha * jet
    blended = np.clip(blended, 0.0, 1.0)

    out = Image.fromarray((blended * 255).astype(np.uint8), mode="RGB")
    out = out.resize(orig_img.size, Image.BILINEAR)
    return out


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# --------------------------------------------------
# Endpoints
# --------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "heatmap_enabled": ENABLE_HEATMAP}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_img = read_imagefile(contents)

        img_resized = pil_img.resize((IMG_SIZE, IMG_SIZE))
        img_np = np.array(img_resized).astype("float32")
        prob = predict_prob_pneumonia(img_np)

        label_idx = int(prob >= 0.5)
        prediction = label_map.get(label_idx, "Pneumonia")
        confidence = prob if label_idx == 1 else 1.0 - prob

        heatmap_b64 = None
        if ENABLE_HEATMAP:
            # less blocky than 64/64, but still moderate for Render memory
            heatmap = make_occlusion_heatmap_colab_style(
                img_np_224=img_np,
                patch_size=48,
                stride=24,
                thr=0.10,
                gamma=0.85,
                blur_radius=2.0,
            )
            overlay = overlay_heatmap_jet_like_colab(pil_img, heatmap, alpha=0.5)
            heatmap_b64 = pil_to_base64(overlay)

        original_b64 = pil_to_base64(pil_img)

        return JSONResponse(
            {
                "prediction": prediction,
                "confidence": confidence,
                "heatmap": None,  # null if disabled
                "original_image": None,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        print("Unexpected backend error:", repr(e))
        raise HTTPException(status_code=500, detail="Internal server error")