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

# Render free tier (512MB) can OOM if occlusion is too dense.
ENABLE_HEATMAP = os.getenv("ENABLE_HEATMAP", "0").strip() == "1"

# Heatmap quality vs speed knobs (safe defaults for Render)
# Suggest on Render: PATCH=64, STRIDE=48 (faster than 48/24, smoother than 64/64)
OCCLUSION_PATCH = int(os.getenv("OCCLUSION_PATCH", "64"))
OCCLUSION_STRIDE = int(os.getenv("OCCLUSION_STRIDE", "48"))

# Cleanup knobs
HEATMAP_THR = float(os.getenv("HEATMAP_THR", "0.15"))         # higher = less speckle
HEATMAP_GAMMA = float(os.getenv("HEATMAP_GAMMA", "0.9"))       # <1 boosts hotspots slightly
HEATMAP_BLUR = float(os.getenv("HEATMAP_BLUR", "3.0"))         # stronger blur => less blocky
HEATMAP_ALPHA = float(os.getenv("HEATMAP_ALPHA", "0.5"))       # overlay strength
HEATMAP_BORDER = float(os.getenv("HEATMAP_BORDER", "0.12"))    # 12% border masked out

app = FastAPI()

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
print("[Backend] OCCLUSION_PATCH:", OCCLUSION_PATCH, "OCCLUSION_STRIDE:", OCCLUSION_STRIDE)

# --------------------------------------------------
# Utility helpers
# --------------------------------------------------
def read_imagefile(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def prepare_array_for_model(img_np: np.ndarray) -> np.ndarray:
    arr = img_np.astype("float32")
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_prob_pneumonia(img_np: np.ndarray) -> float:
    batch = prepare_array_for_model(img_np)
    prob = float(model.predict(batch, verbose=0)[0][0])
    return prob


def _mask_borders(hm: np.ndarray, border_frac: float) -> np.ndarray:
    """Zero out a border region (prevents hotspots on corners/text/borders)."""
    if border_frac <= 0:
        return hm
    H, W = hm.shape
    by = int(H * border_frac)
    bx = int(W * border_frac)
    hm[:by, :] = 0.0
    hm[-by:, :] = 0.0
    hm[:, :bx] = 0.0
    hm[:, -bx:] = 0.0
    return hm


def _remove_small_blobs(hm: np.ndarray, thr: float) -> np.ndarray:
    """
    Cheap blob cleanup without extra deps:
    - threshold
    - use max-filter then keep only regions that survive local max smoothing
    This removes isolated speckles but keeps bigger areas.
    """
    if thr <= 0:
        return hm

    x = np.where(hm >= thr, hm, 0.0)

    # Max-filter (approx) using dilation via PIL
    x_img = Image.fromarray((x * 255).astype(np.uint8))
    # MaxFilter size 5 is a decent cheap denoise
    x_img = x_img.filter(ImageFilter.MaxFilter(size=5))
    x2 = np.array(x_img).astype(np.float32) / 255.0

    # Keep only areas that have support after max-filter
    x = np.where(x2 > 0, x, 0.0)

    m = float(x.max())
    if m > 0:
        x = x / m
    return x


def make_occlusion_heatmap(
    img_np_224: np.ndarray,
    patch_size: int,
    stride: int,
    thr: float = 0.15,
    gamma: float = 0.9,
    blur_radius: float = 3.0,
    border_frac: float = 0.12,
) -> np.ndarray:
    """
    Fast-ish occlusion map with:
    - bicubic upsample (smoother)
    - border masking (reduces "wrong place" highlights)
    - small blob cleanup (less speckle)
    - gaussian blur (reduces blockiness)
    """
    base_pred = predict_prob_pneumonia(img_np_224)

    H, W, _ = img_np_224.shape
    out_h = (H - patch_size) // stride + 1
    out_w = (W - patch_size) // stride + 1

    heatmap = np.zeros((out_h, out_w), dtype=np.float32)

    for i in range(out_h):
        y = i * stride
        for j in range(out_w):
            x = j * stride
            occluded = img_np_224.copy()
            occluded[y : y + patch_size, x : x + patch_size, :] = 0.0
            occl_pred = predict_prob_pneumonia(occluded)
            heatmap[i, j] = base_pred - occl_pred

    heatmap = np.maximum(heatmap, 0.0)
    maxv = float(heatmap.max())
    if maxv > 0:
        heatmap = heatmap / maxv

    # Upsample smoother
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        (IMG_SIZE, IMG_SIZE),
        method="bicubic",
    ).numpy().squeeze()

    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)

    # Mask borders (big improvement for "wrong place" highlights)
    heatmap_resized = _mask_borders(heatmap_resized, border_frac)

    # Remove speckle blobs
    heatmap_resized = _remove_small_blobs(heatmap_resized, thr)

    # Gamma shaping
    if gamma is not None and gamma > 0:
        heatmap_resized = np.power(heatmap_resized, gamma)

    # Smooth edges
    if blur_radius is not None and blur_radius > 0:
        hm_img = Image.fromarray((heatmap_resized * 255).astype(np.uint8))
        hm_img = hm_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        heatmap_resized = np.array(hm_img).astype(np.float32) / 255.0

    return np.clip(heatmap_resized, 0.0, 1.0)


def overlay_heatmap_jet(
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
    return {
        "status": "ok",
        "heatmap_enabled": ENABLE_HEATMAP,
        "occlusion_patch": OCCLUSION_PATCH,
        "occlusion_stride": OCCLUSION_STRIDE,
        "heatmap_thr": HEATMAP_THR,
        "heatmap_blur": HEATMAP_BLUR,
        "heatmap_border": HEATMAP_BORDER,
    }


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
            heatmap = make_occlusion_heatmap(
                img_np_224=img_np,
                patch_size=OCCLUSION_PATCH,
                stride=OCCLUSION_STRIDE,
                thr=HEATMAP_THR,
                gamma=HEATMAP_GAMMA,
                blur_radius=HEATMAP_BLUR,
                border_frac=HEATMAP_BORDER,
            )
            overlay = overlay_heatmap_jet(pil_img, heatmap, alpha=HEATMAP_ALPHA)
            heatmap_b64 = pil_to_base64(overlay)

        original_b64 = pil_to_base64(pil_img)

        return JSONResponse(
            {
                "prediction": prediction,
                "confidence": confidence,
                "heatmap": heatmap_b64,
                "original_image": original_b64,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        print("Unexpected backend error:", repr(e))
        raise HTTPException(status_code=500, detail="Internal server error")