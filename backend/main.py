from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import io
import base64
import numpy as np
from PIL import Image, ImageFilter
import os
import json

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

import matplotlib.cm as cm  # for "jet" colormap like Colab

IMG_SIZE = 224

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this for production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Load model + label map
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "pneumonia_model.keras")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")

print(f"[Backend] Loading model from {MODEL_PATH}")
model = load_model(MODEL_PATH)

if os.path.exists(LABEL_MAP_PATH):
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
else:
    label_map = {0: "Normal", 1: "Pneumonia"}

print("[Backend] Label map:", label_map)


# --------------------------------------------------
# Utility helpers
# --------------------------------------------------
def read_imagefile(file_bytes: bytes) -> Image.Image:
    """Load bytes into an RGB PIL image."""
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def prepare_array_for_model(img_np: np.ndarray) -> np.ndarray:
    """
    Take a (H, W, 3) float32 image in [0,255],
    apply EfficientNet preprocessing and add batch dim.
    """
    arr = img_np.astype("float32")
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr


def predict_prob_pneumonia(img_np: np.ndarray) -> float:
    """
    Run model on a single (H, W, 3) image array and
    return pneumonia probability (class 1).
    """
    batch = prepare_array_for_model(img_np)
    prob = float(model.predict(batch, verbose=0)[0][0])
    return prob


def make_occlusion_heatmap_colab_style(
    img_np_224: np.ndarray,
    patch_size: int = 32,
    stride: int = 16,
    # --- Noise cleanup knobs (tune these) ---
    thr: float = 0.18,          # raise to remove more noise, lower to keep more signal
    gamma: float = 0.85,        # <1 boosts hotspots, >1 softens
    blur_radius: float = 1.2,   # 0 disables blur
) -> np.ndarray:
    """
    Colab-style occlusion sensitivity + optional noise cleanup.

    Colab logic:
      - baseline pred on original image (224x224)
      - slide black occlusion patch
      - heatmap[i,j] = base_pred - occl_pred
      - clamp >=0, normalize by max
      - resize back to (224,224) using tf.image.resize

    Added cleanup:
      - threshold weak values
      - re-normalize after threshold
      - gamma shaping
      - light gaussian blur for smoother heatmaps
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
            occluded[y:y + patch_size, x:x + patch_size, :] = 0.0

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

    # -----------------------------
    # Noise removal / smoothing
    # -----------------------------
    if thr is not None and thr > 0:
        heatmap_resized = np.where(heatmap_resized >= thr, heatmap_resized, 0.0)

        m = float(heatmap_resized.max())
        if m > 0:
            heatmap_resized = heatmap_resized / m

    if gamma is not None and gamma > 0:
        heatmap_resized = np.power(heatmap_resized, gamma)

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
    """
    Matches Colab visualization:
      plt.imshow(orig_arr, cmap="gray")
      plt.imshow(heatmap_resized, cmap="jet", alpha=0.5)

    We produce a single blended RGB image for the frontend.
    """
    base_gray = orig_img.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    base_gray = base_gray.convert("RGB")
    base = np.array(base_gray).astype(np.float32) / 255.0  # (224,224,3)

    hm = np.clip(heatmap_224, 0.0, 1.0)
    jet = cm.get_cmap("jet")(hm)[..., :3]  # (224,224,3) in [0,1]

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
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_img = read_imagefile(contents)

        # --- 1) Classification (224x224) ---
        img_resized = pil_img.resize((IMG_SIZE, IMG_SIZE))
        img_np = np.array(img_resized).astype("float32")  # [0..255]
        prob = predict_prob_pneumonia(img_np)

        label_idx = int(prob >= 0.5)
        prediction = label_map.get(label_idx, "Pneumonia")
        confidence = prob if label_idx == 1 else 1.0 - prob

        # --- 2) Occlusion heatmap (Colab style + noise cleanup) ---
        heatmap = make_occlusion_heatmap_colab_style(
            img_np_224=img_np,
            patch_size=32,
            stride=16,
            thr=0.18,
            gamma=0.85,
            blur_radius=1.2,
        )
        overlay = overlay_heatmap_jet_like_colab(pil_img, heatmap, alpha=0.5)
        heatmap_b64 = pil_to_base64(overlay)

        # --- 3) Original image (frontend display) ---
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
