import base64
import io
import math
import os
import time
import uuid

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, Security, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image

from gradcam import generate_heatmap_with_cam
from model_loader import get_model

app = FastAPI(title="Creative Intelligence Scorer")
security = HTTPBearer()

BEARER_TOKEN = os.environ["API_TOKEN"]

ERROR_MESSAGES = {
    "model_not_loaded": "Model is warming up. Please retry in 30 seconds.",
    "invalid_image": "Could not process image. Please upload a JPG or PNG.",
    "inference_failed": "Scoring failed. Please try again.",
}

BENCHMARKS = {
    "gaming":    {"median_ctr": 0.119, "median_halflife": 10.7, "sample_size": 7485},
    "ecommerce": {"median_ctr": 0.125, "median_halflife": 11.2, "sample_size": 7078},
    "finance":   {"median_ctr": 0.111, "median_halflife": 10.0, "sample_size": 6520},
    "other":     {"median_ctr": 0.118, "median_halflife": 10.6, "sample_size": 1000},
}

GRID_LABELS = [
    "top-left", "top-center", "top-right",
    "mid-left", "center",     "mid-right",
    "bot-left", "bot-center", "bot-right",
]


def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> None:
    if credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")


def _open_image(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail=ERROR_MESSAGES["invalid_image"])


def _cam_region_labels(cam_16: np.ndarray) -> tuple[list[str], list[str]]:
    """3×3 grid aggregation over a 16×16 cam → top-3 high and top-3 low cells."""
    h, w = cam_16.shape
    cells = []
    for row in range(3):
        for col in range(3):
            r0, r1 = row * h // 3, (row + 1) * h // 3
            c0, c1 = col * w // 3, (col + 1) * w // 3
            cells.append(float(cam_16[r0:r1, c0:c1].mean()))

    ranked = sorted(range(9), key=lambda i: cells[i], reverse=True)
    return (
        [GRID_LABELS[i] for i in ranked[:3]],
        [GRID_LABELS[i] for i in ranked[-3:]],
    )


@app.on_event("startup")
async def startup() -> None:
    # Pre-load model so first request doesn't time out
    try:
        get_model()
    except Exception:
        pass  # health endpoint will report model_loaded=False; don't crash the server


@app.get("/health")
async def health() -> dict:
    try:
        model, _ = get_model()
        loaded = model is not None
    except Exception:
        loaded = False
    return {"status": "ok", "model_loaded": loaded}


@app.get("/benchmark")
async def benchmark(vertical: str = Query(...)) -> dict:
    key = vertical.lower()
    if key not in BENCHMARKS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown vertical '{vertical}'. Valid: {list(BENCHMARKS)}",
        )
    return {"vertical": key, **BENCHMARKS[key]}


@app.post("/score")
async def score(
    image: UploadFile = File(...),
    vertical: str = Form(...),
    _: None = Security(verify_token),
) -> dict:
    try:
        model, processor = get_model()
    except Exception:
        raise HTTPException(status_code=503, detail=ERROR_MESSAGES["model_not_loaded"])

    try:
        pil_image = _open_image(await image.read())
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=422, detail=ERROR_MESSAGES["invalid_image"])

    try:
        t0 = time.monotonic()

        inputs = processor(images=pil_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"]  # (1, 3, 224, 224), CPU

        with torch.no_grad():
            clip_out = model.clip(pixel_values=pixel_values)
            embedding = clip_out.pooler_output  # (1, 768)
            outputs = model(embedding=embedding)

        ctr_score = float(outputs["ctr_score"].squeeze())

        log_scale = float(outputs["weibull_params"][0, 0].clamp(-10, 10))
        log_shape = float(outputs["weibull_params"][0, 1].clamp(-10, 10))
        scale = math.exp(log_scale)
        shape = math.exp(log_shape)
        # Weibull median = scale * ln(2)^(1/shape) — survival-theoretic halflife
        halflife_days = scale * (math.log(2) ** (1.0 / shape))

        # Confidence proxy: certainty peaks when prediction is near 0 or 1
        confidence = float(1.0 - 2.0 * abs(ctr_score - 0.5))

        inference_ms = int((time.monotonic() - t0) * 1000)

    except Exception:
        raise HTTPException(status_code=500, detail=ERROR_MESSAGES["inference_failed"])

    return {
        "ad_id": str(uuid.uuid4()),
        "ctr_score": round(ctr_score, 4),
        "halflife_days": round(halflife_days, 2),
        "confidence": round(confidence, 4),
        "inference_ms": inference_ms,
    }


@app.post("/heatmap")
async def heatmap(
    image: UploadFile = File(...),
    _: None = Security(verify_token),
) -> dict:
    try:
        model, _ = get_model()
    except Exception:
        raise HTTPException(status_code=503, detail=ERROR_MESSAGES["model_not_loaded"])

    try:
        pil_image = _open_image(await image.read())
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=422, detail=ERROR_MESSAGES["invalid_image"])

    try:
        # Single pass: returns overlay (for base64) + raw cam_16 (for region labels)
        overlay, cam_16 = generate_heatmap_with_cam(model, pil_image, device="cpu")

        high_attention, low_attention = _cam_region_labels(cam_16)

        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG")
        heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    except Exception:
        raise HTTPException(status_code=500, detail=ERROR_MESSAGES["inference_failed"])

    return {
        "heatmap_b64": heatmap_b64,
        "high_attention": high_attention,
        "low_attention": low_attention,
    }
