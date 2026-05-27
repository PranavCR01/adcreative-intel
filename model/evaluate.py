import argparse
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader

from model.clip_head import CreativeScorer
from model.dataset import CreativeDataset


def _load_model(checkpoint_dir: str) -> CreativeScorer:
    model = CreativeScorer()
    ckpt = Path(checkpoint_dir)
    safetensors = ckpt / "model.safetensors"
    bin_file = ckpt / "pytorch_model.bin"
    if safetensors.exists():
        from safetensors.torch import load_file
        model.load_state_dict(load_file(str(safetensors)))
    elif bin_file.exists():
        model.load_state_dict(torch.load(str(bin_file), map_location="cpu"))
    else:
        raise FileNotFoundError(f"No model weights found in {checkpoint_dir}")
    return model


def infer_single(
    image_path: str,
    vertical: str,
    checkpoint_dir: str = "model/best_alpha",
) -> dict:
    """Single-image inference. vertical is accepted for API contract parity but unused in computation."""
    from PIL import Image
    from transformers import AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(checkpoint_dir).to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)

    with torch.no_grad():
        out = model(pixel_values=pixel_values)

    ctr = float(out["ctr_score"].squeeze())

    # Clamp matches WeibullNLLLoss clamp — never remove
    log_scale = float(out["weibull_params"][0, 0].clamp(-10, 10))
    log_shape = float(out["weibull_params"][0, 1].clamp(-10, 10))
    scale = math.exp(log_scale)   # λ
    shape = math.exp(log_shape)   # k
    # Weibull median: S(t)=0.5 → t = λ * ln(2)^(1/k)
    halflife = scale * (math.log(2) ** (1.0 / shape))

    # Distance from decision boundary → proxy for prediction confidence
    confidence = abs(ctr - 0.5) * 2

    return {
        "ctr_score": round(ctr, 4),
        "halflife_days": round(halflife, 2),
        "confidence": round(confidence, 4),
    }


def run_eval(
    checkpoint_dir: str = "./model/best",
    csv_path: str = "data/labels_clean.csv",
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(checkpoint_dir).to(device)
    model.eval()

    ds = CreativeDataset(csv_path, split="test")
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    ctr_preds, ctr_trues = [], []
    with torch.no_grad():
        for batch in loader:
            emb = batch["embedding"].to(device)
            out = model(embedding=emb)
            ctr_preds.extend(out["ctr_score"].squeeze(-1).cpu().numpy())
            ctr_trues.extend(batch["ctr_score"].numpy())

    ctr_preds = np.array(ctr_preds)
    ctr_trues = np.array(ctr_trues)

    spearman_r, spearman_p = spearmanr(ctr_trues, ctr_preds)
    mae = mean_absolute_error(ctr_trues, ctr_preds)

    print(f"Spearman r:  {spearman_r:.4f}  (p={spearman_p:.4e})")
    print(f"MAE:         {mae:.4f}")

    return {"spearman_r": float(spearman_r), "spearman_p": float(spearman_p), "mae": float(mae)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="./model/best")
    parser.add_argument("--csv", default="data/labels_clean.csv")
    args = parser.parse_args()
    run_eval(args.checkpoint_dir, args.csv)


if __name__ == "__main__":
    main()
