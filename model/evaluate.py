import argparse
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
