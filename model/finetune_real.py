import argparse
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

from model.clip_head import CreativeScorer
from model.dataset import CreativeDataset, CACHE_PATH
from model.survival import WeibullNLLLoss
from model.train import creative_collator

ALPHA = 0.5


def _best_checkpoint(checkpoint_root: str) -> str:
    # trainer_state.json at the output_dir root carries best_model_checkpoint
    state_file = Path(checkpoint_root) / "trainer_state.json"
    if state_file.exists():
        state = json.loads(state_file.read_text())
        best = state.get("best_model_checkpoint")
        if best and Path(best).exists():
            return best
    # Fallback: highest-numbered checkpoint
    ckpts = sorted(
        Path(checkpoint_root).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_root}")
    return str(ckpts[-1])


def _load_model(checkpoint_dir: str) -> CreativeScorer:
    model = CreativeScorer()
    ckpt = Path(checkpoint_dir)
    safetensors_path = ckpt / "model.safetensors"
    bin_path = ckpt / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file
        model.load_state_dict(load_file(str(safetensors_path)))
    elif bin_path.exists():
        model.load_state_dict(torch.load(str(bin_path), map_location="cpu"))
    else:
        raise FileNotFoundError(f"No model weights found in {checkpoint_dir}")
    return model


class RealAdsDataset(Dataset):
    """labels_clean.csv filtered to source='apify', then 80/10/10 split."""

    def __init__(self, csv_path: str, split: str = "train", cache_path: str = CACHE_PATH):
        df = pd.read_csv(csv_path)
        df = df[df["source"] == "apify"].reset_index(drop=True)

        if not Path(cache_path).exists():
            raise FileNotFoundError(
                f"Embedding cache not found at {cache_path}. Run dataset.py first."
            )
        self.cache = torch.load(cache_path, map_location="cpu")

        n = len(df)
        if split == "train":
            df = df.iloc[: int(0.8 * n)]
        elif split == "val":
            df = df.iloc[int(0.8 * n) : int(0.9 * n)]
        elif split == "test":
            df = df.iloc[int(0.9 * n) :]
        df = df.reset_index(drop=True)

        in_cache = df["ad_id"].astype(str).isin(self.cache)
        dropped = (~in_cache).sum()
        if dropped:
            print(f"Dropped {dropped} rows missing from embedding cache ({split} split)")
        self.df = df[in_cache].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "embedding": self.cache[str(row["ad_id"])],
            "ctr_score": torch.tensor(row["ctr_score"], dtype=torch.float32),
            "halflife_days": torch.tensor(
                row["halflife_days"] if pd.notna(row["halflife_days"]) else 1.0,
                dtype=torch.float32,
            ),
            "censored": torch.tensor(
                row["censored"] in (True, "True"),
                dtype=torch.bool,
            ),
        }


class FinetuneTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(embedding=inputs["embedding"])
        bce = nn.BCELoss()(outputs["ctr_score"].squeeze(), inputs["ctr_score"])
        weibull = WeibullNLLLoss()(
            outputs["weibull_params"],
            inputs["halflife_days"],
            inputs["censored"],
        )
        loss = ALPHA * bce + (1 - ALPHA) * weibull
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(embedding=inputs["embedding"])
            loss = self.compute_loss(model, inputs)
        return (loss, None, None)


def run_eval(model: CreativeScorer, csv_path: str, device: str) -> dict:
    # Evaluate on the full test split so the result is comparable to the
    # baseline 0.2342 Spearman r (which used all 22,248 rows, last 10%).
    model.eval()
    ds = CreativeDataset(csv_path, split="test")
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    ctr_preds, ctr_trues = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(embedding=batch["embedding"].to(device))
            ctr_preds.extend(out["ctr_score"].squeeze(-1).cpu().numpy())
            ctr_trues.extend(batch["ctr_score"].numpy())

    ctr_preds = np.array(ctr_preds)
    ctr_trues = np.array(ctr_trues)
    spearman_r, spearman_p = spearmanr(ctr_trues, ctr_preds)
    mae = mean_absolute_error(ctr_trues, ctr_preds)
    print(f"Spearman r: {spearman_r:.4f}  (p={spearman_p:.4e})")
    print(f"MAE:        {mae:.4f}")
    return {"spearman_r": float(spearman_r), "spearman_p": float(spearman_p), "mae": float(mae)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", default="./model/checkpoints")
    parser.add_argument("--csv", default="data/labels_clean.csv")
    parser.add_argument("--output-dir", default="./model/best_finetuned")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Resume from best checkpoint of the initial training run
    best_ckpt = _best_checkpoint(args.checkpoints_dir)
    print(f"Loading checkpoint: {best_ckpt}")
    model = _load_model(best_ckpt)
    # CLIP backbone must remain frozen — assert before training starts
    assert not any(p.requires_grad for p in model.clip.parameters())

    # 2. Real-ads-only datasets (3,502 apify rows, 80/10/10)
    train_ds = RealAdsDataset(args.csv, split="train")
    val_ds = RealAdsDataset(args.csv, split="val")
    print(f"Fine-tune split — train: {len(train_ds)}, val: {len(val_ds)}")

    # 3. Fine-tune — only MLP head has gradients, no CLIP params touched
    training_args = TrainingArguments(
        output_dir="./model/finetune_checkpoints",
        num_train_epochs=30,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=1e-4,
        weight_decay=1e-3,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=20,
        dataloader_num_workers=2,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = FinetuneTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=creative_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Best fine-tuned model saved → {args.output_dir}")

    # 4. Report on the same test split as the baseline run
    print("\n--- Evaluation on full test split ---")
    model.to(device)
    run_eval(model, args.csv, device)


if __name__ == "__main__":
    main()
