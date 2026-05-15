import argparse
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from model.dataset import CreativeDataset
from model.finetune_real import RealAdsDataset, _load_model, run_eval
from model.survival import WeibullNLLLoss
from model.train import creative_collator

BASE_CHECKPOINT = "model/checkpoints/checkpoint-1953"
ALPHAS = [0.6, 0.7, 0.8, 0.9]


class AlphaTrainer(Trainer):
    def __init__(self, alpha: float, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(embedding=inputs["embedding"])
        bce = nn.BCELoss()(outputs["ctr_score"].squeeze(), inputs["ctr_score"])
        weibull = WeibullNLLLoss()(
            outputs["weibull_params"],
            inputs["halflife_days"],
            inputs["censored"],
        )
        loss = self.alpha * bce + (1 - self.alpha) * weibull
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(embedding=inputs["embedding"])
            loss = self.compute_loss(model, inputs)
        return (loss, None, None)


def train_alpha(alpha: float, csv_path: str, device: str) -> dict:
    # Fresh load for every alpha — never reuse a trained model instance
    model = _load_model(BASE_CHECKPOINT)
    assert not any(p.requires_grad for p in model.clip.parameters())

    train_ds = RealAdsDataset(csv_path, split="train")
    val_ds = RealAdsDataset(csv_path, split="val")
    print(f"  train: {len(train_ds)} | val: {len(val_ds)}")

    training_args = TrainingArguments(
        output_dir=f"./model/alpha_{alpha}_checkpoints",
        num_train_epochs=20,
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

    trainer = AlphaTrainer(
        alpha=alpha,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=creative_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    trainer.train()
    trainer.save_model(f"./model/alpha_{alpha}")

    model.to(device)
    return run_eval(model, csv_path, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/labels_clean.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results: dict[float, dict] = {}

    for alpha in ALPHAS:
        print(f"\n{'=' * 52}")
        print(f"  ALPHA = {alpha}")
        print(f"{'=' * 52}")
        results[alpha] = train_alpha(alpha, args.csv, device)

    # Summary table
    print(f"\n{'=' * 38}")
    print(f"{'ALPHA':<8} {'Spearman r':<14} {'MAE'}")
    print(f"{'-' * 38}")
    for alpha, m in results.items():
        print(f"{alpha:<8} {m['spearman_r']:<14.4f} {m['mae']:.4f}")
    print(f"{'=' * 38}")

    best_alpha = max(results, key=lambda a: results[a]["spearman_r"])
    best_r = results[best_alpha]["spearman_r"]
    print(f"\nBest ALPHA: {best_alpha}  (Spearman r = {best_r:.4f})")

    best_dst = Path("./model/best_alpha")
    if best_dst.exists():
        shutil.rmtree(best_dst)
    shutil.copytree(Path(f"./model/alpha_{best_alpha}"), best_dst)
    print(f"Best checkpoint copied → model/best_alpha/")


if __name__ == "__main__":
    main()
