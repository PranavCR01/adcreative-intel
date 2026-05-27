import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

from model.clip_head import CreativeScorer
from model.dataset import CreativeDataset
from model.survival import WeibullNLLLoss

ALPHA = 0.5


def creative_collator(batch):
    # Explicit collator — stacks each key's tensors without any
    # column-filtering or padding logic the Trainer might apply.
    return {key: torch.stack([sample[key] for sample in batch]) for key in batch[0]}


class CreativeTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(embedding=inputs["embedding"])
            loss = self.compute_loss(model, inputs)
        return (loss, None, None)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(embedding=inputs["embedding"])
        bce = nn.BCELoss()(
            outputs["ctr_score"].squeeze(),
            inputs["ctr_score"],
        )
        weibull = WeibullNLLLoss()(
            outputs["weibull_params"],
            inputs["halflife_days"],
            inputs["censored"],
        )
        loss = ALPHA * bce + (1 - ALPHA) * weibull
        return (loss, outputs) if return_outputs else loss


def main():
    csv = "data/labels_clean.csv"
    train_ds = CreativeDataset(csv, split="train")
    val_ds = CreativeDataset(csv, split="val")

    # Verify dataset keys match what compute_loss expects
    sample = train_ds[0]
    assert set(sample.keys()) == {"embedding", "ctr_score", "halflife_days", "censored"}, (
        f"Unexpected dataset keys: {set(sample.keys())}"
    )
    print("Dataset keys OK:", list(sample.keys()))

    model = CreativeScorer()
    assert not any(p.requires_grad for p in model.backbone.parameters())

    args = TrainingArguments(
        output_dir="./model/checkpoints",
        num_train_epochs=15,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        learning_rate=1e-3,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        dataloader_num_workers=2,
        report_to="none",
        remove_unused_columns=False,  # Trainer strips non-forward() keys by default
    )

    trainer = CreativeTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=creative_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()
    trainer.save_model("./model/best")


if __name__ == "__main__":
    main()
