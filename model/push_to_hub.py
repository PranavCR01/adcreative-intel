import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import CLIPProcessor

MODEL_CARD = """\
---
language: en
tags:
- ad-creative
- ctr-prediction
- survival-analysis
- multi-task-learning
- clip
- fatigue-prediction
license: mit
---

# Creative Intelligence Scorer

**Multi-Task Creative Lifespan Prediction** — predicts ad creative CTR score and
fatigue half-life from a raw image using a frozen CLIP backbone and a trainable
multi-task head.

## Architecture

```
Input image (224×224 RGB)
        ↓
[FROZEN] CLIP-ViT-B/32 (openai/clip-vit-base-patch32)
        ↓  512-dim embedding
Projection: Linear(512→256) → ReLU → Dropout(0.2)
        ↓  256-dim shared representation
  ┌─────────────────────┐
  ↓                     ↓
CTR head             Fatigue head
Linear(256→1)        Linear(256→2)
Sigmoid              Weibull params (log_scale, log_shape)
```

Loss = 0.5 × BCELoss(ctr) + 0.5 × WeibullNLLLoss(fatigue, right-censored)

## Training data

- Meta Ad Library (Apify scrape): 3,502 real ad images — gaming, ecommerce, finance verticals
- PIL-generated synthetic ads: 18,746 images with rule-based CTR and half-life labels
- Total: 22,248 images | 80/10/10 train/val/test split

## Metrics (test set)

| Metric | Value | Target |
|--------|-------|--------|
| Spearman r (CTR ranking) | TBD | > 0.30 |
| MAE (CTR calibration) | TBD | < 0.15 |

## Limitations

- **CTR labels are proxy scores**, not real click-through rates — derived from ad
  activity signals, not A/B test data.
- **GradCAM is a spatial approximation** — CLIP's pooler_output discards spatial
  structure; the 16×16 heatmap is gradient-weighted feature attribution on the
  projection layer, not true spatial GradCAM.
- Trained on a dataset with known label imbalance (wear-out >> cut-out).

## Intended use

Portfolio project demonstrating Multi-Task Creative Lifespan Prediction for ad
creative scoring. Not intended for production ad serving decisions.
"""


def push(
    checkpoint_dir: str = "./model/best",
    repo_id: str = None,
    hf_token: str = None,
) -> str:
    hf_token = hf_token or os.environ["HF_TOKEN"]
    api = HfApi(token=hf_token)

    if repo_id is None:
        username = api.whoami()["name"]
        repo_id = f"{username}/creative-intelligence-scorer"

    create_repo(repo_id, token=hf_token, private=False, exist_ok=True)

    # Save processor config so the repo is self-contained
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    processor.save_pretrained(checkpoint_dir)

    (Path(checkpoint_dir) / "README.md").write_text(MODEL_CARD, encoding="utf-8")

    api.upload_folder(folder_path=checkpoint_dir, repo_id=repo_id)

    url = f"https://huggingface.co/{repo_id}"
    print(f"Pushed to: {url}")
    return url


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="./model/best")
    parser.add_argument("--repo-id", default=None)
    args = parser.parse_args()
    push(args.checkpoint_dir, args.repo_id)


if __name__ == "__main__":
    main()
