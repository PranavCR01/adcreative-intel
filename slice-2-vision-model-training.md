# Slice 2 — Vision Model Training

## Goal
Trained multi-task model with meaningful eval metrics, pushed to HF Hub.
By end of this slice: model weights on HF Hub, metrics notebook showing
CTR AUC > 0.65, 5 GradCAM heatmaps that look visually sensible.

## Architecture decisions locked in this slice
- CLIP-ViT-B/32 backbone is ALWAYS frozen — never set requires_grad=True on it
- Embeddings are cached to disk after first extraction — never recomputed
- Multi-task loss: 0.5 * BCELoss(ctr) + 0.5 * WeibullNLLLoss(fatigue)
- Censored rows ARE included in fatigue loss with censored=True flag
- GradCAM runs on the projection layer output, not raw CLIP features
- Model pushed to HF Hub as public repo — linkable in portfolio

## Model architecture (exact, do not deviate)
```
Input image (224×224 RGB)
        ↓
[FROZEN] CLIP-ViT-B/32 — openai/clip-vit-base-patch32
        ↓ (512-dim embedding, no_grad)
Projection: Linear(512, 256) → ReLU → Dropout(0.2)
        ↓ (256-dim shared representation)
      ┌──────────────────────┐
      ↓                      ↓
CTR head                 Fatigue head
Linear(256, 1)           Linear(256, 2)
Sigmoid                  → log_scale, log_shape (Weibull params)
      ↓                      ↓
BCELoss                  WeibullNLLLoss (handles censored)
```

## File structure for this slice
```
model/
├── clip_head.py       ← model definition
├── survival.py        ← Weibull loss implementation
├── dataset.py         ← PyTorch Dataset class for labels.csv
├── train.py           ← Hugging Face Trainer setup + main()
├── evaluate.py        ← metrics: CTR AUC, D-calibration
├── gradcam.py         ← GradCAM heatmap generation
└── push_to_hub.py     ← uploads model + card to HF Hub
notebooks/
└── 02_training.ipynb  ← training curves, eval results, sample heatmaps
```

## clip_head.py — model definition
```python
import torch
import torch.nn as nn
from transformers import CLIPVisionModel

class CreativeScorer(nn.Module):
    def __init__(self):
        super().__init__()
        # Frozen CLIP backbone
        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.clip.parameters():
            param.requires_grad = False  # ALWAYS frozen

        # Trainable head
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.ctr_head = nn.Linear(256, 1)
        self.fatigue_head = nn.Linear(256, 2)  # log_scale, log_shape

    def forward(self, pixel_values):
        with torch.no_grad():
            clip_out = self.clip(pixel_values=pixel_values)
            embedding = clip_out.pooler_output  # (batch, 512)
        shared = self.projection(embedding)      # (batch, 256)
        ctr_logit = self.ctr_head(shared)        # (batch, 1)
        ctr_score = torch.sigmoid(ctr_logit)
        weibull_params = self.fatigue_head(shared)  # (batch, 2)
        return {
            "ctr_score": ctr_score,
            "weibull_params": weibull_params,
            "shared_repr": shared  # needed for GradCAM
        }
```

## survival.py — Weibull loss
```python
import torch
import torch.nn as nn

class WeibullNLLLoss(nn.Module):
    """
    Negative log-likelihood loss for Weibull survival distribution.
    Handles right-censored observations.

    weibull_params: (batch, 2) — log_scale, log_shape
    halflife_days:  (batch,)   — observed time (0 for censored rows)
    censored:       (batch,)   — bool, True = right-censored
    """
    def forward(self, weibull_params, halflife_days, censored):
        log_scale, log_shape = weibull_params[:, 0], weibull_params[:, 1]
        # Clamp to prevent NaN — critical, do not remove
        log_scale = torch.clamp(log_scale, -10, 10)
        log_shape = torch.clamp(log_shape, -10, 10)

        scale = torch.exp(log_scale)   # λ (lambda)
        shape = torch.exp(log_shape)   # k

        # Replace null halflife with 1.0 for censored rows (won't affect loss)
        t = torch.clamp(halflife_days, min=1e-6)

        # Log-likelihood: uncensored = log PDF, censored = log survival function
        log_pdf = (log_shape + (shape - 1) * torch.log(t)
                   - shape * log_scale - (t / scale) ** shape)
        log_sf = -((t / scale) ** shape)

        loss = torch.where(censored, -log_sf, -log_pdf)
        return loss.mean()
```

## dataset.py — PyTorch Dataset
```python
from torch.utils.data import Dataset
from PIL import Image
from transformers import CLIPProcessor
import pandas as pd, torch

class CreativeDataset(Dataset):
    def __init__(self, csv_path: str, split: str = "train"):
        self.df = pd.read_csv(csv_path)
        # 80/10/10 split by index — deterministic, reproducible
        n = len(self.df)
        if split == "train":   self.df = self.df.iloc[:int(0.8*n)]
        elif split == "val":   self.df = self.df.iloc[int(0.8*n):int(0.9*n)]
        elif split == "test":  self.df = self.df.iloc[int(0.9*n):]
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # (3, 224, 224)
        return {
            "pixel_values": pixel_values,
            "ctr_score": torch.tensor(row["ctr_score"], dtype=torch.float32),
            "halflife_days": torch.tensor(
                row["halflife_days"] if pd.notna(row["halflife_days"]) else 1.0,
                dtype=torch.float32
            ),
            "censored": torch.tensor(bool(row["censored"]), dtype=torch.bool),
        }
```

## train.py — training setup
```python
# Key decisions:
# - Use Hugging Face Trainer for standardized training loop
# - Custom compute_loss since multi-task loss is non-standard
# - Mixed precision (fp16) to fit 1650 Ti VRAM
# - Batch size: 64 (reduce to 32 if OOM)
# - LR: 1e-3 for head only (CLIP is frozen so only head params update)
# - Epochs: 15 (early stopping on val loss, patience=3)
# - Save best checkpoint by val loss

from transformers import Trainer, TrainingArguments
from model.clip_head import CreativeScorer
from model.survival import WeibullNLLLoss
import torch.nn as nn

ALPHA = 0.5  # weight between CTR loss and fatigue loss

class CreativeTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs["pixel_values"])
        bce = nn.BCELoss()(
            outputs["ctr_score"].squeeze(),
            inputs["ctr_score"]
        )
        weibull = WeibullNLLLoss()(
            outputs["weibull_params"],
            inputs["halflife_days"],
            inputs["censored"]
        )
        loss = ALPHA * bce + (1 - ALPHA) * weibull
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir="./model/checkpoints",
    num_train_epochs=15,
    per_device_train_batch_size=64,   # reduce to 32 if OOM
    per_device_eval_batch_size=64,
    learning_rate=1e-3,
    fp16=True,                        # mixed precision for 1650 Ti
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_steps=50,
    dataloader_num_workers=2,
)
```

## evaluate.py — metrics
```python
# CTR AUC: standard sklearn roc_auc_score on test set predictions
# D-calibration: for survival models — bins predictions into deciles,
#   checks if observed event rate matches predicted in each bin
#   (a well-calibrated model has roughly equal counts per bin)
# Baseline comparisons:
#   - Random: AUC = 0.5
#   - Duration-only: predict halflife from ad duration alone (no image)
#   - CTR-only: single-task CTR head, no fatigue prediction
```

## gradcam.py — heatmap generation
```python
# GradCAM on the projection layer (not CLIP internals)
# Process:
# 1. Run forward pass, keep shared_repr
# 2. Compute gradient of ctr_score w.r.t. shared_repr (256-dim vector)
# 3. Weight shared_repr channels by gradient magnitude
# 4. Reshape weighted repr to 16x16 spatial grid (approximate spatial attention)
# 5. Upsample to 224x224 with bilinear interpolation
# 6. Normalize to 0-1, apply colormap (cv2.COLORMAP_JET)
# 7. Overlay on original image at 40% opacity
# Output: numpy array (224, 224, 3) — save as PNG

# Note: true GradCAM needs spatial features — CLIP's pooler_output loses
# spatial structure. We use gradient-weighted feature attribution on the
# projection layer as an approximation. Document this limitation in README.
```

## Training hardware plan — 1650 Ti (4GB VRAM)
| Step | VRAM | Estimated time |
|---|---|---|
| CLIP feature extraction (50K imgs, batch 64) | ~2.5GB | 2–3 hrs |
| Head training (frozen backbone, batch 64, fp16) | ~2GB | 2–4 hrs |
| GradCAM on test set (200 images) | ~2GB | 20 min |
| Total | — | ~5–7 hrs |

If OOM at batch 64: reduce to 32. If still OOM: reduce to 16 and increase epochs.
Do NOT unfreeze CLIP to compensate — that breaks the architecture.

## HF Hub — push_to_hub.py
```python
# Push: model weights + processor config
# Model card (README.md) must include:
#   - Architecture description
#   - Training data (Meta Ad Library + synthetic, N images)
#   - Metrics (CTR AUC on test set, D-calibration plot as image)
#   - Limitations (GradCAM approximation, CTR proxy not real CTR)
#   - Intended use (portfolio project, ad creative scoring)
# Repo name: {your-hf-username}/creative-intelligence-scorer
# Visibility: public (linkable from portfolio)
```

## Bug prevention checklist for this slice
- [ ] Confirm CLIP has no gradients: `assert not any(p.requires_grad for p in model.clip.parameters())`
- [ ] Cache CLIP embeddings: run extraction once, save to `data/embeddings.pt`, load from disk in dataset
- [ ] Weibull loss: clamp log inputs to [-10, 10] — already in survival.py above, do not remove
- [ ] BCELoss pos_weight: if CTR AUC on val < 0.58 after epoch 3, add `pos_weight=torch.tensor([2.0])` to BCELoss
- [ ] fp16 overflow: if loss goes to NaN, add `--fp16_opt_level O1` or switch to bf16 if GPU supports it
- [ ] GradCAM note: document the spatial approximation limitation in notebook and README — do not pretend it's exact
- [ ] HF Hub: set `private=False` explicitly — default may be private depending on account tier

## Eval targets (minimum to proceed to Slice 3)
- CTR AUC on test set > 0.65 (random baseline = 0.50)
- Fatigue D-calibration: at least 7/10 decile bins within 20% of expected count
- GradCAM heatmaps: visually, high-attention regions should correlate with
  CTA buttons, faces, and bright colors — spot check 5 images manually

## Start prompt for Claude Code
```
Starting Slice 2 of Creative Intelligence Agent. Read CLAUDE.md first,
then read slice-2-vision-model-training.md in full before writing anything.

Goals this session:
1. Write model/clip_head.py exactly as specified in the slice doc
2. Write model/survival.py with the WeibullNLLLoss — do not simplify it
3. Write model/dataset.py with 80/10/10 deterministic split
4. Write model/train.py using CreativeTrainer with custom compute_loss
5. Write model/evaluate.py with CTR AUC and D-calibration
6. Write model/gradcam.py — document the spatial approximation
7. Write model/push_to_hub.py

Then run training. Monitor VRAM with `nvidia-smi -l 1` in a separate terminal.
Use /plan first. Do not deviate from the architecture in the slice doc.
```

## Done when
- [ ] Training runs to completion without OOM crash
- [ ] CTR AUC on test set > 0.65
- [ ] D-calibration: 7/10 bins within expected range
- [ ] 5 GradCAM heatmaps generated, visually inspected and reasonable
- [ ] Model pushed to HF Hub, public URL confirmed working
- [ ] 02_training.ipynb renders cleanly with all plots

## Next slice
Slice 3 — Model Serving on HF Spaces. Come back to claude.ai first.
