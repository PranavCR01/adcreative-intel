import torch
import torch.nn as nn
from transformers import CLIPVisionModel


class CreativeScorer(nn.Module):
    def __init__(self, local_files_only: bool = False):
        super().__init__()
        # Frozen CLIP backbone — NEVER set requires_grad=True on these params
        self.clip = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            use_safetensors=True,
            local_files_only=local_files_only,
        )
        for param in self.clip.parameters():
            param.requires_grad = False

        # Fail fast if backbone accidentally gets unfrozen anywhere downstream
        assert not any(p.requires_grad for p in self.clip.parameters())

        # Trainable head only
        self.projection = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.ctr_head = nn.Linear(256, 1)
        self.fatigue_head = nn.Linear(256, 2)  # outputs: log_scale, log_shape

    def forward(self, pixel_values=None, embedding=None):
        if embedding is not None:
            # Fast path: pre-computed 768-dim embedding from cache
            pass
        else:
            with torch.no_grad():
                clip_out = self.clip(pixel_values=pixel_values)
                embedding = clip_out.pooler_output  # (batch, 768)

        shared = self.projection(embedding)         # (batch, 256)
        ctr_logit = self.ctr_head(shared)           # (batch, 1)
        ctr_score = torch.sigmoid(ctr_logit)
        weibull_params = self.fatigue_head(shared)  # (batch, 2): log_scale, log_shape

        return {
            "ctr_score": ctr_score,
            "weibull_params": weibull_params,
            "shared_repr": shared,  # retained for GradCAM
        }
