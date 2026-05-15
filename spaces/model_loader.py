import os

import torch
from huggingface_hub import hf_hub_download
from transformers import CLIPProcessor

from clip_head import CreativeScorer

_model: CreativeScorer | None = None
_processor: CLIPProcessor | None = None


def get_model() -> tuple[CreativeScorer, CLIPProcessor]:
    global _model, _processor
    if _model is None:
        hf_repo = os.environ["HF_MODEL_REPO"]
        hf_token = os.getenv("HF_TOKEN")  # optional — needed for private repos

        # Processor config was saved into the repo by push_to_hub.py
        _processor = CLIPProcessor.from_pretrained(hf_repo, token=hf_token)

        _model = CreativeScorer()

        # Prefer safetensors (upload_folder produces these); fall back to .bin
        try:
            weights_path = hf_hub_download(
                repo_id=hf_repo,
                filename="model.safetensors",
                token=hf_token,
            )
            from safetensors.torch import load_file
            state_dict = load_file(weights_path, device="cpu")
        except Exception:
            weights_path = hf_hub_download(
                repo_id=hf_repo,
                filename="pytorch_model.bin",
                token=hf_token,
            )
            # map_location='cpu' — CRITICAL: HF Spaces free tier has no GPU
            state_dict = torch.load(weights_path, map_location="cpu")

        _model.load_state_dict(state_dict)
        _model.eval()

    return _model, _processor
