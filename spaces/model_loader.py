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
        try:
            hf_repo = os.environ["HF_MODEL_REPO"]
            hf_token = os.getenv("HF_TOKEN")

            print(f"[model_loader] Loading from repo: {hf_repo}", flush=True)

            _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("[model_loader] Processor loaded", flush=True)

            _model = CreativeScorer()
            print("[model_loader] Model instantiated", flush=True)

            try:
                weights_path = hf_hub_download(
                    repo_id=hf_repo,
                    filename="model.safetensors",
                    token=hf_token,
                )
                from safetensors.torch import load_file
                state_dict = load_file(weights_path, device="cpu")
                print("[model_loader] Weights loaded from safetensors", flush=True)
            except Exception as e:
                print(f"[model_loader] safetensors failed: {e}, trying .bin", flush=True)
                weights_path = hf_hub_download(
                    repo_id=hf_repo,
                    filename="pytorch_model.bin",
                    token=hf_token,
                )
                state_dict = torch.load(weights_path, map_location="cpu")

            _model.load_state_dict(state_dict)
            _model.eval()
            print("[model_loader] Model ready", flush=True)

        except Exception as e:
            import traceback
            print(f"[model_loader] FATAL: {e}", flush=True)
            print(traceback.format_exc(), flush=True)
            raise

    return _model, _processor
