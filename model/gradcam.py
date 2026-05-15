import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor

from model.clip_head import CreativeScorer


def generate_heatmap(model: CreativeScorer, image: Image.Image, device: str = "cpu") -> np.ndarray:
    """
    Gradient-weighted feature attribution on the projection layer output.

    True GradCAM requires spatial feature maps. CLIP's pooler_output collapses
    the 197 patch tokens into a single 512-dim vector, discarding spatial structure,
    so genuine pixel-level heatmaps are not possible here. Instead we:
      - compute gradients of ctr_score w.r.t. the 256-dim shared_repr
      - weight each channel by its gradient magnitude
      - reshape the 256-dim weighted vector into a 16x16 grid (256 = 16*16)
      - upsample to 224x224 and overlay on the original image

    This approximates which projection-layer feature dimensions drive the CTR
    prediction but does not map faithfully to image regions. Treat output as an
    attribution proxy, not a spatial saliency map.
    """
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    model = model.to(device)
    model.eval()
    model.zero_grad()

    # Forward outside torch.no_grad() so autograd builds the computation graph.
    # The clip block inside forward() uses its own no_grad, so the backbone stays
    # frozen while the projection layer's params remain in the graph.
    outputs = model(pixel_values)
    shared_repr = outputs["shared_repr"]  # (1, 256)
    shared_repr.retain_grad()

    outputs["ctr_score"].squeeze().backward()

    grad = shared_repr.grad                                              # (1, 256)
    weights = grad.abs().squeeze(0)                                     # (256,)
    weighted = (weights * shared_repr.detach().squeeze(0)).cpu().numpy()  # (256,)

    cam = weighted.reshape(16, 16)
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)
    cam_uint8 = (cam * 255).astype(np.uint8)

    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    orig = np.array(image.resize((224, 224))).astype(np.float32)
    overlay = np.clip(0.6 * orig + 0.4 * heatmap, 0, 255).astype(np.uint8)
    return overlay


def save_heatmaps(
    model: CreativeScorer,
    image_paths: List[str],
    output_dir: str,
    device: str = "cpu",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        overlay = generate_heatmap(model, image, device)
        stem = Path(path).stem
        Image.fromarray(overlay).save(out / f"{stem}_heatmap.png")
        print(f"Saved: {stem}_heatmap.png")


def _load_model(checkpoint_dir: str) -> CreativeScorer:
    model = CreativeScorer()
    ckpt = Path(checkpoint_dir)
    if (ckpt / "model.safetensors").exists():
        from safetensors.torch import load_file
        model.load_state_dict(load_file(str(ckpt / "model.safetensors")))
    elif (ckpt / "pytorch_model.bin").exists():
        model.load_state_dict(torch.load(str(ckpt / "pytorch_model.bin"), map_location="cpu"))
    else:
        raise FileNotFoundError(f"No weights in {checkpoint_dir}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_paths", nargs="+")
    parser.add_argument("--checkpoint-dir", default="./model/best")
    parser.add_argument("--output-dir", default="./model/heatmaps")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    model = _load_model(args.checkpoint_dir)
    save_heatmaps(model, args.image_paths, args.output_dir, args.device)


if __name__ == "__main__":
    main()
