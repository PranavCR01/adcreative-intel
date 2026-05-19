from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor

from clip_head import CreativeScorer


def _compute_cam(
    model: CreativeScorer,
    processor: CLIPProcessor,
    image: Image.Image,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
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

    Returns (overlay uint8 224x224x3, cam_16x16 float32 normalized 0-1).
    """
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

    cam_upsampled = cv2.resize(cam.astype(np.float32), (224, 224), interpolation=cv2.INTER_LINEAR)
    cam_uint8 = (cam_upsampled * 255).astype(np.uint8)

    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    orig = np.array(image.resize((224, 224))).astype(np.float32)
    overlay = np.clip(0.6 * orig + 0.4 * heatmap, 0, 255).astype(np.uint8)

    return overlay, cam.astype(np.float32)


def generate_heatmap(
    model: CreativeScorer,
    processor: CLIPProcessor,
    image: Image.Image,
    device: str = "cpu",
) -> np.ndarray:
    """Returns overlay uint8 ndarray (224x224x3)."""
    overlay, _ = _compute_cam(model, processor, image, device)
    return overlay


def generate_heatmap_with_cam(
    model: CreativeScorer,
    processor: CLIPProcessor,
    image: Image.Image,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (overlay uint8 224x224x3, cam_16x16 float32 normalized 0-1)."""
    return _compute_cam(model, processor, image, device)


def save_heatmaps(
    model: CreativeScorer,
    processor: CLIPProcessor,
    image_paths: List[str],
    output_dir: str,
    device: str = "cpu",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        overlay = generate_heatmap(model, processor, image, device)
        stem = Path(path).stem
        Image.fromarray(overlay).save(out / f"{stem}_heatmap.png")
        print(f"Saved: {stem}_heatmap.png")
