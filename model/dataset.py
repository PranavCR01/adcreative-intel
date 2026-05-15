from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor, CLIPVisionModel

CACHE_PATH = "data/clip_embeddings.pt"


def _build_cache(csv_path: str, cache_path: str = CACHE_PATH) -> dict:
    """Extract 768-dim CLIP pooler_output for every image and save to disk.

    Runs once. Produces ~68MB file vs ~13GB for pixel_values cache.
    """
    df = pd.read_csv(csv_path)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        use_safetensors=True,
    )
    clip.eval()

    cache = {}
    total = len(df)
    batch_size = 32

    for start in range(0, total, batch_size):
        if start % 1000 == 0:
            print(f"Extracting embeddings: {start}/{total}")
        batch_df = df.iloc[start : start + batch_size]

        images, ad_ids = [], []
        for _, row in batch_df.iterrows():
            try:
                images.append(Image.open(row["image_path"]).convert("RGB"))
                ad_ids.append(str(row["ad_id"]))
            except Exception as e:
                print(f"Skipping {row['ad_id']}: {e}")

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt")
        with torch.no_grad():
            out = clip(pixel_values=inputs["pixel_values"])
            embeddings = out.pooler_output  # (batch, 768)

        for ad_id, emb in zip(ad_ids, embeddings):
            cache[ad_id] = emb.cpu()

    torch.save(cache, cache_path)
    print(f"Embedding cache saved → {cache_path} ({len(cache)} entries)")
    return cache


class CreativeDataset(Dataset):
    def __init__(self, csv_path: str, split: str = "train", cache_path: str = CACHE_PATH):
        self.df = pd.read_csv(csv_path)

        if Path(cache_path).exists():
            self.cache = torch.load(cache_path, map_location="cpu")
        else:
            print("Building CLIP embedding cache — runs once, ~5 min on GPU / ~20 min on CPU...")
            self.cache = _build_cache(csv_path, cache_path)

        # 80/10/10 split by index — deterministic, reproducible
        n = len(self.df)
        if split == "train":
            self.df = self.df.iloc[: int(0.8 * n)]
        elif split == "val":
            self.df = self.df.iloc[int(0.8 * n) : int(0.9 * n)]
        elif split == "test":
            self.df = self.df.iloc[int(0.9 * n) :]

        self.df = self.df.reset_index(drop=True)

        # Drop rows whose image failed to embed (corrupt/missing at build time)
        in_cache = self.df["ad_id"].astype(str).isin(self.cache)
        dropped = (~in_cache).sum()
        if dropped:
            print(f"Dropped {dropped} rows missing from cache in '{split}' split")
        self.df = self.df[in_cache].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "embedding": self.cache[str(row["ad_id"])],  # (768,) — tensor lookup, no I/O
            "ctr_score": torch.tensor(row["ctr_score"], dtype=torch.float32),
            "halflife_days": torch.tensor(
                row["halflife_days"] if pd.notna(row["halflife_days"]) else 1.0,
                dtype=torch.float32,
            ),
            # csv.DictWriter stores bools as strings "True"/"False";
            # bool("False") == True in Python, so we check membership instead.
            "censored": torch.tensor(
                row["censored"] in (True, "True"),
                dtype=torch.bool,
            ),
        }
