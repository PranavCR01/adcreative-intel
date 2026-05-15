"""
Re-classify existing apify rows that landed in "other" by re-reading
the original Apify JSON files with the updated VERTICAL_MAP.
Moves image files to the correct vertical subdirectory and patches
labels.csv in place. Safe to re-run — skips rows already classified.
"""
import hashlib, json, os, shutil
from pathlib import Path

import pandas as pd

APIFY_DIR = Path("data/apify_raw")
RAW_DIR = Path("data/raw")
LABELS_FILE = Path("data/labels.csv")

# Must match VERTICAL_MAP in apify_downloader.py exactly
VERTICAL_MAP = {
    "mobile game": "gaming", "download game": "gaming",
    "play now": "gaming", "play free": "gaming",
    "game": "gaming", "play": "gaming", "download": "gaming",
    "install": "gaming", "level": "gaming", "battle": "gaming", "rpg": "gaming",
    "shop now": "ecommerce", "buy now": "ecommerce",
    "free shipping": "ecommerce", "limited offer": "ecommerce",
    "order now": "ecommerce", "shop": "ecommerce", "sale": "ecommerce",
    "discount": "ecommerce", "delivery": "ecommerce", "checkout": "ecommerce",
    "fashion": "ecommerce", "clothing": "ecommerce",
    "skincare": "ecommerce", "beauty": "ecommerce",
    "trading app": "finance", "invest now": "finance",
    "invest": "finance", "trading": "finance", "crypto": "finance",
    "stock": "finance", "wealth": "finance", "savings": "finance",
    "loan": "finance", "insurance": "finance", "bank": "finance",
    "financial": "finance",
}


def infer_vertical(ad: dict) -> str:
    raw = (ad.get("snapshot") or {}).get("body") or ""
    body = (raw.get("text", "") if isinstance(raw, dict) else raw).lower()
    for keyword, vertical in VERTICAL_MAP.items():
        if keyword in body:
            return vertical
    return "other"


def build_id_map() -> dict[str, str]:
    """Read all Apify JSON files; return {ad_id: vertical}."""
    id_map: dict[str, str] = {}
    json_files = sorted(APIFY_DIR.glob("*.json"))
    print(f"Reading {len(json_files)} JSON file(s)...")
    for json_file in json_files:
        with open(json_file, encoding="utf-8") as f:
            ads = json.load(f)
        for ad in ads:
            snapshot = ad.get("snapshot") or {}
            images = snapshot.get("images") or []
            img_url = images[0].get("original_image_url") if images else None
            if not img_url:
                continue
            url_hash = hashlib.md5(img_url.encode()).hexdigest()[:12]
            ad_id = f"apify_{url_hash}"
            id_map[ad_id] = infer_vertical(ad)
    print(f"  Built map for {len(id_map)} ads.")
    return id_map


def main():
    id_map = build_id_map()

    df = pd.read_csv(LABELS_FILE)
    before_counts = df["vertical"].value_counts().to_dict()

    candidates = df[(df["source"] == "apify") & (df["vertical"] == "other")].copy()
    print(f"\nRows currently in 'other': {len(candidates)}")

    moved = 0
    skipped_no_map = 0
    skipped_file_missing = 0

    for idx, row in candidates.iterrows():
        new_vertical = id_map.get(row["ad_id"])
        if not new_vertical or new_vertical == "other":
            skipped_no_map += 1
            continue

        old_path = Path(row["image_path"])
        new_dir = RAW_DIR / new_vertical
        new_path = new_dir / old_path.name

        if old_path.exists():
            new_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_path), str(new_path))
        elif new_path.exists():
            pass  # already moved from a previous run
        else:
            skipped_file_missing += 1
            continue

        df.at[idx, "vertical"] = new_vertical
        df.at[idx, "image_path"] = str(new_path)
        moved += 1

    tmp = LABELS_FILE.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, LABELS_FILE)  # atomic rename; avoids Windows lock on the original

    after_counts = df["vertical"].value_counts().to_dict()
    print(f"\nReclassified: {moved}")
    print(f"Stayed 'other' (no keyword match): {skipped_no_map}")
    print(f"Skipped (file missing): {skipped_file_missing}")
    print("\nVertical distribution:")
    print(f"  {'Vertical':<12}  {'Before':>7}  {'After':>7}  {'Delta':>7}")
    all_verts = sorted(set(before_counts) | set(after_counts))
    for v in all_verts:
        b = before_counts.get(v, 0)
        a = after_counts.get(v, 0)
        print(f"  {v:<12}  {b:>7}  {a:>7}  {a - b:>+7}")


if __name__ == "__main__":
    main()
