"""
Analyze real Apify ad images to extract color, brightness, contrast,
and text-density distributions per vertical. Saves results to
data/real_ad_analysis.json for use in synthetic_gen.py parameterization.

Note: labels.csv has no body-text column, so "text density" is estimated
from image edge density (numpy diff on grayscale), which strongly correlates
with the presence of text overlays and busy visual regions.
"""
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

LABELS_FILE = Path("data/labels.csv")
OUT_FILE = Path("data/real_ad_analysis.json")


def dominant_colors(img: Image.Image, n: int = 3) -> list[tuple[int, int, int]]:
    """Return the n most-common colors via quantization."""
    small = img.resize((64, 64), Image.LANCZOS).convert("RGB")
    quantized = small.quantize(colors=n, method=Image.Quantize.FASTOCTREE)
    palette = quantized.getpalette()  # flat [R,G,B, R,G,B, ...]
    # Count pixel occurrences per palette index
    pixel_indices = list(quantized.getdata())
    from collections import Counter
    freq = Counter(pixel_indices)
    ordered = [idx for idx, _ in freq.most_common(n)]
    colors = [(palette[i * 3], palette[i * 3 + 1], palette[i * 3 + 2])
              for i in ordered]
    return colors


def image_stats(img: Image.Image) -> dict:
    """Extract brightness, contrast, and edge-density (text proxy) from image."""
    rgb = np.array(img.convert("RGB"), dtype=np.float32)
    gray = np.array(img.convert("L"), dtype=np.float32)

    brightness = float(rgb.mean())
    contrast = float(rgb.std())

    # Edge density via finite differences — higher = more text / detail
    edges_h = np.abs(np.diff(gray, axis=0))
    edges_v = np.abs(np.diff(gray, axis=1))
    text_density = float((edges_h.mean() + edges_v.mean()) / 2)

    return {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "text_density": round(text_density, 2),
    }


def rgb_to_hsv_hue(r: int, g: int, b: int) -> float:
    """Return hue in [0, 360) for the RGB triple."""
    r_, g_, b_ = r / 255, g / 255, b / 255
    cmax, cmin = max(r_, g_, b_), min(r_, g_, b_)
    delta = cmax - cmin
    if delta == 0:
        return 0.0
    if cmax == r_:
        h = 60 * (((g_ - b_) / delta) % 6)
    elif cmax == g_:
        h = 60 * ((b_ - r_) / delta + 2)
    else:
        h = 60 * ((r_ - g_) / delta + 4)
    return round(h % 360, 1)


def color_bucket(r: int, g: int, b: int) -> str:
    """Classify an RGB color into a broad bucket for easy reading."""
    brightness = (r + g + b) / 3
    saturation = max(r, g, b) - min(r, g, b)
    if brightness < 40:
        return "near-black"
    if brightness > 215:
        return "near-white"
    if saturation < 30:
        return "gray"
    hue = rgb_to_hsv_hue(r, g, b)
    if hue < 20 or hue >= 340:
        return "red"
    if hue < 40:
        return "orange"
    if hue < 70:
        return "yellow"
    if hue < 160:
        return "green"
    if hue < 195:
        return "cyan"
    if hue < 260:
        return "blue"
    if hue < 290:
        return "purple"
    return "magenta"


def main():
    df = pd.read_csv(LABELS_FILE)
    real = df[df["source"] == "apify"].copy()
    print(f"Apify rows in labels.csv: {len(real)}")

    # Only keep rows whose images actually exist
    real = real[real["image_path"].apply(lambda p: Path(p).exists())]
    print(f"Images found on disk:      {len(real)}")

    if real.empty:
        print("\nNo real ad images found on disk. Run apify_downloader.py first.")
        return

    verticals = sorted(real["vertical"].unique())
    print(f"Verticals: {verticals}\n")

    # Per-vertical accumulators
    stats_by_vertical: dict[str, dict] = {v: defaultdict(list) for v in verticals}
    color_buckets_by_vertical: dict[str, list] = {v: [] for v in verticals}

    total = len(real)
    errors = 0
    for i, (_, row) in enumerate(real.iterrows(), 1):
        if i % 100 == 0 or i == total:
            print(f"  Processed {i}/{total}...")
        try:
            img = Image.open(row["image_path"])
            s = image_stats(img)
            v = row["vertical"]
            stats_by_vertical[v]["brightness"].append(s["brightness"])
            stats_by_vertical[v]["contrast"].append(s["contrast"])
            stats_by_vertical[v]["text_density"].append(s["text_density"])

            colors = dominant_colors(img)
            for r, g, b in colors:
                color_buckets_by_vertical[v].append(color_bucket(r, g, b))
        except Exception as e:
            errors += 1

    if errors:
        print(f"\n  Skipped {errors} unreadable images.")

    # Build summary
    summary: dict = {}
    for v in verticals:
        s = stats_by_vertical[v]
        n = len(s["brightness"])
        if n == 0:
            continue

        from collections import Counter
        bucket_counts = Counter(color_buckets_by_vertical[v])
        top_colors = bucket_counts.most_common(5)

        summary[v] = {
            "n": n,
            "brightness": {
                "mean": round(float(np.mean(s["brightness"])), 2),
                "std":  round(float(np.std(s["brightness"])), 2),
                "p25":  round(float(np.percentile(s["brightness"], 25)), 2),
                "p75":  round(float(np.percentile(s["brightness"], 75)), 2),
            },
            "contrast": {
                "mean": round(float(np.mean(s["contrast"])), 2),
                "std":  round(float(np.std(s["contrast"])), 2),
                "p25":  round(float(np.percentile(s["contrast"], 25)), 2),
                "p75":  round(float(np.percentile(s["contrast"], 75)), 2),
            },
            "text_density": {
                "mean": round(float(np.mean(s["text_density"])), 2),
                "std":  round(float(np.std(s["text_density"])), 2),
                "p25":  round(float(np.percentile(s["text_density"], 25)), 2),
                "p75":  round(float(np.percentile(s["text_density"], 75)), 2),
            },
            "dominant_color_buckets": [
                {"bucket": b, "count": c, "pct": round(c / sum(x for _, x in top_colors) * 100, 1)}
                for b, c in top_colors
            ],
        }

    # Print human-readable summary
    print("\n" + "=" * 60)
    print("REAL AD ANALYSIS SUMMARY")
    print("=" * 60)
    for v, s in summary.items():
        print(f"\n-- {v.upper()} (n={s['n']}) --")
        b = s["brightness"]
        print(f"  Brightness : mean={b['mean']:6.1f}  std={b['std']:5.1f}  "
              f"p25={b['p25']:6.1f}  p75={b['p75']:6.1f}")
        c = s["contrast"]
        print(f"  Contrast   : mean={c['mean']:6.1f}  std={c['std']:5.1f}  "
              f"p25={c['p25']:6.1f}  p75={c['p75']:6.1f}")
        t = s["text_density"]
        print(f"  Text dens. : mean={t['mean']:6.1f}  std={t['std']:5.1f}  "
              f"p25={t['p25']:6.1f}  p75={t['p75']:6.1f}")
        print(f"  Top colors : ", end="")
        print("  ".join(f"{x['bucket']}({x['pct']}%)" for x in s["dominant_color_buckets"]))

    print()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved -> {OUT_FILE}")


if __name__ == "__main__":
    main()
