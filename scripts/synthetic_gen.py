from PIL import Image, ImageDraw
import fiftyone as fo
import fiftyone.zoo as foz
import argparse, random, uuid, csv, numpy as np
from pathlib import Path
from datetime import datetime

SYNTHETIC_DIR = Path("data/synthetic")
LABELS_FILE = Path("data/labels.csv")

# Brand bar / CTA colors updated from real_ad_analysis.json:
#   gaming:    orange 37%, near-black 25%, blue 11%  -> warm orange + electric blue
#   ecommerce: orange 26%, near-white 23%, blue 19%, red 11% -> orange, red, blue trust
#   finance:   blue 44%, near-white 22%, gray 22%    -> blue dominant, dark navy, slate
VERTICALS = {
    "gaming": {
        "headlines": ["Play Now — Free!", "Battle with millions",
                      "Top rated RPG 2026", "Join 50M players"],
        "ctas": ["Play Free", "Install Now", "Join Battle"],
        "colors": [(220, 90, 20), (30, 64, 175), (88, 28, 135), (5, 150, 105)],
    },
    "ecommerce": {
        "headlines": ["Shop Now — 50% Off", "Free shipping today",
                      "Limited time offer", "Members get more"],
        "ctas": ["Shop Now", "Buy Now", "Get Deal"],
        "colors": [(234, 88, 12), (220, 38, 38), (59, 130, 246), (202, 138, 4)],
    },
    "finance": {
        "headlines": ["Start investing today", "0% commission trades",
                      "Beat inflation now", "Earn up to 8% APY"],
        "ctas": ["Invest Now", "Start Free", "Learn More"],
        "colors": [(30, 58, 138), (59, 130, 246), (15, 23, 42), (100, 116, 139)],
    },
}

# Per-vertical base-image brightness targets (from real_ad_analysis.json):
#   gaming    mean=108 -> prefer dark bases
#   ecommerce mean=151 -> prefer light bases
#   finance   mean=130 -> prefer mid-brightness; also lower edge density
BRIGHTNESS_TARGETS = {
    "gaming":    (0,   120),
    "ecommerce": (140, 255),
    "finance":   (120, 150),
}
BRIGHTNESS_ATTEMPTS = 8  # max skips before accepting any brightness


def base_brightness(img: Image.Image) -> float:
    return float(np.array(img.convert("RGB"), dtype=np.float32).mean())


def make_ad(base: Image.Image, vertical: str) -> tuple[Image.Image, dict]:
    cfg = VERTICALS[vertical]
    img = base.resize((224, 224)).convert("RGB")
    draw = ImageDraw.Draw(img)
    brand_color = random.choice(cfg["colors"])

    # Brand bar
    draw.rectangle([0, 190, 224, 224], fill=brand_color)

    # CTA button
    cta = random.choice(cfg["ctas"])
    cx, cy = random.randint(80, 144), random.randint(150, 175)
    draw.rounded_rectangle([cx-40, cy-12, cx+40, cy+12], radius=6, fill=(255, 255, 255))
    draw.text((cx, cy), cta, fill=brand_color, anchor="mm")

    # Headline — finance skips it 40% of the time to reduce visual density
    # (real finance mean text_density=2.9 vs gaming/ecommerce ~5.1)
    headline = random.choice(cfg["headlines"])
    if vertical != "finance" or random.random() > 0.4:
        draw.rectangle([0, 0, 224, 32], fill=(0, 0, 0))
        draw.text((112, 16), headline[:28], fill=(255, 255, 255), anchor="mm")

    # Synthetic label rules
    halflife = max(1.0, min(60.0, random.gauss(10, 4)))
    if len(headline) > 20: halflife *= 0.75  # dense text decays faster
    if cy < 160:           halflife *= 1.20  # central CTA helps
    halflife = round(halflife, 2)
    ctr = round(min(1.0, halflife / 90.0), 4)

    disc_type = "cut-out" if halflife < 7 else "wear-out"

    return img, {
        "halflife_days": halflife,
        "ctr_score": ctr,
        "censored": False,
        "discontinuation_type": disc_type,
    }


def ensure_csv_header():
    if not LABELS_FILE.exists():
        with open(LABELS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "ad_id", "image_path", "ctr_score", "halflife_days",
                "censored", "vertical", "source", "scraped_at",
                "discontinuation_type",
            ])
            writer.writeheader()


def main(target_per_vertical: int = 5000, test_mode: bool = False):
    if test_mode:
        target_per_vertical = 10
        max_fetch = 150
        print("TEST MODE: generating 10 images per vertical (30 total) then stopping.")
    else:
        max_fetch = target_per_vertical * 12  # large pool for brightness filtering

    ensure_csv_header()

    # Load with caching — avoids re-downloading on subsequent runs
    dataset_name = f"open-images-v7-train-{max_fetch}"
    if fo.dataset_exists(dataset_name):
        print(f"Loading cached dataset '{dataset_name}'...")
        dataset = fo.load_dataset(dataset_name)
    else:
        print(f"Downloading dataset '{dataset_name}' ({max_fetch} images)...")
        dataset = foz.load_zoo_dataset(
            "open-images-v7", split="train",
            label_types=[], max_samples=max_fetch,
            dataset_name=dataset_name,
        )

    samples = list(dataset)
    random.shuffle(samples)
    count = {v: 0 for v in VERTICALS}

    sample_idx = 0
    total_samples = len(samples)

    while sample_idx < total_samples:
        if all(c >= target_per_vertical for c in count.values()):
            break

        # Pick a vertical that still needs images
        remaining = [v for v in VERTICALS if count[v] < target_per_vertical]
        vertical = random.choice(remaining)
        bmin, bmax = BRIGHTNESS_TARGETS[vertical]

        # Find a base image matching the brightness target for this vertical
        accepted = None
        for attempt in range(BRIGHTNESS_ATTEMPTS):
            if sample_idx >= total_samples:
                break
            candidate = samples[sample_idx]
            sample_idx += 1
            try:
                img_check = Image.open(candidate.filepath)
                b = base_brightness(img_check)
                if bmin <= b <= bmax or attempt == BRIGHTNESS_ATTEMPTS - 1:
                    accepted = (candidate, img_check)
                    break
            except Exception:
                continue

        if accepted is None:
            continue

        candidate, base_img = accepted
        try:
            img, labels = make_ad(base_img, vertical)
            ad_id = f"synthetic_{uuid.uuid4().hex[:12]}"
            out_path = SYNTHETIC_DIR / vertical / f"{ad_id}.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path, "JPEG", quality=92)
            with open(LABELS_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "ad_id", "image_path", "ctr_score", "halflife_days",
                    "censored", "vertical", "source", "scraped_at",
                    "discontinuation_type",
                ])
                writer.writerow({
                    "ad_id": ad_id,
                    "image_path": str(out_path),
                    "ctr_score": labels["ctr_score"],
                    "halflife_days": labels["halflife_days"],
                    "censored": labels["censored"],
                    "vertical": vertical,
                    "source": "synthetic",
                    "scraped_at": datetime.utcnow().isoformat(),
                    "discontinuation_type": labels["discontinuation_type"],
                })
            count[vertical] += 1
            if not test_mode and sum(count.values()) % 1000 == 0:
                print(f"Progress: {count}")
        except Exception:
            continue

    print(f"Done: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Generate 10 images per vertical to validate, then exit.")
    args = parser.parse_args()
    main(test_mode=args.test)
