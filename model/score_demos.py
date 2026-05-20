import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.evaluate import infer_single

DEMOS = [
    ("data/raw/gaming/apify_f321af3f4e59.jpg",            "gaming"),
    ("data/raw/ecommerce/apify_6d945f5a1b9d.jpg",         "ecommerce"),
    ("data/synthetic/finance/synthetic_591690dcf2f2.jpg", "finance"),
]

if __name__ == "__main__":
    for image_path, vertical in DEMOS:
        if not Path(image_path).exists():
            print(f"MISSING: {image_path}")
            sys.exit(1)

    for image_path, vertical in DEMOS:
        print(f"\n[{vertical}] {image_path}")
        result = infer_single(image_path, vertical)
        print(f"  ctr_score:     {result['ctr_score']}")
        print(f"  halflife_days: {result['halflife_days']}")
        print(f"  confidence:    {result['confidence']}")
