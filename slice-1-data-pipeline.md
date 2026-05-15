# Slice 1 — Data Pipeline (Revised)

## Goal
Dataset built, schema locked, EDA complete. Nothing else.
By end of this slice: `data/labels_clean.csv` exists with 50K+ rows,
Apify scraper has run, synthetic generator has padded the dataset,
EDA notebook renders cleanly.

## Why the original plan changed
Meta Ad Library API only provides political/social issue ads in the EU.
Commercial ads (gaming, ecommerce, finance) are NOT accessible via the
official API. Revised strategy: Apify scraper for real commercial ads
(10-15K) + synthetic PIL generation for the rest (35-40K). Total: ~50K.

## Data strategy
| Source | Tool | Expected count | Notes |
|---|---|---|---|
| Meta Ad Library (real) | Apify scraper | 10–15K | Real commercial ads |
| Synthetic | PIL on Open Images V7 | 35–40K | Rule-based labels |
| Total | — | ~50K | Enough for meaningful training |

## Architecture decisions locked in this slice
- Labels computed once, stored in CSV, never recomputed from images
- Right-censored ads (no end date) → censored=True, halflife_days=null
- Images: data/raw/{vertical}/{ad_id}.jpg and data/synthetic/{vertical}/{ad_id}.jpg
- Embeddings NOT computed this slice — Slice 2
- Supabase NOT touched this slice — Slice 3
- No Meta developer account needed — Apify handles the web scraping

## Repo structure to initialize
```
adcreative-intel/
├── data/
│   ├── raw/
│   │   ├── gaming/
│   │   ├── ecommerce/
│   │   └── finance/
│   ├── synthetic/
│   │   ├── gaming/
│   │   ├── ecommerce/
│   │   └── finance/
│   ├── apify_raw/          ← raw JSON dumps from Apify (manual download)
│   └── labels.csv          ← appended to by both scripts
├── notebooks/
│   └── 01_eda.ipynb
├── scripts/
│   ├── apify_downloader.py
│   ├── synthetic_gen.py
│   └── build_labels.py
├── model/                  ← empty placeholder
├── agent/                  ← empty placeholder
├── api/                    ← empty placeholder
├── spaces/                 ← empty placeholder
├── frontend/               ← empty placeholder
├── .env.example
├── requirements.txt
└── CLAUDE.md
```

## Data schema — labels.csv (locked, do not change later)
```
ad_id         string   unique id (apify_{hash} or synthetic_{uuid})
image_path    string   relative path e.g. data/raw/gaming/abc123.jpg
ctr_score     float    0.0–1.0 normalized proxy
halflife_days float    days until discontinued (null if censored)
censored      bool     True = ad still running when scraped
vertical      string   gaming | ecommerce | finance | other
source        string   apify | synthetic
scraped_at    string   ISO 8601 datetime
```

## STEP 1 — Apify setup (manual, you do this in the browser)

### What Apify is
Apify is a web scraping platform with a pre-built Facebook Ad Library
Scraper actor that handles browser automation and anti-bot bypassing.
Free signup gives ~$5 credits — enough for 10-15K ads.

### Setup steps
1. apify.com → sign up free
2. Search Apify Store for "Facebook Ad Library Scraper"
3. Configure the actor:
   ```
   Search terms: ["mobile game", "shop now", "buy now",
                  "trading app", "play free", "free shipping"]
   Country: US
   Ad type: ALL
   Max ads: 5000 per run
   ```
4. Run 2-3 times with different search terms (~30-40 min per run)
5. Download JSON output → save to data/apify_raw/run1.json, run2.json, etc.
6. THEN run apify_downloader.py to process the JSON

### apify_downloader.py
```python
import json, requests, hashlib, csv, os, time, random
from pathlib import Path
from datetime import datetime

RAW_DIR = Path("data/raw")
APIFY_DIR = Path("data/apify_raw")
LABELS_FILE = Path("data/labels.csv")

VERTICAL_MAP = {
    "mobile game": "gaming", "play now": "gaming", "play free": "gaming",
    "download game": "gaming", "shop now": "ecommerce", "buy now": "ecommerce",
    "free shipping": "ecommerce", "limited offer": "ecommerce",
    "trading app": "finance", "invest now": "finance", "crypto": "finance",
}

def infer_vertical(ad: dict) -> str:
    body = " ".join(ad.get("ad_creative_bodies", [])).lower()
    for keyword, vertical in VERTICAL_MAP.items():
        if keyword in body:
            return vertical
    return "other"

def compute_labels(start: str, stop: str | None) -> tuple[float, float | None, bool]:
    if not stop:
        return 0.7, None, True
    from datetime import datetime as dt
    try:
        days = (dt.strptime(stop[:10], "%Y-%m-%d") -
                dt.strptime(start[:10], "%Y-%m-%d")).days
        days = max(1, days)
        return round(min(1.0, days / 90.0), 4), float(days), False
    except Exception:
        return 0.7, None, True

def download_image(url: str, path: Path) -> bool:
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200 and "image" in r.headers.get("content-type", ""):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(r.content)
            return True
    except Exception:
        pass
    return False

def load_seen_ids() -> set:
    if not LABELS_FILE.exists():
        return set()
    import pandas as pd
    return set(pd.read_csv(LABELS_FILE)["ad_id"].tolist())

def ensure_csv_header():
    if not LABELS_FILE.exists():
        with open(LABELS_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "ad_id","image_path","ctr_score","halflife_days",
                "censored","vertical","source","scraped_at"
            ])
            writer.writeheader()

def main():
    ensure_csv_header()
    seen = load_seen_ids()
    total_added = 0

    for json_file in sorted(APIFY_DIR.glob("*.json")):
        print(f"\nProcessing {json_file.name}...")
        with open(json_file) as f:
            ads = json.load(f)

        added = 0
        for ad in ads:
            img_url = (ad.get("snapshot_url") or
                       ad.get("ad_snapshot_url") or
                       (ad.get("images") or [None])[0])
            if not img_url:
                continue
            url_hash = hashlib.md5(img_url.encode()).hexdigest()[:12]
            ad_id = f"apify_{url_hash}"
            if ad_id in seen:
                continue

            vertical = infer_vertical(ad)
            image_path = RAW_DIR / vertical / f"{ad_id}.jpg"

            if not download_image(img_url, image_path):
                continue

            ctr, halflife, censored = compute_labels(
                ad.get("ad_delivery_start_time", "2024-01-01"),
                ad.get("ad_delivery_stop_time")
            )

            with open(LABELS_FILE, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "ad_id","image_path","ctr_score","halflife_days",
                    "censored","vertical","source","scraped_at"
                ])
                writer.writerow({
                    "ad_id": ad_id,
                    "image_path": str(image_path),
                    "ctr_score": ctr,
                    "halflife_days": round(halflife, 2) if halflife else "",
                    "censored": censored,
                    "vertical": vertical,
                    "source": "apify",
                    "scraped_at": datetime.utcnow().isoformat(),
                })

            seen.add(ad_id)
            added += 1
            total_added += 1
            time.sleep(random.uniform(0.5, 1.5))  # polite delay

        print(f"  Added {added} ads from {json_file.name}")

    print(f"\nTotal added: {total_added}")

if __name__ == "__main__":
    main()
```

## STEP 2 — synthetic_gen.py (run in parallel with downloader)
```python
from PIL import Image, ImageDraw
import fiftyone.zoo as foz
import random, uuid, csv
from pathlib import Path
from datetime import datetime

SYNTHETIC_DIR = Path("data/synthetic")
LABELS_FILE = Path("data/labels.csv")

VERTICALS = {
    "gaming": {
        "headlines": ["Play Now — Free!", "Battle with millions",
                      "Top rated RPG 2026", "Join 50M players"],
        "ctas": ["Play Free", "Install Now", "Join Battle"],
        "colors": [(88, 28, 135), (30, 64, 175), (5, 150, 105)],
    },
    "ecommerce": {
        "headlines": ["Shop Now — 50% Off", "Free shipping today",
                      "Limited time offer", "Members get more"],
        "ctas": ["Shop Now", "Buy Now", "Get Deal"],
        "colors": [(220, 38, 38), (234, 88, 12), (202, 138, 4)],
    },
    "finance": {
        "headlines": ["Start investing today", "0% commission trades",
                      "Beat inflation now", "Earn up to 8% APY"],
        "ctas": ["Invest Now", "Start Free", "Learn More"],
        "colors": [(15, 23, 42), (20, 83, 45), (30, 58, 138)],
    },
}

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
    draw.rounded_rectangle([cx-40, cy-12, cx+40, cy+12], radius=6, fill=(255,255,255))
    draw.text((cx, cy), cta, fill=brand_color, anchor="mm")

    # Headline
    headline = random.choice(cfg["headlines"])
    draw.rectangle([0, 0, 224, 32], fill=(0, 0, 0))
    draw.text((112, 16), headline[:28], fill=(255,255,255), anchor="mm")

    # Synthetic label rules
    halflife = max(1.0, min(60.0, random.gauss(10, 4)))
    if len(headline) > 20: halflife *= 0.75  # dense text decays faster
    if cy < 160:           halflife *= 1.20  # central CTA helps
    halflife = round(halflife, 2)
    ctr = round(min(1.0, halflife / 90.0), 4)

    return img, {"halflife_days": halflife, "ctr_score": ctr, "censored": False}

def ensure_csv_header():
    if not LABELS_FILE.exists():
        with open(LABELS_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "ad_id","image_path","ctr_score","halflife_days",
                "censored","vertical","source","scraped_at"
            ])
            writer.writeheader()

def main(target_per_vertical: int = 12000):
    ensure_csv_header()
    dataset = foz.load_zoo_dataset(
        "open-images-v7", split="train",
        label_types=[], max_samples=target_per_vertical * 4,
    )
    samples = list(dataset)
    random.shuffle(samples)
    count = {v: 0 for v in VERTICALS}

    for sample in samples:
        if all(c >= target_per_vertical for c in count.values()):
            break
        vertical = random.choice(list(VERTICALS.keys()))
        if count[vertical] >= target_per_vertical:
            continue
        try:
            base = Image.open(sample.filepath)
            img, labels = make_ad(base, vertical)
            ad_id = f"synthetic_{uuid.uuid4().hex[:12]}"
            out_path = SYNTHETIC_DIR / vertical / f"{ad_id}.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path, "JPEG", quality=92)
            with open(LABELS_FILE, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "ad_id","image_path","ctr_score","halflife_days",
                    "censored","vertical","source","scraped_at"
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
                })
            count[vertical] += 1
            if sum(count.values()) % 1000 == 0:
                print(f"Progress: {count}")
        except Exception:
            continue

    print(f"Done: {count}")

if __name__ == "__main__":
    main()
```

## STEP 3 — build_labels.py
```python
import pandas as pd
from pathlib import Path

df = pd.read_csv("data/labels.csv")
before = len(df)

# Remove bad rows
df = df[df["ctr_score"].between(0, 1)]
df = df[df["image_path"].apply(lambda p: Path(p).exists())]
df = df[~((df["censored"] == False) & (df["halflife_days"].isna()))]
df = df.drop_duplicates(subset=["ad_id"])
after = len(df)

print(f"Removed {before - after} rows. Final: {after}")
print(df["vertical"].value_counts())
print(df["source"].value_counts())

df.to_csv("data/labels_clean.csv", index=False)
```

## Supabase migration SQL (define now, run in Slice 3)
```sql
CREATE TABLE cia_uploads (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at    TIMESTAMPTZ DEFAULT now(),
  r2_key        TEXT NOT NULL,
  vertical      TEXT NOT NULL CHECK (vertical IN ('gaming','ecommerce','finance','other')),
  user_session  UUID NOT NULL
);

CREATE TABLE cia_scores (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  upload_id     UUID NOT NULL REFERENCES cia_uploads(id),
  ctr_score     FLOAT NOT NULL,
  halflife_days FLOAT,
  confidence    FLOAT NOT NULL,
  scored_at     TIMESTAMPTZ DEFAULT now(),
  UNIQUE(upload_id)
);

CREATE TABLE cia_sessions (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at    TIMESTAMPTZ DEFAULT now(),
  messages      JSONB NOT NULL DEFAULT '[]'::jsonb
);
```

## requirements.txt
```
requests==2.31.0
Pillow==10.3.0
pandas==2.2.0
numpy==1.26.4
jupyter==1.0.0
matplotlib==3.8.0
tqdm==4.66.0
python-dotenv==1.0.0
fiftyone==0.23.0
```

## Parallel workflow
```
MANUAL (you, day 1):
  apify.com → run Facebook Ad Library Scraper → download JSON to data/apify_raw/

TERMINAL 1 (once JSON downloaded):
  python scripts/apify_downloader.py

TERMINAL 2 (immediately, no waiting):
  python scripts/synthetic_gen.py        ← runs for ~3-4 hrs

ML INTERN (immediately):
  Run Prompt 1A from ml-intern-prompts.md

YOU (while both terminals run):
  Write model/clip_head.py and model/survival.py
  Read about Weibull survival analysis
```

## Bug prevention checklist
- [ ] Both scripts use newline="" in csv.DictWriter open() call
- [ ] Image download in try/except — never crashes the loop
- [ ] halflife_days stored as float — Weibull needs floats
- [ ] censored=False + halflife_days=null rows removed in build_labels.py
- [ ] Dedup on ad_id before writing — no duplicate rows
- [ ] No Supabase calls this slice
- [ ] lsof -i :8000 before starting any server

## Start prompt for Claude Code
```
Starting Slice 1 of Creative Intelligence Agent.
Read CLAUDE.md first, then slice-1-data-pipeline.md in full.

Goals this session:
1. Initialize repo structure exactly as in the slice doc
2. Write scripts/apify_downloader.py exactly as specified
3. Write scripts/synthetic_gen.py exactly as specified
4. Write scripts/build_labels.py
5. Create notebooks/01_eda.ipynb with the 7 cells defined in the slice doc
6. Write requirements.txt and .env.example

Use /plan first. Do not write any code until I approve the plan.
Apify JSON files will be manually placed in data/apify_raw/ —
the downloader script just processes them, it does not call Apify directly.
```

## Done when
- [ ] Apify actor ran on apify.com, JSON files in data/apify_raw/
- [ ] apify_downloader.py processes JSON, images appear in data/raw/
- [ ] synthetic_gen.py running, images appearing in data/synthetic/
- [ ] build_labels.py produces labels_clean.csv with 40K+ rows
- [ ] EDA notebook: all 7 cells render without errors
- [ ] 3+ verticals with >5K rows each

## Next slice
Slice 2 — Vision Model Training. Come back to claude.ai first.
