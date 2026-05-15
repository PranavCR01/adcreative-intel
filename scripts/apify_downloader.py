import json, requests, hashlib, csv, os, time, random
from pathlib import Path
from datetime import datetime

RAW_DIR = Path("data/raw")
APIFY_DIR = Path("data/apify_raw")
LABELS_FILE = Path("data/labels.csv")

VERTICAL_MAP = {
    # Gaming — ordered longest-match first to avoid "game" swallowing "mobile game"
    "mobile game": "gaming", "download game": "gaming",
    "play now": "gaming", "play free": "gaming",
    "game": "gaming", "play": "gaming", "download": "gaming",
    "install": "gaming", "level": "gaming", "battle": "gaming", "rpg": "gaming",
    # Ecommerce
    "shop now": "ecommerce", "buy now": "ecommerce",
    "free shipping": "ecommerce", "limited offer": "ecommerce",
    "order now": "ecommerce", "shop": "ecommerce", "sale": "ecommerce",
    "discount": "ecommerce", "delivery": "ecommerce", "checkout": "ecommerce",
    "fashion": "ecommerce", "clothing": "ecommerce",
    "skincare": "ecommerce", "beauty": "ecommerce",
    # Finance — longer phrases first
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

def compute_labels(start_ts, stop_ts) -> tuple[float, float | None, bool]:
    if not stop_ts:
        return 0.7, None, True
    from datetime import datetime as dt
    try:
        days = (dt.utcfromtimestamp(int(stop_ts)) -
                dt.utcfromtimestamp(int(start_ts))).days
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
        with open(LABELS_FILE, "w", newline="", encoding="utf-8") as f:
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
        with open(json_file, encoding="utf-8") as f:
            ads = json.load(f)

        added = 0
        for ad in ads:
            snapshot = ad.get("snapshot") or {}
            images = snapshot.get("images") or []
            img_url = images[0].get("original_image_url") if images else None
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
                ad.get("start_date"),
                ad.get("end_date")
            )

            with open(LABELS_FILE, "a", newline="", encoding="utf-8") as f:
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
