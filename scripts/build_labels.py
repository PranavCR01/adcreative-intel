import csv
import pandas as pd
from pathlib import Path

COLUMNS = [
    "ad_id", "image_path", "ctr_score", "halflife_days",
    "censored", "vertical", "source", "scraped_at",
    "discontinuation_type",
]

rows = []
with open("data/labels.csv", encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for line in reader:
        if len(line) == 8:
            line.append("")
        elif len(line) != 9:
            continue
        rows.append(dict(zip(COLUMNS, line)))

df = pd.DataFrame(rows, columns=COLUMNS)
before = len(df)

# Coerce types
df["censored"] = df["censored"].isin([True, "True"])
df["ctr_score"] = pd.to_numeric(df["ctr_score"], errors="coerce")
df["halflife_days"] = pd.to_numeric(df["halflife_days"], errors="coerce")
df["discontinuation_type"] = df["discontinuation_type"].replace("", None)

# Backfill missing discontinuation_type
def _disc_type(r):
    if pd.notna(r["discontinuation_type"]):
        return r["discontinuation_type"]
    if r["censored"]:
        return "censored"
    if pd.isna(r["halflife_days"]):
        return pd.NA
    return "cut-out" if r["halflife_days"] < 7 else "wear-out"

df["discontinuation_type"] = df.apply(_disc_type, axis=1)

# Cleaning
df = df[df["ctr_score"].between(0, 1)]
df = df[df["image_path"].apply(lambda p: Path(p).exists())]
df = df[~((df["censored"] == False) & (df["halflife_days"].isna()))]
df = df.drop_duplicates(subset=["ad_id"])

after = len(df)

print(f"Removed {before - after} rows. Final: {after}")
print(df["vertical"].value_counts())
print(df["source"].value_counts())
print(df["discontinuation_type"].value_counts())

df.to_csv("data/labels_clean.csv", index=False, encoding="utf-8")
