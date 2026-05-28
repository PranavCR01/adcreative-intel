"""
model/train_vertical_classifier.py
====================================
Trains a logistic regression classifier on frozen SigLIP 2 embeddings
to auto-detect ad vertical (gaming / ecommerce / finance / other).

Data: data/labels_clean.csv, source=apify rows only, sentinels (ctr_score==1.0) removed.
      2,542 clean rows across 4 verticals after filtering.

Output: model/vertical_classifier.pkl  — fitted sklearn LogisticRegression
        (clf.predict() returns vertical string; clf.predict_proba() gives confidence)

Usage:
    python model/train_vertical_classifier.py

Requirements (add to requirements-dev.txt if missing):
    scikit-learn, joblib, transformers, torch, Pillow, pandas, tqdm
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from transformers import AutoProcessor, SiglipVisionModel
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

REPO_ROOT    = Path(__file__).resolve().parent.parent
CSV_PATH     = REPO_ROOT / "data" / "labels_clean.csv"
OUTPUT_PATH  = REPO_ROOT / "model" / "vertical_classifier.pkl"
MODEL_NAME   = "google/siglip2-base-patch16-224"
BATCH_SIZE   = 32    # CLAUDE.md: reduce to 32 if OOM on 1650 Ti
SENTINEL_CTR = 1.0   # rows with ctr_score == 1.0 are label sentinels, not real data
N_FOLDS      = 5

# ── Data loading ──────────────────────────────────────────────────────────────

def load_apify_rows(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    total_apify = (df["source"] == "apify").sum()
    df = df[df["source"] == "apify"].copy()
    print(f"Apify rows total          : {total_apify}")

    sentinels = (df["ctr_score"] == SENTINEL_CTR).sum()
    df = df[df["ctr_score"] != SENTINEL_CTR].copy()
    print(f"Sentinel rows removed     : {sentinels}")
    print(f"Clean rows for training   : {len(df)}")
    print()
    print("Vertical distribution:")
    for v, n in df["vertical"].value_counts().items():
        print(f"  {v:<12} {n:>5}  ({n/len(df)*100:.1f}%)")
    print()
    return df.reset_index(drop=True)


# ── Embedding extraction ───────────────────────────────────────────────────────

def extract_embeddings(
    df: pd.DataFrame,
    processor: AutoProcessor,
    backbone: SiglipVisionModel,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Batch-extracts 768-dim SigLIP 2 pooler_output for every image in df.

    Returns:
        X      : float32 ndarray (N, 768)
        y      : object ndarray (N,) of vertical strings
        skipped: list of df indices that failed to load (logged, excluded from X/y)
    """
    X_list:   list[np.ndarray] = []
    y_list:   list[str]        = []
    skipped:  list[int]        = []

    indices = list(range(len(df)))

    for batch_start in tqdm(
        range(0, len(df), BATCH_SIZE),
        desc="Extracting embeddings",
        unit="batch",
    ):
        batch_idx = indices[batch_start : batch_start + BATCH_SIZE]
        batch_df  = df.iloc[batch_idx]

        images:   list[Image.Image] = []
        row_idxs: list[int]         = []

        for i, (_, row) in zip(batch_idx, batch_df.iterrows()):
            path = REPO_ROOT / row["image_path"]
            try:
                images.append(Image.open(path).convert("RGB"))
                row_idxs.append(i)
            except Exception as exc:
                print(f"\n  SKIP {row['image_path']}: {exc}", file=sys.stderr)
                skipped.append(i)

        if not images:
            continue

        inputs       = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"]   # (B, 3, 224, 224)

        with torch.no_grad():
            out        = backbone(pixel_values=pixel_values)
            embeddings = out.pooler_output.cpu().numpy()   # (B, 768)

        for emb, idx in zip(embeddings, row_idxs):
            X_list.append(emb)
            y_list.append(df.at[idx, "vertical"])

    X = np.array(X_list, dtype=np.float32)   # (N, 768)
    y = np.array(y_list)                     # (N,)
    return X, y, skipped


# ── Classifier training ────────────────────────────────────────────────────────

def train_and_evaluate(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """
    5-fold stratified CV with cross_val_predict, then final fit on all data.
    Returns the fitted classifier (all data).
    """
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=42,
    )

    print("Running 5-fold stratified cross-validation ...")
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(clf, X, y, cv=cv)

    overall_acc = (y_pred_cv == y).mean()
    print(f"\nOverall CV accuracy: {overall_acc:.3f}  ({overall_acc*100:.1f}%)")
    print()

    # classification_report shows per-class recall (= per-class accuracy)
    print("Per-vertical results (recall = accuracy within that class):")
    print(classification_report(y, y_pred_cv, digits=3))

    # Final fit on all data
    print("Fitting final classifier on all data ...")
    clf.fit(X, y)
    print(f"Classes: {list(clf.classes_)}")
    print()

    return clf


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── 1. Load data ─────────────────────────────────────────────────────────
    df = load_apify_rows(CSV_PATH)

    # ── 2. Load frozen backbone ───────────────────────────────────────────────
    print(f"Loading SigLIP 2 backbone ({MODEL_NAME}) ...")
    try:
        processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        )
        backbone = SiglipVisionModel.from_pretrained(
            MODEL_NAME,
            use_safetensors=True,
            local_files_only=True,
        )
    except OSError:
        print(
            f"\nERROR: {MODEL_NAME} not found in local HF cache.\n"
            "Run any training script first to download it, or set local_files_only=False.",
            file=sys.stderr,
        )
        sys.exit(1)

    backbone.eval()
    # Backbone must stay frozen — never set requires_grad=True (CLAUDE.md constraint)
    for param in backbone.parameters():
        param.requires_grad = False

    print(f"Backbone loaded (frozen). Extracting embeddings with batch_size={BATCH_SIZE} ...")
    print()

    # ── 3. Extract embeddings ─────────────────────────────────────────────────
    X, y, skipped = extract_embeddings(df, processor, backbone)
    print(f"\nExtracted: {len(X)} embeddings  |  skipped: {len(skipped)} images")
    print(f"X shape: {X.shape}  |  y classes: {sorted(set(y))}")
    print()

    if len(X) == 0:
        print("ERROR: no embeddings extracted. Check image paths.", file=sys.stderr)
        sys.exit(1)

    # ── 4. Train and evaluate ─────────────────────────────────────────────────
    clf = train_and_evaluate(X, y)

    # ── 5. Smoke-test the saved artifact ─────────────────────────────────────
    sample_emb = X[:1]                              # first row
    pred       = clf.predict(sample_emb)[0]
    proba      = clf.predict_proba(sample_emb)[0]
    conf       = float(proba.max())
    print(f"Smoke test — predicted: {pred}  confidence: {conf:.3f}")
    print()

    # ── 6. Save ───────────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
