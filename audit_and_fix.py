"""
Data quality audit and fix for CLIP + Weibull CTR model.

Usage:
    python audit_and_fix.py

Outputs:
    - *_exp1_no_sentinels.csv
    - *_exp2_balanced.csv
    - *_exp3_censored.csv
"""

import pandas as pd
import numpy as np
import os

# ============================================
# CONFIGURATION
# ============================================

DATA_PATH = "data/labels_clean.csv"

print(f"Using data file: {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    print(f"\nERROR: Data file not found: {DATA_PATH}")
    print("Please check the path and try again.")
    exit(1)

# ============================================
# LOAD DATA
# ============================================

if DATA_PATH.endswith('.parquet'):
    df = pd.read_parquet(DATA_PATH)
else:
    df = pd.read_csv(DATA_PATH)

print(f"\n=== INITIAL DATA ===")
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# ============================================
# IDENTIFY KEY COLUMNS
# ============================================

# Find CTR score column
ctr_col = None
for col in df.columns:
    if 'ctr' in col.lower() and 'score' in col.lower():
        ctr_col = col
        break
if ctr_col is None:
    for col in df.columns:
        if 'ctr' in col.lower():
            ctr_col = col
            break

# Find source column (apify vs synthetic)
source_col = None
for col in df.columns:
    if 'source' in col.lower():
        source_col = col
        break
if source_col is None:
    # Try to infer from unique values
    for col in df.columns:
        if df[col].dtype == 'object':
            uniq = df[col].unique()
            if any('apify' in str(v).lower() for v in uniq) or any('synthetic' in str(v).lower() for v in uniq):
                source_col = col
                break

# Find halflife column
halflife_col = None
for col in df.columns:
    if 'halflife' in col.lower() or 'half_life' in col.lower():
        halflife_col = col
        break

print(f"\nIdentified columns:")
print(f"  CTR score: {ctr_col}")
print(f"  Source: {source_col}")
print(f"  Halflife: {halflife_col}")

if ctr_col is None:
    print("\nERROR: Could not find CTR score column. Please update script.")
    exit(1)

# ============================================
# AUDIT CURRENT DATA
# ============================================

print(f"\n=== DATA QUALITY AUDIT ===")

# Sentinel detection
sentinels = df[df[ctr_col] == 1.0]
print(f"\nRows with ctr_score=1.0 (sentinels): {len(sentinels)}")

if source_col:
    print(f"\nSource distribution (all data):")
    print(df[source_col].value_counts())
    
    print(f"\nSource distribution (sentinels only):")
    print(sentinels[source_col].value_counts())
    
    # Count clean rows by source
    non_sentinel = df[df[ctr_col] != 1.0]
    print(f"\nSource distribution (after removing sentinels):")
    print(non_sentinel[source_col].value_counts())

# CTR distribution
print(f"\nCTR score statistics:")
print(df[ctr_col].describe())

print(f"\nCTR score distribution:")
print(f"  = 1.0 (sentinels): {(df[ctr_col] == 1.0).sum()}")
print(f"  > 0.9 and < 1.0: {((df[ctr_col] > 0.9) & (df[ctr_col] < 1.0)).sum()}")
print(f"  0.5 - 0.9: {((df[ctr_col] >= 0.5) & (df[ctr_col] <= 0.9)).sum()}")
print(f"  < 0.5: {(df[ctr_col] < 0.5).sum()}")

# ============================================
# EXPERIMENT 1: DROP SENTINELS
# ============================================

print(f"\n=== EXPERIMENT 1: Drop sentinel rows ===")

df_exp1 = df[df[ctr_col] != 1.0].copy()
print(f"Rows after dropping sentinels: {len(df_exp1)}")

if source_col:
    print(f"Source distribution:")
    source_counts = df_exp1[source_col].value_counts()
    print(source_counts)
    
    total = len(df_exp1)
    for src, count in source_counts.items():
        pct = 100 * count / total
        print(f"  {src}: {count} ({pct:.1f}%)")

output_path_1 = DATA_PATH.replace('.csv', '_exp1_no_sentinels.csv').replace('.parquet', '_exp1_no_sentinels.csv')
df_exp1.to_csv(output_path_1, index=False)
print(f"Saved to: {output_path_1}")

# ============================================
# EXPERIMENT 2: DROP SENTINELS + DOWNSAMPLE SYNTHETIC
# ============================================

print(f"\n=== EXPERIMENT 2: Drop sentinels + downsample synthetic to 60/40 ===")

df_exp2 = df[df[ctr_col] != 1.0].copy()

if source_col:
    # Identify apify vs synthetic
    apify_mask = df_exp2[source_col].str.contains('apify', case=False, na=False)
    df_apify = df_exp2[apify_mask]
    df_synthetic = df_exp2[~apify_mask]
    
    n_apify = len(df_apify)
    n_synthetic = len(df_synthetic)
    
    print(f"Before downsampling:")
    print(f"  Apify: {n_apify}")
    print(f"  Synthetic: {n_synthetic}")
    print(f"  Ratio: {100*n_apify/(n_apify+n_synthetic):.1f}% / {100*n_synthetic/(n_apify+n_synthetic):.1f}%")
    
    # Target: 60% real (apify), 40% synthetic
    target_total = int(n_apify / 0.6)
    target_synthetic = int(target_total * 0.4)
    
    if target_synthetic < n_synthetic:
        df_synthetic_sampled = df_synthetic.sample(n=target_synthetic, random_state=42)
        df_exp2 = pd.concat([df_apify, df_synthetic_sampled], ignore_index=True)
        
        print(f"\nAfter downsampling:")
        print(f"  Apify: {len(df_apify)}")
        print(f"  Synthetic: {len(df_synthetic_sampled)}")
        print(f"  Total: {len(df_exp2)}")
        print(f"  Ratio: {100*len(df_apify)/len(df_exp2):.1f}% / {100*len(df_synthetic_sampled)/len(df_exp2):.1f}%")
    else:
        print(f"\nNOTE: Not enough synthetic data to downsample. Keeping all {n_synthetic} synthetic rows.")

output_path_2 = DATA_PATH.replace('.csv', '_exp2_balanced.csv').replace('.parquet', '_exp2_balanced.csv')
df_exp2.to_csv(output_path_2, index=False)
print(f"Saved to: {output_path_2}")

# ============================================
# EXPERIMENT 3: KEEP SENTINELS AS CENSORED
# ============================================

print(f"\n=== EXPERIMENT 3: Keep sentinels, mark as censored ===")

df_exp3 = df.copy()

# Add censoring flag
df_exp3['is_censored'] = (df_exp3[ctr_col] == 1.0).astype(int)

# Mask sentinel CTR values (set to NaN)
df_exp3.loc[df_exp3['is_censored'] == 1, ctr_col] = np.nan

# Set halflife to null for censored rows
if halflife_col:
    df_exp3.loc[df_exp3['is_censored'] == 1, halflife_col] = np.nan

print(f"Total rows: {len(df_exp3)}")
print(f"Censored rows: {df_exp3['is_censored'].sum()}")
print(f"Clean rows: {(df_exp3['is_censored'] == 0).sum()}")

output_path_3 = DATA_PATH.replace('.csv', '_exp3_censored.csv').replace('.parquet', '_exp3_censored.csv')
df_exp3.to_csv(output_path_3, index=False)
print(f"Saved to: {output_path_3}")

# ============================================
# SUMMARY
# ============================================

print(f"\n=== SUMMARY ===")
print(f"Original data: {len(df)} rows")
print(f"  Sentinels (ctr=1.0): {len(sentinels)}")
print(f"  Clean: {len(df) - len(sentinels)}")

print(f"\nGenerated datasets:")
print(f"1. {output_path_1}")
print(f"   - Drops {len(sentinels)} sentinel rows")
print(f"   - {len(df_exp1)} rows remaining")

print(f"\n2. {output_path_2}")
print(f"   - Drops sentinels + balances real/synthetic to 60/40")
print(f"   - {len(df_exp2)} rows remaining")

print(f"\n3. {output_path_3}")
print(f"   - Keeps all rows, marks sentinels as censored")
print(f"   - {len(df_exp3)} rows (same as original)")
print(f"   - Added 'is_censored' column")

print(f"\n=== RECOMMENDATIONS ===")

if source_col and len(df_exp1) > 0:
    apify_count = (df_exp1[source_col].str.contains('apify', case=False, na=False)).sum()
    synthetic_count = len(df_exp1) - apify_count
    real_ratio = 100 * apify_count / len(df_exp1)
    
    print(f"\nCurrent data quality:")
    print(f"  - Sentinels comprise {100*len(sentinels)/len(df):.1f}% of original data")
    print(f"  - After removal: {real_ratio:.1f}% real data")
    
    if real_ratio < 15:
        print(f"\nWARNING: Only {real_ratio:.1f}% real data after cleaning!")
        print(f"  Recommendation: Use Experiment 2 (balanced)")
    elif real_ratio < 30:
        print(f"\nWARNING: Imbalanced: {real_ratio:.1f}% real data")
        print(f"  Recommendation: Use Experiment 2 (balanced)")
    else:
        print(f"\nOK: Reasonable balance at {real_ratio:.1f}% real data")
        print(f"  Recommendation: Start with Experiment 1")

print(f"\nExpected performance improvement:")
print(f"  - Current Spearman r: 0.254")
print(f"  - With clean data: realistic target 0.30-0.40")

print(f"\nNext steps:")
print(f"  1. Retrain on Experiment 1 dataset")
print(f"  2. Compare with Experiment 2 if real data < 30%")
print(f"  3. Experiment 3 requires custom loss (skip for now)")
