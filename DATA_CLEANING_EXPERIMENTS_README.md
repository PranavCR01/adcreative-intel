# Data Cleaning Experiments: Sentinel Row Removal

## Problem Statement

Your CLIP + Weibull CTR model achieves **Spearman r=0.254** on the test set, but training data has a quality issue:
- **960 apify rows** have `ctr_score=1.0` as a sentinel value (ads still running at scrape time)
- These are NOT real CTR values, distorting training
- Clean apify data: **2,542 rows**
- Synthetic data: **18,746 rows**

## Questions to Answer

1. Should you drop the 960 sentinel rows and retrain?
2. Should you downsample synthetic data to 60/40 real/synthetic?
3. Is there a smarter fix (treating sentinels as censored)?
4. Can Spearman r improve to 0.35+ with clean data?

## Solution: 3 Experiments

### Experiment 1: Drop Sentinels (Baseline)
- Remove 960 sentinel rows
- Train on 2,542 clean apify + 18,746 synthetic
- **Hypothesis**: Removing bad labels improves model

### Experiment 2: Drop Sentinels + Balance Data
- Remove sentinels + downsample synthetic to 60/40 ratio
- Train on ~2,542 apify + ~1,695 synthetic = ~4,237 total
- **Hypothesis**: Balanced real/synthetic prevents overfitting to synthetic distribution

### Experiment 3: Treat Sentinels as Censored
- Keep all rows, add `is_censored` flag
- Mask CTR values for sentinels (set to NaN)
- **Hypothesis**: Preserves information (image features) while handling missing labels

## Files Created

### 1. `audit_and_fix.py`
**What it does:**
- Auto-finds your training data file
- Identifies sentinel rows (ctr_score=1.0)
- Analyzes apify vs synthetic distribution
- Generates 3 cleaned datasets

**How to run:**
```bash
python audit_and_fix.py
```

**What you get:**
- `*_exp1_no_sentinels.csv` - Sentinels dropped
- `*_exp2_balanced.csv` - Sentinels dropped + synthetic downsampled
- `*_exp3_censored.csv` - Sentinels marked as censored

### 2. `retrain_experiments.py`
**What it does:**
- Trains CLIP-ViT-B/32 + Weibull model on cleaned data
- Uses your existing setup (AdamW lr=5e-4, 30 epochs, batch 64)
- Multi-task loss: BCE + Weibull NLL + MSE
- Evaluates Spearman r on test set

**How to run:**
```bash
# Experiment 1
python retrain_experiments.py --dataset exp1

# Experiment 2
python retrain_experiments.py --dataset exp2

# Experiment 3 (advanced - requires censored data handling)
python retrain_experiments.py --dataset exp3
```

**What you get:**
- Trained model: `model_{exp}_best.pt`
- Metrics: `results_{exp}_{timestamp}.json`

### 3. `run_all_experiments.py` ⭐ **RECOMMENDED**
**What it does:**
- Runs ALL 3 experiments automatically
- Compares Spearman r improvements
- Generates final recommendations

**How to run:**
```bash
python run_all_experiments.py
```

This is the **one-click solution** - it will:
1. Run data audit
2. Train 3 models (one per experiment)
3. Compare results
4. Tell you which approach works best
5. Answer all 4 of your questions

**What you get:**
- All 3 trained models
- `experiment_comparison_{timestamp}.csv` - Side-by-side comparison
- Clear recommendation on which dataset to use

## Quick Start (Recommended Path)

### Option A: Run Everything (Automated)
```bash
# 1. Update DATA_PATH in audit_and_fix.py if needed (or let it auto-find)
# 2. Run all experiments
python run_all_experiments.py

# 3. Check results
# - Best model saved as model_exp*_best.pt
# - Comparison table in experiment_comparison_*.csv
```

### Option B: Run Step-by-Step
```bash
# 1. Audit data and create cleaned datasets
python audit_and_fix.py

# 2. Train on each dataset
python retrain_experiments.py --dataset exp1  # Drop sentinels
python retrain_experiments.py --dataset exp2  # Balanced
python retrain_experiments.py --dataset exp3  # Censored

# 3. Compare results manually
```

## Configuration

### Update Data Path
If auto-detection fails, edit `audit_and_fix.py`:
```python
DATA_PATH = "path/to/your/training_data.csv"  # or .parquet
```

### Update Column Names
If your columns have different names, update `Config` in `retrain_experiments.py`:
```python
class Config:
    image_col = 'your_image_column'  # default: 'image_path'
    # Script auto-detects: ctr_score, source, halflife
```

### Adjust Hyperparameters
In `retrain_experiments.py` or via CLI:
```python
class Config:
    lr = 5e-4            # Learning rate
    batch_size = 64      # Batch size
    epochs = 30          # Training epochs
    hidden_dim = 256     # Hidden layer size
    
    bce_weight = 1.0     # Binary cross-entropy weight
    weibull_weight = 1.0 # Weibull NLL weight
```

Or:
```bash
python retrain_experiments.py --dataset exp1 --epochs 50 --batch_size 128 --lr 1e-4
```

## Expected Results

### Realistic Expectations

**Current performance:** Spearman r = 0.254

**After cleaning:**
- **Best case:** r = 0.30-0.40 (18-57% improvement)
- **Likely case:** r = 0.27-0.35 (6-38% improvement)
- **Worst case:** r = 0.25-0.28 (0-10% improvement)

**Why these ranges?**
- Dataset size (~21k samples) can support r ≈ 0.4-0.5 with good features
- Current gap suggests feature quality is the main bottleneck
- Data cleaning alone may give 5-15% improvement
- Reaching r > 0.35 likely requires:
  - Better features (text, metadata)
  - Larger CLIP model (ViT-L/14)
  - More real data (>2,542 samples)

### Interpreting Results

**If r improves to 0.30+:**
✅ Data cleaning worked! Sentinels were hurting performance.
→ Use the best experiment dataset going forward

**If r stays < 0.28:**
⚠️ Sentinels weren't the main issue
→ Focus on feature engineering (add text, metadata)
→ Consider collecting more real data

**If Exp2 (balanced) beats Exp1:**
→ Synthetic data was overfitting - use 60/40 ratio

**If Exp3 (censored) works best:**
→ Sentinels still contained useful image features
→ Advanced approach worth implementing in production

## Troubleshooting

### "Could not auto-find data"
→ Set `DATA_PATH` manually in `audit_and_fix.py`

### "File not found: image_path"
→ Images need to be accessible at paths in your CSV
→ Update paths or set `Config.image_col` to correct column

### CUDA out of memory
→ Reduce batch size: `--batch_size 32` or `--batch_size 16`

### Training is slow
→ Ensure CUDA is available: check `Device: cuda` in output
→ Reduce epochs for quick test: `--epochs 10`

## After Experiments

### Use Best Model in Production

```python
import torch
from retrain_experiments import CLIPWeibullModel, Config

# Load best model
config = Config()
model = CLIPWeibullModel(config)
model.load_state_dict(torch.load('model_exp1_best.pt'))  # or exp2/exp3
model.eval()

# Use for inference
# ... (same as your current inference pipeline)
```

### Next Steps to Reach r > 0.35

1. **Feature Engineering**
   - Add text features (ad headlines, copy)
   - Include metadata (audience, placement, time)
   - Multi-modal fusion (CLIP + text embeddings)

2. **Model Improvements**
   - Upgrade to CLIP ViT-L/14 (larger model)
   - Try ensemble (multiple CLIPs + metadata)
   - Experiment with different architectures

3. **Data Collection**
   - Collect more real apify data (target: 5,000-10,000 samples)
   - Improve synthetic data quality (use better generation)
   - Balance categories/ad types

4. **Hyperparameter Tuning**
   - Learning rate sweep
   - Architecture search (hidden dims, layers)
   - Loss weight tuning (BCE vs Weibull)

## Summary

### What These Scripts Do
✅ Automatically find and audit your data
✅ Generate 3 cleaned datasets (drop, balance, censor)
✅ Train CLIP + Weibull models on each
✅ Compare Spearman r improvements
✅ Give clear recommendation

### What You Get
✅ Answer to: "Should I drop sentinels?" (YES/NO + evidence)
✅ Answer to: "Should I balance real/synthetic?" (YES/NO + evidence)
✅ Answer to: "Can I reach r > 0.35?" (realistic assessment)
✅ Best model saved and ready to deploy

### Time Investment
- **Setup:** 5 min (update DATA_PATH if needed)
- **Runtime:** 2-6 hours (30 epochs × 3 experiments)
- **Analysis:** 5 min (read experiment_comparison.csv)

**Bottom line:** Run `python run_all_experiments.py` and get definitive answers to your data quality questions.

---

**Need help?** Check the output of each script - they include detailed diagnostics and recommendations.
