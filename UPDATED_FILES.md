# Files Updated with DATA_PATH

## Summary

✅ **Updated `DATA_PATH = "data/labels_clean.csv"` in:**

1. **audit_and_fix.py** - Main data audit script
2. **check_setup.py** - Pre-flight diagnostic

## Files That Auto-Detect Data

These scripts don't need hardcoded paths - they auto-find the experiment datasets:

- **retrain_experiments.py** - Looks for `*exp1*.csv`, `*exp2*.csv`, `*exp3*.csv`
- **run_all_experiments.py** - Calls audit_and_fix.py first, then uses generated files

## How to Run

### Option 1: Use the batch file
```bash
run_check.bat
```

### Option 2: Direct command
```bash
py -3.11 check_setup.py
```

### Option 3: If Python 3.11 isn't your default
```bash
python check_setup.py
```

## What check_setup.py Will Do

1. ✓ Check Python version (3.8+ needed)
2. ✓ Verify required packages are installed:
   - pandas, numpy, torch, transformers
   - scikit-learn, scipy, PIL, tqdm
3. ✓ Check GPU availability (CUDA)
4. ✓ Verify `data/labels_clean.csv` exists and is readable
5. ✓ Inspect data structure:
   - CTR score column
   - Source column (apify vs synthetic)
   - Image path column
   - Halflife column
6. ✓ Check for sentinel rows (ctr_score=1.0)
7. ✓ Verify image paths are accessible

## Expected Output

If everything is OK:
```
================================================================================
PRE-FLIGHT CHECK: Data Cleaning Experiments
================================================================================

Python version: 3.11.x
✓ Python version OK

================================================================================
Checking Required Packages
================================================================================
✓ pandas
✓ numpy
✓ torch
✓ transformers
✓ scikit-learn
✓ scipy
✓ PIL
✓ tqdm

================================================================================
Checking GPU
================================================================================
✓ GPU: NVIDIA GeForce RTX 3090
  CUDA version: 11.8
  GPU memory: 24.0 GB

================================================================================
Checking Data Files
================================================================================

✓ Data file: data/labels_clean.csv
  Rows: 100 (showing first 100 if file is larger)
  Columns: ['image_path', 'ctr_score', 'source', 'halflife', ...]

✓ CTR column found: ctr_score
  Rows with value=1.0 (potential sentinels): 960

✓ Source column found: source
  Values: {'apify': 3502, 'synthetic': 18746}

✓ Image column found: image_path
  ✓ First image accessible: data/images/001.jpg

✓ Halflife column found: halflife

================================================================================
SUMMARY
================================================================================

✓ Setup looks good! Ready to run experiments.

Next step:
  python run_all_experiments.py

Or run step-by-step:
  1. python audit_and_fix.py
  2. python retrain_experiments.py --dataset exp1
  3. python retrain_experiments.py --dataset exp2
```

## Troubleshooting

### Missing packages
```bash
pip install pandas numpy torch transformers scikit-learn scipy pillow tqdm
```

### Data file not found
- Verify the file exists: `data/labels_clean.csv`
- Or update `DATA_PATH` in `audit_and_fix.py` if it's in a different location

### Images not accessible
- Check that image paths in the CSV are correct
- Verify images exist at those paths
- Update paths if needed or set up symbolic links

## Next Steps After check_setup.py Passes

1. **Run the full experiment suite:**
   ```bash
   python run_all_experiments.py
   ```

2. **Or run individual experiments:**
   ```bash
   python audit_and_fix.py
   python retrain_experiments.py --dataset exp1
   python retrain_experiments.py --dataset exp2
   ```

3. **Quick test (10 epochs instead of 30):**
   ```bash
   python audit_and_fix.py
   python retrain_experiments.py --dataset exp1 --epochs 10
   ```

## Files You Can Now Run

✅ **check_setup.py** - Ready to run (updated with correct DATA_PATH)
✅ **audit_and_fix.py** - Ready to run (updated with correct DATA_PATH)
✅ **retrain_experiments.py** - Ready to run (auto-finds experiment datasets)
✅ **run_all_experiments.py** - Ready to run (orchestrates everything)

---

**Current Status:** All files updated with `DATA_PATH = "data/labels_clean.csv"`

**Action Required:** Run `py -3.11 check_setup.py` or `run_check.bat`
