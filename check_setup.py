"""
Quick diagnostic check before running experiments.

This script checks:
1. Required packages installed
2. Data file exists and is readable
3. GPU availability
4. Data format compatibility

Run this FIRST to catch issues early.
"""

import sys
import os

def check_package(name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = name
    
    try:
        __import__(import_name)
        print(f"✓ {name}")
        return True
    except ImportError:
        print(f"✗ {name} - NOT INSTALLED")
        return False

def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU: {device_name}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("⚠️  GPU: Not available (will use CPU - training will be SLOW)")
            return False
    except:
        print("✗ GPU: Cannot check (torch not installed)")
        return False

def check_data_file(filepath):
    """Check if data file is readable and has required structure."""
    import pandas as pd
    
    try:
        # Try to read first few rows
        if filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        else:
            df = pd.read_csv(filepath, nrows=100)
        
        print(f"\n✓ Data file: {filepath}")
        print(f"  Rows: {len(df)} (showing first 100 if file is larger)")
        print(f"  Columns: {list(df.columns)}")
        
        # Check for expected columns
        # CTR column (required)
        ctr_cols = [c for c in df.columns if 'ctr' in c.lower()]
        if ctr_cols:
            print(f"\n✓ CTR column found: {ctr_cols[0]}")
            if 'score' in ctr_cols[0].lower():
                # Check for sentinels
                sentinel_count = (df[ctr_cols[0]] == 1.0).sum()
                print(f"  Rows with value=1.0 (potential sentinels): {sentinel_count}")
        else:
            print(f"\n✗ CTR column NOT found - scripts may fail")
        
        # Source column (optional but important)
        source_cols = [c for c in df.columns if 'source' in c.lower()]
        if source_cols:
            print(f"\n✓ Source column found: {source_cols[0]}")
            print(f"  Values: {df[source_cols[0]].value_counts().to_dict()}")
        else:
            print(f"\n⚠️  Source column NOT found - cannot distinguish apify vs synthetic")
        
        # Image path column
        image_cols = [c for c in df.columns if 'image' in c.lower() or 'path' in c.lower()]
        if image_cols:
            print(f"\n✓ Image column found: {image_cols[0]}")
            # Check if first path exists
            first_path = df[image_cols[0]].iloc[0]
            if pd.notna(first_path) and os.path.exists(str(first_path)):
                print(f"  ✓ First image accessible: {first_path}")
            else:
                print(f"  ⚠️  First image not found: {first_path}")
                print(f"     (Training will fail if images aren't accessible)")
        else:
            print(f"\n✗ Image column NOT found - training will fail")
        
        # Halflife column (optional)
        halflife_cols = [c for c in df.columns if 'halflife' in c.lower() or 'half_life' in c.lower()]
        if halflife_cols:
            print(f"\n✓ Halflife column found: {halflife_cols[0]}")
        else:
            print(f"\n⚠️  Halflife column NOT found - Weibull loss will use defaults")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Cannot read data file: {filepath}")
        print(f"  Error: {e}")
        return False

def main():
    print("="*80)
    print("PRE-FLIGHT CHECK: Data Cleaning Experiments")
    print("="*80)
    
    # Check Python version
    print(f"\nPython version: {sys.version}")
    if sys.version_info < (3, 8):
        print("⚠️  Python 3.8+ recommended")
    else:
        print("✓ Python version OK")
    
    # Check required packages
    print("\n" + "="*80)
    print("Checking Required Packages")
    print("="*80)
    
    packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('scikit-learn', 'sklearn'),
        ('scipy', 'scipy'),
        ('PIL', 'PIL'),
        ('tqdm', 'tqdm'),
    ]
    
    missing_packages = []
    for pkg_name, import_name in packages:
        if not check_package(pkg_name, import_name):
            missing_packages.append(pkg_name)
    
    if missing_packages:
        print(f"\n✗ Missing packages: {', '.join(missing_packages)}")
        print(f"\nInstall with:")
        print(f"  pip install {' '.join(missing_packages)}")
        return False
    
    # Check GPU
    print("\n" + "="*80)
    print("Checking GPU")
    print("="*80)
    check_gpu()
    
    # Check data files
    print("\n" + "="*80)
    print("Checking Data Files")
    print("="*80)
    
    # Use the configured data path
    data_path = "data/labels_clean.csv"
    
    if not os.path.exists(data_path):
        print(f"\n✗ Data file not found: {data_path}")
        print("\nPlease ensure:")
        print("  1. The file exists at this location")
        print("  2. Or update DATA_PATH in audit_and_fix.py")
        return False
    
    if not check_data_file(data_path):
        return False
    
    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\n✓ Setup looks good! Ready to run experiments.")
    print("\nNext step:")
    print("  python run_all_experiments.py")
    print("\nOr run step-by-step:")
    print("  1. python audit_and_fix.py")
    print("  2. python retrain_experiments.py --dataset exp1")
    print("  3. python retrain_experiments.py --dataset exp2")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
