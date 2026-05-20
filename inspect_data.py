import os
import glob

# Find data files
print("=== Data Files ===")
for ext in ['*.csv', '*.parquet', '*.json']:
    files = glob.glob(ext) + glob.glob(f'data/{ext}') + glob.glob(f'**/{ext}', recursive=True)
    for f in files[:10]:
        print(f)

print("\n=== Python Scripts ===")
py_files = glob.glob('*.py') + glob.glob('**/*.py', recursive=True)
for f in py_files[:20]:
    if 'train' in f.lower() or 'model' in f.lower():
        print(f)
