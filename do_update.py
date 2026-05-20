import os
import shutil

# Replace the old file with the new one
shutil.copy('audit_and_fix_updated.py', 'audit_and_fix.py')
print("✓ Updated audit_and_fix.py with DATA_PATH = 'data/labels_clean.csv'")

# Clean up
if os.path.exists('audit_and_fix_updated.py'):
    os.remove('audit_and_fix_updated.py')
    print("✓ Removed temporary file")

print("\nNow run: py -3.11 check_setup.py")
