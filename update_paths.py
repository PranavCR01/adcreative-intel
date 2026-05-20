"""Quick script to update DATA_PATH in all files."""

import os
import re

# Target path
NEW_DATA_PATH = 'data/labels_clean.csv'

def update_file(filepath, old_pattern, new_value):
    """Update DATA_PATH in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace the auto-find logic with direct path
        if 'audit_and_fix.py' in filepath:
            # Replace the entire DATA_PATH section
            pattern = r'DATA_PATH = None\nfor path in possible_paths:.*?break'
            replacement = f'DATA_PATH = "{NEW_DATA_PATH}"'
            
            if re.search(pattern, content, re.DOTALL):
                new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"✓ Updated {filepath}")
                return True
        
        print(f"⊘ Skipped {filepath} (no changes needed)")
        return False
        
    except Exception as e:
        print(f"✗ Error updating {filepath}: {e}")
        return False

# Update audit_and_fix.py
print("Updating DATA_PATH in scripts...\n")
update_file('audit_and_fix.py', None, NEW_DATA_PATH)

print(f"\n✓ Updated DATA_PATH to: {NEW_DATA_PATH}")
print(f"\nNow run: py -3.11 check_setup.py")
