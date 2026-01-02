#!/usr/bin/env python3
"""Fix smart quotes in dashboard.py"""

file_path = r'c:\Users\KarimJaber\Downloads\FalconOne App\falconone\ui\dashboard.py'

print("Reading file...")
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

print(f"Original length: {len(content)}")

# Replace smart quotes with straight quotes
replacements = {
    '\u2018': "'",  # ' -> '
    '\u2019': "'",  # ' -> '
    '\u201C': '"',  # " -> "
    '\u201D': '"',  # " -> "
}

for old, new in replacements.items():
    count = content.count(old)
    if count > 0:
        print(f"Replacing {count} instances of {repr(old)} with {repr(new)}")
        content = content.replace(old, new)

print(f"After replacement length: {len(content)}")

print("Writing file...")
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Done! Smart quotes fixed.")
