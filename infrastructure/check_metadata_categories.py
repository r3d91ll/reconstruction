#!/usr/bin/env python3
"""Check metadata files for category information."""

import json
from pathlib import Path
import random

metadata_dir = Path("/mnt/data/arxiv_data/metadata")

# Sample random files
metadata_files = list(metadata_dir.glob("*.json"))
sample_files = random.sample(metadata_files, min(20, len(metadata_files)))

print(f"Checking {len(sample_files)} metadata files for category information...\n")

files_with_categories = 0
empty_categories = 0
missing_categories = 0

# Check arxiv_id patterns to infer categories
arxiv_patterns = {
    'cs': 'Computer Science',
    'math': 'Mathematics', 
    'physics': 'Physics',
    'stat': 'Statistics',
    'q-bio': 'Quantitative Biology',
    'q-fin': 'Quantitative Finance',
    'econ': 'Economics',
    'eess': 'Electrical Engineering and Systems Science'
}

category_from_id = []

for file_path in sample_files[:10]:  # Show first 10
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    arxiv_id = data.get('arxiv_id', 'unknown')
    categories = data.get('categories', None)
    
    print(f"\nFile: {file_path.name}")
    print(f"  arxiv_id: {arxiv_id}")
    print(f"  categories field: {categories}")
    
    # Try to infer category from arxiv_id format
    # Old format: cs/0301001 or math.GT/0301001
    # New format: 2301.00001
    if '/' in arxiv_id:
        prefix = arxiv_id.split('/')[0]
        if '.' in prefix:
            main_cat = prefix.split('.')[0]
        else:
            main_cat = prefix
        print(f"  Inferred from ID: {main_cat}")
        category_from_id.append(main_cat)
    
    if categories is None:
        missing_categories += 1
    elif len(categories) == 0:
        empty_categories += 1
    else:
        files_with_categories += 1

print(f"\n\nSummary of {len(sample_files)} files:")
print(f"  Files with categories: {files_with_categories}")
print(f"  Files with empty categories: {empty_categories}")
print(f"  Files missing categories field: {missing_categories}")

# Check if arxiv IDs contain category info
print(f"\n\nChecking if arXiv IDs contain category prefixes...")
old_format_count = 0
for file_path in metadata_files[:1000]:  # Check first 1000
    arxiv_id = file_path.stem
    if '/' in arxiv_id or any(arxiv_id.startswith(cat) for cat in arxiv_patterns.keys()):
        old_format_count += 1

print(f"Files with old format IDs (containing categories): {old_format_count}/1000")

# Check for other potential category fields
print("\n\nChecking for alternative category fields...")
with open(sample_files[0], 'r') as f:
    sample = json.load(f)
    
print("All fields in metadata:")
for key in sorted(sample.keys()):
    print(f"  - {key}: {type(sample[key]).__name__}")
    if 'cat' in key.lower() or 'subj' in key.lower() or 'topic' in key.lower():
        print(f"    ^^ POTENTIAL CATEGORY FIELD: {sample[key]}")