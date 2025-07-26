#!/usr/bin/env python3
"""Comprehensive check of ALL metadata files for category information."""

import json
from pathlib import Path
from collections import Counter
import sys

metadata_dir = Path("/mnt/data/arxiv_data/metadata")

print(f"Scanning ALL metadata files in {metadata_dir}...\n")

# Get all JSON files
all_files = list(metadata_dir.glob("*.json"))
print(f"Total metadata files: {len(all_files)}")

# Track statistics
files_with_categories = 0
category_counter = Counter()
sample_with_categories = []
files_checked = 0

# Check ALL files
for file_path in all_files:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        categories = data.get('categories', [])
        
        if categories and len(categories) > 0:
            files_with_categories += 1
            
            # Count each category
            for cat in categories:
                category_counter[cat] += 1
            
            # Keep samples
            if len(sample_with_categories) < 10:
                sample_with_categories.append({
                    'file': file_path.name,
                    'arxiv_id': data.get('arxiv_id'),
                    'categories': categories,
                    'title': data.get('title', '')[:80]
                })
        
        files_checked += 1
        
        # Progress indicator
        if files_checked % 10000 == 0:
            print(f"  Checked {files_checked} files... found {files_with_categories} with categories")
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

print(f"\n\nFINAL RESULTS:")
print(f"Total files checked: {files_checked}")
print(f"Files with categories: {files_with_categories}")
print(f"Files without categories: {files_checked - files_with_categories}")
print(f"Percentage with categories: {files_with_categories/files_checked*100:.2f}%")

if files_with_categories > 0:
    print(f"\n\nSample files WITH categories:")
    for sample in sample_with_categories:
        print(f"\nFile: {sample['file']}")
        print(f"  ArXiv ID: {sample['arxiv_id']}")
        print(f"  Categories: {sample['categories']}")
        print(f"  Title: {sample['title']}...")
    
    print(f"\n\nTop 30 categories found:")
    for cat, count in category_counter.most_common(30):
        print(f"  {cat}: {count}")
else:
    print("\n\n⚠️  NO FILES FOUND WITH CATEGORIES!")
    
    # Let's check if there's a different field name
    print("\n\nChecking for alternative field names in first 10 files:")
    for file_path in all_files[:10]:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"\n{file_path.name} fields:")
        for key in sorted(data.keys()):
            value = data[key]
            if isinstance(value, list) and len(value) > 0:
                print(f"  {key}: {value[:3]}... (list with {len(value)} items)")
            elif isinstance(value, str) and ('cat' in key.lower() or 'subj' in key.lower()):
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {type(value).__name__}")