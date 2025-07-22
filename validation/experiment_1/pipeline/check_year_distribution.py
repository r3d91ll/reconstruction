#!/usr/bin/env python3
"""
Check year distribution of papers in source directory
"""

import json
import os
from glob import glob
from collections import Counter

papers_dir = "/home/todd/olympus/Erebus/unstructured/papers/"

print("Analyzing year distribution...")
print("=" * 60)

# Get all JSON files
json_files = glob(os.path.join(papers_dir, "*.json"))
print(f"Total papers found: {len(json_files)}")

# Count papers by year
year_counts = Counter()
no_year_count = 0

for json_file in json_files:
    try:
        with open(json_file, 'r') as f:
            paper = json.load(f)
        year = paper.get('year', None)
        if year:
            year_counts[year] += 1
        else:
            no_year_count += 1
    except:
        pass

# Show distribution
print("\nPapers by year:")
for year in sorted(year_counts.keys()):
    count = year_counts[year]
    print(f"  {year}: {count:,} papers")

print(f"\nPapers without year: {no_year_count}")

# Calculate cumulative counts
cumulative = 0
print("\nCumulative count (oldest to newest):")
for year in sorted(year_counts.keys()):
    cumulative += year_counts[year]
    print(f"  Up to {year}: {cumulative:,} papers")
    if cumulative >= 4000:
        print(f"\n✓ First 4000 papers will cover years up to {year}")
        break

# Show what 4000 papers will include
years_sorted = sorted(year_counts.keys())
papers_included = 0
years_in_4000 = []
for year in years_sorted:
    if papers_included < 4000:
        papers_to_add = min(year_counts[year], 4000 - papers_included)
        if papers_to_add > 0:
            years_in_4000.append((year, papers_to_add))
            papers_included += papers_to_add

print("\nFirst 4000 papers will include:")
for year, count in years_in_4000:
    print(f"  {year}: {count} papers")