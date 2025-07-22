#!/usr/bin/env python3
"""
Check for milestone papers in our dataset
"""

import json
import os
from glob import glob

# Milestone papers to look for
MILESTONE_PAPERS = [
    "word2vec",
    "Attention is All You Need",
    "GPT-2",
    "GPT-3", 
    "GPT3",
    "Toolformer",
    "ReAct",
    "ViLBERT",
    "LXMERT"
]

def check_papers():
    # Path to raw data
    data_path = "/mnt/raid/datasets/information-reconstructionism/raw/ai/"
    
    print("Checking for milestone papers in dataset...")
    print("=" * 60)
    
    found_papers = []
    
    # Search through JSON files
    json_files = glob(os.path.join(data_path, "*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                paper = json.load(f)
                title = paper.get('title', '').strip()
                
                # Check if this is one of our milestone papers
                for milestone in MILESTONE_PAPERS:
                    if milestone.lower() in title.lower():
                        found_papers.append({
                            'title': title,
                            'id': paper.get('id', 'unknown'),
                            'file': os.path.basename(json_file),
                            'milestone': milestone
                        })
                        print(f"✓ Found: {milestone}")
                        print(f"  Title: {title}")
                        print(f"  File: {os.path.basename(json_file)}")
                        print()
                        
        except Exception as e:
            continue
    
    print("\nSummary:")
    print(f"Found {len(found_papers)} milestone papers")
    
    # Check which ones are missing
    found_milestones = [p['milestone'] for p in found_papers]
    missing = [m for m in MILESTONE_PAPERS if not any(m.lower() in f.lower() for f in found_milestones)]
    
    if missing:
        print("\nMissing papers:")
        for m in missing:
            print(f"  - {m}")
    
    return found_papers

if __name__ == "__main__":
    check_papers()