#!/usr/bin/env python3
"""
Check if papers are ready with PDF content for pipeline run
"""

import json
import os
from glob import glob

papers_dir = "/home/todd/olympus/Erebus/unstructured/papers/"

# Check for JSON files
json_files = glob(os.path.join(papers_dir, "*.json"))
pdf_files = glob(os.path.join(papers_dir, "*.pdf"))

print(f"Papers directory: {papers_dir}")
print(f"Found {len(json_files)} JSON files")
print(f"Found {len(pdf_files)} PDF files")

# Check for papers with full content
with_content = 0
with_embeddings = 0
milestone_papers = []

milestone_keywords = [
    "word2vec", "attention is all you need", "gpt-2", "gpt-3",
    "toolformer", "react", "vilbert", "lxmert"
]

for json_file in json_files[:10]:  # Sample first 10
    try:
        with open(json_file, 'r') as f:
            paper = json.load(f)
        
        title = paper.get('title', '').lower()
        
        # Check for milestone papers
        for keyword in milestone_keywords:
            if keyword in title:
                milestone_papers.append({
                    'title': paper.get('title', ''),
                    'has_content': 'pdf_content' in paper,
                    'has_embedding': 'embeddings' in paper
                })
        
        if 'pdf_content' in paper:
            with_content += 1
        if 'embeddings' in paper:
            with_embeddings += 1
            
    except:
        pass

print(f"\nSample of first 10 papers:")
print(f"  With PDF content: {with_content}")
print(f"  With embeddings: {with_embeddings}")

if milestone_papers:
    print(f"\nFound milestone papers:")
    for mp in milestone_papers:
        print(f"  - {mp['title'][:60]}...")
        print(f"    Has content: {mp['has_content']}, Has embedding: {mp['has_embedding']}")
else:
    print(f"\nNo milestone papers found in sample")

print("\nReady to run pipeline!" if json_files else "\nNo papers found!")