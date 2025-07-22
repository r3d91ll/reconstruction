#!/usr/bin/env python3
"""
Check what content is actually in the papers
"""

import json
import os
from glob import glob

papers_dir = "/home/todd/olympus/Erebus/unstructured/papers/"

# Check first 10 papers
json_files = sorted(glob(os.path.join(papers_dir, "*.json")))[:10]

for i, json_file in enumerate(json_files):
    print(f"\n{'='*60}")
    print(f"Paper {i+1}: {os.path.basename(json_file)}")
    
    try:
        with open(json_file, 'r') as f:
            paper = json.load(f)
        
        print(f"Title: {paper.get('title', 'N/A')[:60]}...")
        print(f"Year: {paper.get('year', 'N/A')}")
        
        # Check what content we have
        has_abstract = 'abstract' in paper
        has_pdf_content = 'pdf_content' in paper
        has_embeddings = 'embeddings' in paper or ('dimensions' in paper and 'WHAT' in paper['dimensions'])
        
        print(f"Has abstract: {has_abstract}")
        print(f"Has PDF content: {has_pdf_content}")
        print(f"Has embeddings: {has_embeddings}")
        
        if has_abstract:
            print(f"Abstract length: {len(paper['abstract'])} chars")
        
        if has_pdf_content:
            pdf_content = paper['pdf_content']
            if 'markdown' in pdf_content:
                print(f"PDF markdown length: {len(pdf_content['markdown'])} chars")
            if 'metadata' in pdf_content:
                meta = pdf_content['metadata']
                print(f"PDF pages: {meta.get('num_pages', 'N/A')}")
                print(f"Figures: {meta.get('num_figures', 0)}")
                print(f"Tables: {meta.get('num_tables', 0)}")
        
        # Check content being used
        if has_pdf_content and 'markdown' in paper['pdf_content']:
            content_len = len(paper['pdf_content']['markdown'])
            print(f"\n✓ Would use FULL PDF content: {content_len} chars")
        else:
            abstract_len = len(paper.get('abstract', ''))
            print(f"\n⚠️  Would use ABSTRACT ONLY: {abstract_len} chars")
            
    except Exception as e:
        print(f"Error reading paper: {e}")

print(f"\n{'='*60}")
print("If you're seeing ABSTRACT ONLY, the PDFs haven't been extracted yet!")