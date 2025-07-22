#!/usr/bin/env python3
"""
Clean up partial PDF extraction run
"""

import json
import os
from glob import glob
import time
from datetime import datetime

papers_dir = '/home/todd/olympus/Erebus/unstructured/papers/'
recent_time = time.time() - (30 * 60)  # 30 minutes ago to be safe

cleaned_count = 0
print(f"Cleaning up partial extraction from last 30 minutes...")
print(f"Time threshold: {datetime.fromtimestamp(recent_time)}")

for json_file in glob(os.path.join(papers_dir, '*.json')):
    if os.path.getmtime(json_file) > recent_time:
        try:
            # Load JSON
            with open(json_file, 'r') as f:
                paper = json.load(f)
            
            # Check if it has new fields
            needs_cleaning = False
            if 'pdf_content' in paper:
                del paper['pdf_content']
                needs_cleaning = True
            
            if 'embeddings' in paper:
                del paper['embeddings']
                needs_cleaning = True
                
            if 'dimensions' in paper and 'WHAT' in paper['dimensions']:
                if 'embedding_method' in paper['dimensions']['WHAT']:
                    if 'docling' in paper['dimensions']['WHAT']['embedding_method']:
                        del paper['dimensions']['WHAT']
                        if not paper['dimensions']:
                            del paper['dimensions']
                        needs_cleaning = True
            
            # Save cleaned version
            if needs_cleaning:
                with open(json_file, 'w') as f:
                    json.dump(paper, f, indent=2)
                cleaned_count += 1
                
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

print(f"\nCleaned {cleaned_count} JSON files")
print("Ready for fresh extraction run!")