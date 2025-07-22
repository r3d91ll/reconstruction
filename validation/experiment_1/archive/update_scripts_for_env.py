#!/usr/bin/env python3
"""
Update core scripts to use VALIDATION_OUTPUT_DIR environment variable
"""

import os
import re

def update_script(file_path, old_pattern, new_code):
    """Update a script with new output directory logic"""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace the old pattern with new code
    updated_content = re.sub(old_pattern, new_code, content, flags=re.MULTILINE)
    
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    print(f"Updated: {os.path.basename(file_path)}")

def main():
    """Update all core scripts to use environment variable for output"""
    
    print("Updating scripts to use VALIDATION_OUTPUT_DIR...\n")
    
    # Step 1: Update Jina embedding script
    update_script(
        "/home/todd/reconstructionism/validation/python/step1_generate_embeddings_jina.py",
        r'output_dir = "/home/todd/reconstructionism/validation/data/papers_with_embeddings"',
        'output_dir = os.environ.get("VALIDATION_OUTPUT_DIR", "/home/todd/reconstructionism/validation/data/papers_with_embeddings")'
    )
    
    # Step 2: Update ArangoDB loader
    update_script(
        "/home/todd/reconstructionism/validation/python/step2_load_arangodb.py",
        r'input_dir = "/home/todd/reconstructionism/validation/data/papers_with_embeddings"',
        'input_dir = os.environ.get("VALIDATION_OUTPUT_DIR", "/home/todd/reconstructionism/validation/data/papers_with_embeddings")'
    )
    
    # Also need to update the advanced Jina script
    update_script(
        "/home/todd/reconstructionism/validation/python/step1_jina_advanced.py",
        r'output_dir = "/home/todd/reconstructionism/validation/data/papers_with_embeddings"',
        'output_dir = os.environ.get("VALIDATION_OUTPUT_DIR", "/home/todd/reconstructionism/validation/data/papers_with_embeddings")'
    )
    
    print("\n✓ Scripts updated to use VALIDATION_OUTPUT_DIR environment variable")
    print("\nWhen run through run_validation.py, output will go to timestamped directories.")
    print("When run directly, output will go to default location.")

if __name__ == "__main__":
    main()