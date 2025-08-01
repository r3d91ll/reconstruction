#!/usr/bin/env python3
"""
Monitor the PDF directory and database status
Shows what PDFs are present and their processing status
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from collections import Counter
from arango import ArangoClient

def main():
    """Monitor PDF directory and processing status"""
    
    # Check required environment variables
    required_env_vars = {
        'ARANGO_HOST': 'http://192.168.1.69:8529',  # default value
        'ARANGO_DATABASE': 'base',  # default value
        'ARANGO_USERNAME': 'root',  # default value
        'ARANGO_PASSWORD': None  # required, no default
    }
    
    env_config = {}
    for var, default in required_env_vars.items():
        value = os.environ.get(var, default)
        if value is None:
            print(f"ERROR: {var} environment variable not set")
            sys.exit(1)
        env_config[var] = value
        
    pdf_dir = Path('/mnt/data-cold/arxiv_data/pdf')
    
    print("=" * 80)
    print("PDF Directory Monitor")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Check directory
    if not pdf_dir.exists():
        print(f"ERROR: PDF directory not found: {pdf_dir}")
        sys.exit(1)
        
    # Get all PDFs
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    print(f"\nPDFs in directory: {len(pdf_files)}")
    
    if not pdf_files:
        print("Directory is empty - all PDFs processed!")
        sys.exit(0)
        
    # Extract arxiv IDs
    arxiv_ids = []
    for pdf in pdf_files:
        filename = pdf.stem
        if '_' in filename and '.' not in filename:
            filename = filename.replace('_', '.', 1)
        arxiv_ids.append(filename)
        
    # Connect to database
    client = ArangoClient(hosts=env_config['ARANGO_HOST'])
    db = client.db(
        env_config['ARANGO_DATABASE'],
        username=env_config['ARANGO_USERNAME'],
        password=env_config['ARANGO_PASSWORD']
    )
    collection = db.collection('arxiv_documents')
    
    # Check status in database
    print("\nChecking database status...")
    
    query = """
    FOR id IN @ids
        LET doc = DOCUMENT(@@collection, id)
        RETURN {
            id: id,
            exists: doc != null,
            has_pdf: doc != null AND doc.pdf_content != null AND doc.pdf_content.markdown != null,
            pdf_state: doc != null ? doc.pdf_status.state : null,
            categories: doc != null ? doc.categories : null
        }
    """
    
    cursor = db.aql.execute(
        query,
        bind_vars={
            'ids': arxiv_ids,
            '@collection': 'arxiv_documents'
        }
    )
    
    results = list(cursor)
    
    # Analyze results
    status_counts = Counter()
    category_counts = Counter()
    orphaned = []
    already_processed = []
    ready_to_process = []
    
    for result in results:
        arxiv_id = result['id']
        
        if not result['exists']:
            status_counts['orphaned'] += 1
            orphaned.append(arxiv_id)
        elif result['has_pdf']:
            status_counts['already_processed'] += 1
            already_processed.append(arxiv_id)
        else:
            status_counts['ready_to_process'] += 1
            ready_to_process.append(arxiv_id)
            
        # Count categories
        if result['categories']:
            for cat in result['categories']:
                category_counts[cat] += 1
                
    # Print summary
    print("\n" + "-" * 40)
    print("Summary:")
    print("-" * 40)
    print(f"Total PDFs: {len(pdf_files)}")
    print(f"  - Ready to process: {status_counts['ready_to_process']}")
    print(f"  - Already processed: {status_counts['already_processed']}")
    print(f"  - Orphaned (no metadata): {status_counts['orphaned']}")
    
    # Show category distribution
    if category_counts:
        print("\nTop categories:")
        for cat, count in category_counts.most_common(10):
            print(f"  - {cat}: {count}")
            
    # Show samples
    if orphaned:
        print(f"\nOrphaned PDFs (first 10 of {len(orphaned)}):")
        for pdf_id in orphaned[:10]:
            print(f"  - {pdf_id}.pdf")
            
    if already_processed:
        print(f"\nAlready processed (first 10 of {len(already_processed)}):")
        for pdf_id in already_processed[:10]:
            print(f"  - {pdf_id}.pdf (can be deleted)")
            
    if ready_to_process:
        print(f"\nReady to process (first 10 of {len(ready_to_process)}):")
        for pdf_id in ready_to_process[:10]:
            print(f"  - {pdf_id}.pdf")
            
    # Recommendations
    print("\n" + "=" * 80)
    print("Recommendations:")
    
    if status_counts['already_processed'] > 0:
        print(f"- Run pipeline to delete {status_counts['already_processed']} already processed PDFs")
        
    if status_counts['ready_to_process'] > 0:
        print(f"- Process {status_counts['ready_to_process']} PDFs that have metadata")
        
    if status_counts['orphaned'] > 0:
        print(f"- {status_counts['orphaned']} PDFs have no metadata and will remain")
        print("  Consider adding metadata for these or moving them elsewhere")
        
    print("=" * 80)

if __name__ == '__main__':
    main()