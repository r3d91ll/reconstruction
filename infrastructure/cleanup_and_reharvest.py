#!/usr/bin/env python3
"""Clean up old metadata and prepare for re-harvesting with fixed categories."""

import os
import shutil
from pathlib import Path
from arango import ArangoClient

print("=== Metadata Cleanup and Re-harvest Preparation ===\n")

# 1. Check current metadata directory
metadata_dir = Path("/mnt/data/arxiv_data/metadata")
if metadata_dir.exists():
    file_count = len(list(metadata_dir.glob("*.json")))
    print(f"Current metadata directory: {metadata_dir}")
    print(f"Files to clean: {file_count:,}")
    
    response = input("\nMove old metadata to backup? (y/n): ")
    if response.lower() == 'y':
        # Create backup
        backup_dir = metadata_dir.parent / "metadata_backup_no_categories"
        print(f"Moving to: {backup_dir}")
        shutil.move(str(metadata_dir), str(backup_dir))
        
        # Create fresh metadata directory
        metadata_dir.mkdir(exist_ok=True)
        print("✅ Old metadata backed up, fresh directory created")
    else:
        print("Aborted")
        exit(1)

# 2. Drop the old database
print("\n\nCleaning database...")
try:
    client = ArangoClient(hosts='http://192.168.1.69:8529')
    sys_db = client.db('_system', username='root', password=os.environ.get('ARANGO_PASSWORD', ''))
    
    db_name = 'arxiv_abstracts_61463'
    if sys_db.has_database(db_name):
        response = input(f"\nDrop database '{db_name}'? (y/n): ")
        if response.lower() == 'y':
            sys_db.delete_database(db_name)
            print(f"✅ Database '{db_name}' dropped")
        else:
            print("Keeping existing database")
    else:
        print(f"Database '{db_name}' doesn't exist")
        
except Exception as e:
    print(f"Database cleanup error: {e}")

print("\n\n=== Ready for Re-harvesting ===")
print("\nNext steps:")
print("1. Go to /home/todd/git/arxiv_downloader/")
print("2. Run the metadata harvester with your desired date range")
print("3. Example commands:")
print("   python3 metadata_harvester.py --days-back 30")
print("   python3 metadata_harvester.py --start-date 2024-01-01 --end-date 2024-12-31")
print("\n4. Then re-run the abstract processing pipeline:")
print("   cd /home/todd/reconstructionism/infrastructure")
print("   python3 setup/process_abstracts_only.py --count 61463")
print("\nThe harvester will now correctly capture categories!")