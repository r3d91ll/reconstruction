#!/usr/bin/env python3
"""
Organize PhotoRec recovery results into categorized directories.
Prioritizes family photos and ArXiv TAR files.
"""

import os
import shutil
import hashlib
import magic
import argparse
from pathlib import Path
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS
import tarfile
import json
from collections import defaultdict

class PhotoRecOrganizer:
    def __init__(self, source_dir, dest_dir, dry_run=False):
        self.source_dir = Path(source_dir)
        self.dest_dir = Path(dest_dir)
        self.dry_run = dry_run
        self.stats = defaultdict(int)
        self.duplicates = set()
        self.file_magic = magic.Magic(mime=True)
        
        # Create destination directories
        self.photo_dir = self.dest_dir / "recovered_photos"
        self.arxiv_dir = self.dest_dir / "recovered_arxiv"
        self.other_dir = self.dest_dir / "recovered_other"
        
        if not dry_run:
            self.photo_dir.mkdir(parents=True, exist_ok=True)
            self.arxiv_dir.mkdir(parents=True, exist_ok=True)
            self.other_dir.mkdir(parents=True, exist_ok=True)
    
    def get_file_hash(self, filepath, chunk_size=8192):
        """Calculate SHA256 hash of file for duplicate detection."""
        sha256 = hashlib.sha256()
        try:
            with open(filepath, 'rb') as f:
                while chunk := f.read(chunk_size):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            print(f"Error hashing {filepath}: {e}")
            return None
    
    def get_image_metadata(self, filepath):
        """Extract EXIF data from images to help identify family photos."""
        try:
            image = Image.open(filepath)
            exifdata = image.getexif()
            
            if exifdata:
                metadata = {}
                for tag_id, value in exifdata.items():
                    tag = TAGS.get(tag_id, tag_id)
                    metadata[tag] = value
                
                # Extract key fields that might help identify family photos
                date_taken = metadata.get('DateTimeOriginal', metadata.get('DateTime'))
                camera_model = metadata.get('Model')
                camera_make = metadata.get('Make')
                
                return {
                    'date_taken': date_taken,
                    'camera': f"{camera_make} {camera_model}" if camera_make and camera_model else None,
                    'all_metadata': metadata
                }
        except Exception as e:
            print(f"Error reading EXIF from {filepath}: {e}")
        
        return None
    
    def check_arxiv_tar(self, filepath):
        """Check if TAR file contains ArXiv papers."""
        try:
            with tarfile.open(filepath, 'r') as tar:
                # ArXiv TAR files typically contain .tex files
                for member in tar.getmembers()[:10]:  # Check first 10 files
                    if member.name.endswith('.tex') or 'arxiv' in member.name.lower():
                        return True
        except Exception as e:
            print(f"Error reading TAR {filepath}: {e}")
        
        return False
    
    def organize_file(self, filepath):
        """Organize a single file based on type and content."""
        file_hash = self.get_file_hash(filepath)
        
        # Skip duplicates
        if file_hash in self.duplicates:
            self.stats['duplicates'] += 1
            return
        
        self.duplicates.add(file_hash)
        
        # Determine file type
        mime_type = self.file_magic.from_file(str(filepath))
        file_ext = filepath.suffix.lower()
        
        # Handle images (potential family photos)
        if mime_type.startswith('image/'):
            self.stats['images'] += 1
            
            metadata = self.get_image_metadata(filepath)
            
            # Create subdirectory based on date if available
            if metadata and metadata.get('date_taken'):
                try:
                    date_str = metadata['date_taken']
                    # Parse common EXIF date format
                    date_obj = datetime.strptime(date_str.split()[0], '%Y:%m:%d')
                    year_month = date_obj.strftime('%Y-%m')
                    dest_subdir = self.photo_dir / year_month
                except:
                    dest_subdir = self.photo_dir / 'no_date'
            else:
                dest_subdir = self.photo_dir / 'no_date'
            
            if not self.dry_run:
                dest_subdir.mkdir(exist_ok=True)
                dest_path = dest_subdir / f"{file_hash[:8]}{file_ext}"
                shutil.copy2(filepath, dest_path)
                
                # Save metadata
                if metadata:
                    with open(dest_path.with_suffix('.json'), 'w') as f:
                        json.dump(metadata, f, indent=2, default=str)
        
        # Handle TAR files (potential ArXiv)
        elif file_ext in ['.tar', '.tgz'] or mime_type == 'application/x-tar':
            if self.check_arxiv_tar(filepath):
                self.stats['arxiv_tars'] += 1
                if not self.dry_run:
                    dest_path = self.arxiv_dir / f"{file_hash[:8]}{file_ext}"
                    shutil.copy2(filepath, dest_path)
            else:
                self.stats['other_tars'] += 1
                if not self.dry_run:
                    dest_path = self.other_dir / 'tars' / f"{file_hash[:8]}{file_ext}"
                    dest_path.parent.mkdir(exist_ok=True)
                    shutil.copy2(filepath, dest_path)
        
        # Handle other files
        else:
            self.stats['other'] += 1
            # You might want to organize other file types here
    
    def run(self):
        """Process all recovered files."""
        print(f"Starting organization of PhotoRec recovery...")
        print(f"Source: {self.source_dir}")
        print(f"Destination: {self.dest_dir}")
        print(f"Dry run: {self.dry_run}")
        print()
        
        # Find all recup_dir.* directories
        recup_dirs = sorted(self.source_dir.glob("recup_dir.*"))
        print(f"Found {len(recup_dirs)} recovery directories")
        
        total_files = 0
        for i, recup_dir in enumerate(recup_dirs):
            if i % 100 == 0:
                print(f"Processing directory {i+1}/{len(recup_dirs)}...")
            
            for filepath in recup_dir.iterdir():
                if filepath.is_file():
                    total_files += 1
                    self.organize_file(filepath)
                    
                    if total_files % 1000 == 0:
                        print(f"Processed {total_files} files...")
                        self.print_stats()
        
        print("\nFinal statistics:")
        self.print_stats()
    
    def print_stats(self):
        """Print current statistics."""
        print(f"  Images (potential photos): {self.stats['images']}")
        print(f"  ArXiv TAR files: {self.stats['arxiv_tars']}")
        print(f"  Other TAR files: {self.stats['other_tars']}")
        print(f"  Other files: {self.stats['other']}")
        print(f"  Duplicates skipped: {self.stats['duplicates']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize PhotoRec recovery results")
    parser.add_argument("source", help="Source directory containing recup_dir.* folders")
    parser.add_argument("dest", help="Destination directory for organized files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without copying files")
    
    args = parser.parse_args()
    
    organizer = PhotoRecOrganizer(args.source, args.dest, args.dry_run)
    organizer.run()