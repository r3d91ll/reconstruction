#!/bin/bash
# Basic PhotoRec recovery organization script
# Organizes files by extension into categorized directories

SOURCE_DIR="/mnt/photorec-temp"
DEST_DIR="/home/todd/olympus/photorec-organized"

# Create destination directories
echo "Creating destination directories..."
mkdir -p "$DEST_DIR/photos"
mkdir -p "$DEST_DIR/archives"
mkdir -p "$DEST_DIR/documents"
mkdir -p "$DEST_DIR/other"

# Count files
echo "Counting recovered files..."
echo "Total recovery directories: $(find "$SOURCE_DIR" -name "recup_dir.*" -type d | wc -l)"

# Count by type
echo "Counting by file type..."
echo "  Images: $(find "$SOURCE_DIR" -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.gif" -o -iname "*.bmp" | wc -l)"
echo "  Archives: $(find "$SOURCE_DIR" -iname "*.tar" -o -iname "*.tar.gz" -o -iname "*.tgz" -o -iname "*.zip" | wc -l)"
echo "  Documents: $(find "$SOURCE_DIR" -iname "*.pdf" -o -iname "*.doc*" -o -iname "*.txt" | wc -l)"

# Function to copy files with progress
copy_files_by_pattern() {
    local pattern="$1"
    local dest_subdir="$2"
    local description="$3"
    
    echo "Processing $description..."
    
    # Create file list
    find "$SOURCE_DIR" -iname "$pattern" > /tmp/photorec_${dest_subdir}.list
    
    local total=$(wc -l < /tmp/photorec_${dest_subdir}.list)
    echo "Found $total $description files"
    
    if [ "$total" -gt 0 ]; then
        # Copy files with progress
        local count=0
        while IFS= read -r file; do
            # Get hash for unique naming
            hash=$(sha256sum "$file" | cut -d' ' -f1 | head -c 16)
            ext="${file##*.}"
            
            cp -n "$file" "$DEST_DIR/$dest_subdir/${hash}.${ext}"
            
            count=$((count + 1))
            if [ $((count % 100)) -eq 0 ]; then
                echo "  Copied $count/$total files..."
            fi
        done < /tmp/photorec_${dest_subdir}.list
    fi
    
    rm -f /tmp/photorec_${dest_subdir}.list
}

# Check available space
echo "Checking available space..."
df -h "$DEST_DIR"

# Ask for confirmation
echo
echo "This will copy files from $SOURCE_DIR to $DEST_DIR"
echo "Press Enter to continue or Ctrl+C to cancel..."
read

# Process different file types
copy_files_by_pattern "*.jpg" "photos" "JPG images"
copy_files_by_pattern "*.jpeg" "photos" "JPEG images"
copy_files_by_pattern "*.png" "photos" "PNG images"
copy_files_by_pattern "*.gif" "photos" "GIF images"
copy_files_by_pattern "*.tar" "archives" "TAR archives"
copy_files_by_pattern "*.tar.gz" "archives" "TAR.GZ archives"
copy_files_by_pattern "*.tgz" "archives" "TGZ archives"

echo "Basic organization complete!"
echo
echo "Next steps:"
echo "1. Check $DEST_DIR/photos for family photos"
echo "2. Use 'file' command on TAR files to identify ArXiv archives"
echo "3. Consider using the Python script for more advanced organization"