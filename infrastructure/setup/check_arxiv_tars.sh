#!/bin/bash
# Check sample of TAR files for ArXiv content

SOURCE_DIR="/mnt/photorec-temp"
SAMPLE_SIZE=20

echo "Sampling TAR files to check for ArXiv content..."
echo

# Find some TAR files
tar_files=$(find "$SOURCE_DIR" -name "*.tar" -o -name "*.tar.gz" -o -name "*.tgz" | head -n $SAMPLE_SIZE)

for tar_file in $tar_files; do
    echo "Checking: $tar_file"
    echo "Size: $(du -h "$tar_file" | cut -f1)"
    
    # Try to list contents
    if tar -tf "$tar_file" 2>/dev/null | head -10 | grep -E '\.(tex|bbl|sty|cls|pdf)$' >/dev/null; then
        echo "  ✓ Likely ArXiv archive (contains LaTeX files)"
        echo "  Sample contents:"
        tar -tf "$tar_file" 2>/dev/null | head -5 | sed 's/^/    /'
    else
        echo "  ✗ Not ArXiv (no LaTeX files found)"
    fi
    echo
done

echo "To check a specific TAR file:"
echo "  tar -tf /path/to/file.tar | head -20"