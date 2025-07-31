#!/bin/bash
# Explore the 1TB rescued image for tar files
# Run as root

echo "=== Exploring Rescued 1TB Image for TAR Archives ==="
echo "Date: $(date)"
echo

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (sudo -i)"
    exit 1
fi

IMAGE_PATH="/home/todd/olympus/sda1_rescue.img"
MOUNT_POINT="/mnt/recovery/partial_rescue"

# Check image exists
if [ ! -f "$IMAGE_PATH" ]; then
    echo "ERROR: Image not found at $IMAGE_PATH"
    exit 1
fi

echo "=== Image Details ==="
ls -lh "$IMAGE_PATH"
echo

# Create mount point
mkdir -p "$MOUNT_POINT"

# Check if already mounted
if mountpoint -q "$MOUNT_POINT"; then
    echo "Already mounted, unmounting first..."
    umount "$MOUNT_POINT"
fi

# Find available loop device
echo "=== Setting up loop device ==="
LOOP_DEV=$(losetup -f)
echo "Using loop device: $LOOP_DEV"

# Attach image to loop device
losetup "$LOOP_DEV" "$IMAGE_PATH"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to set up loop device"
    exit 1
fi

echo "Loop device attached"
echo

# Try to mount the partial image
echo "=== Attempting to mount partial image ==="
MOUNTED=false

if mount -t xfs -o ro,norecovery "$LOOP_DEV" "$MOUNT_POINT" 2>/dev/null; then
    echo "✓ Mounted successfully!"
    MOUNTED=true
else
    echo "✗ Standard mount failed, trying with additional options..."
    
    if mount -t xfs -o ro,norecovery,inode64 "$LOOP_DEV" "$MOUNT_POINT" 2>/dev/null; then
        echo "✓ Mounted with inode64!"
        MOUNTED=true
    else
        echo "✗ Mount failed - filesystem may be incomplete in rescued portion"
    fi
fi

if [ "$MOUNTED" = true ]; then
    echo
    echo "=== Exploring mounted filesystem for TAR files ==="
    
    # Show disk usage
    echo "--- Disk usage ---"
    df -h "$MOUNT_POINT"
    echo
    
    # List top-level directories
    echo "--- Top-level directories ---"
    ls -la "$MOUNT_POINT/" 2>/dev/null | head -20
    echo
    
    # Search for arxiv data
    echo "--- Searching for arxiv directories ---"
    find "$MOUNT_POINT" -maxdepth 4 -type d -name "*arxiv*" 2>/dev/null
    echo
    
    # Look for TAR files
    echo "--- Searching for TAR archives ---"
    echo "Looking for .tar files..."
    TAR_COUNT=$(find "$MOUNT_POINT" -name "*.tar" -o -name "*.tar.gz" -o -name "*.tgz" 2>/dev/null | wc -l)
    echo "Total TAR archives found: $TAR_COUNT"
    
    if [ $TAR_COUNT -gt 0 ]; then
        echo
        echo "TAR files found:"
        find "$MOUNT_POINT" \( -name "*.tar" -o -name "*.tar.gz" -o -name "*.tgz" \) -ls 2>/dev/null | head -20
        
        echo
        echo "--- TAR file size distribution ---"
        find "$MOUNT_POINT" \( -name "*.tar" -o -name "*.tar.gz" -o -name "*.tgz" \) -exec ls -lh {} \; 2>/dev/null | awk '{print $5}' | sort -h | uniq -c
        
        echo
        echo "--- Checking TAR contents (first few) ---"
        FIRST_TAR=$(find "$MOUNT_POINT" -name "*.tar" -o -name "*.tar.gz" 2>/dev/null | head -1)
        if [ -n "$FIRST_TAR" ]; then
            echo "Sample from: $FIRST_TAR"
            tar -tf "$FIRST_TAR" 2>/dev/null | head -10 || echo "Could not read tar contents"
        fi
    fi
    
    # Look for specific arxiv tar structure
    echo
    echo "--- Looking for ArXiv TAR patterns ---"
    # ArXiv files often have patterns like arXiv_src_YYMM_NNN.tar
    find "$MOUNT_POINT" -name "*arXiv*.tar" -o -name "*arxiv*.tar" -o -name "src-*" 2>/dev/null | head -20
    
    # Check for directories that might contain tars
    echo
    echo "--- Directories containing TAR files ---"
    find "$MOUNT_POINT" \( -name "*.tar" -o -name "*.tar.gz" \) -type f 2>/dev/null | sed 's|/[^/]*$||' | sort | uniq -c | sort -rn | head -10
    
    # Unmount
    echo
    echo "=== Cleanup ==="
    umount "$MOUNT_POINT"
    echo "✓ Unmounted"
else
    echo
    echo "=== Alternative: Searching for TAR signatures in raw image ==="
    
    # Search for TAR headers in the image
    echo "--- Looking for TAR magic signatures ---"
    echo "Searching for 'ustar' signatures (TAR format indicator)..."
    
    # TAR files have 'ustar' at offset 257
    grep -abo "ustar" "$LOOP_DEV" 2>/dev/null | head -20
    
    echo
    echo "--- Looking for gzip signatures (for .tar.gz files) ---"
    # Gzip files start with 1f 8b
    od -An -tx1 -N 1000000 "$LOOP_DEV" 2>/dev/null | grep -m 10 "1f 8b"
    
    echo
    echo "The image appears to contain partial filesystem data."
    echo "Full recovery will be possible once we have the complete image."
fi

# Clean up loop device
losetup -d "$LOOP_DEV"
echo "✓ Loop device detached"

echo
echo "=== Summary ==="
echo "1. We have successfully rescued 1TB of the 5.5TB drive"
echo "2. The rescued portion may contain arxiv TAR archives"
echo "3. TAR files are often large, so many might be beyond the 1TB mark"
echo "4. Full recovery requires completing the ddrescue process"
echo
echo "=== Quick TAR Check Commands ==="
echo "If you want to check specific locations:"
echo "  # List all TAR files if mounted:"
echo "  find /mnt/recovery -name '*.tar' -ls"
echo
echo "  # Check a TAR file's contents:"
echo "  tar -tvf /path/to/file.tar | head"
echo
echo "  # Extract specific files from TAR:"
echo "  tar -xf /path/to/file.tar --wildcards '*.pdf'"