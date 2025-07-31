#!/bin/bash
# Explore the 1TB rescued image
# Run as root

echo "=== Exploring Rescued 1TB Image ==="
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

# Try to get XFS info
echo "=== XFS Filesystem Info ==="
xfs_info "$LOOP_DEV" 2>&1 || echo "Cannot read XFS info directly"
echo

# Try to mount the partial image
echo "=== Attempting to mount partial image ==="
echo "Note: This may fail or show errors due to incomplete image"
echo

MOUNTED=false

# Try mounting with various recovery options
if mount -t xfs -o ro,norecovery "$LOOP_DEV" "$MOUNT_POINT" 2>/dev/null; then
    echo "✓ Mounted successfully!"
    MOUNTED=true
else
    echo "✗ Standard mount failed, trying with additional options..."
    
    # Try with explicit inode size
    if mount -t xfs -o ro,norecovery,inode64 "$LOOP_DEV" "$MOUNT_POINT" 2>/dev/null; then
        echo "✓ Mounted with inode64!"
        MOUNTED=true
    else
        echo "✗ Mount failed - filesystem may be incomplete in rescued portion"
    fi
fi

if [ "$MOUNTED" = true ]; then
    echo
    echo "=== Exploring mounted filesystem ==="
    
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
    
    # Look for PDFs
    echo "--- Searching for PDF files ---"
    PDF_COUNT=$(find "$MOUNT_POINT" -name "*.pdf" 2>/dev/null | wc -l)
    echo "Total PDFs found: $PDF_COUNT"
    
    if [ $PDF_COUNT -gt 0 ]; then
        echo
        echo "Sample PDFs:"
        find "$MOUNT_POINT" -name "*.pdf" 2>/dev/null | head -10
        
        echo
        echo "--- PDF distribution by directory ---"
        find "$MOUNT_POINT" -name "*.pdf" -type f 2>/dev/null | sed 's|/[^/]*$||' | sort | uniq -c | sort -rn | head -10
    fi
    
    # Check if we can access specific arxiv data
    if [ -d "$MOUNT_POINT/arxiv_data" ]; then
        echo
        echo "=== Found arxiv_data directory! ==="
        echo "Contents:"
        ls -la "$MOUNT_POINT/arxiv_data/" | head -10
        
        if [ -d "$MOUNT_POINT/arxiv_data/pdf" ]; then
            echo
            echo "PDF directory found!"
            echo "Sample files:"
            ls -la "$MOUNT_POINT/arxiv_data/pdf/" | head -10
            echo "..."
            echo "Total PDFs in arxiv_data/pdf: $(find "$MOUNT_POINT/arxiv_data/pdf" -name "*.pdf" | wc -l)"
        fi
    fi
    
    # Unmount
    echo
    echo "=== Cleanup ==="
    umount "$MOUNT_POINT"
    echo "✓ Unmounted"
else
    echo
    echo "=== Alternative: Using XFS tools to explore ==="
    
    # Try xfs_db to explore the structure
    echo "--- Attempting to read with xfs_db ---"
    echo "quit" | xfs_db -r "$LOOP_DEV" 2>&1 | head -20
    
    echo
    echo "--- Searching for file signatures ---"
    echo "Looking for PDF signatures in first 1GB..."
    
    # Search for PDF headers
    strings -n 8 "$LOOP_DEV" 2>/dev/null | grep -m 20 "^%PDF-" | head -10
    
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
echo "2. The rescued portion may contain partial arxiv data"
echo "3. Full recovery requires completing the ddrescue process"
echo "4. The log file preserves our progress - we can resume anytime"
echo
echo "=== Next Steps ==="
echo "1. Wait for 6TB drives to arrive"
echo "2. Create RAID1 with new drives"
echo "3. Resume ddrescue: sudo ddrescue -n -B /dev/sda1 /path/to/new/storage/sda1_rescue.img /home/todd/olympus/sda1_rescue.log"