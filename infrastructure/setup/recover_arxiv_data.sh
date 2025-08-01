#!/bin/bash
# Recover arxiv data from sda1
# Run as root

echo "=== ArXiv Data Recovery from sda1 ==="
echo "Date: $(date)"
echo

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root"
    exit 1
fi

# Create recovery mount point
RECOVERY_DIR="/mnt/recovery/arxiv_data"
mkdir -p "$RECOVERY_DIR"

echo "=== Attempting XFS recovery mount ==="
echo

# First, let's check the XFS filesystem info
echo "--- XFS Info ---"
xfs_info /dev/sda1 2>&1 || echo "Cannot get XFS info directly"
echo

# Try to mount with various recovery options
echo "--- Attempting mount with recovery options ---"

# Option 1: Mount with no log recovery (safest for read-only)
echo "Trying: norecovery mount..."
if mount -t xfs -o ro,norecovery /dev/sda1 "$RECOVERY_DIR" 2>/dev/null; then
    echo "✓ SUCCESS: Mounted with norecovery"
    MOUNTED=true
else
    echo "✗ Failed with norecovery"
    
    # Option 2: Force mount with sb parameter
    echo "Trying: force mount with sb=0..."
    if mount -t xfs -o ro,norecovery,sb=0 /dev/sda1 "$RECOVERY_DIR" 2>/dev/null; then
        echo "✓ SUCCESS: Mounted with sb=0"
        MOUNTED=true
    else
        echo "✗ Failed with sb=0"
        
        # Option 3: Try inode64 option
        echo "Trying: inode64 mount..."
        if mount -t xfs -o ro,norecovery,inode64 /dev/sda1 "$RECOVERY_DIR" 2>/dev/null; then
            echo "✓ SUCCESS: Mounted with inode64"
            MOUNTED=true
        else
            echo "✗ Failed with inode64"
            MOUNTED=false
        fi
    fi
fi

echo

if [ "$MOUNTED" = true ]; then
    echo "=== Filesystem successfully mounted! ==="
    echo
    echo "--- Disk usage ---"
    df -h "$RECOVERY_DIR"
    echo
    echo "--- Top-level contents ---"
    ls -la "$RECOVERY_DIR" | head -20
    echo
    echo "--- Looking for arxiv_data ---"
    find "$RECOVERY_DIR" -maxdepth 3 -type d -name "*arxiv*" 2>/dev/null | head -20
    echo
    echo "--- PDF count (if found) ---"
    if [ -d "$RECOVERY_DIR/arxiv_data/pdf" ]; then
        echo "Found arxiv_data/pdf directory!"
        pdf_count=$(find "$RECOVERY_DIR/arxiv_data/pdf" -name "*.pdf" 2>/dev/null | wc -l)
        echo "Total PDFs: $pdf_count"
    fi
    echo
    echo "=== Recovery Options ==="
    echo "1. To copy all data to safe location:"
    echo "   rsync -avP $RECOVERY_DIR/arxiv_data/ /home/todd/olympus/arxiv_data_recovery/"
    echo
    echo "2. To unmount when done:"
    echo "   umount $RECOVERY_DIR"
else
    echo "=== Mount failed - trying xfs_repair dry run ==="
    echo
    # Check if repair might help
    echo "Running xfs_repair in check-only mode..."
    xfs_repair -n /dev/sda1 2>&1 | head -50
    echo
    echo "=== Alternative: Direct data recovery ==="
    echo "Since the filesystem has XFS magic number, we might be able to:"
    echo "1. Use xfs_repair -L /dev/sda1 (WARNING: This zeros the log)"
    echo "2. Use photorec or testdisk for raw recovery"
    echo "3. Try mounting after xfs_repair"
    echo
    echo "Next step: Create a backup image first:"
    echo "   dd if=/dev/sda1 of=/home/todd/olympus/sda1_backup.img bs=1M status=progress"
fi

# Also check sdb1 for comparison
echo
echo "=== Checking sdb1 status ==="
echo "This was likely the mirror - checking if it has any recoverable data..."
file -s /dev/sdb1
hexdump -C /dev/sdb1 | head -20

echo
echo "=== Summary ==="
echo "sda1: XFS filesystem detected (primary data source)"
echo "sdb1: No filesystem detected (likely failed mirror)"
echo
echo "This appears to have been a RAID1 that was directly formatted"
echo "without mdadm metadata (explains why it's partition type 'Linux RAID')"