#!/bin/bash
# Immediate recovery options while waiting for 6TB drives
# Run as root

echo "=== Immediate Recovery Options ==="
echo "While waiting for new drives, we can try targeted recovery"
echo

# Check available space
echo "=== Current Storage Status ==="
df -h | grep -E "Filesystem|olympus|root|vg0"
echo

# Option 1: Mount and selective copy
echo "=== Option 1: Selective Recovery ==="
echo "Try to mount sda1 and copy only critical PDFs"
echo

MOUNT_POINT="/mnt/recovery/sda1"
mkdir -p "$MOUNT_POINT"

# Try to mount
if mount -t xfs -o ro,norecovery,noatime,nodiratime /dev/sda1 "$MOUNT_POINT" 2>/dev/null; then
    echo "✓ sda1 mounted successfully!"
    
    # Find arxiv directories
    echo "Looking for arxiv data..."
    ARXIV_DIR=$(find "$MOUNT_POINT" -maxdepth 3 -type d -name "*arxiv*" 2>/dev/null | head -1)
    
    if [ -n "$ARXIV_DIR" ]; then
        echo "Found: $ARXIV_DIR"
        
        # Count PDFs
        PDF_COUNT=$(find "$ARXIV_DIR" -name "*.pdf" 2>/dev/null | wc -l)
        echo "Total PDFs: $PDF_COUNT"
        
        # Check sizes
        echo "Total size: $(du -sh "$ARXIV_DIR" 2>/dev/null | cut -f1)"
        
        # Show sample of PDFs
        echo
        echo "Sample PDFs:"
        find "$ARXIV_DIR" -name "*.pdf" 2>/dev/null | head -10
        
        # Calculate available space for selective copy
        AVAIL_SPACE=$(df /home/todd/olympus | tail -1 | awk '{print $4}')
        echo
        echo "Available space on olympus: ${AVAIL_SPACE}K"
        
        # Suggest selective copy
        echo
        echo "=== Selective Copy Commands ==="
        echo "1. Copy most recent PDFs (by name):"
        echo "   find $ARXIV_DIR -name '*.pdf' | sort -r | head -1000 | xargs -I {} cp {} /tmp/arxiv_critical/"
        echo
        echo "2. Copy specific year:"
        echo "   rsync -av --include='*2024*.pdf' --include='*/' --exclude='*' $ARXIV_DIR/ /tmp/arxiv_2024/"
        echo
        echo "3. Copy by size (smaller files first):"
        echo "   find $ARXIV_DIR -name '*.pdf' -printf '%s %p\n' | sort -n | head -1000 | cut -d' ' -f2- | xargs -I {} cp {} /tmp/arxiv_small/"
    fi
    
    umount "$MOUNT_POINT"
else
    echo "✗ Cannot mount sda1 - drive may be too damaged"
fi

echo
echo "=== Option 2: Check the 1TB already rescued ==="
echo "Mount the partial image we already have:"
echo

# Check if we can mount the partial image
if [ -f "/home/todd/olympus/sda1_rescue.img" ]; then
    echo "Found rescue image: 1TB copied"
    echo "Commands to explore it:"
    echo "  # Create loop device"
    echo "  losetup -f /home/todd/olympus/sda1_rescue.img"
    echo "  # Try to mount (may fail due to incomplete image)"
    echo "  mount -t xfs -o ro,norecovery /dev/loop0 /mnt/recovery/partial"
    echo
    echo "  # Or use xfs_db to explore:"
    echo "  xfs_db -r /home/todd/olympus/sda1_rescue.img"
fi

echo
echo "=== Option 3: Clear space on olympus ==="
echo "Current usage on olympus:"
du -sh /home/todd/olympus/* 2>/dev/null | sort -rh | head -10
echo
echo "Consider moving less critical data to make room"

echo
echo "=== When 6TB drives arrive ==="
echo "1. Install new drives (let's say they become sdc and sdd)"
echo "2. Create new RAID1:"
echo "   mdadm --create /dev/md2 --level=1 --raid-devices=2 /dev/sdc /dev/sdd"
echo "3. Resume ddrescue:"
echo "   ddrescue -n -B /dev/sda1 /dev/md2 /home/todd/sda1_full_rescue.log"
echo "4. Or continue from where we left off:"
echo "   ddrescue -n -B /dev/sda1 /mnt/new_raid/sda1_rescue.img /home/todd/olympus/sda1_rescue.log"
echo
echo "=== IMPORTANT ==="
echo "- Do NOT write anything to sda"
echo "- Check sdb health too: smartctl -a /dev/sdb"
echo "- The ddrescue log file is precious - it tracks what's been copied"