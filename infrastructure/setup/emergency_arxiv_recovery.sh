#!/bin/bash
set -euo pipefail
# Emergency ArXiv Data Recovery Script
# For failing sda1 with XFS corruption

echo "=== EMERGENCY ARXIV DATA RECOVERY ==="
echo "Date: $(date)"
echo "WARNING: sda showing hardware failures - time critical!"
echo

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (sudo -i)"
    exit 1
fi

# Set recovery paths
RECOVERY_SOURCE="/mnt/recovery/sda1"
BACKUP_DEST="/home/todd/olympus/arxiv_emergency_backup"
LOG_FILE="/home/todd/arxiv_recovery_$(date +%Y%m%d_%H%M%S).log"

# Create destinations
mkdir -p "$RECOVERY_SOURCE"
mkdir -p "$BACKUP_DEST"

echo "=== Step 1: Check drive health ==="
echo "Checking SMART status of sda..."
smartctl -H /dev/sda | tee -a "$LOG_FILE"
echo

# Show critical SMART attributes
echo "Critical SMART attributes:"
smartctl -A /dev/sda | grep -E "Reallocated|Pending|Uncorrectable|UDMA_CRC|Temperature" | tee -a "$LOG_FILE"
echo

echo "=== Step 2: Attempt read-only mount ==="
# Try to mount with maximum safety options
echo "Attempting safe mount of sda1..."

# First unmount if already mounted
umount "$RECOVERY_SOURCE" 2>/dev/null

# Try mounting with various options
MOUNT_SUCCESS=false

# Option 1: Most conservative
if mount -t xfs -o ro,norecovery,noatime,nodiratime,inode64 /dev/sda1 "$RECOVERY_SOURCE" 2>/dev/null; then
    echo "✓ Mounted successfully with safe options"
    MOUNT_SUCCESS=true
else
    echo "✗ Safe mount failed, trying alternate superblock..."
    
    # Option 2: Try backup superblock
    for sb in 0 1 2; do
        if mount -t xfs -o ro,norecovery,noatime,nodiratime,sb=$sb /dev/sda1 "$RECOVERY_SOURCE" 2>/dev/null; then
            echo "✓ Mounted with superblock $sb"
            MOUNT_SUCCESS=true
            break
        fi
    done
fi

if [ "$MOUNT_SUCCESS" = false ]; then
    echo "✗ ERROR: Cannot mount sda1 - filesystem too damaged"
    echo
    echo "=== Alternative: Raw disk copy ==="
    echo "Since mount failed, we should try raw disk imaging:"
    echo "1. Use ddrescue for failing drives:"
    echo "   ddrescue -n -B /dev/sda1 /home/todd/olympus/sda1_rescue.img /home/todd/olympus/sda1_rescue.log"
    echo
    echo "2. Then try to recover from the image"
    exit 1
fi

echo
echo "=== Step 3: Locate arxiv data ==="
# Find arxiv directories
ARXIV_DIRS=$(find "$RECOVERY_SOURCE" -maxdepth 4 -type d -name "*arxiv*" 2>/dev/null)

if [ -z "$ARXIV_DIRS" ]; then
    echo "✗ No arxiv directories found!"
    echo "Showing top-level directories:"
    ls -la "$RECOVERY_SOURCE/" | head -20
else
    echo "✓ Found arxiv directories:"
    echo "$ARXIV_DIRS"
fi

echo
echo "=== Step 4: Emergency data copy ==="
echo "Using rsync with aggressive error handling..."
echo "Logs will be saved to: $LOG_FILE"
echo

# Function to copy with retries
copy_with_retry() {
    local src="$1"
    local dst="$2"
    local attempt=1
    local max_attempts=3
    
    while [ $attempt -le $max_attempts ]; do
        echo "Copy attempt $attempt of $max_attempts..."
        
        # Use rsync with error recovery options
        rsync -av \
            --ignore-errors \
            --partial \
            --inplace \
            --no-checksum \
            --timeout=10 \
            --contimeout=10 \
            --progress \
            "$src" "$dst" 2>&1 | tee -a "$LOG_FILE"
        
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "✓ Copy successful"
            return 0
        else
            echo "✗ Copy failed on attempt $attempt"
            attempt=$((attempt + 1))
            sleep 5
        fi
    done
    
    return 1
}

# Copy arxiv data if found
if [ -d "$RECOVERY_SOURCE/arxiv_data" ]; then
    echo "Copying arxiv_data directory..."
    copy_with_retry "$RECOVERY_SOURCE/arxiv_data/" "$BACKUP_DEST/"
elif [ -n "$ARXIV_DIRS" ]; then
    echo "Copying found arxiv directories..."
    for dir in $ARXIV_DIRS; do
        echo "Copying: $dir"
        copy_with_retry "$dir/" "$BACKUP_DEST/$(basename $dir)/"
    done
else
    echo "WARNING: No specific arxiv directories found"
    echo "Copying entire filesystem (this may take a long time)..."
    copy_with_retry "$RECOVERY_SOURCE/" "$BACKUP_DEST/full_recovery/"
fi

echo
echo "=== Step 5: Verify backup ==="
if [ -d "$BACKUP_DEST" ]; then
    echo "Backup location: $BACKUP_DEST"
    echo "Total size: $(du -sh "$BACKUP_DEST" 2>/dev/null | cut -f1)"
    echo "PDF count: $(find "$BACKUP_DEST" -name "*.pdf" 2>/dev/null | wc -l)"
    echo "Directory structure:"
    tree -d -L 3 "$BACKUP_DEST" 2>/dev/null || ls -la "$BACKUP_DEST"
fi

# Unmount
echo
echo "=== Step 6: Cleanup ==="
umount "$RECOVERY_SOURCE" 2>/dev/null && echo "✓ Unmounted recovery source"

echo
echo "=== RECOVERY SUMMARY ==="
echo "1. Check backup at: $BACKUP_DEST"
echo "2. Review log at: $LOG_FILE"
echo "3. DO NOT use sda for any new data!"
echo "4. Replace the failing drive ASAP"
echo
echo "=== Next Steps ==="
echo "1. Verify the backed up PDFs are readable"
echo "2. Check other drives' health: smartctl -a /dev/sdb"
echo "3. Plan new RAID1 configuration without sda/sdb"
echo
echo "To check if PDFs are corrupted:"
echo "  find $BACKUP_DEST -name '*.pdf' -exec pdfinfo {} \; >/dev/null 2>pdf_errors.log"
echo "  Then check pdf_errors.log for any damaged files"