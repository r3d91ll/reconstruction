#!/bin/bash
set -euo pipefail

cleanup() {
  mountpoint -q "$MOUNT_POINT" && umount "$MOUNT_POINT"
}
trap cleanup EXIT

# Check sdb for recoverable data
# Run as root
echo "=== Checking sdb for Recoverable Data ==="
echo "Date: $(date)"
echo "sdb was likely the RAID1 mirror of sda"
echo

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (sudo -i)"
    exit 1
fi

# First, check the drive health
echo "=== Drive Health Check ==="
echo "--- SMART Status ---"
smartctl -H /dev/sdb 2>/dev/null | grep -A2 "SMART overall" || echo "Could not read SMART status"

echo
echo "--- Critical Attributes ---"
smartctl -A /dev/sdb 2>/dev/null | grep -E "Reallocated_Sector|Current_Pending|Offline_Uncorrectable|UDMA_CRC|Temperature" || echo "Could not read SMART attributes"

echo
echo "=== Partition and Filesystem Detection ==="
# Check partition table
echo "--- Partition Table ---"
fdisk -l /dev/sdb 2>/dev/null | grep -E "^/dev|Disk model|Disk /dev"

echo
echo "--- Checking sdb1 ---"
# Check what's on sdb1
file -s /dev/sdb1

echo
echo "--- Looking for filesystem signatures ---"
# Check first 1MB for any recognizable signatures
dd if=/dev/sdb1 bs=1M count=1 2>/dev/null | strings | grep -E "XFS|ext4|NTFS|FAT|BTRFS" | head -5

echo
echo "--- Checking for RAID metadata ---"
mdadm --examine /dev/sdb1 2>&1

echo
echo "=== Attempting Recovery Options ==="

# Option 1: Try XFS recovery
echo "--- Option 1: XFS Recovery Attempt ---"
xfs_db -r -c "sb 0" -c "p" /dev/sdb1 2>&1 | grep -E "magicnum|blocksize|dblocks|uuid" | head -10

# Check for backup superblocks
echo
echo "--- Searching for XFS backup superblocks ---"
echo "This may take a moment..."
xfs_repair -n -f /dev/sdb1 2>&1 | grep -E "found candidate|Phase 1|superblock" | head -20

echo
echo "--- Option 2: Direct filesystem scan ---"
# Try testdisk to scan for partitions
which testdisk >/dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "Running testdisk in non-interactive mode..."
    echo -e "l\nq" | testdisk /dev/sdb1 2>&1 | grep -E "FAT|NTFS|ext|XFS|Linux" | head -10
else
    echo "testdisk not installed (apt install testdisk)"
fi

echo
echo "--- Option 3: Search for TAR/data signatures ---"
echo "Looking for TAR file signatures..."
# Search for TAR headers (ustar at offset 257)
dd if=/dev/sdb1 bs=512 count=10000 skip=0 2>/dev/null | strings | grep -m 5 "ustar"

echo
echo "Looking for gzip signatures (tar.gz files)..."
# Look for gzip magic bytes
od -An -tx1 -N 1000000 /dev/sdb1 2>/dev/null | grep -m 5 "1f 8b"

echo
echo "=== Recovery Recommendations ==="

# Try to mount read-only
MOUNT_POINT="/mnt/recovery/sdb1_test"
mkdir -p "$MOUNT_POINT"

echo
echo "--- Attempting test mount ---"
if mount -t xfs -o ro,norecovery /dev/sdb1 "$MOUNT_POINT" 2>/dev/null; then
    echo "✓ SUCCESS: sdb1 can be mounted!"
    echo
    echo "Quick content check:"
    ls -la "$MOUNT_POINT" | head -10
    echo
    echo "Looking for arxiv data:"
    find "$MOUNT_POINT" -maxdepth 3 -type d -name "*arxiv*" 2>/dev/null | head -10
    echo
    echo "TAR file count:"
    find "$MOUNT_POINT" -name "*.tar" -o -name "*.tar.gz" 2>/dev/null | wc -l
    
    umount "$MOUNT_POINT"
    echo "✓ Unmounted test mount"
    
    echo
    echo "=== GOOD NEWS: sdb1 appears to have recoverable data! ==="
    echo "Recommended actions:"
    echo "1. Create a backup image immediately:"
    echo "   ddrescue -n -B /dev/sdb1 /home/todd/olympus/sdb1_backup.img /home/todd/olympus/sdb1_backup.log"
    echo
    echo "2. Or mount and copy directly:"
    echo "   mount -t xfs -o ro,norecovery /dev/sdb1 /mnt/recovery/sdb1"
    echo "   rsync -avP /mnt/recovery/sdb1/arxiv_data/ /safe/location/"
else
    echo "✗ Cannot mount sdb1 as XFS"
    echo
    echo "=== Alternative Recovery Options ==="
    echo "1. The filesystem structure may be damaged"
    echo "2. Try photorec for raw file recovery"
    echo "3. Create an image first before further attempts:"
    echo "   ddrescue -n -B /dev/sdb1 /path/to/sdb1_image.img /path/to/sdb1.log"
    echo
    echo "4. Then work on the image instead of the physical drive"
fi

echo
echo "=== Summary ==="
echo "sda1: Has XFS filesystem with CRC errors (failing drive)"
echo "sdb1: Check results above - may be recoverable"
echo
echo "Since these were RAID1 mirrors, sdb1 should have the same data as sda1"
echo "If sdb1 is healthier, it's the better recovery source!"