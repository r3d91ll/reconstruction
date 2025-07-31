#!/bin/bash
# RAID1 Recovery Investigation Script
# Run as root

echo "=== RAID1 Recovery Investigation ==="
echo "Date: $(date)"
echo

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (sudo -i first)"
    exit 1
fi

echo "=== Examining sda and sdb ==="
echo

# Check partition tables
echo "--- Partition Tables ---"
fdisk -l /dev/sda | grep -E "^/dev|Disk model"
echo
fdisk -l /dev/sdb | grep -E "^/dev|Disk model"
echo

# Check for filesystem signatures
echo "--- Filesystem Detection ---"
file -s /dev/sda1
file -s /dev/sdb1
echo

# Check for RAID superblocks
echo "--- RAID Superblock Search ---"
mdadm --examine /dev/sda1 2>&1
echo
mdadm --examine /dev/sdb1 2>&1
echo

# Check for any RAID metadata on the whole disk
echo "--- Checking whole disks for RAID ---"
mdadm --examine /dev/sda 2>&1 | head -20
echo
mdadm --examine /dev/sdb 2>&1 | head -20
echo

# Look for XFS filesystem headers
echo "--- XFS Filesystem Headers ---"
xfs_db -r -c "sb 0" -c "p" /dev/sda1 2>&1 | head -10
echo
xfs_db -r -c "sb 0" -c "p" /dev/sdb1 2>&1 | head -10
echo

# Check if data is directly accessible
echo "--- Direct Mount Attempts (read-only) ---"
mkdir -p /mnt/recovery/sda1 /mnt/recovery/sdb1

# Try mounting as XFS
if mount -t xfs -o ro,norecovery /dev/sda1 /mnt/recovery/sda1 2>/dev/null; then
    echo "✓ sda1 mounted successfully as XFS"
    echo "Contents preview:"
    ls -la /mnt/recovery/sda1/ | head -10
    df -h /mnt/recovery/sda1
    umount /mnt/recovery/sda1
else
    echo "✗ sda1 failed to mount as XFS"
fi
echo

if mount -t xfs -o ro,norecovery /dev/sdb1 /mnt/recovery/sdb1 2>/dev/null; then
    echo "✓ sdb1 mounted successfully as XFS"
    echo "Contents preview:"
    ls -la /mnt/recovery/sdb1/ | head -10
    df -h /mnt/recovery/sdb1
    umount /mnt/recovery/sdb1
else
    echo "✗ sdb1 failed to mount as XFS"
fi
echo

# Look for backup superblocks
echo "--- Searching for XFS backup superblocks ---"
echo "sda1:"
xfs_repair -n -v /dev/sda1 2>&1 | grep -E "^Phase|superblock|found"
echo
echo "sdb1:"
xfs_repair -n -v /dev/sdb1 2>&1 | grep -E "^Phase|superblock|found"
echo

# Check for old mdadm metadata
echo "--- Checking for old RAID configurations ---"
cat /etc/mdadm/mdadm.conf 2>/dev/null || echo "No mdadm.conf found"
echo

# Look for LVM signatures (in case RAID1 had LVM on top)
echo "--- LVM Signature Check ---"
pvs 2>&1 | grep -E "sda|sdb"
echo

# Summary
echo "=== Recovery Options ==="
echo "Based on the investigation above, here are your options:"
echo "1. If XFS mounted successfully, data is directly accessible"
echo "2. If RAID superblocks found, can reassemble RAID1"
echo "3. If only filesystem found, can recover from single disk"
echo "4. If LVM signatures found, need to activate VG first"
echo
echo "Next steps will depend on what we found above."