# Storage Architecture Options for New 6TB Drives

## Current Issues
- RAID not properly saved to mdadm.conf
- RAID0 failure caused system crash
- LVM adds complexity layer
- Mixed hot/cold data on same arrays

## Option 1: RAID1 + LVM (Traditional)
**Pros:**
- Flexible volume management
- Can resize volumes dynamically
- Snapshots capability
- Familiar setup

**Cons:**
- Extra complexity layer
- More points of failure
- Need to maintain both mdadm and LVM configs

```bash
# Setup
mdadm --create /dev/md2 --level=1 --raid-devices=2 /dev/sdc /dev/sdd
pvcreate /dev/md2
vgcreate vg_cold /dev/md2
lvcreate -L 5.5T -n arxiv_data vg_cold
mkfs.xfs /dev/vg_cold/arxiv_data
```

## Option 2: RAID1 Direct (No LVM)
**Pros:**
- Simpler stack
- Fewer things to break
- Direct filesystem on RAID
- Easier recovery

**Cons:**
- Less flexibility
- No easy resizing
- No snapshots

```bash
# Setup
mdadm --create /dev/md2 --level=1 --raid-devices=2 /dev/sdc /dev/sdd
mkfs.xfs /dev/md2
mount /dev/md2 /mnt/data-cold
```

## Option 3: ZFS Mirror
**Pros:**
- Built-in checksums (detects corruption)
- Snapshots
- Compression
- No separate RAID/LVM layers
- Self-healing with redundancy

**Cons:**
- Different toolset
- More RAM usage
- Not in mainline kernel

```bash
# Setup
zpool create -o ashift=12 cold_storage mirror /dev/sdc /dev/sdd
zfs set compression=lz4 cold_storage
zfs set atime=off cold_storage
```

## Option 4: Btrfs RAID1
**Pros:**
- Built-in RAID
- Checksums
- Snapshots
- In mainline kernel
- Easy scrubbing

**Cons:**
- Btrfs RAID5/6 still unstable
- Some consider it less mature

```bash
# Setup
mkfs.btrfs -m raid1 -d raid1 /dev/sdc /dev/sdd
mount /dev/sdc /mnt/data-cold
```

## Recommendation: RAID1 Direct for Cold Storage

For your arxiv TAR files (cold storage), I'd recommend **Option 2: RAID1 Direct**:

1. **Simplicity**: Fewer layers = fewer failure points
2. **Recovery**: Easier to recover from either drive
3. **Performance**: No LVM overhead
4. **Reliability**: mdadm RAID1 is rock-solid

### Setup Steps:
```bash
# 1. Create RAID1
mdadm --create /dev/md2 --level=1 --raid-devices=2 --metadata=1.2 /dev/sdc /dev/sdd

# 2. Wait for sync
watch cat /proc/mdstat

# 3. Create filesystem
mkfs.xfs -L arxiv_cold /dev/md2

# 4. Mount
mkdir -p /mnt/data-cold
mount /dev/md2 /mnt/data-cold

# 5. CRITICAL: Save RAID config
mdadm --detail --scan >> /etc/mdadm/mdadm.conf
update-initramfs -u

# 6. Add to fstab
echo "UUID=$(blkid -s UUID -o value /dev/md2) /mnt/data-cold xfs defaults,noatime 0 2" >> /etc/fstab
```

### Monitoring Setup:
```bash
# Email alerts
apt install mailutils
echo "MAILADDR=your-email@example.com" >> /etc/mdadm/mdadm.conf

# Or script alerts
cat > /etc/mdadm/mdadm.conf << 'EOF'
PROGRAM /usr/local/bin/raid-alert.sh
EOF

# Regular scrubbing (monthly)
echo '0 2 1 * * root /usr/share/mdadm/checkarray --all' > /etc/cron.d/mdadm-scrub
```

## For the Existing Arrays:

### Convert md0 from RAID0 to RAID10:
- Requires backup/restore (no in-place conversion)
- Or keep as RAID0 but ONLY for truly temporary data

### Keep md1 RAID1:
- Already good for reliability
- Just ensure config is saved

What do you think? Simple RAID1 for the cold storage?