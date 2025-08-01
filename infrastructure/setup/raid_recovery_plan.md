# RAID Recovery and Reconfiguration Plan

## Current Status
- md0: RAID0 (4x 1TB NVMe) - RECOVERED but unstable
- md1: RAID1 (2x 4TB NVMe) - Healthy
- nvme3n1p1 temporarily dropped from md0 but rejoined after reboot

## Immediate Actions

### 1. Check Drive Health
```bash
# Check SMART status for all NVMe drives
for drive in nvme{0..6}n1; do
    echo "=== $drive ==="
    sudo nvme smart-log /dev/$drive | grep -E "critical_warning|temperature|available_spare|percentage_used"
done

# Check for errors in dmesg
sudo dmesg | grep -i "nvme\|error\|fail" | tail -50
```

### 2. Backup Critical Data
```bash
# Ensure arxiv PDFs are backed up
rsync -avP /mnt/data-cold/arxiv_data/ /home/todd/olympus/arxiv_backup/
```

### 3. Monitor RAID Health
```bash
# Configure proper alerting in /etc/mdadm/mdadm.conf:
echo "MAILADDR=your-email@example.com" >> /etc/mdadm/mdadm.conf
# Or use a custom alert script:
echo "PROGRAM /usr/local/bin/raid-alert.sh" >> /etc/mdadm/mdadm.conf

# Then enable monitoring daemon:
systemctl enable mdadm-monitor
systemctl start mdadm-monitor

# Alternative: Add to crontab with custom alert script
*/5 * * * * /usr/sbin/mdadm --monitor --scan --oneshot --program=/usr/local/bin/raid-alert.sh
```

## Safer RAID Configuration Options

### Option 1: Convert md0 to RAID10 (Recommended)
**Pros:** Good balance of performance and redundancy
**Cons:** Only 50% space efficiency
```bash
# Steps (DESTRUCTIVE - backup first!):
1. Backup all data from md0
2. Stop all services using md0
3. Unmount all LVs on vg0
4. Remove LVM setup: lvremove, vgremove, pvremove
5. Stop and remove md0
6. Zero old superblocks to prevent auto-assembly:
   for dev in /dev/nvme3n1p1 /dev/nvme4n1p1 /dev/nvme5n1p1 /dev/nvme6n1p1; do
       mdadm --zero-superblock $dev
   done
7. Create new RAID10:
   mdadm --create /dev/md0 --level=10 --raid-devices=4 /dev/nvme3n1p1 /dev/nvme4n1p1 /dev/nvme5n1p1 /dev/nvme6n1p1
```

### Option 2: Split Workloads
**Pros:** Isolate risk, optimize for workload
**Cons:** More complex management
```bash
# Create two arrays:
# md2: RAID1 for important data (2x 1TB)
# md3: RAID0 for truly temporary data (2x 1TB)
```

### Option 3: RAID5
**Pros:** Better space efficiency (75%)
**Cons:** Write performance penalty, rebuild stress
```bash
mdadm --create /dev/md0 --level=5 --raid-devices=4 /dev/nvme3n1p1 /dev/nvme4n1p1 /dev/nvme5n1p1 /dev/nvme6n1p1
```

## PDF Processing Pipeline Improvements

### 1. Add I/O throttling
```python
# Limit concurrent reads
import time
import threading

class IOThrottler:
    def __init__(self, max_iops=100):
        self.max_iops = max_iops
        self.lock = threading.Lock()
        self.last_io = 0
    
    def throttle(self):
        with self.lock:
            now = time.time()
            min_interval = 1.0 / self.max_iops
            elapsed = now - self.last_io
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            self.last_io = time.time()
```

### 2. Reduce worker count
```python
# Conservative settings
docling_workers: int = 2  # Was 4
docling_memory_fraction: float = 0.4  # Was 0.22 per worker
```

### 3. Add health checks
```python
# Check system resources before processing
def check_system_health():
    # Check RAID status
    raid_status = subprocess.run(['cat', '/proc/mdstat'], capture_output=True, text=True)
    if 'degraded' in raid_status.stdout or '[_' in raid_status.stdout:
        raise Exception("RAID array degraded!")
    
    # Check disk space
    stat = os.statvfs('/mnt/data-cold')
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    if free_gb < 100:
        raise Exception(f"Low disk space: {free_gb:.1f}GB free")
```

## Long-term Recommendations

1. **Hardware Monitoring**
   - Set up smartd for all drives
   - Configure email alerts for drive failures
   - Monitor temperatures during heavy workloads

2. **Workload Isolation**
   - Separate database I/O from batch processing
   - Use different storage tiers for different data types
   - Consider ZFS for better data integrity

3. **Backup Strategy**
   - Regular snapshots of critical data
   - Off-site backup for arxiv PDFs
   - Test restore procedures regularly