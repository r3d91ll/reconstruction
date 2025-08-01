#!/bin/bash
# Check health of all drives
# Run as root

echo "=== DRIVE HEALTH CHECK ==="
echo "Date: $(date)"
echo

# Function to check drive health
check_drive() {
    local drive=$1
    echo "=== Checking $drive ==="
    
    # Basic info
    smartctl -i /dev/$drive 2>/dev/null | grep -E "Model|Serial|Capacity|Sector Size"
    
    # Health status
    echo
    smartctl -H /dev/$drive 2>/dev/null | grep -A1 "SMART overall"
    
    # Critical attributes
    echo
    echo "Critical SMART attributes:"
    smartctl -A /dev/$drive 2>/dev/null | grep -E "Reallocated_Sector_Ct|Current_Pending_Sector|Offline_Uncorrectable|UDMA_CRC_Error_Count|Temperature_Celsius|Media_Wearout_Indicator|Wear_Leveling_Count|Runtime_Bad_Block|Reported_Uncorrect" | grep -v "Pre-fail"
    
    # For NVMe drives
    if [[ $drive == nvme* ]]; then
        echo
        echo "NVMe specific health:"
        nvme smart-log /dev/$drive 2>/dev/null | grep -E "critical_warning|temperature|available_spare|percentage_used|data_units_read|data_units_written|host_read_commands|host_write_commands|media_errors|num_err_log_entries"
    fi
    
    echo
    echo "----------------------------------------"
    echo
}

# Check all drives
echo "=== Checking traditional drives ==="
for drive in sda sdb; do
    if [ -e "/dev/$drive" ]; then
        check_drive $drive
    fi
done

echo "=== Checking NVMe drives ==="
for drive in nvme{0..6}n1; do
    if [ -e "/dev/$drive" ]; then
        check_drive $drive
    fi
done

echo
echo "=== RAID Status ==="
cat /proc/mdstat

echo
echo "=== Storage Overview ==="
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT,MODEL | grep -v loop

echo
echo "=== Recommendations ==="
echo "1. sda shows hardware errors - DO NOT use for new data"
echo "2. Check if sdb is healthy enough to recover RAID1 data"
echo "3. Monitor NVMe drives' percentage_used for wear level"
echo "4. Any drive with reallocated sectors or pending sectors needs replacement"