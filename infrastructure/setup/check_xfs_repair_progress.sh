#!/bin/bash
# Check on the xfs_repair progress
# Run as root

echo "=== Checking XFS Repair Progress ==="
echo "Date: $(date)"
echo

# Find the xfs_repair process
XFS_PID=$(pgrep -f "xfs_repair.*sdb1")

if [ -n "$XFS_PID" ]; then
    echo "xfs_repair is still running (PID: $XFS_PID)"
    echo
    
    # Check how long it's been running
    echo "Process info:"
    ps -p $XFS_PID -o pid,etime,cmd
    echo
    
    # Check system I/O to see if it's actually doing work
    echo "I/O activity on sdb:"
    iostat -x 1 2 | grep -E "sdb|Device"
    echo
    
    # Check current disk position being read
    echo "Current disk activity:"
    cat /proc/$XFS_PID/io 2>/dev/null | grep -E "read_bytes|write_bytes" || echo "Cannot read I/O stats"
    echo
    
    # Estimate progress (rough)
    echo "Disk position estimate:"
    DISK_SIZE_BYTES=$((6 * 1024 * 1024 * 1024 * 1024))  # 6TB
    if [ -f /proc/$XFS_PID/fdinfo/3 ]; then
        POS=$(grep pos /proc/$XFS_PID/fdinfo/3 2>/dev/null | awk '{print $2}')
        if [ -n "$POS" ]; then
            PERCENT=$(echo "scale=2; $POS * 100 / $DISK_SIZE_BYTES" | bc)
            echo "Approximate position: $PERCENT% of disk"
        fi
    fi
    
    echo
    echo "=== What's happening ==="
    echo "xfs_repair is searching for backup superblocks across the entire 6TB drive."
    echo "Each period (.) represents a checked location."
    echo "This can take 1-2 hours for a 6TB drive."
    echo
    echo "You can:"
    echo "1. Let it finish (recommended if not urgent)"
    echo "2. Kill it with: kill $XFS_PID"
    echo "3. Skip sdb and focus on sda recovery with new drives"
else
    echo "xfs_repair is not running. The scan may have completed."
    echo "Check the terminal where you ran the recovery script."
fi

echo
echo "=== Alternative: Quick Check ==="
echo "If you want to stop waiting and check sdb differently:"
echo "1. Kill the xfs_repair process"
echo "2. Try a faster mount test:"
echo "   mount -t xfs -o ro,norecovery /dev/sdb1 /mnt/test"
echo "3. Or skip to setting up new drives for recovery"