#!/bin/bash
# Check on the xfs_repair progress
# Run as root
set -euo pipefail

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Error: This script must be run as root to access /proc information" >&2
    exit 1
fi

echo "=== Checking XFS Repair Progress ==="
echo "Date: $(date)"
echo

# Find the xfs_repair process(es)
mapfile -t XFS_PIDS < <(pgrep -f "xfs_repair.*sdb1")

if [ ${#XFS_PIDS[@]} -gt 0 ]; then
    for XFS_PID in "${XFS_PIDS[@]}"; do
        echo "xfs_repair is still running (PID: $XFS_PID)"
        echo
        
        # Check how long it's been running
        echo "Process info:"
        ps -p "$XFS_PID" -o pid,etime,cmd
    echo
    
        # Check system I/O to see if it's actually doing work
        echo "I/O activity on sdb:"
        iostat -x 1 2 | grep -E "sdb|Device" || true
        echo
        
        # Check current disk position being read
        echo "Current disk activity:"
        cat "/proc/$XFS_PID/io" 2>/dev/null | grep -E "read_bytes|write_bytes" || echo "Cannot read I/O stats"
        echo
        
        # Find the file descriptor for /dev/sdb1
        device_fd=""
        if [ -d "/proc/$XFS_PID/fd" ]; then
            for fd in /proc/$XFS_PID/fd/*; do
                if [ -L "$fd" ]; then
                    link_target=$(readlink -f "$fd" 2>/dev/null || true)
                    if [ "$link_target" = "/dev/sdb1" ]; then
                        device_fd=$(basename "$fd")
                        break
                    fi
                fi
            done
        fi
        
        # Estimate progress (rough)
        echo "Disk position estimate:"
        # Get actual disk size dynamically
        if command -v blockdev >/dev/null 2>&1; then
            DISK_SIZE_BYTES=$(blockdev --getsize64 /dev/sdb1 2>/dev/null || echo "0")
        else
            DISK_SIZE_BYTES=0
        fi
        
        if [ -n "$device_fd" ] && [ -f "/proc/$XFS_PID/fdinfo/$device_fd" ] && [ "$DISK_SIZE_BYTES" -gt 0 ]; then
            POS=$(grep pos "/proc/$XFS_PID/fdinfo/$device_fd" 2>/dev/null | awk '{print $2}')
            if [ -n "$POS" ]; then
                PERCENT=$(echo "scale=2; $POS * 100 / $DISK_SIZE_BYTES" | bc)
                echo "Approximate position: $PERCENT% of disk"
                echo "Device FD: $device_fd, Position: $POS bytes, Disk size: $DISK_SIZE_BYTES bytes"
            else
                echo "Could not read position from fdinfo"
            fi
        else
            if [ -z "$device_fd" ]; then
                echo "Could not find file descriptor for /dev/sdb1"
            elif [ "$DISK_SIZE_BYTES" -eq 0 ]; then
                echo "Could not determine disk size"
            else
                echo "Could not read fdinfo"
            fi
        fi
    
        echo
    done
    
    echo "=== What's happening ==="
    echo "xfs_repair is searching for backup superblocks across the drive."
    echo "Each period (.) represents a checked location."
    echo "This can take 1-2 hours for a large drive."
    echo
    echo "You can:"
    echo "1. Let it finish (recommended if not urgent)"
    if [ ${#XFS_PIDS[@]} -eq 1 ]; then
        echo "2. Kill it with: kill ${XFS_PIDS[0]}"
    else
        echo "2. Kill all processes with: kill ${XFS_PIDS[*]}"
    fi
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