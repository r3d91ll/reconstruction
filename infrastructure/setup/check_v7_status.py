#!/usr/bin/env python3
"""Check V7 pipeline status"""

import os
import subprocess
import time
from datetime import datetime

# Check running processes
print("=== Running Processes ===")
try:
    result = subprocess.run(
        ["ps", "aux"],
        capture_output=True,
        text=True,
        check=True
    )
    # Filter for process_pdfs_continuous_gpu_v7
    for line in result.stdout.splitlines():
        if "process_pdfs_continuous_gpu_v7" in line and "grep" not in line:
            print(line)
except subprocess.CalledProcessError as e:
    print(f"Error checking processes: {e}")

# Check GPU usage
print("\n=== GPU Status ===")
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv"],
        capture_output=True,
        text=True,
        check=True
    )
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"Error checking GPU status: {e}")
except FileNotFoundError:
    print("nvidia-smi not found - GPU monitoring not available")

# Check recent logs
print("\n=== Recent Activity ===")
log_files = [
    "v7_test.log",
    "late_chunking_pipeline.log", 
    "/tmp/pdf_pipeline_v7_late.log"
]

for log_file in log_files:
    if os.path.exists(log_file):
        print(f"\nChecking {log_file}:")
        try:
            # Use tail to get last 20 lines
            result = subprocess.run(
                ["tail", "-20", log_file],
                capture_output=True,
                text=True,
                check=True
            )
            # Filter for relevant keywords
            for line in result.stdout.splitlines():
                if any(keyword in line.lower() for keyword in ["worker", "chunk", "error", "late"]):
                    print(line)
        except subprocess.CalledProcessError as e:
            print(f"Error reading log file: {e}")

# Check checkpoint database
checkpoint_path = "checkpoints/pdf_pipeline_v7_late/checkpoints.lmdb"
if os.path.exists(checkpoint_path):
    print(f"\n=== Checkpoint Status ===")
    stat = os.stat(checkpoint_path)
    print(f"Checkpoint DB size: {stat.st_size / 1024:.1f} KB")
    print(f"Last modified: {datetime.fromtimestamp(stat.st_mtime)}")