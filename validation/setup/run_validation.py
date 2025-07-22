#!/usr/bin/env python3
"""
Master runner for Information Reconstructionism validation
Creates timestamped output directories for each run
"""

import os
import sys
import subprocess
from datetime import datetime
import shutil

def create_run_directory():
    """Create timestamped directory for this run"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"/home/todd/reconstructionism/validation/results/run_{timestamp}"
    
    # Create directory structure
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "analysis"), exist_ok=True)
    
    # Create run info file
    info_file = os.path.join(run_dir, "run_info.txt")
    with open(info_file, 'w') as f:
        f.write(f"Run timestamp: {timestamp}\n")
        f.write(f"Start time: {datetime.now().isoformat()}\n")
        f.write(f"Purpose: Information Reconstructionism Validation\n")
    
    return run_dir

def run_validation(run_dir):
    """Run the validation sequence"""
    
    # Set environment variable for output directory
    os.environ['VALIDATION_OUTPUT_DIR'] = os.path.join(run_dir, "data")
    
    # Log file for this run
    log_file = os.path.join(run_dir, "logs", "validation.log")
    
    print(f"Starting validation run in: {run_dir}")
    print(f"Logs will be saved to: {log_file}")
    print("-" * 60)
    
    # Run the proof sequence
    core_dir = "/home/todd/reconstructionism/validation/core"
    runner_script = os.path.join(core_dir, "run_proof_sequence.py")
    
    with open(log_file, 'w') as log:
        process = subprocess.Popen(
            [sys.executable, runner_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=core_dir
        )
        
        # Stream output to both console and log file
        for line in process.stdout:
            print(line, end='')
            log.write(line)
            log.flush()
        
        process.wait()
        
    # Update run info with completion
    info_file = os.path.join(run_dir, "run_info.txt")
    with open(info_file, 'a') as f:
        f.write(f"End time: {datetime.now().isoformat()}\n")
        f.write(f"Exit code: {process.returncode}\n")
    
    return process.returncode

def main():
    print("INFORMATION RECONSTRUCTIONISM - VALIDATION RUNNER")
    print("=" * 60)
    
    # Create run directory
    run_dir = create_run_directory()
    
    # Run validation
    exit_code = run_validation(run_dir)
    
    if exit_code == 0:
        print(f"\n✓ Validation completed successfully!")
        print(f"Results saved in: {run_dir}")
    else:
        print(f"\n✗ Validation failed with exit code: {exit_code}")
        print(f"Check logs in: {run_dir}/logs/")
    
    # Create a symlink to latest run
    latest_link = "/home/todd/reconstructionism/validation/results/latest"
    if os.path.exists(latest_link):
        os.remove(latest_link)
    os.symlink(run_dir, latest_link)
    print(f"\nLatest run linked at: {latest_link}")

if __name__ == "__main__":
    main()
