#!/usr/bin/env python3
"""
Launcher Script for Dual-GPU Pipeline with Monitoring

This script:
1. Starts the dual-GPU processing pipeline in the background
2. Launches the monitoring dashboard in the foreground
3. Handles graceful shutdown of both processes
"""

import os
import sys
import subprocess
import time
import signal
import argparse
from pathlib import Path
import json
from datetime import datetime

class PipelineLauncher:
    """Manages launching and monitoring the dual-GPU pipeline."""
    
    def __init__(self):
        self.pipeline_process = None
        self.monitor_process = None
        self.log_dir = Path("./logs")
        self.log_dir.mkdir(exist_ok=True)
        
    def launch_pipeline(self, args):
        """Launch the processing pipeline in background."""
        # Determine which pipeline to use
        if args.production:
            script_name = "setup/process_abstracts_production.py"
        else:
            script_name = "setup/process_abstracts_dual_gpu.py"
            
        # Build command
        cmd = [
            sys.executable,
            script_name,
            "--db-name", args.db_name,
            "--db-host", args.db_host,
            "--batch-size", str(args.batch_size),
            "--metadata-dir", args.metadata_dir
        ]
        
        if args.count:
            cmd.extend(["--count", str(args.count)])
        
        if args.clean_start:
            cmd.append("--clean-start")
            
        if args.checkpoint_dir:
            cmd.extend(["--checkpoint-dir", args.checkpoint_dir])
            
        if args.resume:
            cmd.append("--resume")
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pipeline_log = self.log_dir / f"pipeline_{timestamp}.log"
        
        print(f"Starting pipeline process...")
        print(f"Command: {' '.join(cmd)}")
        print(f"Log file: {pipeline_log}")
        
        # Start pipeline process
        with open(pipeline_log, 'w') as log_file:
            self.pipeline_process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid  # Create new process group for clean shutdown
            )
        
        print(f"Pipeline started with PID: {self.pipeline_process.pid}")
        
        # Wait a moment for pipeline to initialize
        time.sleep(5)
        
        # Check if process is still running
        if self.pipeline_process.poll() is not None:
            print("ERROR: Pipeline process died immediately!")
            with open(pipeline_log, 'r') as f:
                print("Last log lines:")
                print(f.read()[-1000:])  # Print last 1000 chars
            return False
            
        return True
    
    def launch_monitor(self, checkpoint_dir):
        """Launch the monitoring dashboard."""
        # Determine checkpoint path
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir) / "checkpoint.pkl"
        else:
            # Default checkpoint location
            checkpoint_path = Path(f"./checkpoints/{self.db_name}") / "checkpoint.pkl"
        
        # Build monitoring command
        cmd = [
            sys.executable,
            "gpu_monitor_dashboard.py",
            "--checkpoint", str(checkpoint_path),
            "--metrics-file", str(self.log_dir / "metrics.jsonl"),
            "--save-interval", "30"
        ]
        
        print(f"\nStarting monitoring dashboard...")
        print(f"Checkpoint: {checkpoint_path}")
        
        # Run monitor in foreground
        try:
            self.monitor_process = subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\nMonitor stopped by user")
    
    def cleanup(self):
        """Clean shutdown of all processes."""
        print("\nShutting down...")
        
        # Monitor process runs in foreground and completes when this cleanup runs
        # No explicit cleanup needed for monitor
        # Stop pipeline gracefully
        if self.pipeline_process and self.pipeline_process.poll() is None:
            print("Stopping pipeline process...")
            # Send SIGTERM to process group
            try:
                os.killpg(os.getpgid(self.pipeline_process.pid), signal.SIGTERM)
                # Wait up to 30 seconds for graceful shutdown
                self.pipeline_process.wait(timeout=30)
                print("Pipeline stopped gracefully")
            except subprocess.TimeoutExpired:
                print("Pipeline didn't stop gracefully, forcing...")
                os.killpg(os.getpgid(self.pipeline_process.pid), signal.SIGKILL)
            except Exception as e:
                print(f"Error stopping pipeline: {e}")
    
    def run(self, args):
        """Main execution flow."""
        # Set up signal handlers
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}")
            self.cleanup()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Launch pipeline
            if not self.launch_pipeline(args):
                print("Failed to start pipeline!")
                return 1
            
            # Give pipeline time to initialize
            print("\nWaiting for pipeline to initialize...")
            time.sleep(10)
            
            # Launch monitor
            self.launch_monitor(args.checkpoint_dir)
            
            # When monitor exits, check if pipeline is still running
            if self.pipeline_process.poll() is None:
                print("\nMonitor exited but pipeline still running.")
                print("Options:")
                print("  1. Press Ctrl+C to stop pipeline")
                print("  2. Run monitor again: python gpu_monitor_dashboard.py --checkpoint <path>")
                
                # Wait for pipeline to complete or user interrupt with timeout
                try:
                    # Wait up to 300 seconds (5 minutes) for pipeline to complete
                    self.pipeline_process.wait(timeout=300)
                except subprocess.TimeoutExpired:
                    print("\nPipeline process timeout reached (5 minutes).")
                    print("Pipeline may still be running. Options:")
                    print("  1. Terminate the pipeline process")
                    print("  2. Let it continue running in the background")
                    
                    try:
                        response = input("\nTerminate pipeline? (y/N): ").strip().lower()
                        if response == 'y':
                            print("Terminating pipeline process...")
                            self.pipeline_process.terminate()
                            # Give it 10 seconds to terminate gracefully
                            try:
                                self.pipeline_process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                print("Process didn't terminate gracefully, killing it...")
                                self.pipeline_process.kill()
                                self.pipeline_process.wait()
                        else:
                            print("Pipeline process will continue running in the background.")
                            print(f"Pipeline PID: {self.pipeline_process.pid}")
                    except KeyboardInterrupt:
                        print("\nReceived interrupt, cleaning up...")
                        
        except Exception as e:
            print(f"\nError: {e}")
            return 1
        finally:
            self.cleanup()
            
        return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Launch dual-GPU pipeline with monitoring dashboard"
    )
    
    # Pipeline arguments
    parser.add_argument('--metadata-dir', type=str, 
                        default='/mnt/data/arxiv_data/metadata',
                        help='Directory containing metadata JSON files')
    parser.add_argument('--count', type=int, 
                        help='Number of documents to process (default: all)')
    parser.add_argument('--db-name', type=str, 
                        default='arxiv_abstracts_enhanced',
                        help='Database name')
    parser.add_argument('--db-host', type=str, 
                        default=os.getenv('DB_HOST', 'localhost'),
                        help='Database host/IP')
    parser.add_argument('--clean-start', action='store_true',
                        help='Drop existing database and start fresh')
    parser.add_argument('--batch-size', type=int, default=200,
                        help='Base batch size per GPU')
    parser.add_argument('--checkpoint-dir', type=str,
                        help='Directory for checkpoints')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--production', action='store_true',
                        help='Use production pipeline with advanced features')
    
    args = parser.parse_args()
    
    # Print configuration
    print("DUAL-GPU PIPELINE LAUNCHER")
    print("="*60)
    print(f"Database: {args.db_name} @ {args.db_host}")
    print(f"Metadata: {args.metadata_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Mode: {'Resume' if args.resume else 'Fresh start'}")
    print("="*60)
    
    # Create and run launcher
    launcher = PipelineLauncher()
    return launcher.run(args)


if __name__ == "__main__":
    sys.exit(main())