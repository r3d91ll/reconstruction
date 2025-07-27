#!/usr/bin/env python3
"""
Pipeline Performance Monitor - Real-time stats for GPU pipeline
"""

import subprocess
import time
import sys
import shutil
from datetime import datetime
from pathlib import Path
from collections import deque

def get_gpu_stats():
    """Get GPU memory and utilization stats"""
    # Check if nvidia-smi is available
    if not shutil.which('nvidia-smi'):
        print("Warning: nvidia-smi not found. GPU monitoring disabled.")
        return []
    
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            stats = []
            lines = result.stdout.strip().split('\n')
            for line in lines:
                parts = line.split(', ')
                if len(parts) == 6:
                    try:
                        idx, name, used, total, util, temp = parts
                        stats.append({
                            'idx': int(idx),
                            'name': name,
                            'mem_used_gb': float(used) / 1024,
                            'mem_total_gb': float(total) / 1024,
                            'mem_percent': (float(used) / float(total)) * 100,
                            'gpu_util': int(util),
                            'temp': int(temp)
                        })
                    except ValueError as e:
                        print(f"Error parsing GPU stats line '{line}': {e}")
                        continue
            return stats
    except subprocess.SubprocessError as e:
        print(f"Error running nvidia-smi: {e}")
    except Exception as e:
        print(f"Unexpected error getting GPU stats: {e}")
    return []

def count_running_processes(process_pattern='process_pdfs_continuous_gpu'):
    """Get count of running processes matching pattern"""
    try:
        result = subprocess.run(
            ['pgrep', '-f', process_pattern],
            capture_output=True,
            text=True,
            check=False
        )
        return len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
    except subprocess.SubprocessError as e:
        print(f"Error counting processes: {e}")
        return 0
    except Exception as e:
        print(f"Unexpected error in process count: {e}")
        return 0

def get_pipeline_status(log_file='pipeline_v8_optimized.log', max_lines=1000):
    """Get stats from log file efficiently"""
    stats = {
        'processed': 0,
        'extracted': 0,
        'chunked': 0,
        'errors': 0,
        'last_arxiv_id': '',
        'rate': 0.0
    }
    
    if not Path(log_file).exists():
        return stats
        
    try:
        # Efficiently read only the last N lines using deque
        with open(log_file, 'r') as f:
            lines = deque(f, maxlen=max_lines)
            
        for line in lines:
            if 'Extracted' in line and 'chars' in line:
                stats['extracted'] += 1
                # Extract arxiv ID
                if ':' in line:
                    arxiv_part = line.split('Extracted')[1].split(':')[0].strip()
                    stats['last_arxiv_id'] = arxiv_part
                    
            elif 'Late chunked' in line and 'chunks' in line:
                stats['chunked'] += 1
                
            elif 'ERROR' in line:
                stats['errors'] += 1
                
            elif 'Progress:' in line and 'docs/sec' in line:
                # Extract rate
                try:
                    rate_part = line.split('(')[1].split('docs/sec')[0]
                    stats['rate'] = float(rate_part.strip())
                except (IndexError, ValueError) as e:
                    # Log specific error but continue
                    print(f"Warning: Could not parse rate from line: {e}")
                    
        stats['processed'] = stats['chunked']  # Processed = successfully chunked
        
    except IOError as e:
        print(f"Error reading log file '{log_file}': {e}")
    except Exception as e:
        print(f"Unexpected error processing log file: {e}")
        
    return stats

def format_time(seconds):
    """Format seconds to human readable time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# Configuration constants
DEFAULT_TOTAL_PDFS = 4598
UPDATE_INTERVAL = 5  # seconds

def supports_ansi_escape():
    """Check if terminal supports ANSI escape codes"""
    # Check if output is a TTY
    if not sys.stdout.isatty():
        return False
    
    # Check platform and terminal type
    import os
    term = os.environ.get('TERM', '')
    if term == 'dumb':
        return False
        
    # Most modern terminals support ANSI
    return True

def clear_screen():
    """Clear screen if supported"""
    if supports_ansi_escape():
        print("\033[2J\033[H", end='')
    else:
        # Just add some spacing for non-ANSI terminals
        print("\n" * 2)

def main():
    """Main monitoring loop"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline Performance Monitor")
    parser.add_argument('--total-pdfs', type=int, default=DEFAULT_TOTAL_PDFS,
                        help=f'Total number of PDFs to process (default: {DEFAULT_TOTAL_PDFS})')
    parser.add_argument('--log-file', default='pipeline_v8_optimized.log',
                        help='Log file to monitor')
    parser.add_argument('--update-interval', type=int, default=UPDATE_INTERVAL,
                        help=f'Update interval in seconds (default: {UPDATE_INTERVAL})')
    parser.add_argument('--process-pattern', default='process_pdfs_continuous_gpu',
                        help='Process name pattern to monitor')
    args = parser.parse_args()
    
    print("Pipeline Performance Monitor")
    print("=" * 80)
    
    start_time = time.time()
    last_processed = 0
    use_ansi = supports_ansi_escape()
    
    while True:
        try:
            # Clear screen
            clear_screen()
            
            # Header
            print(f"Pipeline Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            
            # GPU Stats
            gpu_stats = get_gpu_stats()
            if gpu_stats:
                print("\nGPU Status:")
                print("-" * 80)
                for gpu in gpu_stats:
                    # Use simple markers if no ANSI support
                    if use_ansi:
                        status = "🟢" if gpu['mem_percent'] < 85 else "🟡" if gpu['mem_percent'] < 95 else "🔴"
                    else:
                        status = "[OK]" if gpu['mem_percent'] < 85 else "[WARN]" if gpu['mem_percent'] < 95 else "[CRIT]"
                    print(f"{status} GPU {gpu['idx']}: {gpu['mem_used_gb']:.1f}/{gpu['mem_total_gb']:.1f}GB "
                          f"({gpu['mem_percent']:.1f}%) | Util: {gpu['gpu_util']}% | Temp: {gpu['temp']}°C")
            else:
                print("\nGPU monitoring not available")
            
            # Pipeline Stats
            log_stats = get_pipeline_status(args.log_file)
            elapsed = time.time() - start_time
            
            print("\nPipeline Progress:")
            print("-" * 80)
            print(f"Extracted: {log_stats['extracted']} PDFs")
            print(f"Chunked: {log_stats['chunked']} documents")
            print(f"Errors: {log_stats['errors']}")
            print(f"Last PDF: {log_stats['last_arxiv_id']}")
            
            # Performance
            print("\nPerformance:")
            print("-" * 80)
            
            # Calculate rates
            if elapsed > 0:
                overall_rate = log_stats['processed'] / elapsed
                print(f"Overall Rate: {overall_rate:.2f} docs/sec")
            
            if log_stats['rate'] > 0:
                print(f"Current Rate: {log_stats['rate']:.2f} docs/sec")
                
            # Time estimate
            if log_stats['processed'] > 0 and log_stats['rate'] > 0:
                remaining = args.total_pdfs - log_stats['processed']
                if remaining > 0:
                    eta_seconds = remaining / log_stats['rate']
                    print(f"ETA: {format_time(eta_seconds)} ({remaining} PDFs remaining)")
                else:
                    print("Processing complete!")
            
            print(f"\nRuntime: {format_time(elapsed)}")
            
            # Process count
            proc_count = count_running_processes(args.process_pattern)
            print(f"Active Processes: {proc_count}")
            
            # Instructions
            print("\n" + "=" * 80)
            print("Press Ctrl+C to exit monitor (pipeline will continue running)")
            
            time.sleep(args.update_interval)
            
        except KeyboardInterrupt:
            print("\nMonitor stopped. Pipeline continues running.")
            break
        except Exception as e:
            print(f"Monitor error: {e}")
            time.sleep(args.update_interval)

if __name__ == "__main__":
    main()