#!/usr/bin/env python3
"""
Real-time GPU Monitoring Dashboard
Displays live statistics for dual-GPU processing pipeline.
"""

import os
import time
import psutil
import torch
import curses
from datetime import datetime, timedelta
from collections import deque
import threading
import json
from pathlib import Path
from typing import Dict, List, Optional
import pickle
import logging

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


class RestrictedUnpickler(pickle.Unpickler):
    """Restricted unpickler that only allows safe built-in types."""
    
    def find_class(self, module, name):
        # Only allow safe built-in types
        ALLOWED_CLASSES = {
            'builtins': {'dict', 'list', 'set', 'tuple', 'str', 'int', 'float', 'bool', 'bytes', 'NoneType'},
            '__builtin__': {'dict', 'list', 'set', 'tuple', 'str', 'int', 'float', 'bool', 'bytes', 'NoneType'}
        }
        
        if module in ALLOWED_CLASSES and name in ALLOWED_CLASSES[module]:
            return super().find_class(module, name)
        
        raise pickle.UnpicklingError(f"Restricted unpickling: {module}.{name} not allowed")


class GPUMonitorDashboard:
    """Real-time monitoring dashboard for dual-GPU processing."""
    
    def __init__(self, checkpoint_path: Optional[Path] = None, 
                 metrics_file: Optional[Path] = None,
                 gpu_ids: Optional[List[int]] = None):
        self.checkpoint_path = checkpoint_path
        self.metrics_file = metrics_file
        self.logger = logging.getLogger(__name__)
        
        # Configurable GPU list
        if gpu_ids is None:
            # Default to GPUs 0 and 1 if available
            if torch.cuda.is_available():
                self.gpu_ids = list(range(min(2, torch.cuda.device_count())))
            else:
                self.gpu_ids = [0, 1]  # Default fallback
        else:
            self.gpu_ids = gpu_ids
        
        # Initialize NVML if available
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
            except (pynvml.NVMLError, Exception) as e:
                self.nvml_initialized = False
                self.logger.warning(f"Failed to initialize NVML: {e}")
        else:
            self.nvml_initialized = False
            self.logger.warning("pynvml not available. GPU temperature/power monitoring disabled.")
        
        # Data storage
        self.gpu_stats = {}
        for gpu_id in self.gpu_ids:
            self.gpu_stats[gpu_id] = {
                'memory_history': deque(maxlen=60),
                'util_history': deque(maxlen=60),
                'temp_history': deque(maxlen=60),
                'power_history': deque(maxlen=60),
                'throughput_history': deque(maxlen=60)
            }
        
        self.system_stats = {
            'cpu_history': deque(maxlen=60),
            'memory_history': deque(maxlen=60),
            'network_history': deque(maxlen=60)
        }
        
        self.processing_stats = {
            'total_processed': 0,
            'total_failed': 0,
            'start_time': time.time(),
            'checkpoint_data': {}
        }
        
        # Update thread
        self.stop_event = threading.Event()
        self.update_thread = None
        
    def start_monitoring(self):
        """Start background monitoring thread."""
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring thread."""
        self.stop_event.set()
        if self.update_thread:
            self.update_thread.join()
        
        if self.nvml_initialized:
            pynvml.nvmlShutdown()
    
    def _update_loop(self):
        """Background thread to update statistics."""
        while not self.stop_event.is_set():
            try:
                self._update_gpu_stats()
                self._update_system_stats()
                self._update_processing_stats()
                time.sleep(1)  # Update every second
            except Exception as e:
                # Don't crash on monitoring errors
                self.logger.error(f"Error in monitoring update loop: {e}")
    
    def _update_gpu_stats(self):
        """Update GPU statistics."""
        for gpu_id in self.gpu_ids:
            try:
                # Memory stats
                torch.cuda.set_device(gpu_id)
                allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
                memory_percent = (allocated / total) * 100
                
                self.gpu_stats[gpu_id]['memory_history'].append({
                    'time': time.time(),
                    'percent': memory_percent,
                    'allocated_gb': allocated,
                    'total_gb': total
                })
                
                # NVML stats (temperature, utilization, power)
                if self.nvml_initialized:
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                        
                        # Utilization
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        self.gpu_stats[gpu_id]['util_history'].append({
                            'time': time.time(),
                            'gpu': util.gpu,
                            'memory': util.memory
                        })
                        
                        # Temperature
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        self.gpu_stats[gpu_id]['temp_history'].append({
                            'time': time.time(),
                            'temp': temp
                        })
                        
                        # Power
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to watts
                        max_power = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000
                        self.gpu_stats[gpu_id]['power_history'].append({
                            'time': time.time(),
                            'power': power,
                            'max_power': max_power
                        })
                        
                    except (pynvml.NVMLError, AttributeError) as e:
                        self.logger.debug(f"NVML error for GPU {gpu_id}: {e}")
                        
            except (torch.cuda.CudaError, RuntimeError) as e:
                self.logger.debug(f"CUDA error for GPU {gpu_id}: {e}")
    
    def _update_system_stats(self):
        """Update system statistics."""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.system_stats['cpu_history'].append({
            'time': time.time(),
            'percent': cpu_percent
        })
        
        # Memory
        mem = psutil.virtual_memory()
        self.system_stats['memory_history'].append({
            'time': time.time(),
            'percent': mem.percent,
            'used_gb': mem.used / (1024**3),
            'total_gb': mem.total / (1024**3)
        })
        
        # Network (if processing involves network)
        try:
            net = psutil.net_io_counters()
            self.system_stats['network_history'].append({
                'time': time.time(),
                'bytes_sent': net.bytes_sent,
                'bytes_recv': net.bytes_recv
            })
        except (AttributeError, OSError) as e:
            self.logger.debug(f"Network IO error: {e}")
    
    def _update_processing_stats(self):
        """Update processing statistics from checkpoint."""
        if self.checkpoint_path and self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'rb') as f:
                    data = RestrictedUnpickler(f).load()
                    self.processing_stats['checkpoint_data'] = data
                    
                    # Calculate totals
                    self.processing_stats['total_processed'] = len(data.get('processed_files', []))
                    self.processing_stats['total_failed'] = len(data.get('failed_files', {}))
                    
                    # Calculate throughput
                    for gpu_id in self.gpu_ids:
                        if gpu_id in data.get('gpu_stats', {}):
                            processed = data['gpu_stats'][gpu_id].get('processed', 0)
                            elapsed = time.time() - self.processing_stats['start_time']
                            throughput = processed / elapsed if elapsed > 0 else 0
                            
                            self.gpu_stats[gpu_id]['throughput_history'].append({
                                'time': time.time(),
                                'throughput': throughput,
                                'total': processed
                            })
            except (FileNotFoundError, IOError, pickle.UnpicklingError) as e:
                self.logger.debug(f"Error loading checkpoint: {e}")
    
    def run_dashboard(self, stdscr):
        """Run the dashboard with curses."""
        # Setup curses
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(1)   # Non-blocking input
        stdscr.timeout(100) # Refresh every 100ms
        
        # Colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)   # Good
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Warning
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)     # Critical
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)    # Info
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLACK)   # Normal
        
        # Start monitoring
        self.start_monitoring()
        
        try:
            while True:
                # Clear screen
                stdscr.clear()
                height, width = stdscr.getmaxyx()
                
                # Header
                self._draw_header(stdscr, width)
                
                # GPU sections
                gpu_y = 3
                for gpu_id in self.gpu_ids:
                    gpu_y = self._draw_gpu_section(stdscr, gpu_id, gpu_y, width)
                    gpu_y += 1
                
                # System section
                system_y = gpu_y
                system_y = self._draw_system_section(stdscr, system_y, width)
                
                # Processing stats
                if system_y < height - 5:
                    self._draw_processing_section(stdscr, system_y + 1, width)
                
                # Footer
                self._draw_footer(stdscr, height - 1, width)
                
                # Refresh
                stdscr.refresh()
                
                # Check for quit
                key = stdscr.getch()
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('s') or key == ord('S'):
                    # Save snapshot
                    self.save_metrics_snapshot()
                    stdscr.addstr(height - 2, 2, "Metrics snapshot saved!", curses.color_pair(1))
                    
        finally:
            self.stop_monitoring()
    
    def _draw_header(self, stdscr, width):
        """Draw dashboard header."""
        title = "GPU Processing Monitor Dashboard"
        stdscr.attron(curses.color_pair(4) | curses.A_BOLD)
        stdscr.addstr(0, (width - len(title)) // 2, title)
        stdscr.attroff(curses.color_pair(4) | curses.A_BOLD)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stdscr.addstr(1, width - len(timestamp) - 1, timestamp)
        
        # Separator
        stdscr.addstr(2, 0, "=" * (width - 1))
    
    def _draw_gpu_section(self, stdscr, gpu_id, start_y, width):
        """Draw GPU statistics section."""
        y = start_y
        
        # GPU header
        gpu_name = f"GPU {gpu_id}"
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            gpu_name += f" - {torch.cuda.get_device_name(gpu_id)}"
        
        stdscr.attron(curses.A_BOLD)
        stdscr.addstr(y, 2, gpu_name[:width-4])
        stdscr.attroff(curses.A_BOLD)
        y += 1
        
        # Memory bar
        if self.gpu_stats[gpu_id]['memory_history']:
            latest_mem = self.gpu_stats[gpu_id]['memory_history'][-1]
            mem_percent = latest_mem['percent']
            
            # Color based on usage
            if mem_percent > 90:
                color = curses.color_pair(3)  # Red
            elif mem_percent > 75:
                color = curses.color_pair(2)  # Yellow
            else:
                color = curses.color_pair(1)  # Green
            
            bar_width = min(40, width - 30)
            filled = int(bar_width * mem_percent / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            
            stdscr.addstr(y, 4, "Memory: ")
            stdscr.attron(color)
            stdscr.addstr(y, 12, bar)
            stdscr.attroff(color)
            stdscr.addstr(y, 12 + bar_width + 1, 
                         f"{latest_mem['allocated_gb']:.1f}/{latest_mem['total_gb']:.1f} GB "
                         f"({mem_percent:.1f}%)")
            y += 1
        
        # Utilization
        if self.gpu_stats[gpu_id]['util_history']:
            latest_util = self.gpu_stats[gpu_id]['util_history'][-1]
            stdscr.addstr(y, 4, f"Utilization: GPU {latest_util['gpu']}%, Memory {latest_util['memory']}%")
            y += 1
        
        # Temperature and Power
        temp_power_line = []
        if self.gpu_stats[gpu_id]['temp_history']:
            latest_temp = self.gpu_stats[gpu_id]['temp_history'][-1]
            temp_str = f"Temp: {latest_temp['temp']}°C"
            
            # Color code temperature
            if latest_temp['temp'] > 85:
                temp_str = f"Temp: {latest_temp['temp']}°C ⚠"
            
            temp_power_line.append(temp_str)
        
        if self.gpu_stats[gpu_id]['power_history']:
            latest_power = self.gpu_stats[gpu_id]['power_history'][-1]
            temp_power_line.append(f"Power: {latest_power['power']:.0f}W/{latest_power['max_power']:.0f}W")
        
        if temp_power_line:
            stdscr.addstr(y, 4, " | ".join(temp_power_line))
            y += 1
        
        # Throughput
        if self.gpu_stats[gpu_id]['throughput_history']:
            latest_throughput = self.gpu_stats[gpu_id]['throughput_history'][-1]
            stdscr.addstr(y, 4, f"Throughput: {latest_throughput['throughput']:.1f} docs/sec "
                                f"(Total: {latest_throughput['total']:,})")
            y += 1
        
        return y
    
    def _draw_system_section(self, stdscr, start_y, width):
        """Draw system statistics section."""
        y = start_y
        
        stdscr.attron(curses.A_BOLD)
        stdscr.addstr(y, 2, "System Resources")
        stdscr.attroff(curses.A_BOLD)
        y += 1
        
        # CPU
        if self.system_stats['cpu_history']:
            latest_cpu = self.system_stats['cpu_history'][-1]
            cpu_color = curses.color_pair(1)
            if latest_cpu['percent'] > 90:
                cpu_color = curses.color_pair(3)
            elif latest_cpu['percent'] > 80:
                cpu_color = curses.color_pair(2)
                
            stdscr.addstr(y, 4, "CPU: ")
            stdscr.attron(cpu_color)
            stdscr.addstr(f"{latest_cpu['percent']:.1f}%")
            stdscr.attroff(cpu_color)
            stdscr.addstr(f" ({psutil.cpu_count()} cores)")
            y += 1
        
        # Memory
        if self.system_stats['memory_history']:
            latest_mem = self.system_stats['memory_history'][-1]
            stdscr.addstr(y, 4, f"Memory: {latest_mem['used_gb']:.1f}/{latest_mem['total_gb']:.1f} GB "
                               f"({latest_mem['percent']:.1f}%)")
            y += 1
        
        return y
    
    def _draw_processing_section(self, stdscr, start_y, width):
        """Draw processing statistics section."""
        y = start_y
        
        stdscr.attron(curses.A_BOLD)
        stdscr.addstr(y, 2, "Processing Statistics")
        stdscr.attroff(curses.A_BOLD)
        y += 1
        
        # Overall stats
        elapsed = time.time() - self.processing_stats['start_time']
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        total = self.processing_stats['total_processed']
        failed = self.processing_stats['total_failed']
        rate = total / elapsed if elapsed > 0 else 0
        
        stdscr.addstr(y, 4, f"Elapsed: {elapsed_str} | "
                           f"Processed: {total:,} | "
                           f"Failed: {failed:,} | "
                           f"Rate: {rate:.1f} docs/sec")
        y += 1
        
        # Per-GPU stats from checkpoint
        checkpoint_data = self.processing_stats.get('checkpoint_data', {})
        gpu_stats = checkpoint_data.get('gpu_stats', {})
        
        if gpu_stats:
            for gpu_id in self.gpu_ids:
                if gpu_id in gpu_stats:
                    gpu_processed = gpu_stats[gpu_id].get('processed', 0)
                    gpu_errors = gpu_stats[gpu_id].get('errors', 0)
                    gpu_rate = gpu_processed / elapsed if elapsed > 0 else 0
                    
                    color = curses.color_pair(1) if gpu_errors == 0 else curses.color_pair(2)
                    stdscr.attron(color)
                    stdscr.addstr(y, 4, f"GPU {gpu_id}: {gpu_processed:,} processed, "
                                       f"{gpu_errors:,} errors, "
                                       f"{gpu_rate:.1f} docs/sec")
                    stdscr.attroff(color)
                    y += 1
        
        # Estimated completion
        if 'last_update' in checkpoint_data and total > 0:
            # This is a simplified estimate - you'd need total file count for accurate ETA
            eta_str = "Calculating..."
            stdscr.addstr(y, 4, f"ETA: {eta_str}")
            y += 1
    
    def _draw_footer(self, stdscr, y, width):
        """Draw dashboard footer."""
        footer = "Press 'q' to quit, 's' to save metrics snapshot"
        stdscr.addstr(y, (width - len(footer)) // 2, footer)
    
    def save_metrics_snapshot(self):
        """Save current metrics to file."""
        if not self.metrics_file:
            return
            
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'gpu_stats': {},
            'system_stats': {},
            'processing_stats': self.processing_stats
        }
        
        # Get latest GPU stats
        for gpu_id in self.gpu_ids:
            gpu_data = {}
            for stat_type in ['memory_history', 'util_history', 'temp_history', 
                            'power_history', 'throughput_history']:
                if self.gpu_stats[gpu_id][stat_type]:
                    gpu_data[stat_type.replace('_history', '')] = self.gpu_stats[gpu_id][stat_type][-1]
            snapshot['gpu_stats'][gpu_id] = gpu_data
        
        # Get latest system stats
        for stat_type in ['cpu_history', 'memory_history']:
            if self.system_stats[stat_type]:
                snapshot['system_stats'][stat_type.replace('_history', '')] = self.system_stats[stat_type][-1]
        
        # Append to file
        try:
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(snapshot) + '\n')
        except (IOError, OSError) as e:
            self.logger.error(f"Failed to save metrics snapshot: {e}")


def main():
    """Run the monitoring dashboard."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU Processing Monitor Dashboard")
    parser.add_argument('--checkpoint', type=str,
                       help='Path to checkpoint file')
    parser.add_argument('--metrics-file', type=str,
                       help='File to save metrics snapshots')
    parser.add_argument('--save-interval', type=int, default=60,
                       help='Interval in seconds to auto-save metrics')
    parser.add_argument('--gpu-ids', type=int, nargs='+',
                       help='GPU IDs to monitor (default: 0 1)')
    
    args = parser.parse_args()
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("Warning: No CUDA devices available")
    elif torch.cuda.device_count() < 2:
        print(f"Warning: This dashboard is designed for dual-GPU systems")
        print(f"Found {torch.cuda.device_count()} GPU(s)")
    
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create dashboard
    dashboard = GPUMonitorDashboard(
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        metrics_file=Path(args.metrics_file) if args.metrics_file else None,
        gpu_ids=args.gpu_ids
    )
    
    # Run with curses
    try:
        curses.wrapper(dashboard.run_dashboard)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"\nDashboard error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Save final metrics
        if args.metrics_file:
            dashboard.save_metrics_snapshot()
            print(f"\nFinal metrics saved to: {args.metrics_file}")


if __name__ == "__main__":
    main()