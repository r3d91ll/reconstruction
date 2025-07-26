#!/usr/bin/env python3
"""
Fixes and additions for the V3 pipeline
"""

import logging
import time
import threading
from typing import Dict, List
import torch

logger = logging.getLogger(__name__)


class GPUUtilizationMonitor(threading.Thread):
    """Monitor GPU utilization and adapt batch sizes"""
    
    def __init__(
        self,
        gpu_queue,
        output_queue,
        config,
        stop_event,
        batch_size_callback=None
    ):
        super().__init__(daemon=True)
        self.gpu_queue = gpu_queue
        self.output_queue = output_queue
        self.config = config
        self.stop_event = stop_event
        self.batch_size_callback = batch_size_callback
        
        # Tracking
        self.gpu_stats = {gpu_id: {'util_history': [], 'mem_history': []} 
                         for gpu_id in config.gpu_devices}
        self.low_util_counts = {gpu_id: 0 for gpu_id in config.gpu_devices}
        
    def run(self):
        """Monitor GPU metrics and queue depths"""
        logger.info("GPU Utilization Monitor started")
        
        # Try to import pynvml for GPU monitoring
        has_nvml = False
        try:
            import pynvml
            pynvml.nvmlInit()
            has_nvml = True
            logger.info("NVML initialized for GPU monitoring")
        except:
            logger.warning("NVML not available, GPU monitoring limited")
            
        while not self.stop_event.is_set():
            try:
                # Monitor queue depths
                gpu_queue_size = self.gpu_queue.qsize()
                output_queue_size = self.output_queue.qsize()
                
                # Log queue status
                queue_percent = (gpu_queue_size / self.config.max_gpu_queue_size * 100 
                               if self.config.max_gpu_queue_size > 0 else 0)
                
                if queue_percent > 80:
                    logger.warning(f"GPU queue near capacity: {gpu_queue_size}/{self.config.max_gpu_queue_size}")
                elif queue_percent < 20 and gpu_queue_size < 10:
                    logger.info("GPU queue running low - preprocessing may be too slow")
                
                # Monitor GPU stats if available
                if has_nvml:
                    for gpu_id in self.config.gpu_devices:
                        stats = self._get_gpu_stats(pynvml, gpu_id)
                        if stats:
                            self._process_gpu_stats(gpu_id, stats)
                            
                # Summary log
                self._log_summary(gpu_queue_size, output_queue_size)
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                
            # Wait for next interval
            self.stop_event.wait(self.config.monitor_interval)
            
        # Cleanup
        if has_nvml:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
                
        self._log_final_stats()
        
    def _get_gpu_stats(self, pynvml, gpu_id: int) -> Dict:
        """Get GPU statistics"""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            
            # Get utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # Power if available
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
            except:
                power = 0
                
            return {
                'utilization': util.gpu,
                'memory_used_gb': mem_info.used / 1024**3,
                'memory_total_gb': mem_info.total / 1024**3,
                'memory_percent': (mem_info.used / mem_info.total) * 100,
                'temperature': temp,
                'power_watts': power
            }
            
        except Exception as e:
            logger.debug(f"Failed to get GPU {gpu_id} stats: {e}")
            return None
            
    def _process_gpu_stats(self, gpu_id: int, stats: Dict):
        """Process GPU statistics and trigger adaptations"""
        # Update history
        self.gpu_stats[gpu_id]['util_history'].append(stats['utilization'])
        self.gpu_stats[gpu_id]['mem_history'].append(stats['memory_percent'])
        
        # Keep only recent history
        max_history = 20
        if len(self.gpu_stats[gpu_id]['util_history']) > max_history:
            self.gpu_stats[gpu_id]['util_history'].pop(0)
            self.gpu_stats[gpu_id]['mem_history'].pop(0)
            
        # Check for low utilization
        if stats['utilization'] < self.config.low_util_threshold:
            self.low_util_counts[gpu_id] += 1
            if self.low_util_counts[gpu_id] >= self.config.low_util_alert_threshold:
                logger.warning(
                    f"GPU {gpu_id} consistently underutilized: {stats['utilization']}% "
                    f"(Memory: {stats['memory_percent']:.1f}%)"
                )
        else:
            self.low_util_counts[gpu_id] = 0
            
        # Adaptive batch sizing based on memory
        if self.batch_size_callback and len(self.gpu_stats[gpu_id]['mem_history']) >= 5:
            avg_memory = sum(self.gpu_stats[gpu_id]['mem_history'][-5:]) / 5
            
            if avg_memory > 85:
                logger.info(f"GPU {gpu_id} memory pressure ({avg_memory:.1f}%), suggesting smaller batches")
                self.batch_size_callback(gpu_id, 'decrease')
            elif avg_memory < 50 and stats['utilization'] > 80:
                logger.info(f"GPU {gpu_id} has memory headroom ({avg_memory:.1f}%), suggesting larger batches")
                self.batch_size_callback(gpu_id, 'increase')
                
        # Temperature warning
        if stats['temperature'] > 80:
            logger.warning(f"GPU {gpu_id} running hot: {stats['temperature']}°C")
            
    def _log_summary(self, gpu_queue_size: int, output_queue_size: int):
        """Log summary statistics"""
        summary_parts = [
            f"Queues - GPU: {gpu_queue_size}, Output: {output_queue_size}"
        ]
        
        # Add GPU stats if available
        for gpu_id, stats in self.gpu_stats.items():
            if stats['util_history']:
                recent_util = stats['util_history'][-1]
                recent_mem = stats['mem_history'][-1]
                summary_parts.append(f"GPU{gpu_id}: {recent_util}% util, {recent_mem:.1f}% mem")
                
        logger.info(" | ".join(summary_parts))
        
    def _log_final_stats(self):
        """Log final statistics summary"""
        logger.info("GPU Utilization Summary:")
        
        for gpu_id, stats in self.gpu_stats.items():
            if stats['util_history']:
                avg_util = sum(stats['util_history']) / len(stats['util_history'])
                avg_mem = sum(stats['mem_history']) / len(stats['mem_history'])
                max_util = max(stats['util_history'])
                max_mem = max(stats['mem_history'])
                
                logger.info(
                    f"  GPU {gpu_id} - Avg: {avg_util:.1f}% util, {avg_mem:.1f}% mem | "
                    f"Peak: {max_util}% util, {max_mem:.1f}% mem"
                )


class AdaptiveBatchSizeManager:
    """Manage batch sizes adaptively per GPU"""
    
    def __init__(self, initial_batch_size: int, min_size: int = 16, max_size: int = 512):
        self.gpu_batch_sizes = {}
        self.initial_size = initial_batch_size
        self.min_size = min_size
        self.max_size = max_size
        self.lock = threading.Lock()
        
    def get_batch_size(self, gpu_id: int) -> int:
        """Get current batch size for GPU"""
        with self.lock:
            return self.gpu_batch_sizes.get(gpu_id, self.initial_size)
            
    def adjust_batch_size(self, gpu_id: int, direction: str):
        """Adjust batch size for a GPU"""
        with self.lock:
            current = self.gpu_batch_sizes.get(gpu_id, self.initial_size)
            
            if direction == 'decrease':
                new_size = max(self.min_size, int(current * 0.8))
            elif direction == 'increase':
                new_size = min(self.max_size, int(current * 1.2))
            else:
                return current
                
            if new_size != current:
                self.gpu_batch_sizes[gpu_id] = new_size
                logger.info(f"Adjusted GPU {gpu_id} batch size: {current} -> {new_size}")
                
            return new_size


def add_enhanced_health_monitor_methods(health_monitor_class):
    """Add missing methods to HealthMonitor class"""
    
    def _restart_preprocessing_worker(self, index: int):
        """Restart a dead preprocessing worker"""
        try:
            from process_abstracts_continuous_gpu_v3 import PreprocessingWorker
            
            dead_worker = self.pipeline.preprocessing_workers[index]
            
            # Create new worker with same config
            new_worker = PreprocessingWorker(
                document_queue=self.pipeline.document_queue,
                gpu_queue=self.pipeline.gpu_queue,
                config=self.pipeline.config,
                stop_event=self.pipeline.preprocessing_stop,
                worker_id=dead_worker.worker_id
            )
            
            new_worker.start()
            self.pipeline.preprocessing_workers[index] = new_worker
            logger.info(f"Successfully restarted preprocessing worker {dead_worker.worker_id}")
            
        except Exception as e:
            logger.error(f"Failed to restart preprocessing worker: {e}")
            
    def _restart_db_writer(self):
        """Restart database writer if it dies"""
        try:
            from process_abstracts_continuous_gpu_v3 import RobustDatabaseWriter
            
            # Create new writer
            new_writer = RobustDatabaseWriter(
                config=self.pipeline.config,
                write_queue=self.pipeline.db_write_queue,
                stop_event=self.pipeline.output_stop,
                checkpoint_manager=self.pipeline.checkpoint_manager
            )
            
            new_writer.start()
            self.pipeline.db_writer = new_writer
            logger.info("Successfully restarted database writer")
            
        except Exception as e:
            logger.error(f"Failed to restart database writer: {e}")
            
    # Add these methods to the HealthMonitor class
    health_monitor_class._restart_preprocessing_worker = _restart_preprocessing_worker
    health_monitor_class._restart_db_writer = _restart_db_writer


# Fixed warmup code for GPUWorker
def fixed_gpu_warmup(self, model):
    """Fixed warmup without underscore assignment"""
    try:
        dummy_batch = ["warmup text"] * min(10, self.config.batch_size)
        warmup_result = model.encode_batch(dummy_batch, batch_size=len(dummy_batch))
        torch.cuda.synchronize(device=self.gpu_id)
        torch.cuda.empty_cache()
        logger.debug(f"GPU {self.gpu_id} warmed up successfully")
    except Exception as e:
        logger.warning(f"GPU {self.gpu_id} warmup failed: {e}")


# Backpressure implementation for preprocessing
def add_backpressure_to_preprocessing(worker_class):
    """Add backpressure when GPU queue is full"""
    
    original_send_batch = worker_class._send_batch
    
    def _send_batch_with_backpressure(self, texts, metadata, batch_num):
        """Send batch with backpressure control"""
        if not texts:
            return
            
        # Check queue size and apply backpressure
        max_attempts = 100
        for attempt in range(max_attempts):
            try:
                queue_size = self.gpu_queue.qsize()
                queue_percent = (queue_size / self.config.max_gpu_queue_size * 100 
                               if self.config.max_gpu_queue_size > 0 else 0)
                
                if queue_percent > 90:
                    if attempt == 0:
                        logger.debug(f"GPU queue at {queue_percent:.1f}% capacity, applying backpressure")
                    time.sleep(0.1 * (attempt + 1))  # Progressive backoff
                    continue
                    
                # Call original method
                return original_send_batch(self, texts, metadata, batch_num)
                
            except Exception as e:
                logger.error(f"Error checking queue size: {e}")
                break
                
        # If we couldn't send after all attempts, still try
        return original_send_batch(self, texts, metadata, batch_num)
        
    worker_class._send_batch = _send_batch_with_backpressure


# Example integration code
"""
# In your main pipeline, integrate these fixes:

# 1. Add GPU monitor with batch size adaptation
self.batch_size_manager = AdaptiveBatchSizeManager(config.batch_size)
self.gpu_monitor = GPUUtilizationMonitor(
    gpu_queue=self.gpu_queue,
    output_queue=self.output_queue,
    config=self.config,
    stop_event=self.stop_event,
    batch_size_callback=self.batch_size_manager.adjust_batch_size
)
self.gpu_monitor.start()

# 2. Fix the health monitor
add_enhanced_health_monitor_methods(HealthMonitor)

# 3. Fix GPU warmup
GPUWorker._warmup_gpu = fixed_gpu_warmup

# 4. Add backpressure
add_backpressure_to_preprocessing(PreprocessingWorker)
"""