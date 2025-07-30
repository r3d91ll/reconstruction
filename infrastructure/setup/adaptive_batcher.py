"""
Adaptive Batch Size Management for Pipeline Optimization
"""

import time
import logging
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class BatchMetrics:
    """Metrics for a single batch"""
    size: int
    duration: float
    throughput: float
    timestamp: float

class AdaptiveBatcher:
    """
    Dynamically adjusts batch size based on performance metrics.
    
    The batcher monitors:
    - Processing time per batch
    - Throughput (docs/sec)
    - Queue depths
    
    And adjusts batch size to optimize for:
    - Target processing time per batch
    - Maximum sustainable throughput
    """
    
    def __init__(
        self, 
        initial_size: int = 1000,
        min_size: int = 100,
        max_size: int = 5000,
        target_time: float = 10.0,
        history_size: int = 20,
        min_adjustment_interval: float = 30.0,
        bucket_size: int = 500,
        adjustment_step: int = 200
    ):
        """
        Initialize adaptive batcher.
        
        Args:
            initial_size: Starting batch size
            min_size: Minimum allowed batch size
            max_size: Maximum allowed batch size
            target_time: Target seconds per batch
            history_size: Number of batches to track for analysis
            min_adjustment_interval: Minimum seconds between batch size adjustments
            bucket_size: Size for bucketing batch sizes when analyzing throughput
            adjustment_step: Step size for gradual batch size adjustments
        """
        # Validate parameters
        if min_size <= 0:
            raise ValueError("min_size must be positive")
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if initial_size <= 0:
            raise ValueError("initial_size must be positive")
        if min_size > max_size:
            raise ValueError("min_size cannot be greater than max_size")
        if initial_size < min_size or initial_size > max_size:
            raise ValueError("initial_size must be between min_size and max_size")
        if target_time <= 0:
            raise ValueError("target_time must be positive")
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        
        self.batch_size = initial_size
        self.min_size = min_size
        self.max_size = max_size
        self.target_time = target_time
        self.min_adjustment_interval = min_adjustment_interval
        self.bucket_size = bucket_size
        self.adjustment_step = adjustment_step
        
        # Performance tracking
        self.history = deque(maxlen=history_size)
        self.adjustment_cooldown = 0
        self.last_adjustment_time = 0
        
        # Thresholds
        self.time_tolerance = 0.2  # 20% tolerance on target time
        self.throughput_window = 5  # batches to analyze for throughput
        
        logger.info(
            f"AdaptiveBatcher initialized: size={initial_size}, "
            f"range=[{min_size}, {max_size}], target_time={target_time}s"
        )
        
    def record_batch(self, size: int, duration: float) -> None:
        """Record metrics for a completed batch"""
        throughput = size / duration if duration > 0 else 0
        
        metrics = BatchMetrics(
            size=size,
            duration=duration,
            throughput=throughput,
            timestamp=time.time()
        )
        
        self.history.append(metrics)
        
        # Reduce cooldown
        if self.adjustment_cooldown > 0:
            self.adjustment_cooldown -= 1
            
    def should_adjust(self) -> bool:
        """Check if batch size should be adjusted"""
        if len(self.history) < 3:
            return False
            
        if self.adjustment_cooldown > 0:
            return False
            
        # Check if enough time has passed since last adjustment
        if time.time() - self.last_adjustment_time < self.min_adjustment_interval:
            return False
            
        return True
        
    def get_optimal_size(self) -> int:
        """Calculate optimal batch size based on recent performance"""
        if not self.should_adjust():
            return self.batch_size
            
        # Analyze recent batches
        recent = list(self.history)[-self.throughput_window:]
        avg_duration = sum(b.duration for b in recent) / len(recent)
        avg_throughput = sum(b.throughput for b in recent) / len(recent)
        
        # Check if we're meeting target time
        if abs(avg_duration - self.target_time) <= self.target_time * self.time_tolerance:
            # We're within tolerance, optimize for throughput
            return self._optimize_for_throughput()
        elif avg_duration > self.target_time:
            # Taking too long, reduce batch size
            return self._reduce_batch_size(avg_duration)
        else:
            # Going too fast, increase batch size
            return self._increase_batch_size(avg_duration)
            
    def _optimize_for_throughput(self) -> int:
        """Optimize batch size for maximum throughput"""
        # Look for throughput trends
        if len(self.history) < 10:
            return self.batch_size
            
        # Compare throughput at different batch sizes
        size_groups = {}
        for batch in self.history:
            size_bucket = (batch.size // self.bucket_size) * self.bucket_size  # Round to nearest bucket_size
            if size_bucket not in size_groups:
                size_groups[size_bucket] = []
            size_groups[size_bucket].append(batch.throughput)
            
        # Find size with best average throughput
        best_size = self.batch_size
        best_throughput = 0
        
        for size, throughputs in size_groups.items():
            if len(throughputs) >= 2:  # Need at least 2 samples
                avg_throughput = sum(throughputs) / len(throughputs)
                if avg_throughput > best_throughput:
                    best_throughput = avg_throughput
                    best_size = size
                    
        # Make conservative adjustment toward best size
        if best_size > self.batch_size:
            new_size = min(self.batch_size + self.adjustment_step, best_size)
        elif best_size < self.batch_size:
            new_size = max(self.batch_size - self.adjustment_step, best_size)
        else:
            new_size = self.batch_size
            
        return self._apply_adjustment(new_size)
        
    def _reduce_batch_size(self, current_duration: float) -> int:
        """Reduce batch size when processing is too slow"""
        # Calculate reduction factor
        ratio = self.target_time / current_duration
        new_size = int(self.batch_size * ratio * 0.9)  # Conservative 90%
        
        return self._apply_adjustment(new_size)
        
    def _increase_batch_size(self, current_duration: float) -> int:
        """Increase batch size when processing is too fast"""
        # Calculate increase factor
        ratio = self.target_time / current_duration
        new_size = int(self.batch_size * ratio * 0.9)  # Conservative 90%
        
        return self._apply_adjustment(new_size)
        
    def _apply_adjustment(self, new_size: int) -> int:
        """Apply adjustment with bounds checking and logging"""
        old_size = self.batch_size
        
        # Apply bounds
        new_size = max(self.min_size, min(self.max_size, new_size))
        
        # Only adjust if change is significant (>10%)
        if abs(new_size - old_size) / old_size < 0.1:
            return self.batch_size
            
        self.batch_size = new_size
        self.adjustment_cooldown = 3  # Wait 3 batches before next adjustment
        self.last_adjustment_time = time.time()
        
        logger.info(f"Adjusted batch size: {old_size} -> {new_size}")
        
        return new_size
        
    def get_stats(self) -> Dict:
        """Get current performance statistics"""
        if not self.history:
            return {
                'current_batch_size': self.batch_size,
                'avg_duration': 0,
                'avg_throughput': 0,
                'total_batches': 0
            }
            
        recent = list(self.history)[-10:]  # Last 10 batches
        
        return {
            'current_batch_size': self.batch_size,
            'avg_duration': sum(b.duration for b in recent) / len(recent),
            'avg_throughput': sum(b.throughput for b in recent) / len(recent),
            'total_batches': len(self.history),
            'min_duration': min(b.duration for b in recent),
            'max_duration': max(b.duration for b in recent),
            'adjustment_cooldown': self.adjustment_cooldown
        }

class QueueAwareBatcher(AdaptiveBatcher):
    """
    Extended adaptive batcher that considers queue depths.
    
    Adjusts batch size based on:
    - Processing performance (inherited)
    - Input queue depth
    - Output queue availability
    """
    
    def __init__(self, *args, 
                 output_util_threshold: float = 0.8,
                 input_backlog_threshold: int = 1000,
                 low_output_util_threshold: float = 0.5,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_history = deque(maxlen=10)
        
        # Configurable thresholds
        self.output_util_threshold = output_util_threshold
        self.input_backlog_threshold = input_backlog_threshold
        self.low_output_util_threshold = low_output_util_threshold
        
    def record_queue_state(
        self, 
        input_depth: int, 
        output_depth: int,
        max_output: int
    ) -> None:
        """Record current queue states"""
        self.queue_history.append({
            'timestamp': time.time(),
            'input_depth': input_depth,
            'output_depth': output_depth,
            'output_utilization': output_depth / max_output if max_output > 0 else 0
        })
        
    def get_optimal_size(self) -> int:
        """Calculate optimal size considering queue pressure"""
        base_size = super().get_optimal_size()
        
        if not self.queue_history:
            return base_size
            
        # Check queue pressure
        recent_queues = list(self.queue_history)[-5:]
        avg_input = sum(q['input_depth'] for q in recent_queues) / len(recent_queues)
        avg_output_util = sum(q['output_utilization'] for q in recent_queues) / len(recent_queues)
        
        # Adjust based on queue state
        if avg_output_util > self.output_util_threshold:
            # Output queue is getting full, reduce batch size
            adjusted_size = int(base_size * 0.8)
            logger.debug(f"High output queue pressure, reducing size: {base_size} -> {adjusted_size}")
        elif avg_input > self.input_backlog_threshold and avg_output_util < self.low_output_util_threshold:
            # Large input backlog and output has capacity
            adjusted_size = int(base_size * 1.2)
            logger.debug(f"Input backlog with output capacity, increasing size: {base_size} -> {adjusted_size}")
        else:
            adjusted_size = base_size
            
        return max(self.min_size, min(self.max_size, adjusted_size))

def create_batcher(config: Dict) -> AdaptiveBatcher:
    """Factory function to create appropriate batcher"""
    batcher_type = config.get('batcher_type', 'adaptive')
    
    if batcher_type == 'queue_aware':
        return QueueAwareBatcher(
            initial_size=config.get('batch_size', 1000),
            min_size=config.get('min_batch_size', 100),
            max_size=config.get('max_batch_size', 5000),
            target_time=config.get('target_batch_time', 10.0)
        )
    else:
        return AdaptiveBatcher(
            initial_size=config.get('batch_size', 1000),
            min_size=config.get('min_batch_size', 100),
            max_size=config.get('max_batch_size', 5000),
            target_time=config.get('target_batch_time', 10.0)
        )