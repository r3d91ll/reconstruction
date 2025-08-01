# Production Pipeline Features

## Overview

The production pipeline (`process_abstracts_production.py`) includes several advanced features for reliability and performance:

### 1. Advanced Memory Management
- **Late Chunking Memory Estimation**: Accurately estimates memory requirements considering quadratic attention scaling
- **Dynamic Batch Adjustment**: Automatically reduces batch sizes under memory pressure
- **Predictive Allocation**: Learns from actual memory usage to improve estimates

### 2. Smart Batching
- **Document-aware Batching**: Groups documents optimally based on size and memory constraints
- **Priority Support**: Processes high-priority documents first
- **Automatic Rebalancing**: Splits batches that are too large for available memory

### 3. Enhanced Checkpointing
- **Atomic Writes**: Ensures checkpoint integrity even during crashes
- **Backup System**: Maintains backup checkpoints for recovery
- **Validation**: Checksums and structure validation on load
- **Performance History**: Tracks processing metrics for optimization

### 4. Predictive Load Balancing
- **Performance-based Selection**: Routes batches to faster GPUs
- **Queue-aware Scheduling**: Considers current GPU load
- **Memory-aware Routing**: Avoids overloaded GPUs

### 5. Health Monitoring
- **Real-time Health Reports**: Monitors GPU health every 30 seconds
- **Error Pattern Tracking**: Identifies recurring issues
- **Automatic Recovery**: Attempts to recover from transient errors

## Usage

### Basic Usage
```bash
# Standard dual-GPU pipeline
python launch_dual_gpu_pipeline.py \
    --db-name arxiv_abstracts_enhanced \
    --metadata-dir /mnt/data/arxiv_data/metadata \
    --batch-size 200

# Production pipeline with advanced features
python launch_dual_gpu_pipeline.py \
    --db-name arxiv_abstracts_enhanced \
    --metadata-dir /mnt/data/arxiv_data/metadata \
    --batch-size 200 \
    --production
```

### Testing Features
```bash
# Test production features
python test_production_features.py
```

### Direct Production Pipeline
```bash
python setup/process_abstracts_production.py \
    --metadata-dir /mnt/data/arxiv_data/metadata \
    --db-name arxiv_abstracts_production \
    --db-host 192.168.1.69 \
    --checkpoint-dir ./checkpoints/production
```

## Configuration

### Memory Settings
The memory manager uses these parameters:
- Base model memory: 4.5 GB (Jina v4)
- Attention scaling: O(n²) where n is max sequence length
- Safety margin: 15%
- Critical threshold: 90%

### Batch Size Optimization
The smart batcher optimizes for:
- Target GPU memory usage: 75%
- Minimum batch size: 1 document
- Maximum batch size: 256 documents

### Load Balancing
The predictive balancer considers:
- Recent throughput (last 100 batches)
- Current memory usage
- Queue depth

## Monitoring

The production pipeline provides detailed monitoring:

1. **GPU Metrics**
   - Memory usage and allocation
   - Processing throughput
   - Error rates
   - Temperature and utilization

2. **Performance Metrics**
   - Documents per second per GPU
   - Batch processing times
   - Memory efficiency
   - Load balance ratio

3. **Error Tracking**
   - Failed documents with reasons
   - Error patterns and frequencies
   - Recovery attempts

## Checkpoint Format

The enhanced checkpoint includes:
```python
{
    'processed_files': set(),  # Successfully processed
    'failed_files': {},        # Failed with error info
    'gpu_stats': {},          # Per-GPU statistics
    'metadata': {
        'version': '2.0',
        'checksum': 'sha256...',
        'last_update': datetime
    },
    'performance_history': {}, # Historical performance
    'error_patterns': {}       # Error frequency tracking
}
```

## Best Practices

1. **Memory Management**
   - Start with conservative batch sizes (100-200)
   - Monitor memory usage in dashboard
   - Let the system auto-adjust batch sizes

2. **Error Recovery**
   - Check error patterns in checkpoint
   - Failed files are retried up to 3 times
   - Use `--resume` to continue from failures

3. **Performance Optimization**
   - Use `--production` for large datasets
   - Monitor GPU balance in dashboard
   - Check checkpoint for performance history

## Troubleshooting

### High Memory Usage
- Reduce `--batch-size`
- Check for unusually long documents
- Monitor with `gpu_monitor_dashboard.py`

### Imbalanced GPUs
- Check NVLink status in logs
- Verify both GPUs are healthy
- Review performance history in checkpoint

### Processing Errors
- Check `production_pipeline.log`
- Review error patterns in checkpoint
- Verify Jina model initialization

## Future Enhancements

Planned improvements:
- Adaptive tokenizer for better estimates
- Multi-node support
- Dynamic model switching
- Advanced error recovery strategies