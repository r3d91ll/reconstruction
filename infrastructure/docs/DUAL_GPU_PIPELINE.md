# Dual-GPU Processing Pipeline Documentation

## Overview

The dual-GPU processing pipeline leverages two RTX A6000 GPUs with NVLink for high-throughput document processing and embedding generation. The system includes both an enhanced dual-GPU pipeline and a production-ready variant with advanced features.

## Architecture

### Core Components

1. **Pipeline Launcher** (`launch_dual_gpu_pipeline.py`)
   - Manages the lifecycle of the processing pipeline and monitoring dashboard
   - Handles graceful shutdown and process management
   - Provides unified entry point for both standard and production pipelines

2. **Enhanced Dual-GPU Pipeline** (`process_abstracts_dual_gpu.py`)
   - NVLink-aware memory management for optimal GPU communication
   - Dynamic load balancing between GPUs
   - Robust error recovery and checkpoint/resume capability
   - Real-time GPU health monitoring

3. **Production Pipeline** (`process_abstracts_production.py`)
   - All features of enhanced pipeline plus:
   - Advanced memory management with late chunking awareness
   - Smart document batching based on size and priority
   - Predictive load balancing using performance history
   - Enhanced checkpointing with validation and backup

4. **GPU Monitor Dashboard** (`gpu_monitor_dashboard.py`)
   - Real-time visualization of GPU metrics
   - Processing progress tracking
   - Performance statistics and efficiency metrics

## Key Features

### NVLink Optimization
- Automatic detection and enablement of NVLink between GPUs
- Peer memory access for improved data transfer
- Optimized batch distribution based on NVLink availability

### Memory Management
- **Dynamic Batch Sizing**: Automatically adjusts batch sizes based on GPU memory usage
- **Memory Pressure Detection**: Reduces batch size when memory exceeds 80%
- **Forced Cleanup**: Periodic memory cleanup to prevent fragmentation
- **OOM Recovery**: Graceful handling of out-of-memory errors with automatic retry

### Load Balancing
- **Performance-Based Routing**: Routes work to faster GPU based on recent throughput
- **Queue-Aware Scheduling**: Considers current GPU workload
- **Dynamic Rebalancing**: Adjusts distribution based on processing speed

### Checkpoint System
- **Atomic Operations**: Ensures data integrity during saves
- **Resume Capability**: Continue processing from last checkpoint
- **Failed File Tracking**: Retry failed documents up to 3 times
- **Performance History**: Tracks processing metrics for optimization

### Error Handling
- **Automatic Retry**: Configurable retry attempts with exponential backoff
- **Health Monitoring**: Continuous GPU health checks
- **Graceful Degradation**: Continue with single GPU if one fails
- **Comprehensive Logging**: Detailed error tracking and diagnostics

## Usage

### Basic Usage
```bash
# Standard dual-GPU processing
python launch_dual_gpu_pipeline.py \
    --db-name arxiv_abstracts_enhanced \
    --metadata-dir /mnt/data/arxiv_data/metadata \
    --batch-size 200 \
    --clean-start

# Resume from checkpoint
python launch_dual_gpu_pipeline.py \
    --db-name arxiv_abstracts_enhanced \
    --metadata-dir /mnt/data/arxiv_data/metadata \
    --batch-size 200 \
    --resume \
    --checkpoint-dir ./checkpoints/arxiv_abstracts_enhanced
```

### Production Pipeline
```bash
# Use production pipeline with advanced features
python launch_dual_gpu_pipeline.py \
    --db-name arxiv_abstracts_production \
    --metadata-dir /mnt/data/arxiv_data/metadata \
    --batch-size 200 \
    --production \
    --checkpoint-dir ./checkpoints/production
```

### Process Specific Number of Documents
```bash
# Process first 10,000 documents
python launch_dual_gpu_pipeline.py \
    --db-name arxiv_abstracts_test \
    --metadata-dir /mnt/data/arxiv_data/metadata \
    --count 10000 \
    --batch-size 200
```

## Configuration

### Environment Variables
```bash
# Database connection
ARANGO_HOST=localhost
ARANGO_PORT=8529
ARANGO_USERNAME=root
ARANGO_PASSWORD=your_password

# GPU settings (optional)
CUDA_VISIBLE_DEVICES=0,1  # Specify which GPUs to use
```

### Command Line Arguments
- `--metadata-dir`: Directory containing arXiv metadata JSON files
- `--count`: Number of documents to process (default: all)
- `--db-name`: Target database name
- `--db-host`: Database host (default: localhost)
- `--clean-start`: Drop existing database and start fresh
- `--batch-size`: Base batch size per GPU (default: 200)
- `--checkpoint-dir`: Directory for checkpoint files
- `--resume`: Resume from previous checkpoint
- `--production`: Use production pipeline with advanced features

## Performance Characteristics

### Processing Speed
- **Combined Throughput**: 400-600 documents/second (both GPUs)
- **Per-GPU Rate**: 200-300 documents/second
- **Parallel Efficiency**: 85-95% with NVLink

### Memory Usage
- **Base Model Memory**: ~4.5 GB per GPU (Jina v4)
- **Processing Overhead**: 2-4 GB depending on batch size
- **Recommended Free Memory**: 8+ GB per GPU

### Batch Size Guidelines
- **Small Documents** (< 1000 chars): 200-400 per batch
- **Medium Documents** (1000-5000 chars): 100-200 per batch
- **Large Documents** (> 5000 chars): 50-100 per batch

## Database Schema

The pipeline creates a single collection `abstract_metadata` with the following structure:

```json
{
  "_key": "arxiv_id",
  "arxiv_id": "2310.08560",
  "title": "Paper Title",
  "authors": ["Author 1", "Author 2"],
  "abstract": "Full abstract text...",
  "categories": ["cs.AI", "cs.LG"],
  "published": "2023-10-12T00:00:00",
  "updated": "2023-10-15T00:00:00",
  "doi": "10.xxxx/xxxxx",
  "journal_ref": "Journal reference if available",
  "pdf_url": "https://arxiv.org/pdf/2310.08560.pdf",
  "abs_url": "https://arxiv.org/abs/2310.08560",
  "abstract_embedding": [0.123, 0.456, ...],  // 2048-dimensional
  "processed_gpu": 0,  // Which GPU processed this document
  "encode_time": 0.0234,  // Encoding time in seconds
  "processed_at": "2024-01-15T10:30:00"
}
```

### Indexes
- Hash index on `arxiv_id` (unique)
- Persistent index on `categories[*]` (array)
- Persistent index on `published`
- Persistent index on `processed_gpu`
- Fulltext index on `title`
- Fulltext index on `abstract`

## Monitoring

### Real-time Dashboard
The monitoring dashboard provides:
- GPU memory usage and temperature
- Processing rate per GPU
- Queue depths and load balance
- Error counts and recovery status
- Overall progress and ETA

### Log Files
- `enhanced_dual_gpu_pipeline.log`: Main pipeline logs
- `production_pipeline.log`: Production pipeline logs (if using)
- `logs/pipeline_YYYYMMDD_HHMMSS.log`: Timestamped session logs

### Checkpoint Information
Checkpoints contain:
- Processed file set
- Failed files with error details
- GPU statistics and performance metrics
- Timestamp of last update

## Troubleshooting

### Common Issues

1. **GPU Not Found**
   ```
   ERROR: This pipeline requires 2 GPUs
   ```
   - Ensure both GPUs are properly installed
   - Check CUDA_VISIBLE_DEVICES environment variable
   - Verify with `nvidia-smi`

2. **Database Connection Failed**
   ```
   Database connection failed after 3 attempts
   ```
   - Verify ArangoDB is running
   - Check ARANGO_PASSWORD environment variable
   - Test connection with ArangoDB web interface

3. **Out of Memory Errors**
   - Reduce `--batch-size` parameter
   - Check for other GPU-using processes
   - Monitor with `nvidia-smi` during processing

4. **Imbalanced GPU Usage**
   - Check NVLink status in logs
   - Verify both GPUs are healthy
   - Review load balancer statistics

### Recovery Procedures

1. **Resume After Crash**
   ```bash
   python launch_dual_gpu_pipeline.py \
       --db-name arxiv_abstracts_enhanced \
       --resume \
       --checkpoint-dir ./checkpoints/arxiv_abstracts_enhanced
   ```

2. **Force Restart**
   ```bash
   # Remove checkpoint if corrupted
   rm ./checkpoints/arxiv_abstracts_enhanced/checkpoint.pkl
   
   # Start with clean database
   python launch_dual_gpu_pipeline.py \
       --db-name arxiv_abstracts_enhanced \
       --clean-start
   ```

3. **Single GPU Fallback**
   ```bash
   # If one GPU fails, use single GPU pipeline
   CUDA_VISIBLE_DEVICES=0 python setup/process_abstracts_only.py \
       --metadata-dir /mnt/data/arxiv_data/metadata \
       --db-name arxiv_abstracts_single
   ```

## Performance Optimization

### Tips for Maximum Throughput

1. **Batch Size Tuning**
   - Start with default (200) and monitor memory
   - Increase if memory usage < 70%
   - Decrease if seeing OOM errors

2. **NVLink Utilization**
   - Ensure NVLink is detected in logs
   - Keep GPUs physically adjacent if possible
   - Use matching GPU models

3. **Database Optimization**
   - Place database on SSD for best performance
   - Ensure sufficient database memory
   - Monitor database response times

4. **System Configuration**
   - Disable GPU power limiting
   - Ensure adequate cooling
   - Use performance CPU governor

### Benchmarking
```bash
# Benchmark with 10,000 documents
python launch_dual_gpu_pipeline.py \
    --db-name benchmark_test \
    --count 10000 \
    --batch-size 200 \
    --clean-start

# Results will show:
# - Combined throughput
# - Per-GPU rates
# - Parallel efficiency
# - Memory usage statistics
```

## Advanced Features (Production Pipeline)

### Smart Batching
- Groups documents by size for optimal memory usage
- Supports document priorities
- Automatically splits oversized batches

### Predictive Load Balancing
- Learns GPU performance characteristics
- Routes batches to optimal GPU
- Considers memory pressure and queue depth

### Enhanced Error Recovery
- Pattern detection for systematic errors
- Automatic retry with backoff
- Detailed error diagnostics

### Memory Prediction
- Estimates memory requirements before processing
- Accounts for attention mechanism scaling
- Prevents OOM through proactive management

## Future Enhancements

Planned improvements include:
- Multi-node distributed processing
- Support for additional embedding models
- Real-time embedding similarity computation
- Advanced caching mechanisms
- Automatic performance tuning