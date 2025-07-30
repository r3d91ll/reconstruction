# Single Collection Pipeline - Morning Test Guide

## Overview
This implementation consolidates the 3-collection approach into a single collection to improve write performance by 40-65%.

**Target Performance**: 25-30 documents/second (up from baseline 18 docs/sec)

## Files Created

1. **process_abstracts_single_collection.py** - Main pipeline implementation
   - Single collection schema with all fields
   - ValidationMetrics for performance tracking
   - ErrorRecovery with exponential backoff
   - LMDB checkpoint system with file fallback
   - Resource monitoring

2. **adaptive_batcher.py** - Dynamic batch size optimization
   - Adjusts batch size based on performance
   - Queue-aware variant available

3. **test_single_collection_1k.py** - 1000 document test script
   - Runs proof of concept test
   - Validates performance metrics

4. **arangodb_optimization.conf** - Database tuning settings
   - RocksDB optimizations
   - Write buffer configuration

## Morning Test Procedure

### 1. Ensure Environment is Ready
```bash
# Check ArangoDB is running
sudo systemctl status arangodb3

# Ensure password is set
export ARANGO_PASSWORD='your_password'

# Navigate to setup directory
cd /home/todd/reconstructionism/infrastructure/setup
```

### 2. Run 1000 Document Test
```bash
./test_single_collection_1k.py
```

Expected output:
- Processing rate: 20+ docs/sec (proof of concept target)
- Single write operation per document
- Checkpoint system functional

### 3. Verify Results in ArangoDB
```bash
# Connect to ArangoDB web UI at http://localhost:8529
# Database: arxiv_test_single_collection
# Collection: arxiv_documents

# Check document structure includes:
# - _key (arxiv_id)
# - title, authors, categories
# - abstract
# - abstract_embedding (1024-dim vector)
# - submitted_date, updated_date
# - pdf_status object
```

### 4. Review Metrics
Check `abstracts_single_collection.log` for:
- Checkpoint progress every 10K documents
- Write operations count
- Error tracking
- Resource utilization

## Key Improvements Implemented

1. **Single Collection Design**
   - All data in one document per paper
   - Reduces write operations from 3 to 1
   - Optimized indexes for common queries

2. **Performance Tracking**
   - Real-time metrics collection
   - 10K document checkpoints
   - Batch processing time analysis

3. **Reliability Features**
   - LMDB checkpointing with file fallback
   - Error recovery with exponential backoff
   - Failed batch preservation

4. **Resource Optimization**
   - Batch size: 1000 documents
   - Write buffer: 128MB
   - Commit frequency: 10K documents

## Next Steps After Successful Test

1. **10K Document Validation** (Milestone 2)
   ```bash
   python3 process_abstracts_single_collection.py \
     --max-abstracts 10000 \
     --db-name arxiv_validation_10k
   ```

2. **100K Stability Test** (Milestone 3)
   ```bash
   python3 process_abstracts_single_collection.py \
     --max-abstracts 100000 \
     --db-name arxiv_stability_100k
   ```

3. **Full Production Run** (2.3M documents)
   - Export existing 3-collection data first
   - Run without --max-abstracts limit
   - Monitor for 24-26 hours

## Troubleshooting

### Low Performance
1. Check ArangoDB logs: `sudo journalctl -u arangodb3 -f`
2. Monitor I/O: `iostat -x 1`
3. Verify GPU utilization: `nvidia-smi`

### Checkpoint Issues
- LMDB errors will fallback to file-based checkpoints
- Check `checkpoints/single_collection/` directory
- Resume is automatic on restart

### Memory Issues
- Reduce batch size: `--batch-size 500`
- Decrease queue sizes in config
- Monitor with `htop`

## Success Criteria

✓ 1000 documents process at 20+ docs/sec
✓ Single write operation per document
✓ Checkpointing works correctly
✓ No data loss on resume
✓ Metrics show improvement over baseline

## Contact

Issues or questions: Check the logs first, then review the implementation in `process_abstracts_single_collection.py`.