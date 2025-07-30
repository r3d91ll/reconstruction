# Multi-Source Academic Database MVP Requirements

## Version 4.0 - Data-Driven Architecture with Empirical Performance Targets

### Executive Summary

Build a multi-source academic database based on measured performance characteristics. Current baseline: 18 documents/second across 3 collections. Target: 25-30 documents/second through collection consolidation and targeted optimizations. Design prioritizes incremental improvements over unsubstantiated claims.

---

## Baseline Performance Metrics

### Current System Performance (Measured)

- **Processing rate**: 18 documents/second (1,080 docs/minute)
- **Dataset size**: 2,365,047 documents processed
- **Processing time**: 36.49 hours
- **Storage used**: ~31GB for 2.3M documents (13.4KB/document)
- **Write pattern**: 3 collections, small random writes

### Observed Characteristics

- 33% performance improvement during processing (13.5 → 18 docs/sec)
- No configuration changes required for improvement
- Performance increase likely from cache warming

---

## Architecture Overview

### Two-Phase Implementation

1. **Phase 1 - Collection Consolidation**
   - Merge 3 collections into 1 per source
   - Expected improvement: 25-30 docs/sec (40-65% gain)
   - Rationale: Reduce write operations by 66%

2. **Phase 2 - Analytical Processing**
   - Build purpose-specific views on demand
   - No permanent analytical collections initially
   - Create as research needs emerge

---

## Phase 1: Collection Consolidation

### Single Collection Schema

```javascript
// Collection: arxiv_documents
{
  "_key": "2301.00001",  // arXiv ID
  "title": "...",
  "authors": [...],
  "categories": ["cs.AI", "cs.LG"],
  "abstract": "...",
  "abstract_embedding": [...],  // 1024-dim vector (4KB)
  "submitted_date": "2023-01-01",
  "updated_date": "2023-01-15",
  
  // PDF tracking (when processed)
  "pdf_status": {
    "state": "unprocessed",  // State machine below
    "tar_source": "arXiv_pdf_2301_001.tar",
    "last_updated": null,
    "retry_count": 0,
    "error_message": null
  }
}
```

**Document size**: ~13.4KB average (measured)
**Expected throughput**: 25-30 docs/second
**Expected improvement**: 29-40% reduction in processing time

### I/O Optimization Strategy

Based on current bottleneck analysis:

1. **Batch size**: 1,000 documents (13.4MB)
2. **Write buffer**: 128MB (accommodates ~9,500 documents)
3. **Commit frequency**: Every 10,000 documents
4. **Thread allocation**: 4 write threads (of 24 available cores)

### Validation Metrics (Minimal Set)

To confirm our 40-65% improvement target, we'll track:

```python
class ValidationMetrics:
    """Minimal metrics to validate collection consolidation benefits"""
    
    def __init__(self):
        self.start_time = time.time()
        self.checkpoint_times = []  # Track 10K document intervals
        self.checkpoint_counter = 0   # Efficient modulo replacement
        self.write_operations = 0    # Count actual DB writes
        self.documents_processed = 0
        self.errors = []             # Track failures
        
    def record_batch(self, batch_size, success=True):
        """Record batch processing results"""
        # Validate batch_size
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
            
        if success:
            self.documents_processed += batch_size
            self.checkpoint_counter += batch_size
            
            # Check if we've hit 10K threshold
            if self.checkpoint_counter >= 10000:
                self.checkpoint_counter -= 10000  # Preserve remainder
                self.checkpoint()
        else:
            self.errors.append({
                'timestamp': time.time(),
                'batch_size': batch_size,
                'documents_processed': self.documents_processed
            })
        
    def checkpoint(self):
        """Called every 10,000 documents"""
        current_time = time.time()
        if self.checkpoint_times:
            interval_time = current_time - self.checkpoint_times[-1]
            rate = 10000 / interval_time
            print(f"[{current_time:.0f}] Last 10K docs: {rate:.1f} docs/sec")
        self.checkpoint_times.append(current_time)
        
    def final_report(self):
        """Generate comparison metrics with error summary"""
        # Handle edge cases
        if not self.checkpoint_times:
            print("No checkpoints recorded - insufficient data")
            return
            
        total_time = time.time() - self.start_time
        overall_rate = self.documents_processed / total_time if total_time > 0 else 0
        writes_per_doc = self.write_operations / self.documents_processed if self.documents_processed > 0 else 0
        
        print(f"\nFinal metrics:")
        print(f"  Overall rate: {overall_rate:.1f} docs/sec (target: 25-30)")
        print(f"  Writes per doc: {writes_per_doc:.2f} (target: 1.0)")
        print(f"  Processing time: {total_time/3600:.2f} hours")
        print(f"  Error count: {len(self.errors)}")
        
        if self.errors and self.errors[0].get('documents_processed') is not None:
            print(f"  First error at: {self.errors[0]['documents_processed']} documents")
```

**Key metrics to prove success:**

1. **Throughput**: Must exceed 25 docs/sec consistently
2. **Write reduction**: Should show 1 write/doc vs previous 3
3. **Time to completion**: Under 26 hours for full dataset

### ArangoDB Configuration

```yaml
# Specific settings for write optimization
[rocksdb]
write-buffer-size = 134217728  # 128MB
max-write-buffer-number = 4
max-background-jobs = 8
compaction-read-ahead-size = 2097152  # 2MB

[cache]
size = 8589934592  # 8GB (sufficient for working set)

[server]
threads = 16  # Utilize available cores
```

---

## Phase 2: PDF Processing (Future)

### Storage Architecture

**Current assets**:

- 4TB tar archive in `/mnt/data-cold/arxiv_data/`
- ~60% of arXiv PDFs acquired

**Processing approach**:

- Extract PDFs on demand only
- Process in batches of 100 for efficiency
- Delete after embedding generation
- Track processing state in main document

### PDF Lifecycle State Machine

```
State Transitions:
- unprocessed → processing (extraction initiated)
- processing → completed (successful embedding)
- processing → failed (error occurred)
- failed → unprocessed (retry_count < 3)
- failed → abandoned (retry_count >= 3)
- completed → reprocess (manual trigger only)

Timeout: processing → failed (after 300 seconds)
```

**Storage requirements**:

- Working space: 100GB (100 PDFs @ 1MB average)
- No permanent PDF storage needed

---

## Hardware Utilization

### Current Hardware

- 2x NVIDIA A6000 GPUs (48GB each)
- 256GB System RAM
- 24 CPU cores
- Multiple RAID configurations

### Realistic Allocation

- **GPU 0**: Abstract embeddings (100% during import)
- **GPU 1**: Idle (remove from requirements)
- **RAM**: 32GB active usage (8GB RocksDB + 24GB process)
- **CPU**: 16 threads allocated (67% utilization)

### Storage Requirements

- **Database**: 100GB (3x current size for growth)
- **Working space**: 100GB for PDF processing
- **Tar archives**: 4TB (existing)

---

## Performance Targets

### Realistic Goals (Based on Measurements)

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Write throughput | 18 docs/sec | 25-30 docs/sec | 40-65% |
| Processing time | 36.5 hours | Reduced by 29-40% | Faster completion |
| Storage efficiency | 3 collections | 1 collection | 66% reduction |
| Write operations | 3 per document | 1 per document | 66% reduction |

### Measurement Plan

1. Baseline current 3-collection performance
2. Test single collection with sample dataset
3. Measure write amplification factor
4. Profile RocksDB compaction impact
5. Adjust batch sizes based on results

### Validation Checkpoints

During implementation, we'll validate at:

- **1K documents**: Initial rate check (should exceed 20 docs/sec)
- **10K documents**: Sustained performance (should stabilize at 25+)
- **100K documents**: Long-run stability (no degradation)
- **Full dataset**: Final validation

---

## Phase 1 Milestone Markers

### Milestone 1: Proof of Concept

**Target**: Validate single collection approach

- Process 1,000 documents in single collection
- Achieve 20+ docs/sec (exceeds baseline)
- Confirm write reduction (1 vs 3 operations)
- **Success criteria**: Green light to proceed

### Milestone 2: Performance Validation

**Target**: Confirm scalability at 10K documents  

- Maintain 25+ docs/sec throughput
- Memory usage <1GB
- No performance degradation
- **Success criteria**: Consistent improvement over baseline

### Milestone 3: Stability Test

**Target**: Process 100K documents

- Sustained 25-30 docs/sec
- Checkpoint/resume functionality working
- Metrics show linear scaling
- **Success criteria**: Linear performance scaling confirmed

### Milestone 4: Migration Ready

**Target**: Prepare for full dataset

- Export current 3-collection data
- Configure ArangoDB optimizations
- Validate single-file pipeline code
- **Success criteria**: All systems go for production run

### Milestone 5: Production Import

**Target**: Process full 2.3M documents

- Achieve target throughput of 25-30 docs/sec
- Zero data loss
- All metrics within target range
- **Success criteria**: 40-65% improvement achieved

### Milestone 6: Phase 1 Complete

**Target**: Validated single collection architecture

- Document final performance metrics
- Archive 3-collection export
- Update pipeline for ongoing use
- **Success criteria**: Ready for Phase 2 planning

---

## Risk Mitigation

### Technical Risks

1. **Performance target miss**
   - Mitigation: Conservative estimates based on measurements
   - Fallback: Accept 40% improvement as success

2. **Memory pressure**
   - Current usage: <32GB
   - Available: 256GB
   - Risk: Minimal

3. **Storage growth**
   - Current: 31GB
   - Allocated: 100GB
   - Headroom: 3.2x

### Data Integrity

- Checkpoint every 10,000 documents
- Verify counts post-migration
- Keep original export for 30 days

---

## Success Criteria

1. **Primary**: Achieve 25+ documents/second sustained
2. **Secondary**: Complete import in <26 hours
3. **Validation**: 100% document count match
4. **Storage**: Single collection per source

---

## Operational Procedures

### Error Recovery Strategy

```python
class ErrorRecovery:
    """Batch failure recovery with exponential backoff"""
    
    def __init__(self, collection, dead_letter_path="failed_batches"):
        self.max_retries = 3
        self.base_delay = 1.0  # seconds
        self.collection = collection
        self.dead_letter_path = Path(dead_letter_path)
        self.dead_letter_path.mkdir(exist_ok=True)
        
    def retry_batch(self, batch, attempt=0):
        if attempt >= self.max_retries:
            # Write to dead letter queue
            self.save_failed_batch(batch)
            return False
            
        try:
            # Retry logic with exponential backoff
            delay = self.base_delay * (2 ** attempt)
            time.sleep(delay)
            
            # Attempt write
            self.collection.insert_many(batch)
            return True
            
        except DuplicateKeyError:
            # Skip documents that already exist
            return self.retry_with_upsert(batch)
            
        except ConnectionError:
            # Network issues - retry
            return self.retry_batch(batch, attempt + 1)
            
        except Exception as e:
            # Other errors - log and retry
            print(f"Retry {attempt}: {e}")
            return self.retry_batch(batch, attempt + 1)
    
    def retry_with_upsert(self, batch):
        """Retry batch using upsert operations"""
        try:
            for doc in batch:
                if '_key' in doc:
                    try:
                        self.collection.update({'_key': doc['_key']}, doc)
                    except:
                        self.collection.insert(doc)
                else:
                    self.collection.insert(doc)
            return True
        except Exception as e:
            print(f"Upsert failed: {e}")
            return False
    
    def save_failed_batch(self, batch):
        """Save failed batch to dead letter queue"""
        timestamp = datetime.now().isoformat()
        filename = self.dead_letter_path / f"failed_batch_{timestamp}.json"
        try:
            with open(filename, 'w') as f:
                json.dump({
                    'timestamp': timestamp,
                    'batch_size': len(batch),
                    'documents': batch
                }, f, indent=2)
            print(f"Saved failed batch to {filename}")
        except Exception as e:
            print(f"Failed to save dead letter batch: {e}")
```

### Checkpoint Storage

```python
# Using LMDB for checkpoint persistence with fallback
checkpoint_db = lmdb.open('./checkpoints', map_size=1024*1024*100)  # 100MB

def save_checkpoint(doc_count, last_key):
    # Validate checkpoint data
    if not isinstance(doc_count, int) or doc_count < 0:
        raise ValueError("doc_count must be a non-negative integer")
    if last_key is None:
        raise ValueError("last_key cannot be None")
    
    checkpoint_data = {
        'documents_processed': doc_count,
        'last_document_key': str(last_key),
        'timestamp': time.time(),
        'version': '1.0'
    }
    
    try:
        with checkpoint_db.begin(write=True) as txn:
            txn.put(b'progress', json.dumps(checkpoint_data).encode())
    except lmdb.Error as e:
        # Fallback to atomic file-based checkpoint
        print(f"LMDB error: {e}, using file backup")
        temp_file = 'checkpoint_backup.tmp'
        try:
            # Write to temporary file first
            with open(temp_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            # Atomically rename to final file
            os.replace(temp_file, 'checkpoint_backup.json')
        except Exception as write_error:
            print(f"Failed to write checkpoint: {write_error}")
            # Clean up temp file if it exists
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise
```

### Monitoring Hooks

```python
# Resource monitoring during import
def monitor_resources():
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024,
        'io_wait': psutil.cpu_times().iowait,
        'db_connections': get_active_connections()
    }
```

### Batch Size Adaptation

```python
import logging
import threading
from collections import deque
import time

class AdaptiveBatcher:
    """Production-ready adaptive batch size management with thread safety"""
    
    def __init__(self, initial_size=1000, logger=None):
        self.batch_size = initial_size
        self.min_size = 100
        self.max_size = 5000
        self.target_time = 10.0  # seconds per batch
        
        # Production enhancements
        self._lock = threading.Lock()  # Thread safety
        self.history = deque(maxlen=20)  # Performance history
        self.logger = logger or logging.getLogger(__name__)
        self.smoothing_factor = 0.7  # Exponential smoothing
        self.last_adjustment_time = 0
        self.min_adjustment_interval = 30  # seconds
        
        self.logger.info(f"AdaptiveBatcher initialized with size={initial_size}")
        
    def adjust_size(self, last_batch_time):
        """Thread-safe batch size adjustment with smoothing"""
        with self._lock:
            current_time = time.time()
            
            # Prevent too frequent adjustments
            if current_time - self.last_adjustment_time < self.min_adjustment_interval:
                return self.batch_size
            
            # Add to history
            self.history.append(last_batch_time)
            
            # Calculate smoothed average time
            if len(self.history) >= 3:
                avg_time = sum(self.history) / len(self.history)
                smoothed_time = (self.smoothing_factor * avg_time + 
                                (1 - self.smoothing_factor) * last_batch_time)
            else:
                smoothed_time = last_batch_time
            
            old_size = self.batch_size
            
            # Gradual adjustments to prevent oscillations
            if smoothed_time > self.target_time * 1.2:
                # Too slow, reduce batch size gradually
                adjustment_factor = min(0.9, self.target_time / smoothed_time)
                self.batch_size = max(self.min_size, int(self.batch_size * adjustment_factor))
            elif smoothed_time < self.target_time * 0.8:
                # Too fast, increase batch size gradually
                adjustment_factor = min(1.1, self.target_time / smoothed_time)
                self.batch_size = min(self.max_size, int(self.batch_size * adjustment_factor))
            
            # Log significant changes
            if self.batch_size != old_size:
                self.last_adjustment_time = current_time
                change_pct = ((self.batch_size - old_size) / old_size) * 100
                self.logger.info(
                    f"Batch size adjusted: {old_size} -> {self.batch_size} "
                    f"({change_pct:+.1f}%) based on avg time {smoothed_time:.2f}s"
                )
                
            return self.batch_size
    
    def get_current_size(self):
        """Thread-safe getter for current batch size"""
        with self._lock:
            return self.batch_size
    
    def get_stats(self):
        """Get performance statistics"""
        with self._lock:
            if not self.history:
                return {"current_size": self.batch_size, "avg_time": 0, "samples": 0}
            
            return {
                "current_size": self.batch_size,
                "avg_time": sum(self.history) / len(self.history),
                "min_time": min(self.history),
                "max_time": max(self.history),
                "samples": len(self.history)
            }
```

---

## Future Considerations

### Not in Scope for MVP

- PDF processing pipeline
- Cross-source deduplication
- Library of Congress classification
- Analytical collections
- Second GPU utilization

### Phase 2 Planning

After validating single collection performance:

1. Design PDF processing pipeline
2. Create analytical views as needed
3. Implement on-demand graph generation

---

## Appendix A: Measured Data Points

```
Current Performance:
- Documents: 2,365,047
- Time: 2189.4 minutes
- Rate: 18.0 docs/second
- Storage: ~31GB

Hardware Utilization:
- GPU: 1 of 2 used
- RAM: <32GB of 256GB used
- CPU: Unknown (24 cores available)
- I/O: Primary bottleneck
```

## Appendix C: Metrics Implementation

### Integration Points

```python
# In main processing loop
metrics = ValidationMetrics()

for batch in document_batches:
    try:
        # Process batch
        embeddings = generate_embeddings(batch)
        
        # Single write operation (vs 3 previously)
        collection.insert_many(batch)
        metrics.write_operations += 1
        metrics.record_batch(len(batch), success=True)
        
    except Exception as e:
        print(f"Batch failed: {e}")
        metrics.record_batch(len(batch), success=False)
        # Implement retry logic or continue based on strategy

# Final report
metrics.final_report()
```

### Expected Output

```
Progress checkpoints:
[1722345678] Last 10K docs: 26.3 docs/sec
[1722346012] Last 10K docs: 27.1 docs/sec  
[1722346357] Last 10K docs: 26.8 docs/sec
...

Final metrics:
  Overall rate: 26.7 docs/sec (target: 25-30) ✓
  Writes per doc: 1.00 (target: 1.0) ✓
  Processing time: 24.6 hours
  Error count: 2
  First error at: 145,234 documents

Performance Distribution (last 100 checkpoints):
  P50: 26.5 docs/sec
  P95: 28.2 docs/sec
  P99: 29.1 docs/sec
  Min: 22.3 docs/sec (during compaction)
  Max: 29.7 docs/sec
```

---

## Appendix B: Configuration Changes

### From Version 3.0

- Removed unsubstantiated 100+ docs/sec target
- Eliminated 10x write amplification claim
- Reduced RAM requirement from 256GB to actual usage
- Removed Library of Congress classification requirement
- Based all targets on measured performance

### Empirical Approach

All performance targets now based on:

1. Measured baseline performance
2. Conservative improvement estimates
3. Actual hardware capabilities
4. Observed system behavior

---

## Sign-off

**Prepared by**: Claude Assistant  
**Date**: July 30, 2025  
**Version**: 4.0  
**Status**: Data-Driven Architecture

### Review Checklist

- [x] All performance claims backed by data
- [x] Resource allocations based on measurements
- [x] Conservative improvement targets
- [x] Clear success criteria
- [x] Incremental implementation plan
