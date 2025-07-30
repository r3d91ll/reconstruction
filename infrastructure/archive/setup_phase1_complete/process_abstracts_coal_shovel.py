#!/usr/bin/env python3
"""
Coal Shovel Pipeline - Keep GPU Fed Continuously
Pre-load all batches in CPU RAM, then shovel them to GPU maintaining 7 batches in flight
"""

import os
import sys
import json
import time
import queue
import logging
import threading
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
import psutil

# Import the main pipeline components
from process_abstracts_single_collection import (
    Config, JinaEmbedder, DatabaseWriter, CheckpointManager,
    ValidationMetrics, monitor_resources
)

logger = logging.getLogger(__name__)


class CoalShovelPipeline:
    """Pipeline that pre-loads all batches and keeps GPU constantly fed"""
    
    def __init__(self, config: Config):
        self.config = config
        self.metrics = ValidationMetrics()
        self.all_batches = []  # Pre-loaded batches in CPU RAM
        
    def pre_load_all_batches(self) -> int:
        """Load all documents and create batches in CPU RAM"""
        logger.info("Pre-loading all batches into CPU RAM...")
        
        # Log available RAM at the start
        available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        logger.info(f"Available RAM: {available_ram_gb:.2f} GB / {total_ram_gb:.2f} GB total")
        
        checkpoint_manager = CheckpointManager(self.config.checkpoint_dir)
        metadata_dir = Path(self.config.metadata_dir)
        all_files = list(metadata_dir.glob("*.json"))
        
        if self.config.max_abstracts:
            all_files = all_files[:self.config.max_abstracts]
            
            # Warn if max_abstracts is very high
            if self.config.max_abstracts > 100000:
                logger.warning(f"WARNING: max_abstracts is set to {self.config.max_abstracts}, which may consume significant RAM. "
                             f"Available RAM: {available_ram_gb:.2f} GB. Monitor memory usage closely.")
            
        batch = []
        processed_count = 0
        
        for file_path in all_files:
            try:
                with open(file_path, 'r') as f:
                    metadata = json.load(f)
                    
                arxiv_id = metadata.get('arxiv_id', metadata.get('id', ''))
                if not arxiv_id:
                    continue
                    
                # Skip if already processed
                if self.config.resume and checkpoint_manager.is_processed(arxiv_id):
                    continue
                    
                abstract = metadata.get('abstract', '').strip()
                if len(abstract) < 50:
                    continue
                    
                # Create document
                document = {
                    '_key': arxiv_id,
                    'title': metadata.get('title', ''),
                    'authors': metadata.get('authors', []),
                    'categories': metadata.get('categories', []),
                    'abstract': abstract,
                    'submitted_date': metadata.get('published', ''),
                    'updated_date': metadata.get('updated', ''),
                    'pdf_status': {
                        'state': 'unprocessed',
                        'tar_source': None,
                        'last_updated': None,
                        'retry_count': 0,
                        'error_message': None
                    }
                }
                
                batch.append(document)
                processed_count += 1
                
                # Create batch when full
                if len(batch) >= self.config.batch_size:
                    self.all_batches.append(batch)
                    batch = []
                    
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                
        # Don't forget the last batch
        if batch:
            self.all_batches.append(batch)
            
        logger.info(f"Pre-loaded {len(self.all_batches)} batches ({processed_count} documents) into CPU RAM")
        return len(self.all_batches)
        
    def coal_shovel_worker(self, embedding_queue: mp.Queue, stop_event: mp.Event):
        """The engineer shoveling coal - keeps exactly 7 batches in GPU pipeline"""
        logger.info("Coal shovel worker starting - maintaining 7 batches in GPU pipeline")
        
        batch_index = 0
        batches_in_flight = 0
        max_in_flight = 7
        
        # Initial loading - fill the pipeline
        while batch_index < len(self.all_batches) and batches_in_flight < max_in_flight:
            embedding_queue.put(self.all_batches[batch_index])
            batch_index += 1
            batches_in_flight += 1
            logger.debug(f"Shoveled batch {batch_index}, {batches_in_flight} in flight")
            
        # Keep shoveling until done
        while batch_index < len(self.all_batches) and not stop_event.is_set():
            try:
                # Wait for space in the queue (meaning GPU consumed a batch)
                # This blocks until there's room, maintaining our limit
                embedding_queue.put(self.all_batches[batch_index], timeout=1.0)
                batch_index += 1
                
                if batch_index % 10 == 0:
                    logger.info(f"Shoveled {batch_index}/{len(self.all_batches)} batches")
                    
            except queue.Full:
                # Queue is full, GPU is still processing
                continue
            except Exception as e:
                logger.error(f"Coal shovel error: {e}")
                
        logger.info(f"Coal shovel complete - fed all {batch_index} batches to GPU")
        
    def run(self):
        """Run the coal shovel pipeline"""
        logger.info("Starting Coal Shovel Pipeline")
        logger.info(f"Strategy: Pre-load all batches, maintain 7 in GPU pipeline")
        
        # Pre-load all batches
        start_load = time.time()
        num_batches = self.pre_load_all_batches()
        load_time = time.time() - start_load
        logger.info(f"Pre-loading took {load_time:.1f}s")
        
        if num_batches == 0:
            logger.warning("No batches to process!")
            return
            
        # Create queues
        embedding_queue = mp.Queue(maxsize=7)  # Exactly 7 batches max
        db_queue = mp.Queue(maxsize=7)
        stop_event = mp.Event()
        
        # Start database writer
        db_writer = DatabaseWriter(db_queue, self.config, self.metrics)
        db_writer.start()
        
        # Start GPU embedding worker
        embedding_process = mp.Process(
            target=embedding_worker_process_simple,
            args=(0, embedding_queue, db_queue, stop_event, self.config, self.metrics)
        )
        embedding_process.start()
        
        # Start coal shovel thread
        shovel_thread = threading.Thread(
            target=self.coal_shovel_worker,
            args=(embedding_queue, stop_event)
        )
        shovel_thread.start()
        
        # Monitor progress
        start_time = time.time()
        last_update = time.time()
        
        try:
            while embedding_process.is_alive() or not db_queue.empty():
                time.sleep(5)
                
                if time.time() - last_update > 10:
                    elapsed = time.time() - start_time
                    rate = self.metrics.documents_processed / elapsed if elapsed > 0 else 0
                    
                    logger.info(
                        f"Progress: {self.metrics.documents_processed}/{num_batches * self.config.batch_size} | "
                        f"Rate: {rate:.1f} docs/s | "
                        f"Queues: embed={embedding_queue.qsize()}, db={db_queue.qsize()}"
                    )
                    last_update = time.time()
                    
        except KeyboardInterrupt:
            logger.warning("Interrupted - shutting down")
            stop_event.set()
            
        # Wait for completion
        shovel_thread.join()
        embedding_queue.put(None)  # Poison pill
        embedding_process.join()
        db_writer.stop()
        db_writer.join()
        
        # Final report
        self.metrics.final_report()
        
        total_time = time.time() - start_time
        logger.info(f"\nCoal Shovel Pipeline Complete:")
        logger.info(f"  Pre-load time: {load_time:.1f}s")
        logger.info(f"  Processing time: {total_time:.1f}s")
        logger.info(f"  Total documents: {self.metrics.documents_processed}")


def embedding_worker_process_simple(
    gpu_id: int,
    embedding_queue: mp.Queue,
    db_queue: mp.Queue,
    stop_event: mp.Event,
    config: Config,
    metrics: ValidationMetrics
):
    """Simplified embedding worker for coal shovel approach"""
    import torch
    torch.cuda.set_device(gpu_id)
    
    logger = logging.getLogger(f"GPU-{gpu_id}")
    logger.info(f"GPU worker starting on device {gpu_id}")
    
    embedder = JinaEmbedder(device=f'cuda:{gpu_id}')
    
    while not stop_event.is_set():
        try:
            batch = embedding_queue.get(timeout=1.0)
            if batch is None:
                break
                
            # Process batch
            abstracts = [doc['abstract'] for doc in batch]
            embeddings = embedder.embed_batch(abstracts)
            
            # Add embeddings
            for i, doc in enumerate(batch):
                doc['abstract_embedding'] = embeddings[i].tolist()
                
            # Send to database
            db_queue.put(batch)
            
            # Memory cleanup
            del embeddings, abstracts
            torch.cuda.empty_cache()
            
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"GPU worker error: {e}")
            torch.cuda.empty_cache()


def main():
    """Run coal shovel pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Coal Shovel Pipeline")
    parser.add_argument('--max-abstracts', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--clean-start', action='store_true')
    
    args = parser.parse_args()
    
    config = Config(
        max_abstracts=args.max_abstracts,
        batch_size=args.batch_size,
        clean_start=args.clean_start,
        db_name='arxiv_coal_shovel_test'
    )
    
    pipeline = CoalShovelPipeline(config)
    pipeline.run()


if __name__ == '__main__':
    main()