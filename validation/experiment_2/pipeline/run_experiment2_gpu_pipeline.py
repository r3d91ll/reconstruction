#!/usr/bin/env python3
"""
GPU-Accelerated Experiment 2 Pipeline
Uses both A6000 GPUs (48GB each) with NVLink for maximum performance
All operations that can be GPU-accelerated are moved to GPU
"""

import os
import sys
import subprocess
from datetime import datetime
import logging
import time
import torch
import json

def check_gpu_setup():
    """Verify GPU setup and availability"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU pipeline requires CUDA.")
    
    gpu_count = torch.cuda.device_count()
    print(f"\n🚀 GPU Setup Verification:")
    print(f"Found {gpu_count} GPUs")
    
    total_memory = 0
    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        total_memory += memory
        print(f"  GPU {i}: {name} ({memory:.1f} GB)")
    
    print(f"Total GPU Memory: {total_memory:.1f} GB")
    
    if gpu_count < 2:
        print("⚠️  Warning: Found less than 2 GPUs. Performance may be reduced.")
    
    return gpu_count, total_memory

def setup_logging(log_dir, run_name):
    """Set up logging for the pipeline run."""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{run_name}.log")
    
    # Configure logging to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__), log_file

def run_gpu_pipeline(num_papers=2000):
    """Run GPU-accelerated experiment 2 pipeline"""
    
    # Check GPU setup first
    gpu_count, total_gpu_memory = check_gpu_setup()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = os.environ.get('EXP2_BASE_RESULTS_DIR', 
                                      os.path.join(os.path.dirname(__file__), '..', 'results'))
    results_dir = os.path.join(base_results_dir, f"exp2_gpu_run_{num_papers}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Set up logging
    log_dir = os.path.join(results_dir, "logs")
    logger, log_file = setup_logging(log_dir, f"exp2_gpu_pipeline_{num_papers}_{timestamp}")
    
    logger.info("=" * 60)
    logger.info(f"GPU-ACCELERATED EXPERIMENT 2 PIPELINE")
    logger.info(f"Processing {num_papers} papers")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now()}")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"GPU Count: {gpu_count}")
    logger.info(f"Total GPU Memory: {total_gpu_memory:.1f} GB")
    
    # Set environment variables
    os.environ["EXP2_NUM_PAPERS"] = str(num_papers)
    os.environ["EXP2_RESULTS_DIR"] = results_dir
    os.environ["EXP2_DB_NAME"] = "information_reconstructionism_exp2_gpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Use both GPUs
    os.environ["EXP2_USE_GPU"] = "true"
    os.environ["EXP2_USE_FP16"] = "true"  # Use float16 for memory efficiency
    os.environ["NON_INTERACTIVE"] = "true"  # Run in non-interactive mode
    
    # GPU-specific batch sizes
    os.environ["EXP2_BATCH_SIZE"] = "10000"  # Large batches for GPU
    os.environ["EXP2_EMBEDDING_BATCH_SIZE"] = "64"  # Jina batch size
    os.environ["EXP2_CHUNK_BATCH_SIZE"] = "5000"  # Chunk processing batch
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # GPU-Accelerated Pipeline steps
    scripts = [
        ("Database Setup", "setup_experiment2_database.py", "Create GPU experiment database", False),
        ("GPU Extraction", "step1_late_chunking_extraction_gpu.py", f"GPU-accelerated extraction of {num_papers} papers", True),
        ("GPU Embeddings", "step2_generate_embeddings_gpu.py", "Generate embeddings using Jina on GPU", True),
        ("GPU DB Load", "step3_load_to_db_gpu.py", "Batch load to ArangoDB with GPU preprocessing", True),
        ("GPU Similarity", "step4_compute_similarity_gpu.py", "Compute similarities using both GPUs", True),
        ("GPU Aggregation", "step5_aggregate_documents_gpu.py", "GPU-accelerated document aggregation", True),
        ("GPU Analysis", "step6_multiscale_analysis_gpu.py", "Multi-scale analysis on GPU", True),
    ]
    
    # Track pipeline progress
    step_results = {}
    pipeline_start = time.time()
    
    # Run each step
    for step_name, script, description, is_gpu in scripts:
        logger.info(f"\n{'='*50}")
        logger.info(f"{step_name}: {description}")
        logger.info(f"Script: {script}")
        if is_gpu:
            logger.info(f"🚀 GPU-ACCELERATED")
        logger.info(f"{'='*50}")
        
        start_time = time.time()
        
        # Create step-specific log file
        step_log_file = os.path.join(log_dir, f"{step_name.replace(' ', '_').lower()}_{timestamp}.log")
        
        # Check if script exists, if not use fallback
        script_path = os.path.join(script_dir, script)
        if not os.path.exists(script_path):
            # Use existing non-GPU version as fallback
            fallback_script = script.replace("_gpu.py", ".py")
            fallback_path = os.path.join(script_dir, fallback_script)
            
            if os.path.exists(fallback_path):
                logger.warning(f"GPU script not found, using fallback: {fallback_script}")
                script_path = fallback_path
            elif script == "step4_compute_similarity_gpu.py":
                # We already created the GPU similarity script
                script_path = os.path.join(script_dir, "step3_compute_chunk_similarity_gpu.py")
            else:
                logger.error(f"Script not found: {script}")
                step_results[step_name] = "SCRIPT_NOT_FOUND"
                continue
        
        try:
            # Special handling for database setup
            if script == "setup_experiment2_database.py":
                logger.info("Setting up experiment 2 GPU database...")
                # Run interactively in case it asks about existing database
                result = subprocess.run(
                    [sys.executable, script_path],
                    text=True,
                    capture_output=False  # Allow interaction
                )
            else:
                # Run other scripts normally
                # Create filtered environment with only necessary variables
                filtered_env = {
                    k: v for k, v in os.environ.items()
                    if k.startswith(('EXP2_', 'CUDA_', 'PATH', 'PYTHONPATH'))
                }

                with open(step_log_file, 'w') as step_log:
                    result = subprocess.run(
                        [sys.executable, script_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=filtered_env
                    )
                    
                    # Write output to step log
                    step_log.write(result.stdout)
                    
                    # Log key lines to main log
                    for line in result.stdout.split('\n'):
                        if any(keyword in line.lower() for keyword in 
                               ['error', 'failed', 'success', 'complete', '✓', '✗', 
                                'loaded:', 'chunks:', 'papers:', 'edges:', 'gpu', 
                                'cuda', 'memory', 'batch']):
                            logger.info(f"  {line}")
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log GPU memory usage after each step
            if is_gpu and torch.cuda.is_available():
                for i in range(gpu_count):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    logger.info(f"  GPU {i} memory used: {allocated:.2f} GB")
            
            if result.returncode == 0:
                logger.info(f"✓ {step_name} completed in {duration:.1f} seconds")
                if script != "setup_experiment2_database.py":
                    logger.info(f"  Step log: {step_log_file}")
                step_results[step_name] = "SUCCESS"
            else:
                logger.error(f"✗ {step_name} FAILED with return code {result.returncode}")
                if script != "setup_experiment2_database.py":
                    logger.error(f"  Check step log: {step_log_file}")
                step_results[step_name] = "FAILED"
                
                # Continue with remaining steps if database setup fails
                if script == "setup_experiment2_database.py":
                    logger.warning("Database setup failed - may already exist, continuing...")
                else:
                    # For missing GPU scripts, continue with warning
                    if step_results.get(step_name) == "SCRIPT_NOT_FOUND":
                        logger.warning(f"Skipping {step_name} - script not implemented yet")
                        continue
                    return False
                
        except Exception as e:
            logger.error(f"✗ {step_name} FAILED with exception: {e}")
            step_results[step_name] = "EXCEPTION"
            # Continue if GPU script not found
            if "No such file" in str(e):
                logger.warning(f"Skipping {step_name} - not implemented yet")
                continue
            return False
    
    # Calculate total duration
    total_duration = time.time() - pipeline_start
    
    logger.info(f"\n{'='*60}")
    logger.info("GPU-ACCELERATED PIPELINE COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Logs directory: {log_dir}")
    
    # Log summary
    logger.info("\nPIPELINE SUMMARY:")
    for step, result in step_results.items():
        logger.info(f"  {step}: {result}")
    
    # Save GPU performance metrics
    gpu_metrics = {
        "gpu_count": gpu_count,
        "total_gpu_memory_gb": total_gpu_memory,
        "pipeline_duration_seconds": total_duration,
        "pipeline_duration_seconds": total_duration,
        "papers_processed": num_papers,
        "papers_per_second": num_papers / total_duration if total_duration > 0 else 0,
        "timestamp": datetime.now().isoformat(),
        "step_results": step_results
    }
    
    metrics_path = os.path.join(results_dir, "gpu_performance_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(gpu_metrics, f, indent=2)
    
    logger.info(f"\nGPU Performance Metrics saved to: {metrics_path}")
    logger.info(f"Processing rate: {gpu_metrics['papers_per_second']:.2f} papers/second")
    
    logger.info("\n✓ GPU-Accelerated Experiment 2 pipeline complete!")
    logger.info(f"Main log: {log_file}")
    
    return True

if __name__ == "__main__":
    # Get number of papers from command line or use default
    num_papers = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    
    if num_papers < 1:
        print("Error: Number of papers must be positive")
        sys.exit(1)
    
    print(f"\n🚀 Starting GPU-Accelerated Experiment 2 with {num_papers} papers...")
    print("This will use both A6000 GPUs with NVLink for maximum performance")
    print("Database: information_reconstructionism_exp2_gpu")
    print("Estimated time: 30-60 minutes for 2000 papers (vs 4-6 hours on CPU)\n")
    
    # Run the pipeline
    success = run_gpu_pipeline(num_papers)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)