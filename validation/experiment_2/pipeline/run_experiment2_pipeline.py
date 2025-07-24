#!/usr/bin/env python3
"""
Run Experiment 2 pipeline with late chunking for 2000 documents
Uses separate database to preserve experiment_1 results
"""

import os
import sys
import subprocess
from datetime import datetime
import logging
import time

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

def run_experiment2_pipeline(num_papers=2000):
    """Run experiment 2 pipeline with late chunking"""
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = os.environ.get('EXP2_BASE_RESULTS_DIR', 
                                      os.path.join(os.path.dirname(__file__), '..', 'results'))
    results_dir = os.path.join(base_results_dir, f"exp2_run_{num_papers}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Set up logging
    log_dir = os.path.join(results_dir, "logs")
    logger, log_file = setup_logging(log_dir, f"exp2_pipeline_{num_papers}_{timestamp}")
    
    logger.info("=" * 60)
    logger.info(f"EXPERIMENT 2 PIPELINE - LATE CHUNKING")
    logger.info(f"Processing {num_papers} papers")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now()}")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Log file: {log_file}")
    
    # Set environment variables
    os.environ["EXP2_NUM_PAPERS"] = str(num_papers)
    os.environ["EXP2_RESULTS_DIR"] = results_dir
    os.environ["EXP2_DB_NAME"] = "information_reconstructionism_exp2"
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Pipeline steps
    scripts = [
        ("Database Setup", "setup_experiment2_database.py", "Create experiment 2 database"),
        ("Late Chunking", "step1_late_chunking_extraction.py", f"Extract {num_papers} papers with semantic chunking"),
        ("Load to DB", "step2_load_chunks_to_db.py", "Load papers and chunks to ArangoDB"),
        ("Chunk Similarity", "step3_compute_chunk_similarity.py", "Compute chunk-to-chunk similarities"),
        ("Document Aggregation", "step4_aggregate_doc_similarity.py", "Aggregate to document level"),
        ("Multi-Scale Analysis", "step5_multiscale_analysis.py", "Analyze context amplification at multiple scales"),
    ]
    
    # Track pipeline progress
    step_results = {}
    pipeline_start = time.time()
    
    # Run each step
    for step_name, script, description in scripts:
        logger.info(f"\n{'='*50}")
        logger.info(f"{step_name}: {description}")
        logger.info(f"Script: {script}")
        logger.info(f"{'='*50}")
        
        start_time = time.time()
        
        # Create step-specific log file
        step_log_file = os.path.join(log_dir, f"{step_name.replace(' ', '_').lower()}_{timestamp}.log")
        
        try:
            # Special handling for database setup
            if script == "setup_experiment2_database.py":
                logger.info("Setting up experiment 2 database...")
                # Run interactively in case it asks about existing database
                result = subprocess.run(
                    [sys.executable, script],
                    text=True,
                    capture_output=False  # Allow interaction
                )
            else:
                # Run other scripts normally
                with open(step_log_file, 'w') as step_log:
                    result = subprocess.run(
                        [sys.executable, script],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True
                    )
                    
                    # Write output to step log
                    step_log.write(result.stdout)
                    
                    # Log key lines to main log
                    for line in result.stdout.split('\n'):
                        if any(keyword in line.lower() for keyword in 
                               ['error', 'failed', 'success', 'complete', '✓', '✗', 
                                'loaded:', 'chunks:', 'papers:', 'edges:']):
                            logger.info(f"  {line}")
            
            # Calculate duration
            duration = time.time() - start_time
            
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
                    return False
                
        except Exception as e:
            logger.error(f"✗ {step_name} FAILED with exception: {e}")
            step_results[step_name] = "EXCEPTION"
            return False
    
    # Calculate total duration
    total_duration = time.time() - pipeline_start
    
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT 2 PIPELINE COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Logs directory: {log_dir}")
    
    # Log summary
    logger.info("\nPIPELINE SUMMARY:")
    for step, result in step_results.items():
        logger.info(f"  {step}: {result}")
    
    # Verify outputs
    logger.info("\nOUTPUT VERIFICATION:")
    try:
        from arango import ArangoClient
        client = ArangoClient(hosts='http://192.168.1.69:8529')
        username = os.environ.get('ARANGO_USERNAME', 'root')
        password = os.environ.get('ARANGO_PASSWORD', '')
        db = client.db('information_reconstructionism_exp2', username=username, password=password)
        
        # Check collections
        for collection_name in ['papers_exp2', 'chunks_exp2', 'chunk_similarities_exp2', 'document_similarities_exp2', 'chunk_hierarchy_exp2']:
            if db.has_collection(collection_name):
                count = db.collection(collection_name).count()
                logger.info(f"  {collection_name}: {count:,} documents")
        
        # Special queries for insights
        if db.has_collection('chunks_exp2'):
            query = """
            FOR chunk IN chunks_exp2
                COLLECT paper_id = chunk.paper_id WITH COUNT INTO chunk_count
                COLLECT 
                    avg_chunks = AVG(chunk_count),
                    min_chunks = MIN(chunk_count),
                    max_chunks = MAX(chunk_count),
                    total_papers = COUNT(1)
                RETURN {
                    avg_chunks_per_paper: avg_chunks,
                    min_chunks: min_chunks,
                    max_chunks: max_chunks,
                    total_papers: total_papers
                }
            """
            stats = list(db.aql.execute(query))
            if stats:
                stat = stats[0]
                logger.info(f"\nChunking statistics:")
                logger.info(f"  Papers processed: {stat.get('total_papers', 0)}")
                logger.info(f"  Avg chunks per paper: {stat.get('avg_chunks_per_paper', 0):.1f}")
                logger.info(f"  Min/Max chunks: {stat.get('min_chunks', 0)}/{stat.get('max_chunks', 0)}")
                
    except Exception as e:
        logger.error(f"  Could not verify database: {e}")
    
    # Create analysis scripts
    create_analysis_scripts(results_dir, logger)
    
    logger.info("\n✓ Experiment 2 pipeline complete!")
    logger.info(f"Main log: {log_file}")
    
    return True

def create_analysis_scripts(results_dir, logger):
    """Create analysis scripts for experiment 2"""
    
    analysis_dir = os.path.join(results_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Multi-scale analysis script
    multiscale_script = os.path.join(analysis_dir, "analyze_multiscale_context.py")
    with open(multiscale_script, 'w') as f:
        f.write("""#!/usr/bin/env python3
import sys
import os

# Calculate path relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
analysis_dir = os.path.abspath(os.path.join(script_dir, '..', '..', '..', 'analysis'))
sys.path.append(analysis_dir)

try:
    from multiscale_context_analysis import run_analysis
except ImportError as e:
    print(f"Error importing multiscale_context_analysis: {e}")
    print(f"Attempted to import from: {analysis_dir}")
    sys.exit(1)

if __name__ == "__main__":
    try:
        run_analysis()
    except Exception as e:
        print(f"Error during analysis execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
""")
    os.chmod(multiscale_script, 0o755)
    
    logger.info(f"\nCreated analysis scripts in: {analysis_dir}")

if __name__ == "__main__":
    # Get number of papers from command line or use default
    num_papers = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    
    if num_papers < 1:
        print("Error: Number of papers must be positive")
        sys.exit(1)
    
    print(f"\nStarting Experiment 2 with {num_papers} papers and late chunking...")
    print("This will create a separate database: information_reconstructionism_exp2")
    print("Estimated time: 4-6 hours for 2000 papers\n")
    
    # Run the pipeline
    success = run_experiment2_pipeline(num_papers)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)