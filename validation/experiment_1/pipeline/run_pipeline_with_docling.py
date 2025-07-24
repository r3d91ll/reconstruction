#!/usr/bin/env python3
"""
Run pipeline with Docling-extracted data
Skips step1 (embedding generation) since embeddings are already in the papers
"""

import os
import sys
import subprocess
from datetime import datetime
import logging

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

def run_pipeline(num_papers=1000):
    """Run the pipeline with Docling-extracted papers"""
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"/home/todd/reconstructionism/validation/experiment_1/results/docling_run_{num_papers}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Set up logging
    log_dir = os.path.join(results_dir, "logs")
    logger, log_file = setup_logging(log_dir, f"pipeline_docling_{num_papers}_{timestamp}")
    
    logger.info("=" * 60)
    logger.info(f"PIPELINE RUN WITH DOCLING DATA - {num_papers} PAPERS")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now()}")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Log file: {log_file}")
    
    # Set environment variables
    os.environ["PAPER_LIMIT"] = str(num_papers)
    os.environ["RESULTS_DIR"] = results_dir
    
    # Define pipeline steps (skip step1 since embeddings exist)
    scripts = [
        ("Step 2", "step2_load_arangodb.py", "Load papers with embeddings into ArangoDB"),
        ("Step 3", "step3_compute_similarity.py", "Compute similarities (GPU accelerated)"),
        ("Step 4", "step4_context_amplification_batch.py", "Apply Context^1.5"),
    ]
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Track pipeline progress
    step_results = {}
    pipeline_start = datetime.now()
    
    # Run each step
    for step_name, script, description in scripts:
        logger.info(f"\n{'='*50}")
        logger.info(f"{step_name}: {description}")
        logger.info(f"Script: {script}")
        logger.info(f"{'='*50}")
        
        start_time = datetime.now()
        
        # Create step-specific log file
        step_log_file = os.path.join(log_dir, f"{step_name.replace(' ', '_').lower()}_{timestamp}.log")
        
        try:
            # Run the script and capture output
            with open(step_log_file, 'w') as step_log:
                result = subprocess.run(
                    [sys.executable, script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
                # Write output to step log
                step_log.write(result.stdout)
                
                # Also log key lines to main log
                for line in result.stdout.split('\n'):
                    if any(keyword in line.lower() for keyword in ['error', 'failed', 'success', 'complete', '✓', '✗', 'loaded:', 'results:']):
                        logger.info(f"  {line}")
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            
            if result.returncode == 0:
                logger.info(f"✓ {step_name} completed in {duration:.1f} seconds")
                logger.info(f"  Step log: {step_log_file}")
                step_results[step_name] = "SUCCESS"
            else:
                logger.error(f"✗ {step_name} FAILED with return code {result.returncode}")
                logger.error(f"  Check step log: {step_log_file}")
                step_results[step_name] = "FAILED"
                return False
                
        except Exception as e:
            logger.error(f"✗ {step_name} FAILED with exception: {e}")
            step_results[step_name] = "EXCEPTION"
            return False
    
    # Calculate total duration
    total_duration = (datetime.now() - pipeline_start).total_seconds()
    
    logger.info(f"\n{'='*60}")
    logger.info("PIPELINE COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total time: {total_duration:.1f} seconds")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Logs directory: {log_dir}")
    
    # Log summary
    logger.info("\nPIPELINE SUMMARY:")
    for step, result in step_results.items():
        logger.info(f"  {step}: {result}")
    
    # Check outputs
    logger.info("\nOUTPUT VERIFICATION:")
    
    # Check database
    try:
        from arango import ArangoClient
        client = ArangoClient(hosts='http://192.168.1.69:8529')
        db = client.db('information_reconstructionism', username='root', password='')
        
        if db.has_collection('papers'):
            papers_count = db.collection('papers').count()
            logger.info(f"  Papers in database: {papers_count}")
            
            # Check for full content
            query = """
            FOR paper IN papers
                FILTER paper.has_full_content == true
                COLLECT WITH COUNT INTO full_content_count
                RETURN full_content_count
            """
            cursor = db.aql.execute(query)
            full_content_count = list(cursor)[0] if cursor else 0
            logger.info(f"  Papers with full content: {full_content_count}")
        
        if db.has_collection('semantic_similarity'):
            edges_count = db.collection('semantic_similarity').count()
            logger.info(f"  Similarity edges in database: {edges_count}")
            
    except Exception as e:
        logger.error(f"  Could not verify database: {e}")
    
    # Create analysis scripts
    create_analysis_scripts(results_dir, logger)
    
    logger.info("\n✓ Pipeline run complete!")
    logger.info(f"Main log: {log_file}")
    
    return True

def create_analysis_scripts(results_dir, logger):
    """Create analysis scripts for the results"""
    
    analysis_dir = os.path.join(results_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Create Wolfram validation script
    wolfram_script = os.path.join(analysis_dir, "run_wolfram_validation.py")
    with open(wolfram_script, 'w') as f:
        f.write("""#!/usr/bin/env python3
import sys
import os
sys.path.append('/home/todd/reconstructionism/validation/experiment_1/wolfram')
from context_amplification_validation import run_validation

if __name__ == "__main__":
    run_validation()
""")
    os.chmod(wolfram_script, 0o755)
    
    logger.info(f"\nCreated analysis scripts in: {analysis_dir}")
    logger.info(f"  - {wolfram_script}: Run Wolfram validation on results")

if __name__ == "__main__":
    # Get number of papers from command line or use default
    num_papers = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    
    if num_papers < 1:
        print("Error: Number of papers must be positive")
        sys.exit(1)
    
    # Run the pipeline
    success = run_pipeline(num_papers)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)