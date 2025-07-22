#!/usr/bin/env python3
"""
Run pipeline with configurable number of papers
"""

import os
import sys
import subprocess
from datetime import datetime

def update_step1_for_run(num_papers):
    """Update step1 to process specified number of papers"""
    
    # Read the current step1 script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    step1_path = os.path.join(script_dir, "step1_generate_embeddings_jina.py")
    with open(step1_path, 'r') as f:
        content = f.read()
    
    # Update limit
    test_content = content.replace('limit = 100  # Start small for testing', 
                                  f'limit = {num_papers}  # Pipeline run - {num_papers} papers')
    
    # Save as run version
    run_path = os.path.join(script_dir, f"step1_run_{num_papers}.py")
    with open(run_path, 'w') as f:
        f.write(test_content)
    
    return run_path

def run_pipeline(num_papers=10):
    """Run the pipeline with specified number of papers"""
    
    print("=" * 60)
    print(f"PIPELINE RUN - {num_papers} PAPERS")
    print("=" * 60)
    print(f"Start time: {datetime.now()}")
    print()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = f"/home/todd/reconstructionism/validation/experiment_1/results/run_{num_papers}_{timestamp}"
    os.makedirs(test_dir, exist_ok=True)
    os.environ["VALIDATION_OUTPUT_DIR"] = os.path.join(test_dir, "data/papers_with_embeddings")
    
    # Create/update the run script
    test_script_path = update_step1_for_run(num_papers)
    print(f"Created run script: {test_script_path}")
    
    # Define pipeline steps
    scripts = [
        ("Step 1", test_script_path, f"Generate embeddings for {num_papers} papers"),
        ("Step 2", "step2_load_arangodb.py", "Load into ArangoDB"),
        ("Step 3", "step3_compute_similarity.py", "Compute similarities (GPU accelerated)"),
        ("Step 4", "step4_context_amplification_batch.py", "Apply Context^1.5"),
    ]
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run each step
    for step_name, script, description in scripts:
        print(f"\n{'='*50}")
        print(f"{step_name}: {description}")
        print(f"{'='*50}")
        
        start_time = datetime.now()
        
        try:
            # Run the script
            result = subprocess.run(
                [sys.executable, script],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Print output
            print(result.stdout)
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            print(f"\n✓ {step_name} completed in {duration:.1f} seconds")
            
        except subprocess.CalledProcessError as e:
            print(f"\n✗ {step_name} FAILED")
            print("Error:", e.stderr)
            print("Output:", e.stdout)
            return False
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"Total time: {(datetime.now() - start_time).total_seconds():.1f} seconds")
    print(f"Results in: {test_dir}")
    print(f"{'='*60}")
    
    print("\n✓ Pipeline successful!")
    
    # Cleanup temporary script
    os.remove(test_script_path)
    
    return True

if __name__ == "__main__":
    # Get number of papers from command line or use default
    num_papers = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    
    if num_papers < 1:
        print("Error: Number of papers must be positive")
        sys.exit(1)
    
    # Run the pipeline
    success = run_pipeline(num_papers)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)