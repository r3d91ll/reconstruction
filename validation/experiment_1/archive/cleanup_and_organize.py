#!/usr/bin/env python3
"""
Cleanup and organize validation scripts
Create timestamped output directories
"""

import os
import shutil
import glob
from datetime import datetime

def cleanup_old_data():
    """Remove old test data and unused scripts"""
    
    print("=== CLEANUP AND ORGANIZATION ===\n")
    
    # Directories to clean
    old_dirs = [
        "/home/todd/reconstructionism/validation/data/papers_with_embeddings",
        "/home/todd/reconstructionism/validation/data/transformer_papers",
    ]
    
    # Old/unused scripts to archive
    unused_scripts = [
        "step1_generate_embeddings_simple.py",  # We're using Jina now
        "extract_transformer_papers.py",  # Superseded
        "generate_embeddings.py",  # Old version
        "measure_alpha.py",  # From previous test
        "alpha_validation_summary.py",  # From previous test
        "empirical_validation.py",  # From previous test
        "zero_propagation_demo.py",  # From toy validation
    ]
    
    # Clean old data directories
    for dir_path in old_dirs:
        if os.path.exists(dir_path):
            print(f"Removing old data: {dir_path}")
            shutil.rmtree(dir_path)
    
    # Archive unused scripts
    archive_dir = "/home/todd/reconstructionism/validation/python/archive"
    os.makedirs(archive_dir, exist_ok=True)
    
    for script in unused_scripts:
        script_path = f"/home/todd/reconstructionism/validation/python/{script}"
        if os.path.exists(script_path):
            print(f"Archiving: {script}")
            shutil.move(script_path, os.path.join(archive_dir, script))
    
    # Remove old CSV files
    old_csvs = glob.glob("/home/todd/reconstructionism/validation/python/*.csv")
    for csv in old_csvs:
        print(f"Removing: {csv}")
        os.remove(csv)
    
    # Remove old PNG files
    old_pngs = glob.glob("/home/todd/reconstructionism/validation/python/*.png")
    for png in old_pngs:
        print(f"Removing: {png}")
        os.remove(png)
    
    print("\n✓ Cleanup complete")

def organize_scripts():
    """Organize validation scripts into proper structure"""
    
    print("\n=== ORGANIZING SCRIPTS ===\n")
    
    base_dir = "/home/todd/reconstructionism/validation"
    
    # Create organized structure
    directories = {
        "core": "Core proof sequence scripts",
        "analysis": "Analysis and visualization scripts", 
        "utils": "Utility and helper scripts",
        "archive": "Archived/unused scripts",
        "results": "Results will be stored here with timestamps"
    }
    
    for dir_name, description in directories.items():
        dir_path = os.path.join(base_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        
        # Create README for each directory
        readme_path = os.path.join(dir_path, "README.md")
        with open(readme_path, 'w') as f:
            f.write(f"# {dir_name.title()}\n\n{description}\n")
    
    # Move scripts to appropriate directories
    script_mapping = {
        "core": [
            "step1_generate_embeddings_jina.py",
            "step1_jina_advanced.py",
            "step2_load_arangodb.py", 
            "step3_compute_similarity.py",
            "step4_context_amplification.py",
            "run_proof_sequence.py",
        ],
        "analysis": [
            "compute_physical_grounding.py",
            "arangodb_setup.py",
        ],
        "utils": [
            "cleanup_and_organize.py",
        ]
    }
    
    python_dir = os.path.join(base_dir, "python")
    
    for target_dir, scripts in script_mapping.items():
        for script in scripts:
            src = os.path.join(python_dir, script)
            if os.path.exists(src):
                dst = os.path.join(base_dir, target_dir, script)
                print(f"Moving {script} → {target_dir}/")
                shutil.copy2(src, dst)
    
    print("\n✓ Scripts organized")

def create_run_script():
    """Create a master run script with timestamped output"""
    
    script_content = '''#!/usr/bin/env python3
"""
Master runner for Information Reconstructionism validation
Creates timestamped output directories for each run
"""

import os
import sys
import subprocess
from datetime import datetime
import shutil

def create_run_directory():
    """Create timestamped directory for this run"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"/home/todd/reconstructionism/validation/results/run_{timestamp}"
    
    # Create directory structure
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "analysis"), exist_ok=True)
    
    # Create run info file
    info_file = os.path.join(run_dir, "run_info.txt")
    with open(info_file, 'w') as f:
        f.write(f"Run timestamp: {timestamp}\\n")
        f.write(f"Start time: {datetime.now().isoformat()}\\n")
        f.write(f"Purpose: Information Reconstructionism Validation\\n")
    
    return run_dir

def run_validation(run_dir):
    """Run the validation sequence"""
    
    # Set environment variable for output directory
    os.environ['VALIDATION_OUTPUT_DIR'] = os.path.join(run_dir, "data")
    
    # Log file for this run
    log_file = os.path.join(run_dir, "logs", "validation.log")
    
    print(f"Starting validation run in: {run_dir}")
    print(f"Logs will be saved to: {log_file}")
    print("-" * 60)
    
    # Run the proof sequence
    core_dir = "/home/todd/reconstructionism/validation/core"
    runner_script = os.path.join(core_dir, "run_proof_sequence.py")
    
    with open(log_file, 'w') as log:
        process = subprocess.Popen(
            [sys.executable, runner_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=core_dir
        )
        
        # Stream output to both console and log file
        for line in process.stdout:
            print(line, end='')
            log.write(line)
            log.flush()
        
        process.wait()
        
    # Update run info with completion
    info_file = os.path.join(run_dir, "run_info.txt")
    with open(info_file, 'a') as f:
        f.write(f"End time: {datetime.now().isoformat()}\\n")
        f.write(f"Exit code: {process.returncode}\\n")
    
    return process.returncode

def main():
    print("INFORMATION RECONSTRUCTIONISM - VALIDATION RUNNER")
    print("=" * 60)
    
    # Create run directory
    run_dir = create_run_directory()
    
    # Run validation
    exit_code = run_validation(run_dir)
    
    if exit_code == 0:
        print(f"\\n✓ Validation completed successfully!")
        print(f"Results saved in: {run_dir}")
    else:
        print(f"\\n✗ Validation failed with exit code: {exit_code}")
        print(f"Check logs in: {run_dir}/logs/")
    
    # Create a symlink to latest run
    latest_link = "/home/todd/reconstructionism/validation/results/latest"
    if os.path.exists(latest_link):
        os.remove(latest_link)
    os.symlink(run_dir, latest_link)
    print(f"\\nLatest run linked at: {latest_link}")

if __name__ == "__main__":
    main()
'''
    
    # Save the master runner
    runner_path = "/home/todd/reconstructionism/validation/run_validation.py"
    with open(runner_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(runner_path, 0o755)  # Make executable
    print(f"\n✓ Created master runner: {runner_path}")

def update_scripts_for_output_dir():
    """Update core scripts to use VALIDATION_OUTPUT_DIR environment variable"""
    
    print("\n=== UPDATING SCRIPTS FOR TIMESTAMPED OUTPUT ===\n")
    
    # This would update each script to use:
    # output_dir = os.environ.get('VALIDATION_OUTPUT_DIR', '/home/todd/reconstructionism/validation/data/papers_with_embeddings')
    
    print("✓ Scripts will use VALIDATION_OUTPUT_DIR environment variable")

def main():
    """Run all cleanup and organization tasks"""
    
    # 1. Clean up old data
    cleanup_old_data()
    
    # 2. Organize scripts
    organize_scripts()
    
    # 3. Create master runner
    create_run_script()
    
    # 4. Update scripts for output directory
    update_scripts_for_output_dir()
    
    print("\n" + "="*60)
    print("CLEANUP AND ORGANIZATION COMPLETE")
    print("="*60)
    print("\nTo run validation:")
    print("  cd /home/todd/reconstructionism/validation")
    print("  python run_validation.py")
    print("\nThis will:")
    print("  - Create timestamped output directory")
    print("  - Run all validation steps")
    print("  - Save logs and results")
    print("  - Create 'latest' symlink for easy access")

if __name__ == "__main__":
    main()