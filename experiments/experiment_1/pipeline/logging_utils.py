#!/usr/bin/env python3
"""
Logging utilities for experiment_1 pipeline
Ensures all script outputs are captured with timestamps
"""

import os
import sys
import logging
from datetime import datetime
import functools

def setup_logging(script_name, log_dir=None):
    """
    Set up logging for a pipeline script.
    
    Args:
        script_name: Name of the script (e.g., "step1_embeddings")
        log_dir: Directory for log files (default: ../logs/)
    
    Returns:
        logger: Configured logger instance
        log_file: Path to the log file
    """
    # Default log directory
    if log_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(os.path.dirname(script_dir), "logs")
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{script_name}_{timestamp}.log"
    log_file = os.path.join(log_dir, log_filename)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)  # Also print to console
        ]
    )
    
    logger = logging.getLogger(script_name)
    
    # Log startup info
    logger.info("="*60)
    logger.info(f"Starting {script_name}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info("="*60)
    
    return logger, log_file

def log_function_call(logger):
    """Decorator to log function calls and results."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"Calling {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.info(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed: {str(e)}", exc_info=True)
                raise
        return wrapper
    return decorator

def capture_subprocess_output(logger, command, description=None):
    """
    Run a subprocess and capture its output to the logger.
    
    Args:
        logger: Logger instance
        command: Command to run (list)
        description: Optional description of the command
    
    Returns:
        returncode: Process return code
    """
    import subprocess
    
    if description:
        logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(command)}")
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Stream output line by line
    for line in iter(process.stdout.readline, ''):
        if line:
            logger.info(f"  > {line.rstrip()}")
    
    process.wait()
    
    if process.returncode == 0:
        logger.info(f"Command completed successfully (return code: 0)")
    else:
        logger.error(f"Command failed (return code: {process.returncode})")
    
    return process.returncode

def log_summary(logger, stats_dict):
    """Log a summary of statistics."""
    logger.info("\nSUMMARY STATISTICS:")
    logger.info("-" * 40)
    for key, value in stats_dict.items():
        logger.info(f"{key}: {value}")
    logger.info("-" * 40)

def create_pipeline_log_summary(log_dir):
    """Create a summary of all logs in the directory."""
    summary_file = os.path.join(log_dir, "pipeline_summary.txt")
    
    with open(summary_file, 'w') as f:
        f.write("EXPERIMENT_1 PIPELINE LOG SUMMARY\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("="*60 + "\n\n")
        
        # List all log files
        log_files = sorted(glob.glob(os.path.join(log_dir, "*.log")))
        
        for log_file in log_files:
            f.write(f"\nLog: {os.path.basename(log_file)}\n")
            f.write("-"*40 + "\n")
            
            # Extract key information
            with open(log_file, 'r') as lf:
                lines = lf.readlines()
                
                # Find errors
                errors = [l for l in lines if 'ERROR' in l]
                if errors:
                    f.write(f"ERRORS FOUND: {len(errors)}\n")
                    for error in errors[:3]:  # First 3 errors
                        f.write(f"  - {error.strip()}\n")
                else:
                    f.write("No errors found\n")
                
                # Find completion status
                completed = any('completed successfully' in l for l in lines)
                f.write(f"Status: {'COMPLETED' if completed else 'INCOMPLETE'}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("END OF SUMMARY\n")
    
    return summary_file

# Example usage in a script:
if __name__ == "__main__":
    # Test the logging setup
    logger, log_file = setup_logging("test_logging")
    
    logger.info("This is a test message")
    logger.warning("This is a warning")
    logger.error("This is an error")
    
    # Test subprocess capture
    capture_subprocess_output(logger, ["echo", "Hello from subprocess"], "Test echo command")
    
    # Test summary
    log_summary(logger, {
        "Papers processed": 100,
        "Embeddings generated": 100,
        "Errors": 0,
        "Duration": "5.2 minutes"
    })
    
    print(f"\nLog file created: {log_file}")