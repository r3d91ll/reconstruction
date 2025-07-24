#!/usr/bin/env python3
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
