#!/bin/bash
# Run Docling extraction in a stable session
# Usage: ./run_docling_extraction.sh [num_papers]

NUM_PAPERS=${1:-1000}
LOG_FILE="docling_extraction_$(date +%Y%m%d_%H%M%S).log"

echo "Starting Docling extraction for $NUM_PAPERS papers"
echo "Log file: $LOG_FILE"
echo "Run this in screen/tmux for stability!"
echo ""
echo "Suggested commands:"
echo "  screen -S docling"
echo "  ./run_docling_extraction.sh 4000"
echo "  Ctrl+A, D to detach"
echo "  screen -r docling to reattach"
echo ""
echo "Starting in 5 seconds..."
sleep 5

# Run with unbuffered output for real-time logging
python3 -u extract_pdfs_docling.py \
    --papers-dir /home/todd/olympus/Erebus/unstructured/papers \
    --limit $NUM_PAPERS \
    2>&1 | tee $LOG_FILE