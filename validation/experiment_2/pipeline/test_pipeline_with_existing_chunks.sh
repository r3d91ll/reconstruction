#!/bin/bash
# Test pipeline with existing 5-paper chunks

echo "Testing experiment_2 pipeline with existing chunks..."

# Check if chunks directory exists
# Allow chunks directory to be configured via environment variable with fallback to default
CHUNKS_DIR="${CHUNKS_DIR:-../results/exp2_run_5_20250724_013320/chunks/}"
if [ ! -d "$CHUNKS_DIR" ]; then
    echo "Error: Chunks directory does not exist at $CHUNKS_DIR"
    echo "Please ensure the chunks directory exists before running this script."
    exit 1
fi

echo "Using chunks from: $CHUNKS_DIR"

# Set environment variables
# Use relative path or environment variable for results directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXP2_RESULTS_DIR="${EXP2_RESULTS_DIR:-$SCRIPT_DIR/../results/exp2_run_5_20250724_013320}"
export EXP2_DB_NAME="information_reconstructionism_exp2"
export EXP2_BATCH_SIZE="1000"
export EXP2_SIM_THRESHOLD="0.5"
export EXP2_AGG_METHOD="max"

# Skip step1 (chunks already generated) and run steps 2-5
echo "Step 1: Skipping - chunks already generated"

echo -e "\n=== Step 2: Load chunks to database ==="
python3 step2_load_chunks_to_db.py
if [ $? -ne 0 ]; then
    echo "Step 2 failed!"
    exit 1
fi

echo -e "\n=== Step 3: Compute chunk similarities ==="
python3 step3_compute_chunk_similarity.py
if [ $? -ne 0 ]; then
    echo "Step 3 failed!"
    exit 1
fi

echo -e "\n=== Step 4: Aggregate document similarities ==="
python3 step4_aggregate_doc_similarity.py
if [ $? -ne 0 ]; then
    echo "Step 4 failed!"
    exit 1
fi

echo -e "\n=== Step 5: Multi-scale analysis ==="
python3 step5_multiscale_analysis.py
if [ $? -ne 0 ]; then
    echo "Step 5 failed!"
    exit 1
fi

echo -e "\n✓ All steps completed successfully!"
echo "Results in: $EXP2_RESULTS_DIR"