#!/bin/bash
# Test pipeline with existing 5-paper chunks

echo "Testing experiment_2 pipeline with existing chunks..."
echo "Using chunks from: ../results/exp2_run_5_20250724_013320/chunks/"

# Set environment variables
export EXP2_RESULTS_DIR="/home/todd/reconstructionism/validation/experiment_2/results/exp2_run_5_20250724_013320"
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