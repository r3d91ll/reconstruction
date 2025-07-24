#!/bin/bash
# Script to run step 3 similarity computation with CPU or GPU

# Default to GPU version
USE_GPU=${USE_GPU:-true}

# Set environment variables
export EXP2_DB_NAME="information_reconstructionism_exp2"
export EXP2_RESULTS_DIR="./validation/experiment_2/results"

# ArangoDB connection details
# Allow ARANGO_HOST to be configured via environment variable with default fallback
export ARANGO_HOST="${ARANGO_HOST:-http://192.168.1.69:8529}"
export ARANGO_USERNAME="root"
# You'll need to set ARANGO_PASSWORD manually or source it from a secure location

# GPU-specific settings
export CUDA_VISIBLE_DEVICES="0,1"  # Use both A6000 GPUs
export EXP2_USE_FP16="true"        # Use float16 for memory efficiency
export EXP2_BATCH_SIZE="10000"     # Larger batches for GPU
export EXP2_SIM_THRESHOLD="0.5"

echo "=========================================="
echo "Running Step 3: Chunk Similarity Computation"
echo "=========================================="

if [ "$USE_GPU" = "true" ]; then
    echo "Using GPU-accelerated version with 2x A6000 (48GB each)"
    echo "GPUs: $CUDA_VISIBLE_DEVICES"
    echo ""
    
    # First run a quick GPU test
    echo "Testing GPU setup..."
    python3 validation/experiment_2/pipeline/test_gpu_similarity.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "GPU test passed. Running similarity computation..."
        python3 validation/experiment_2/pipeline/step3_compute_chunk_similarity_gpu.py
    else
        echo "GPU test failed. Please check your CUDA installation."
        exit 1
    fi
else
    echo "Using CPU version (slower)"
    echo ""
    
    # Smaller batch size for CPU
    export EXP2_BATCH_SIZE="1000"
    export EXP2_CHUNK_BATCH_SIZE="100"
    
    python3 validation/experiment_2/pipeline/step3_compute_chunk_similarity.py
fi

echo ""
echo "Step 3 complete. Check results in: $EXP2_RESULTS_DIR/chunk_similarity_summary*.json"