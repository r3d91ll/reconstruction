#!/bin/bash
# Launch the GPU-accelerated experiment 2 pipeline

# Set environment variables
export ARANGO_HOST="http://192.168.1.69:8529"
export ARANGO_USERNAME="root"
# IMPORTANT: Set ARANGO_PASSWORD before running:
# export ARANGO_PASSWORD="your_password_here"

# Check if password is set
if [ -z "$ARANGO_PASSWORD" ]; then
    echo "ERROR: ARANGO_PASSWORD environment variable must be set"
    echo "Run: export ARANGO_PASSWORD='your_password_here'"
    exit 1
fi

# Check for CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. CUDA/GPU drivers may not be installed."
    exit 1
fi

# Display GPU info
echo "🚀 GPU Configuration:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Number of papers to process (default: 2000)
NUM_PAPERS=${1:-2000}

echo ""
echo "=========================================="
echo "GPU-ACCELERATED EXPERIMENT 2 PIPELINE"
echo "=========================================="
echo "Processing $NUM_PAPERS papers"
echo "Database: information_reconstructionism_exp2_gpu"
echo "Using GPUs: 0,1 (both A6000s with NVLink)"
echo ""
echo "This pipeline will:"
echo "1. Extract PDFs with GPU-accelerated Docling"
echo "2. Generate embeddings with Jina on GPU"
echo "3. Load to database with GPU preprocessing"
echo "4. Compute similarities using both GPUs"
echo "5. Aggregate documents on GPU"
echo "6. Perform multiscale analysis on GPU"
echo ""
echo "Estimated time: 30-60 minutes (vs 4-6 hours on CPU)"
echo ""
read -p "Press Enter to start or Ctrl+C to cancel..."

# Run the pipeline
cd "$(dirname "$0")"
python3 run_experiment2_gpu_pipeline.py $NUM_PAPERS

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ GPU Pipeline completed successfully!"
    echo "Check results in: validation/experiment_2/results/"
else
    echo ""
    echo "❌ Pipeline failed. Check logs for details."
    exit 1
fi