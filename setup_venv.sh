#!/bin/bash
# Setup virtual environment for Information Reconstructionism infrastructure

echo "Setting up virtual environment for Information Reconstructionism..."

# Create virtual environment in project root
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install infrastructure requirements
echo "Installing infrastructure requirements..."
pip install -r infrastructure_setup/requirements.txt

# Install the irec_infrastructure package in development mode
echo "Installing irec_infrastructure package..."
pip install -e .

# Create .env file template if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env template..."
    cat > .env << EOL
# Information Reconstructionism Environment Variables

# Jina API (if using cloud API)
JINA_API_KEY=your_api_key_here

# Database
ARANGO_HOST=localhost
ARANGO_PORT=8529
ARANGO_USERNAME=root
ARANGO_PASSWORD=your_password_here
ARANGO_DATABASE=information_reconstructionism

# GPU Settings
USE_GPU=true
GPU_DEVICES=0,1
USE_FP16=true

# Processing
BATCH_SIZE=32
CHUNK_SIZE=1024
CHUNK_OVERLAP=200

# Paths
ARXIV_DATA_PATH=/mnt/data/arxiv_data/pdf
OUTPUT_PATH=./processed_documents_local
EOL
fi

echo ""
echo "Virtual environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To test the GPU setup, run:"
echo "  python infrastructure_setup/test_local_gpu_simple.py"
echo ""
echo "To process documents with local GPU, run:"
echo "  python infrastructure_setup/process_documents_local_gpu.py --input-dir /mnt/data/arxiv_data/pdf --output-dir ./processed_documents_local --num-docs 1960"