#!/bin/bash

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Conda and try again."
    exit 1
fi

# Check if required arguments are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 MODEL_ID USERNAME HF_TOKEN [--method QUANTIZATION_METHOD]"
    exit 1
fi

MODEL_ID="$1"
USERNAME="$2"
HF_TOKEN="$3"
QUANTIZATION_METHOD="gptq"  # Default method

# Parse optional argument
if [ "$4" = "--method" ] && [ -n "$5" ]; then
    QUANTIZATION_METHOD="$5"
fi

# Set up Conda environment
CONDA_ENV_NAME="easyquant"
PYTHON_VERSION="3.10"

# Create Conda environment if it doesn't exist
if ! conda info --envs | grep -q "$CONDA_ENV_NAME"; then
    echo "Creating Conda environment: $CONDA_ENV_NAME"
    conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
else
    echo "Conda environment $CONDA_ENV_NAME already exists"
fi

# Activate Conda environment
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"

# Install dependencies
pip install -r requirements.txt

# Run the Python script
CUDA_VISIBLE_DEVICES=0 python easyquant.py "$MODEL_ID" "$USERNAME" "$HF_TOKEN" --method "$QUANTIZATION_METHOD"

# Deactivate Conda environment
conda deactivate

echo "Please run the cleanup script to remove unnecessary files and folders."
echo "EasyQuant process completed."