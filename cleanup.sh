#!/bin/bash

# Function to print messages with timestamps
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Cleanup function
cleanup() {
    log_message "Starting cleanup process..."


    # Remove llama.cpp repository if it exists
    if [ -d llama.cpp ]; then
        log_message "Removing llama.cpp repository..."
        rm -rf llama.cpp
    else
        log_message "llama.cpp repository not found. Skipping..."
    fi

    # Remove quantized model folders (only those created by EasyQuant)
    if ls ./*-GGUF ./*-GPTQ ./*-AWQ 1> /dev/null 2>&1; then
        log_message "Removing quantized model folders..."
        rm -rf ./*-GGUF ./*-GPTQ ./*-AWQ
    else
        log_message "No quantized model folders found. Skipping..."
    fi

    # Remove downloaded model folders (assuming they're in the current directory)
    # Be cautious with this command as it removes all directories
    find . -maxdepth 1 -type d ! -name '.' ! -name '.git' -exec rm -rf {} +

    # Remove Conda environment
    if conda info --envs | grep -q "easyquant"; then
        log_message "Removing Conda environment 'easyquant'..."
        conda env remove -n easyquant -y
    else
        log_message "Conda environment 'easyquant' not found. Skipping..."
    fi

    log_message "Cleanup completed."
}

# Run the cleanup function
cleanup