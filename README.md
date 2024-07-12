# EasyQuant
Quantize 🤗 model to GGUF, GPTQ, and AWQ. A command-line version of [AutoQuant](https://colab.research.google.com/drive/1b6nqC7UZVt8bx4MksX7s656GXPM-eWw4?usp=sharing).


## Prerequisites

- Anaconda or Miniconda installed on your system
- Git LFS installed on your system

## Project Structure

```
EasyQuant/
│
├── easyquant.sh          # Main shell script to run the tool
├── cleanup.sh            # Cleanup script to remove temporary files and environments
├── easyquant.py          # Python script containing the core functionality
├── requirements.txt      # List of Python dependencies
└── README.md             # This file
```

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/EasyQuant.git
   cd EasyQuant
   ```

2. Make the shell scripts executable:
   ```
   chmod +x easyquant.sh cleanup.sh
   ```

## Usage

Run the script with the following command:

```
./easyquant.sh MODEL_ID USERNAME HF_TOKEN [--method QUANTIZATION_METHOD]
```

- `MODEL_ID`: The Hugging Face model ID (e.g., "mlabonne/Daredevil-8B")
- `USERNAME`: Your Hugging Face username
- `HF_TOKEN`: Your Hugging Face API token
- `QUANTIZATION_METHOD`: (Optional) One of "gguf", "gptq", or "awq" (default is "gptq")


This will:
1. Set up a Conda environment named "easyquant"
2. Install all necessary dependencies from `requirements.txt`
3. Download the specified model to a dedicated EasyQuant cache directory
4. Quantize it using the chosen method
5. Upload the quantized version to your Hugging Face account

## Cleanup

The `cleanup.sh` script is a utility script for cleaning up the artifacts. It performs the following cleanup operations:

- Removes the EasyQuant-specific cache directory
- Removes the llama.cpp repository if it exists
- Removes quantized model folders created by EasyQuant
- Removes the Conda environment

The cleanup process is non-interactive and focuses on removing only the resources created by EasyQuant, ensuring that other Hugging Face caches remain untouched.

To run the cleanup script, execute the following command:
```bash
./cleanup.sh
```

## Note

This tool requires significant computational resources, especially for larger models. Make sure you have sufficient GPU memory and storage space available.