import argparse
import os
import torch
from huggingface_hub import login, create_repo, HfApi, ModelCard
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import huggingface_hub
import subprocess

def download_model(model_id):
    model_name = model_id.split('/')[-1]
    if os.path.exists(model_name):
        print(f"Model directory '{model_name}' already exists. Skipping download.")
    else:
        print(f"Downloading model {model_id}...")
        result = subprocess.run(["git", "clone", f"https://huggingface.co/{model_id}"], check=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error downloading model: {result.stderr}")
            return None
    return model_name


def quantize_gguf(model_name, quantization_format="q4_k_m"):
    os.system("git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make")
    # os.system("pip install -r llama.cpp/requirements.txt")
    
    fp16 = f"{model_name}/{model_name.lower()}.fp16.bin"
    os.system(f"python llama.cpp/convert_hf_to_gguf.py {model_name} --outtype f16 --outfile {fp16}")
    
    qtype = f"{model_name}/{model_name.lower()}.{quantization_format.upper()}.gguf"
    os.system(f"./llama.cpp/llama-quantize {fp16} {qtype} {quantization_format}")
    
    return f"{model_name}"

def quantize_gptq(model_id, bits=4, group_size=128, damp_percent=0.1):
    model_path = model_id.split('/')[-1]
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    quantization_config = GPTQConfig(bits=bits, dataset="c4", tokenizer=tokenizer, group_size=group_size, damp_percent=damp_percent)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", quantization_config=quantization_config, low_cpu_mem_usage=True, local_files_only=True)
    
    save_folder = f"{model_id}-GPTQ"
    model.save_pretrained(save_folder, use_safetensors=True)
    tokenizer.save_pretrained(save_folder)
    
    return save_folder

def quantize_awq(model_name, bits=4, group_size=128, version="GEMM", zero_point=True):
    from awq import AutoAWQForCausalLM
    
    quant_config = {
        "w_bit": bits,
        "q_group_size": group_size,
        "version": version,
        "zero_point": zero_point
    }
    save_folder = f"{model_name}-AWQ"
    model_path = model_name

    model = AutoAWQForCausalLM.from_pretrained(model_path, safetensors=True, low_cpu_mem_usage=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model.quantize(tokenizer, quant_config=quant_config)

    model.save_quantized(save_folder)
    tokenizer.save_pretrained(save_folder)
    
    return save_folder

def upload_model(username, model_name, quantization_method, save_folder, hf_token):
    api = HfApi()
    
    card = ModelCard.load(f"{model_name}/README.md")
    card.data.tags.append("easyquant")
    card.data.tags.append(quantization_method.lower())
    card.save(f'{model_name}/README.md')
    
    repo_id = f"{username}/{model_name}-{quantization_method}"
    create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,
        token=hf_token
    )
    
    try:
        if quantization_method == "GGUF":
            save_folder = f"{model_name}"
            # For GGUF, we're uploading specific files
            api.upload_folder(
                folder_path=save_folder,
                repo_id=repo_id,
                allow_patterns=["*.gguf","*.md"],
                token=hf_token
            )
        else:
            # For other methods, we're uploading a folder
            save_folder = f"{model_name}-{quantization_method}"
            api.upload_folder(
                folder_path=save_folder,
                repo_id=repo_id,
                token=hf_token
            )
        print(f"Model successfully uploaded to {repo_id}")
    except Exception as e:
        print(f"Error uploading model: {e}")

def main():
    parser = argparse.ArgumentParser(description="EasyQuant: Simplified Quantization Tool")
    parser.add_argument("model_id", type=str, help="Hugging Face model ID")
    parser.add_argument("username", type=str, help="Hugging Face username")
    parser.add_argument("hf_token", type=str, help="Hugging Face API token")
    parser.add_argument("--method", type=str, choices=["gguf", "gptq", "awq"], default="gptq", help="Quantization method")
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. This may significantly slow down the quantization process.")
        print("If you have a CUDA-capable GPU, please ensure you have the correct drivers and CUDA toolkit installed.")
    else:
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")


    login(args.hf_token)
    
    model_name = download_model(args.model_id)
    
    if args.method == "gguf":
        save_folder = quantize_gguf(model_name)
    elif args.method == "gptq":
        save_folder = quantize_gptq(args.model_id)
    elif args.method == "awq":
        save_folder = quantize_awq(model_name)

    upload_model(args.username, model_name, args.method.upper(), save_folder, args.hf_token)
    print(f"Quantized model uploaded to: {args.username}/{model_name}-{args.method.upper()}")

if __name__ == "__main__":
    main()