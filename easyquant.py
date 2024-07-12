import argparse
import os
from huggingface_hub import login, create_repo, HfApi, ModelCard
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import huggingface_hub

def set_cache_dir():
    cache_dir = os.path.expanduser("~/.cache/huggingface/easyquant")
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_HOME"] = cache_dir
    huggingface_hub.constants.HUGGINGFACE_HUB_CACHE = cache_dir
    return cache_dir

def download_model(model_id):
    os.system(f"git clone https://huggingface.co/{model_id}")
    return model_id.split('/')[-1]

def quantize_gguf(model_name, quantization_format="q4_k_m"):
    os.system("git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make")
    os.system("pip install -r llama.cpp/requirements.txt")
    
    fp16 = f"{model_name}/{model_name.lower()}.fp16.bin"
    os.system(f"python llama.cpp/convert_hf_to_gguf.py {model_name} --outtype f16 --outfile {fp16}")
    
    qtype = f"{model_name}/{model_name.lower()}.{quantization_format.upper()}.gguf"
    os.system(f"./llama.cpp/llama-quantize {fp16} {qtype} {quantization_format}")
    
    return f"{model_name}-GGUF"

def quantize_gptq(model_id, bits=4, group_size=128, damp_percent=0.1):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    quantization_config = GPTQConfig(bits=bits, dataset="c4", tokenizer=tokenizer, group_size=group_size, damp_percent=damp_percent)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quantization_config, low_cpu_mem_usage=True)
    
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

    model = AutoAWQForCausalLM.from_pretrained(model_name, safetensors=True, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.quantize(tokenizer, quant_config=quant_config)

    model.save_quantized(save_folder)
    tokenizer.save_pretrained(save_folder)
    
    return save_folder

def upload_model(username, model_name, quantization_method, save_folder, hf_token):
    api = HfApi()
    
    card = ModelCard.load(model_name)
    card.data.tags.append("easyquant")
    card.data.tags.append(quantization_method.lower())
    card.save(f'{save_folder}/README.md')

    create_repo(
        repo_id=f"{username}/{model_name}-{quantization_method}",
        repo_type="model",
        exist_ok=True,
        token=hf_token
    )
    api.upload_folder(
        folder_path=save_folder,
        repo_id=f"{username}/{model_name}-{quantization_method}",
        token=hf_token
    )

def main():
    parser = argparse.ArgumentParser(description="EasyQuant: Simplified Quantization Tool")
    parser.add_argument("model_id", type=str, help="Hugging Face model ID")
    parser.add_argument("username", type=str, help="Hugging Face username")
    parser.add_argument("hf_token", type=str, help="Hugging Face API token")
    parser.add_argument("--method", type=str, choices=["gguf", "gptq", "awq"], default="gptq", help="Quantization method")
    args = parser.parse_args()

    cache_dir = set_cache_dir()
    print(f"Using cache directory: {cache_dir}")

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