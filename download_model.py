import os
import argparse
from colorama import Fore, init
from huggingface_hub import snapshot_download
from transformers import (
    AutoModel, AutoTokenizer, AutoProcessor, AutoModelForCausalLM,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
init(autoreset=True)


def get_model_type(model_name: str) -> str:
    model_name_lower = model_name.lower()
    
    if "bytedance/sa2va" in model_name_lower:
        if "qwen3-vl" in model_name_lower:
            return "sa2va_qwen3"
        elif "qwen2_5-vl" in model_name_lower or "qwen2.5-vl" in model_name_lower:
            return "sa2va_qwen2_5"
        elif "internvl3" in model_name_lower:
            return "sa2va_internvl3"
        else:
            return "sa2va"
    
    elif "polyu-chenlab/unipixel" in model_name_lower:
        return "unipixel"
    
    elif "opengvlab/internvl3_5" in model_name_lower or "opengvlab/internvl3.5" in model_name_lower:
        return "internvl3_5"
    elif "opengvlab/internvl3" in model_name_lower:
        return "internvl3"
    elif "qwen/qwen3-vl-235b" in model_name_lower:
        return "qwen3_vl_moe"
    elif "qwen/qwen3-vl" in model_name_lower:
        return "qwen3_vl"
    elif "qwen/qwen2.5-vl" in model_name_lower:
        return "qwen2_5_vl"
    elif "llava-onevision" in model_name_lower:
        return "llava_onevision"
    elif "vst-7b" in model_name_lower:
        return "vst"
    elif "spatial-ssrl" in model_name_lower:
        return "spatial_ssrl"
    elif "spatialladder" in model_name_lower:
        return "spatial_ladder"
    elif "spacer-sft" in model_name_lower:
        return "spacer_sft"
    else:
        raise ValueError(f"Unknown model type for: {model_name}")


def download_model(model_name: str, cache_dir: str = None):

    print(f"Start to download model: {model_name}")
    print(f"Cache directory: {cache_dir if cache_dir else 'HF default path'}")
    
    model_type = get_model_type(model_name)
    print(f"Detected model type: {model_type}")
    
    try:
        local_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            resume_download=True,
            local_files_only=False
        )
        
        print(f"{Fore.GREEN}Model downloaded successfully: {model_name}")
        
        print(f"{Fore.CYAN}Model saved to: {local_path}")
        
        
    except Exception as e:
        raise RuntimeError(f"Failed to download model: {model_name}\n{e}")


def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace models to local cache",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        nargs='+',
        required=True,
        help='model name(s) on HuggingFace, e.g., bytedance/sa2va-internvl3'
    )
    
    parser.add_argument(
        '--cache-dir', '-c',
        type=str,
        default=None,
        help='cache directory for models (optional)'
    )
    
    args = parser.parse_args()
    models = args.model
    cache_dir = args.cache_dir
    
    for i, model_name in enumerate(models, 1):
        print(f"{Fore.CYAN}[{i}/{len(models)}] Downloading model: {model_name}")
        try:
            download_model(model_name, cache_dir)
        except Exception as e:
            print(f"{Fore.YELLOW}WARN: Skipping model {model_name}")
            continue
    
    print(f"{Fore.GREEN}All models downloaded successfully")


if __name__ == "__main__":
    main()
