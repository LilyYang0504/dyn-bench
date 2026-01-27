import os
import argparse
import subprocess
import yaml
from pathlib import Path
from colorama import Fore, init

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


def download_model(model_name: str, local_dir: str):
    """Download model using huggingface-cli to a flat directory structure"""
    
    print(f"{Fore.CYAN}Start to download model: {model_name}")
    print(f"{Fore.CYAN}Target directory: {local_dir}")
    
    model_type = get_model_type(model_name)
    print(f"{Fore.CYAN}Detected model type: {model_type}")
    
    os.makedirs(local_dir, exist_ok=True)
    
    cmd = [
        "huggingface-cli",
        "download",
        model_name,
        "--local-dir", local_dir
    ]
    
    try:
        print(f"{Fore.YELLOW}Running: {' '.join(cmd)}")
        
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env
        )
        
        print(f"{Fore.GREEN}Model downloaded successfully: {model_name}")
        print(f"{Fore.GREEN}Model saved to: {local_dir}")
        
        if result.stdout:
            print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"{Fore.RED}ERROR: Failed to download model: {model_name}")
        if e.stderr:
            try:
                print(f"{Fore.RED}{e.stderr}")
            except:
                print(f"{Fore.RED}Error output contains undecodable characters")
        raise RuntimeError(f"Download failed for {model_name}")
    except FileNotFoundError:
        raise RuntimeError(
            f"{Fore.RED}huggingface-cli not found. Please install: pip install huggingface_hub[cli]"
        )


def load_config():
    """Load config.yaml to get default cache_dir"""
    config_path = Path(__file__).parent / "conf" / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('model', {}).get('cache_dir', None)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace models to local directory using huggingface-cli",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        nargs='+',
        required=True,
        help='model name(s) on HuggingFace, e.g., Qwen/Qwen3-VL-8B-Instruct'
    )
    
    parser.add_argument(
        '--cache-dir', '-c',
        type=str,
        default=None,
        help='Base directory for models (overrides config.yaml). Each model will be saved to {cache_dir}/{model_name}'
    )
    
    args = parser.parse_args()
    models = args.model
    
    cache_dir = args.cache_dir or load_config() or "./models"
    
    print(f"{Fore.CYAN}Base cache directory: {cache_dir}")
    print(f"{Fore.CYAN}Models to download: {len(models)}")
    print()
    
    for i, model_name in enumerate(models, 1):
        print(f"{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}[{i}/{len(models)}] Processing: {model_name}")
        print(f"{Fore.CYAN}{'='*60}")
        
        model_dir_name = model_name.replace('/', '--')
        local_dir = os.path.join(cache_dir, model_dir_name)
        
        try:
            download_model(model_name, local_dir)
            print()
        except Exception as e:
            print(f"{Fore.YELLOW}WARN: Skipping model {model_name}: {e}")
            print()
            continue
    
    print(f"{Fore.GREEN}{'='*60}")
    print(f"{Fore.GREEN}All models processed successfully")
    print(f"{Fore.GREEN}{'='*60}")


if __name__ == "__main__":
    main()
