from typing import Dict, Any
import os
import torch
from colorama import Fore, init
from transformers import (
    AutoTokenizer, AutoModel, AutoProcessor, AutoModelForCausalLM,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
init(autoreset=True)


def is_flash_attn_available() -> bool:
    try:
        import flash_attn
        return True
    except ImportError:
        return False


def extract_model_name_from_path(path: str) -> str:
    import re
    
    match = re.search(r'models--([^/\\]+)--([^/\\]+)', path)
    if match:
        org, model = match.groups()
        return f"{org}/{model}"
    
    return os.path.basename(path.rstrip('/\\'))


def is_local_path(path: str) -> bool:
    if os.path.exists(path):
        return True
    
    if '\\' in path:
        return True
    if len(path) >= 2 and path[1] == ':':
        return True
    
    if path.startswith('./') or path.startswith('../'):
        return True
    
    if path.startswith('/'):
        return True
    
    if path.count('/') > 1 or 'models--' in path:
        return True

    return False


def get_model_type(model_name: str) -> str:
    import os
    import re
    
    original_name = model_name
    
    if os.path.sep in model_name or '\\' in model_name or '/' in model_name:
        match = re.search(r'models--([^\\\/]+)--([^\\\/]+)', model_name)
        if match:
            org = match.group(1)
            model = match.group(2)
            model_name = f"{org}/{model}"
        else:
            for org in ['OpenGVLab', 'Qwen', 'ByteDance', 'lmms-lab', 'rayruiyang', 'internlm', 'hongxingli', 'RUBBISHLIKE']:
                if org.lower() in model_name.lower():
                    parts = model_name.replace('\\', '/').split('/')
                    for i, part in enumerate(parts):
                        if org.lower() in part.lower() and i + 1 < len(parts):
                            model_name = f"{org}/{parts[i+1]}"
                            break
                    break
    
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
    
    elif "polyu-chenlab/unipixel" in model_name_lower or "unipixel" in model_name_lower:
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
        raise ValueError(f"Unknown model type for: {original_name}\nmodel name: {model_name}")


def load_model(config: Dict) -> Dict[str, Any]:
    model_name = config['model']['name']
    device = config['model']['device']
    cache_dir = config['model'].get('cache_dir', None)
    
    model_alias = config['model'].get('alias', None)
    
    local_files_only = is_local_path(model_name)
    
    if model_alias:
        print(f"Model alias: {model_name} -> {model_alias}")
        model_type = get_model_type(model_alias)
        print(f"Model type (from alias): {model_type}")
    else:
        model_type = get_model_type(model_name)
        print(f"Model type (auto-detected): {model_type}")
    torch_dtype_str = config['model'].get('torch_dtype', 'bfloat16')
    if torch_dtype_str == "auto":
        torch_dtype = "auto"
    else:
        torch_dtype = getattr(torch, torch_dtype_str)

    use_flash_attn = config['model'].get('use_flash_attn', True)
    flash_attn_available = is_flash_attn_available()
    
    if use_flash_attn and not flash_attn_available:
        print(f"{Fore.YELLOW}WARN: Flash Attention requested but not installed. Falling back to eager attention.")
        use_flash_attn = False
    
    attn_implementation = "flash_attention_2" if use_flash_attn else "eager"
    
    trust_remote_code = config['model'].get('trust_remote_code', True)

    model = None
    tokenizer = None
    processor = None
    supports_mask = False
    
    if model_type in ["sa2va", "sa2va_internvl3", "sa2va_qwen2_5", "sa2va_qwen3"]:
        supports_mask = True
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            local_files_only=local_files_only
        ).eval().cuda()
        if model_type in ["sa2va_qwen3", "sa2va_qwen2_5"]:
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir, local_files_only=local_files_only)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False, cache_dir=cache_dir, local_files_only=local_files_only)

    elif model_type in ["internvl3", "internvl3_5"]:
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=local_files_only
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False, cache_dir=cache_dir, local_files_only=local_files_only)

    elif model_type == "qwen2_5_vl":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )
        processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=local_files_only)

    elif model_type == "qwen3_vl":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )
        processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=local_files_only)

    elif model_type == "qwen3_vl_moe":
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )
        processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=local_files_only)

    elif model_type == "llava_onevision":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            device_map="auto",
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir, local_files_only=local_files_only)

    elif model_type == "vst":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )
        processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=256*28*28,
            max_pixels=1280*28*28,
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )

    elif model_type == "spatial_ssrl":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )
        processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=local_files_only)

    elif model_type == "spatial_ladder":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )
        processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=local_files_only)
    
    elif model_type == "spacer_sft":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )
        processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=local_files_only)
    
    elif model_type == "unipixel":
        from utils.unipixel_helper import load_unipixel_model
        supports_mask = True
        model, processor = load_unipixel_model(
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )
        tokenizer = None
    
    if model is None:
        raise ValueError(f"Failed to load model: {model_name}")
    
    if processor:
        print(f"{Fore.GREEN}Model and processor loaded successfully")
    else:
        print(f"{Fore.GREEN}Model and tokenizer loaded successfully")
    
    print(f"  - Precision: {torch_dtype_str}")
    print(f"  - Flash Attention: {attn_implementation} {f'{Fore.GREEN}(checked)' if attn_implementation == 'flash_attention_2' else f'{Fore.YELLOW}(eager fallback)'}")
    print(f"  - Supports Mask: {supports_mask}")
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'processor': processor,
        'model_type': model_type,
        'device': device,
        'supports_mask': supports_mask
    }

