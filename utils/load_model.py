from typing import Dict, Any
import torch
from transformers import AutoTokenizer, AutoModel, AutoProcessor


def get_model_type(model_name: str) -> str:
    """
    根据模型名称判断模型类型
    
    Args:
        model_name: 模型名称（如 ByteDance/Sa2VA-4B）
    
    Returns:
        str: 模型类型
            - "sa2va": Sa2VA-1B/4B/8B
            - "internvl3": Sa2VA-InternVL3-2B/8B/14B
            - "qwen2_5": Sa2VA-Qwen2_5-VL-3B/7B
            - "qwen3": Sa2VA-Qwen3-VL-2B/4B
    """

    model_name_lower = model_name.lower()
    
    if "qwen3-vl" in model_name_lower:
        return "qwen3"
    elif "qwen2_5-vl" in model_name_lower or "qwen2.5-vl" in model_name_lower:
        return "qwen2_5"
    elif "internvl3" in model_name_lower:
        return "internvl3"
    elif "sa2va" in model_name_lower:
        # 默认Sa2VA系列（1B/4B/8B）
        return "sa2va"
    else:
        raise ValueError(f"Unknown model type for: {model_name}")


def load_model(config: Dict) -> Dict[str, Any]:
    """
    加载Sa2VA系列模型
    
    根据不同的模型类型，加载对应的模型和tokenizer/processor
    
    Args:
        config: 配置字典，包含model字段
            - name: HF模型名称
            - device: 运行设备
            - torch_dtype: 模型精度
            - use_flash_attn: 是否使用Flash Attention
            - trust_remote_code: 是否信任远程代码
    
    Returns:
        Dict: 包含以下内容
            - model: 加载的模型
            - tokenizer: tokenizer（Sa2VA/InternVL3/Qwen2_5使用）
            - processor: processor（Qwen3使用）
            - model_type: 模型类型
            - device: 设备
    """
    
    model_name = config['model']['name']
    device = config['model']['device']
    
    print(f"\nLoading model from {model_name}...")
    
    # 判断模型类型
    model_type = get_model_type(model_name)
    print(f"  Model type: {model_type}")
    
    # 获取torch数据类型
    torch_dtype_str = config['model'].get('torch_dtype', 'bfloat16')
    torch_dtype = getattr(torch, torch_dtype_str)
    
    # 加载模型
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=config['model'].get('use_flash_attn', True),
        trust_remote_code=config['model'].get('trust_remote_code', True)
    ).eval().to(device)
    
    # 根据模型类型加载tokenizer或processor
    tokenizer = None
    processor = None
    
    if model_type == "qwen3":
        # Qwen3系列使用processor
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False
        )
        print(f"✓ Model and processor loaded successfully on {device}!")
    else:
        # Sa2VA/InternVL3/Qwen2_5系列使用tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False
        )
        print(f"✓ Model and tokenizer loaded successfully on {device}!")
    
    print(f"  - Precision: {torch_dtype_str}")
    print(f"  - Flash Attention: {config['model'].get('use_flash_attn', True)}")
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'processor': processor,
        'model_type': model_type,
        'device': device
    }

