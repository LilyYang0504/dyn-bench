from typing import Dict, Any
import os
import torch
from transformers import (
    AutoTokenizer, AutoModel, AutoProcessor, AutoModelForCausalLM,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)


def is_local_path(path: str) -> bool:
    """检测是否为本地路径"""
    return os.path.exists(path) or os.path.sep in path or '\\' in path or '/' in path


def get_model_type(model_name: str) -> str:
    """
    根据模型名称判断模型类型
    
    Args:
        model_name: 模型名称或本地路径
    
    Returns:
        str: 模型类型
            Sa2VA系列（支持Mask）:
            - "sa2va": ByteDance/Sa2VA-1B/4B/8B
            - "sa2va_internvl3": ByteDance/Sa2VA-InternVL3-2B/8B/14B
            - "sa2va_qwen2_5": ByteDance/Sa2VA-Qwen2_5-VL-3B/7B
            - "sa2va_qwen3": ByteDance/Sa2VA-Qwen3-VL-2B/4B
            
            纯QA模型（不支持Mask）:
            - "internvl3": OpenGVLab/InternVL3-*
            - "internvl3_5": OpenGVLab/InternVL3_5-*
            - "qwen2_5_vl": Qwen/Qwen2.5-VL-*-Instruct
            - "qwen3_vl": Qwen/Qwen3-VL-*-Instruct
            - "qwen3_vl_moe": Qwen/Qwen3-VL-235B-A22B-Instruct
            - "llava_onevision": lmms-lab/LLaVA-OneVision-*
            - "vst": rayruiyang/VST-7B-RL
            - "spatial_ssrl": internlm/Spatial-SSRL-7B
            - "spatial_ladder": hongxingli/SpatialLadder-3B
            - "spacer_sft": RUBBISHLIKE/SpaceR-SFT-3B/7B
    """
    
    # 检测是否为本地路径（包含路径分隔符或 models-- 格式）
    import os
    import re
    
    original_name = model_name
    
    # 如果是本地路径，尝试从路径中提取模型名称
    if os.path.sep in model_name or '\\' in model_name or '/' in model_name:
        # 查找 models--{org}--{model} 格式
        match = re.search(r'models--([^\\\/]+)--([^\\\/]+)', model_name)
        if match:
            org = match.group(1)
            model = match.group(2)
            # 转换为标准格式 org/model
            model_name = f"{org}/{model}"
            print(f"  检测到本地路径，提取模型名称: {model_name}")
        else:
            # 尝试其他方式提取
            # 例如从路径中查找常见组织名
            for org in ['OpenGVLab', 'Qwen', 'ByteDance', 'lmms-lab', 'rayruiyang', 'internlm', 'hongxingli', 'RUBBISHLIKE']:
                if org.lower() in model_name.lower():
                    # 尝试提取模型部分
                    parts = model_name.replace('\\', '/').split('/')
                    for i, part in enumerate(parts):
                        if org.lower() in part.lower() and i + 1 < len(parts):
                            model_name = f"{org}/{parts[i+1]}"
                            print(f"  检测到本地路径，提取模型名称: {model_name}")
                            break
                    break
    
    model_name_lower = model_name.lower()
    
    # Sa2VA系列（原有模型）
    if "bytedance/sa2va" in model_name_lower:
        if "qwen3-vl" in model_name_lower:
            return "sa2va_qwen3"
        elif "qwen2_5-vl" in model_name_lower or "qwen2.5-vl" in model_name_lower:
            return "sa2va_qwen2_5"
        elif "internvl3" in model_name_lower:
            return "sa2va_internvl3"
        else:
            return "sa2va"
    
    # 新增的纯QA模型
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
        raise ValueError(f"Unknown model type for: {original_name}\n提取的模型名称: {model_name}")


def load_model(config: Dict) -> Dict[str, Any]:
    """
    加载多模态模型
    
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
            - tokenizer: tokenizer（部分模型使用）
            - processor: processor（部分模型使用）
            - model_type: 模型类型
            - device: 设备
            - supports_mask: 是否支持Mask任务
    """
    
    model_name = config['model']['name']
    device = config['model']['device']
    cache_dir = config['model'].get('cache_dir', None)
    
    # 支持模型别名映射（用于匿名测试等场景）
    # 如果设置了 alias，用 alias 来识别模型类型，但仍用 name 加载模型文件
    model_alias = config['model'].get('alias', None)
    
    # 检测是否为本地路径
    local_files_only = is_local_path(model_name)
    if local_files_only:
        print(f"  检测到本地路径: {model_name}")

    print(f"\nLoading model from {model_name}...")
    
    # 判断模型类型：
    # - 如果设置了 alias，从 alias 检测类型（推荐用于匿名测试）
    # - 否则从 name 直接检测
    if model_alias:
        print(f"  Model alias: {model_name} -> {model_alias}")
        model_type = get_model_type(model_alias)
        print(f"  Model type (from alias): {model_type}")
    else:
        model_type = get_model_type(model_name)
        print(f"  Model type (auto-detected): {model_type}")

    # 获取torch数据类型
    torch_dtype_str = config['model'].get('torch_dtype', 'bfloat16')
    if torch_dtype_str == "auto":
        torch_dtype = "auto"
    else:
        torch_dtype = getattr(torch, torch_dtype_str)

    use_flash_attn = config['model'].get('use_flash_attn', True)
    trust_remote_code = config['model'].get('trust_remote_code', True)

    model = None
    tokenizer = None
    processor = None
    supports_mask = False  # 默认不支持Mask
    
    # Sa2VA系列 - 支持Mask
    if model_type in ["sa2va", "sa2va_internvl3", "sa2va_qwen2_5", "sa2va_qwen3"]:
        supports_mask = True
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=use_flash_attn,
            trust_remote_code=trust_remote_code,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=local_files_only
        ).eval()
        if model_type == "sa2va_qwen3":
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir, local_files_only=local_files_only)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False, cache_dir=cache_dir, local_files_only=local_files_only)

    # InternVL3系列
    elif model_type in ["internvl3", "internvl3_5"]:
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=use_flash_attn,
            trust_remote_code=trust_remote_code,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=local_files_only
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False, cache_dir=cache_dir, local_files_only=local_files_only)

    # Qwen2.5-VL系列
    elif model_type == "qwen2_5_vl":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )
        # 使用默认的processor参数
        processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=local_files_only)

    # Qwen3-VL系列
    elif model_type == "qwen3_vl":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch_dtype,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )
        processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=local_files_only)

    # Qwen3-VL-MoE (235B-A22B)
    elif model_type == "qwen3_vl_moe":
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch_dtype,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )
        processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=local_files_only)

    # LLaVA-OneVision
    elif model_type == "llava_onevision":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir, local_files_only=local_files_only)

    # VST-7B-RL (基于Qwen2.5-VL)
    elif model_type == "vst":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2" if use_flash_attn else "eager",
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )
        # VST使用特定的pixel范围
        processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=256*28*28,
            max_pixels=1280*28*28,
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )

    # Spatial-SSRL-7B (基于Qwen2.5-VL)
    elif model_type == "spatial_ssrl":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )
        processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=local_files_only)

    # SpatialLadder-3B (基于Qwen2.5-VL)
    elif model_type == "spatial_ladder":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2" if use_flash_attn else "eager",
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )
        processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=local_files_only)
    
    # SpaceR-SFT (基于Qwen2.5-VL)
    elif model_type == "spacer_sft":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )
        processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=local_files_only)
    
    if model is None:
        raise ValueError(f"Failed to load model: {model_name}")
    
    # 打印加载信息
    if processor:
        print(f"✓ Model and processor loaded successfully!")
    else:
        print(f"✓ Model and tokenizer loaded successfully!")
    
    print(f"  - Precision: {torch_dtype_str}")
    print(f"  - Flash Attention: {use_flash_attn}")
    print(f"  - Supports Mask: {supports_mask}")
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'processor': processor,
        'model_type': model_type,
        'device': device,
        'supports_mask': supports_mask
    }

