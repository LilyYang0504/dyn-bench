from typing import List, Dict, Any, Tuple
import torch
import numpy as np
from PIL import Image


def run_mask_task(
    model_dict: Dict[str, Any],
    frame_paths: List[str],
    question: str
) -> Tuple[str, List[np.ndarray]]:
    """
    执行Mask分割任务
    
    根据不同的模型类型使用不同的推理方式
    
    Args:
        model_dict: 模型字典，包含：
            - model: 模型对象
            - tokenizer: tokenizer（Sa2VA/InternVL3/Qwen2_5）
            - processor: processor（Qwen3）
            - model_type: 模型类型
            - device: 设备
        frame_paths: 视频帧路径列表
        question: 分割任务描述
    
    Returns:
        Tuple[str, List[np.ndarray]]: (文本答案, 预测掩码列表)
    """
    model = model_dict['model']
    model_type = model_dict['model_type']
    tokenizer = model_dict.get('tokenizer')
    processor = model_dict.get('processor')
    
    # 加载图像
    images = [Image.open(p).convert('RGB') for p in frame_paths]
    
    # UniPixel 特殊处理
    if model_type == "unipixel":
        from utils.unipixel_helper import run_unipixel_mask
        return run_unipixel_mask(model, processor, frame_paths, question)
    
    # 根据模型类型构建提示词
    if model_type == "sa2va":
        # Sa2VA系列：不需要<image>标签
        text_prompts = question
    else:
        # InternVL3/Qwen2_5/Qwen3系列：需要<image>标签
        text_prompts = f"<image>{question}"
    
    # 构建输入字典
    if model_type == "qwen3":
        # Qwen3系列使用processor
        input_dict = {
            'video': images,
            'text': text_prompts,
            'past_text': '',
            'mask_prompts': None,
            'processor': processor,
        }
    else:
        # Sa2VA/InternVL3/Qwen2_5系列使用tokenizer
        input_dict = {
            'video': images,
            'text': text_prompts,
            'past_text': '',
            'mask_prompts': None,
            'tokenizer': tokenizer,
        }
    
    # 执行推理
    with torch.inference_mode():
        return_dict = model.predict_forward(**input_dict)
    
    # 提取结果
    answer = return_dict.get("prediction", "")
    answer = answer.replace('<|im_end|>', '').strip()
    
    masks = return_dict.get("prediction_masks", [])
    
    return answer, masks
