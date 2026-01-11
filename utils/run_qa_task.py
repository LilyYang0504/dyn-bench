from typing import List, Dict, Any
import torch
from PIL import Image


def run_qa_task(
    model_dict: Dict[str, Any],
    frame_paths: List[str],
    question: str,
    options: List[str] = None
) -> str:
    """
    执行QA任务
    
    根据不同的模型类型使用不同的推理方式
    
    Args:
        model_dict: 模型字典，包含：
            - model: 模型对象
            - tokenizer: tokenizer（Sa2VA/InternVL3/Qwen2_5）
            - processor: processor（Qwen3）
            - model_type: 模型类型
            - device: 设备
        frame_paths: 视频帧路径列表
        question: 问题文本
        options: 选项列表（可选）
    
    Returns:
        str: 模型预测的答案
    """
    model = model_dict['model']
    model_type = model_dict['model_type']
    tokenizer = model_dict.get('tokenizer')
    processor = model_dict.get('processor')
    
    # 加载图像
    images = [Image.open(p).convert('RGB') for p in frame_paths]
    
    # 根据模型类型构建提示词
    if options:
        options_text = "\n".join(options)
        base_question = f"{question}\n{options_text}\nAnswer with the option's letter from the given choices directly."
    else:
        base_question = question
    
    # 构建完整提示词（根据模型类型添加<image>标签）
    if model_type == "sa2va":
        # Sa2VA系列：不需要<image>标签
        text_prompts = base_question
    else:
        # InternVL3/Qwen2_5/Qwen3系列：需要<image>标签
        text_prompts = f"<image>{base_question}"
    
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
    
    # 提取预测结果
    answer = return_dict.get("prediction", "")
    
    # 清理输出（移除特殊标记）
    answer = answer.replace('<|im_end|>', '').strip()
    
    return answer
