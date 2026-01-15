"""
UniPixel 模型辅助模块

由于 UniPixel 不能通过标准 transformers 加载，需要使用其自定义库。
需要先克隆 UniPixel 仓库到 thirdparty/UniPixel 目录。
"""

import sys
import os
from pathlib import Path
from typing import List, Tuple, Any, Dict
import numpy as np
from PIL import Image


def check_unipixel_installed():
    """检查 UniPixel 是否已安装"""
    try:
        import unipixel
        return True
    except ImportError:
        return False


def setup_unipixel_path():
    """设置 UniPixel 的 Python 路径"""
    # 获取 bench 目录
    bench_dir = Path(__file__).parent.parent
    unipixel_dir = bench_dir / "thirdparty" / "UniPixel"
    
    if not unipixel_dir.exists():
        raise RuntimeError(
            f"UniPixel 仓库未找到！\n"
            f"请执行以下命令克隆仓库：\n"
            f"  cd {bench_dir / 'thirdparty'}\n"
            f"  git clone https://github.com/PolyU-ChenLab/UniPixel.git\n"
            f"  cd UniPixel\n"
            f"  pip install -r requirements.txt"
        )
    
    # 添加到 Python 路径
    unipixel_str = str(unipixel_dir)
    if unipixel_str not in sys.path:
        sys.path.insert(0, unipixel_str)
        print(f"  Added UniPixel to Python path: {unipixel_str}")


def load_unipixel_model(
    model_name: str, 
    device: str = "cuda",
    cache_dir: str = None,
    local_files_only: bool = False
):
    """
    加载 UniPixel 模型
    
    Args:
        model_name: 模型名称或本地路径，如 "PolyU-ChenLab/UniPixel-3B"
        device: 设备
        cache_dir: 模型缓存目录（下载模型时使用）
        local_files_only: 是否只使用本地文件（离线模式）
    
    Returns:
        tuple: (model, processor)
    """
    # 设置路径
    setup_unipixel_path()
    
    # 导入 UniPixel 库
    try:
        from unipixel.model.builder import build_model_with_cache
    except ImportError:
        # 如果没有自定义包装器，使用原始的 build_model
        try:
            from unipixel.model.builder import build_model
            print("  Warning: Using original build_model (cache_dir not supported)")
        except ImportError as e:
            raise ImportError(
                f"无法导入 unipixel 库！错误：{e}\n"
                f"请确保已安装 UniPixel 依赖：\n"
                f"  cd thirdparty/UniPixel\n"
                f"  pip install -r requirements.txt"
            )
        
        # 使用原始函数（不支持 cache_dir）
        print(f"  Loading UniPixel model: {model_name}")
        model, processor = build_model(model_name)
        model = model.to(device)
        return model, processor
    
    # 使用支持 cache_dir 的版本
    print(f"  Loading UniPixel model: {model_name}")
    if cache_dir:
        print(f"  Using cache directory: {cache_dir}")
    if local_files_only:
        print(f"  Using local files only (offline mode)")
    
    model, processor = build_model_with_cache(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only
    )
    model = model.to(device)
    
    return model, processor


def run_unipixel_qa(
    model: Any,
    processor: Any,
    frame_paths: List[str],
    question: str
) -> str:
    """
    使用 UniPixel 模型执行 QA 任务
    
    Args:
        model: UniPixel 模型
        processor: UniPixel processor
        frame_paths: 视频帧路径列表
        question: 问题文本
    
    Returns:
        str: 模型预测的答案
    """
    from unipixel.dataset.utils import process_vision_info
    from unipixel.utils.io import load_image, load_video
    from unipixel.utils.transforms import get_sam2_transform
    
    device = next(model.parameters()).device
    sam2_transform = get_sam2_transform(model.config.sam2_image_size)
    
    # 加载帧
    # UniPixel 期望的是原始帧数组，我们从文件加载
    frames_list = [np.array(Image.open(p).convert('RGB')) for p in frame_paths]
    frames = np.stack(frames_list, axis=0)  # [T, H, W, C]
    
    # 构建消息
    messages = [{
        'role': 'user',
        'content': [
            {
                'type': 'video',
                'video': frame_paths,  # 使用路径列表
                'min_pixels': 128 * 28 * 28,
                'max_pixels': 256 * 28 * 28 * max(1, int(16 / len(frame_paths)))
            },
            {
                'type': 'text',
                'text': question
            }
        ]
    }]
    
    # 准备输入
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    images, videos, kwargs = process_vision_info(messages, return_video_kwargs=True)
    
    data = processor(text=[text], images=images, videos=videos, return_tensors='pt', **kwargs)
    # 将 numpy array 转换为 torch tensor
    frames_tensor = torch.from_numpy(frames)  # [T, H, W, C]
    data['frames'] = [sam2_transform(frames_tensor).to(model.sam2.dtype)]
    data['frame_size'] = [frames.shape[1:3]]
    
    # 生成
    with torch.no_grad():
        output_ids = model.generate(
            **data.to(device),
            do_sample=False,
            temperature=None,
            top_k=None,
            top_p=None,
            repetition_penalty=None,
            max_new_tokens=512
        )
    
    # 解码
    assert data.input_ids.size(0) == output_ids.size(0) == 1
    output_ids = output_ids[0, data.input_ids.size(1):]
    
    if output_ids[-1] == processor.tokenizer.eos_token_id:
        output_ids = output_ids[:-1]
    
    response = processor.decode(output_ids, clean_up_tokenization_spaces=False)
    
    return response


def run_unipixel_mask(
    model: Any,
    processor: Any,
    frame_paths: List[str],
    question: str
) -> Tuple[str, List[np.ndarray]]:
    """
    使用 UniPixel 模型执行 Mask 分割任务
    
    Args:
        model: UniPixel 模型
        processor: UniPixel processor
        frame_paths: 视频帧路径列表
        question: 分割任务描述
    
    Returns:
        Tuple[str, List[np.ndarray]]: (文本答案, 预测掩码列表)
    """
    from unipixel.dataset.utils import process_vision_info
    from unipixel.utils.io import load_image, load_video
    from unipixel.utils.transforms import get_sam2_transform
    
    device = next(model.parameters()).device
    sam2_transform = get_sam2_transform(model.config.sam2_image_size)
    
    # 加载帧
    frames_list = [np.array(Image.open(p).convert('RGB')) for p in frame_paths]
    frames = np.stack(frames_list, axis=0)  # [T, H, W, C]
    
    # 构建消息
    messages = [{
        'role': 'user',
        'content': [
            {
                'type': 'video',
                'video': frame_paths,
                'min_pixels': 128 * 28 * 28,
                'max_pixels': 256 * 28 * 28 * max(1, int(16 / len(frame_paths)))
            },
            {
                'type': 'text',
                'text': question
            }
        ]
    }]
    
    # 准备输入
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    images, videos, kwargs = process_vision_info(messages, return_video_kwargs=True)
    
    data = processor(text=[text], images=images, videos=videos, return_tensors='pt', **kwargs)
    # 将 numpy array 转换为 torch tensor
    frames_tensor = torch.from_numpy(frames)  # [T, H, W, C]
    data['frames'] = [sam2_transform(frames_tensor).to(model.sam2.dtype)]
    data['frame_size'] = [frames.shape[1:3]]
    
    # 生成
    with torch.no_grad():
        output_ids = model.generate(
            **data.to(device),
            do_sample=False,
            temperature=None,
            top_k=None,
            top_p=None,
            repetition_penalty=None,
            max_new_tokens=512
        )
    
    # 解码文本
    assert data.input_ids.size(0) == output_ids.size(0) == 1
    output_ids = output_ids[0, data.input_ids.size(1):]
    
    if output_ids[-1] == processor.tokenizer.eos_token_id:
        output_ids = output_ids[:-1]
    
    response = processor.decode(output_ids, clean_up_tokenization_spaces=False)
    
    # 获取分割掩码
    masks = []
    if len(model.seg) >= 1:
        # model.seg 包含分割结果
        # 需要转换为我们的格式 [T, H, W]
        for seg in model.seg:
            # 根据 UniPixel 的实现，seg 应该是掩码数组
            # 这里需要根据实际输出格式调整
            if isinstance(seg, np.ndarray):
                masks.append(seg)
            else:
                # 如果是 tensor，转换为 numpy
                masks.append(seg.cpu().numpy() if hasattr(seg, 'cpu') else np.array(seg))
    
    # 确保掩码数量与帧数匹配
    if len(masks) == 0:
        # 如果没有生成掩码，返回空掩码
        H, W = frames.shape[1:3]
        masks = [np.zeros((H, W), dtype=np.uint8) for _ in range(len(frames_list))]
    
    return response, masks


import torch
