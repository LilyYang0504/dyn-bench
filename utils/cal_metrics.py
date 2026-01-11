from typing import List, Dict
from pathlib import Path
import numpy as np
from PIL import Image


def fuzzy_matching(pred: str) -> str:
    """
    提取预测答案中的选项字母
    
    Args:
        pred: 模型预测的原始文本
    
    Returns:
        str: 提取的选项字母（大写）
    
    Example:
        >>> fuzzy_matching("A. The answer is...")
        'A'
        >>> fuzzy_matching("The answer is B")
        'B'
    """
    pred = pred.strip()
    if not pred:
        return ""
    
    # 取第一个词
    first_char = pred.split()[0] if pred.split() else pred[0]
    
    # 移除标点符号并转大写
    return first_char.rstrip('.,;:!?').upper()


def compute_qa_accuracy(pred: str, gt: str) -> float:
    """
    计算单个QA任务的准确率
    
    Args:
        pred: 模型预测的答案
        gt: Ground Truth答案
    
    Returns:
        float: 准确率（0.0或1.0）
    """

    pred_letter = fuzzy_matching(pred)
    gt_letter = gt.strip().upper()
    
    return 1.0 if pred_letter == gt_letter else 0.0


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    计算IoU (Intersection over Union) / Jaccard Index
    
    Args:
        pred_mask: 预测掩码 (binary array)
        gt_mask: Ground Truth掩码 (binary array)
    
    Returns:
        float: IoU分数 [0, 1]
    """

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def compute_boundary_f1(pred_mask: np.ndarray, gt_mask: np.ndarray, bound_th: int = 2) -> float:
    """
    计算边界F1分数
    
    边界F1评估分割掩码边界的精确度，通过形态学操作提取边界，
    然后计算预测边界和GT边界之间的精确率和召回率
    
    Args:
        pred_mask: 预测掩码 (binary array)
        gt_mask: Ground Truth掩码 (binary array)
        bound_th: 边界阈值（腐蚀/膨胀的迭代次数）
    
    Returns:
        float: 边界F1分数 [0, 1]
    """

    try:
        from scipy import ndimage
        
        # 提取边界：原始mask XOR 腐蚀后的mask
        pred_boundary = pred_mask ^ ndimage.binary_erosion(pred_mask, iterations=bound_th)
        gt_boundary = gt_mask ^ ndimage.binary_erosion(gt_mask, iterations=bound_th)
        
        # 特殊情况处理
        if pred_boundary.sum() == 0 and gt_boundary.sum() == 0:
            return 1.0
        if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
            return 0.0
        
        # 膨胀边界以允许一定容差
        pred_dilated = ndimage.binary_dilation(pred_boundary, iterations=bound_th)
        gt_dilated = ndimage.binary_dilation(gt_boundary, iterations=bound_th)
        
        # 计算精确率和召回率
        precision = np.logical_and(pred_boundary, gt_dilated).sum() / pred_boundary.sum()
        recall = np.logical_and(gt_boundary, pred_dilated).sum() / gt_boundary.sum()
        
        # 计算F1分数
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
        
    except Exception as e:
        print(f"Warning: Error computing boundary F1: {e}")
        return 0.0


def load_gt_masks(mask_dir: str, object_id: str, n_frames: int) -> List[np.ndarray]:
    """
    加载Ground Truth掩码
    
    Args:
        mask_dir: 掩码目录路径
        object_id: 物体ID (例如: "object_1")
        n_frames: 帧数量
    
    Returns:
        List[np.ndarray]: GT掩码列表
    """

    masks = []
    
    # 从object_id中提取物体编号 (例如: "object_1" -> "1")
    obj_num = object_id.split("_")[-1]
    obj_suffix = f"obj{obj_num}"
    
    mask_dir = Path(mask_dir)
    
    for i in range(n_frames):
        # 可能的文件名格式
        possible_names = [
            f"frame_{i:04d}_{obj_suffix}.png",
            f"frame_{i:04d}_{obj_suffix}.jpg",
        ]
        
        mask_path = None
        for name in possible_names:
            path = mask_dir / name
            if path.exists():
                mask_path = path
                break
        
        if mask_path:
            try:
                # 加载掩码并转为灰度图
                mask = np.array(Image.open(mask_path).convert('L'))
                masks.append(mask)
            except Exception as e:
                print(f"Warning: Error loading mask {mask_path}: {e}")
                # 添加零掩码
                if masks:
                    masks.append(np.zeros_like(masks[-1]))
                else:
                    masks.append(np.zeros((480, 640), dtype=np.uint8))
        else:
            # 文件不存在，添加零掩码
            if masks:
                masks.append(np.zeros_like(masks[-1]))
            else:
                masks.append(np.zeros((480, 640), dtype=np.uint8))
    
    return masks


def compute_jf_score(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray], 
                     bound_th: int = 2) -> Dict[str, float]:
    """
    计算J&F分数（Jaccard & F-measure）
    
    J&F是视频对象分割的标准评估指标，结合了：
    - J (Jaccard): 区域相似度（IoU）
    - F (F-measure): 边界准确度（Boundary F1）
    - J&F: 两者的平均值
    
    Args:
        pred_masks: 预测掩码列表
        gt_masks: Ground Truth掩码列表
        bound_th: 边界阈值（用于边界F1计算）
    
    Returns:
        Dict[str, float]: 包含 "J", "F", "J&F" 三个指标
    """

    if not pred_masks or not gt_masks:
        return {"J": 0.0, "F": 0.0, "J&F": 0.0}
    
    j_scores = []
    f_scores = []
    
    n_frames = min(len(pred_masks), len(gt_masks))
    
    for i in range(n_frames):
        # 获取当前帧的掩码
        pred = pred_masks[i] if i < len(pred_masks) else np.zeros_like(gt_masks[0])
        gt = gt_masks[i]
        
        # 确保是2D数组
        if pred.ndim > 2:
            pred = pred.squeeze()
        if gt.ndim > 2:
            gt = gt.squeeze()
        
        # 转为二值掩码
        pred_binary = (pred > 0.5).astype(bool)
        gt_binary = (gt > 0).astype(bool)
        
        # 如果尺寸不匹配，调整预测掩码尺寸
        if pred_binary.shape != gt_binary.shape:
            pred_pil = Image.fromarray(pred_binary.astype(np.uint8) * 255)
            pred_pil = pred_pil.resize((gt_binary.shape[1], gt_binary.shape[0]), Image.NEAREST)
            pred_binary = np.array(pred_pil) > 127
        
        # 计算J和F
        j_scores.append(compute_iou(pred_binary, gt_binary))
        f_scores.append(compute_boundary_f1(pred_binary, gt_binary, bound_th))
    
    # 计算平均分数
    j_mean = np.mean(j_scores) if j_scores else 0.0
    f_mean = np.mean(f_scores) if f_scores else 0.0
    
    return {
        "J": j_mean,
        "F": f_mean,
        "J&F": (j_mean + f_mean) / 2
    }



def compute_category_metrics(results: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    计算各个类别的平均指标
    
    Args:
        results: 评测结果列表
    
    Returns:
        Dict: 各类别的指标统计
    """
    from collections import defaultdict
    
    category_results = defaultdict(lambda: {"qa": [], "mask": []})
    
    for result in results:
        category = result.get("category", "Unknown")
        
        if result.get("is_segmentation"):
            # Mask任务
            jf_score = result.get("J&F", 0.0)
            category_results[category]["mask"].append(jf_score)
        else:
            # QA任务
            accuracy = result.get("accuracy", 0.0)
            category_results[category]["qa"].append(accuracy)
    
    # 计算平均值
    metrics = {}
    for category in ["Camera-Object", "Inter-Object", "Object-Scene"]:
        qa_scores = category_results[category]["qa"]
        mask_scores = category_results[category]["mask"]
        
        metrics[category] = {}
        
        if qa_scores:
            metrics[category]["QA_Accuracy"] = np.mean(qa_scores) * 100
            metrics[category]["QA_Count"] = len(qa_scores)
        
        if mask_scores:
            metrics[category]["Mask_J&F"] = np.mean(mask_scores) * 100
            metrics[category]["Mask_Count"] = len(mask_scores)
    
    return metrics
