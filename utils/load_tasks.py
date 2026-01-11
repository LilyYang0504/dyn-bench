import json
import glob
from pathlib import Path
from typing import List, Dict, Any


TASK_CATEGORIES = {
    "cameramask": "Camera-Object",
    "cameraqa": "Camera-Object",
    "objmask": "Inter-Object",
    "qa": "Inter-Object",
    "scenemask": "Object-Scene",
    "sceneqa": "Object-Scene",
}

QA_TASK_SUFFIXES = ["cameraqa", "qa", "sceneqa"]
MASK_TASK_SUFFIXES = ["cameramask", "objmask", "scenemask"]


def load_all_tasks(datasets_dir: Path, task_type: str = "all") -> List[Dict[str, Any]]:
    """
    加载所有评测任务
    
    Args:
        datasets_dir: 数据集根目录
        task_type: 任务类型过滤
            - "all": 加载所有任务（QA + Mask）
            - "qa": 仅加载QA任务
            - "mask": 仅加载Mask任务
    
    Returns:
        List[Dict]: 任务列表，每个任务包含：
            - scene_name: 场景名称（任务ID）
            - dataset: 数据集名称（场景类别）
            - task_type: 任务类别后缀
            - category: 任务大类（Camera-Object/Inter-Object/Object-Scene）
            - is_segmentation: 是否为分割任务
            - frame_paths: 帧文件路径列表
            - mask_dir: 掩码目录（仅Mask任务）
            - question: 问题/任务描述
            - options: 选项列表（仅QA任务）
            - answer: 答案（仅QA任务）
            - object_id: 物体ID（仅Mask任务）
            - question_idx: 问题索引（仅QA任务）
    """
    
    samples = []
    multi_json_dir = datasets_dir / "multi_json"
    
    if not multi_json_dir.exists():
        print(f"✗ Error: multi_json directory not found: {multi_json_dir}")
        return samples
    
    print(f"\nLoading tasks from {datasets_dir}...")
    
    # 遍历所有场景文件夹
    for dataset_dir in sorted(multi_json_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        print(f"  Processing dataset: {dataset_name}")

        # 遍历所有JSON文件
        json_files = sorted(dataset_dir.glob("*.json"))
        for json_file in json_files:
            # 跳过frame_sampling文件
            if "frame_sampling" in json_file.name:
                continue

            filename = json_file.stem
            
            # 从右往左找第一个下划线来分割任务id和任务类别
            parts = filename.rsplit('_', 1)
            if len(parts) != 2:
                print(f"    ⚠ Skipping invalid filename: {filename}")
                continue
            
            scene_name = parts[0]  # 任务id
            task_suffix = parts[1]  # 任务类别
            
            # 验证任务类别是否有效
            if task_suffix not in QA_TASK_SUFFIXES + MASK_TASK_SUFFIXES:
                print(f"    ⚠ Unknown task type: {task_suffix}")
                continue

            # 应用任务类型过滤
            is_segmentation = task_suffix in MASK_TASK_SUFFIXES
            if task_type == "qa" and is_segmentation:
                continue
            elif task_type == "mask" and not is_segmentation:
                continue

            category = TASK_CATEGORIES.get(task_suffix, "Unknown")

            # 查找对应的frames目录
            frames_dir = datasets_dir / "frames" / dataset_name / scene_name
            if not frames_dir.exists():
                print(f"    ✗ Warning: Frames directory not found: {frames_dir}")
                continue

            # 获取所有帧文件
            frame_paths = sorted(glob.glob(str(frames_dir / "frame_*.jpg")))
            if not frame_paths:
                frame_paths = sorted(glob.glob(str(frames_dir / "*.jpg")))

            if not frame_paths:
                print(f"    ✗ Warning: No frames found in: {frames_dir}")
                continue

            # 获取对应的masks目录
            mask_dir = datasets_dir / "masks" / dataset_name / scene_name

            # 加载JSON文件内容
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    task_data = json.load(f)
            except Exception as e:
                print(f"    ✗ Error loading {json_file}: {e}")
                continue

            # 根据任务类型处理数据
            if is_segmentation:
                # Mask任务
                for obj_id, obj_data in task_data.items():
                    sample = {
                        "scene_name": scene_name,
                        "dataset": dataset_name,
                        "task_type": task_suffix,
                        "category": category,
                        "is_segmentation": True,
                        "frame_paths": frame_paths,
                        "mask_dir": str(mask_dir),
                        "question": obj_data.get("question", ""),
                        "options": None,
                        "answer": None,
                        "object_id": obj_id,
                    }
                    samples.append(sample)
            else:
                # QA任务
                for idx, qa_item in enumerate(task_data):
                    sample = {
                        "scene_name": scene_name,
                        "dataset": dataset_name,
                        "task_type": task_suffix,
                        "category": category,
                        "is_segmentation": False,
                        "frame_paths": frame_paths,
                        "mask_dir": None,
                        "question": qa_item.get("question", ""),
                        "options": qa_item.get("options", []),
                        "answer": qa_item.get("answer", ""),
                        "object_id": None,
                        "question_idx": idx,
                    }
                    samples.append(sample)
    
    # 统计信息
    qa_count = sum(1 for s in samples if not s["is_segmentation"])
    mask_count = sum(1 for s in samples if s["is_segmentation"])
    
    print(f"\n✓ Loaded {len(samples)} tasks total")
    print(f"  - QA tasks: {qa_count}")
    print(f"  - Mask tasks: {mask_count}")
    
    return samples


def get_task_statistics(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    获取任务统计信息
    
    Args:
        tasks: 任务列表
    
    Returns:
        Dict: 统计信息字典
    """
    stats = {
        "total": len(tasks),
        "qa": 0,
        "mask": 0,
        "by_category": {},
        "by_dataset": {},
    }
    
    for task in tasks:
        # 按任务类型统计
        if task["is_segmentation"]:
            stats["mask"] += 1
        else:
            stats["qa"] += 1
        
        # 按大类统计
        category = task["category"]
        if category not in stats["by_category"]:
            stats["by_category"][category] = {"qa": 0, "mask": 0}
        
        if task["is_segmentation"]:
            stats["by_category"][category]["mask"] += 1
        else:
            stats["by_category"][category]["qa"] += 1
        
        # 按数据集统计
        dataset = task["dataset"]
        if dataset not in stats["by_dataset"]:
            stats["by_dataset"][dataset] = 0
        stats["by_dataset"][dataset] += 1
    
    return stats
