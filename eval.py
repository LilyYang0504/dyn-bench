import os
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import OrderedDict, defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.load_datasets import download_dataset, verify_dataset_structure
from utils.load_model import load_model
from utils.load_tasks import load_all_tasks, get_task_statistics
from utils.cal_metrics import (
    compute_qa_accuracy,
    compute_jf_score,
    load_gt_masks,
    compute_category_metrics,
    fuzzy_matching
)
from utils.run_qa_task import run_qa_task
from utils.run_mask_task import run_mask_task
from utils.save_results import save_results


def load_config(config_path: str = "conf/config.yaml") -> Dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Evaluation Script"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="conf/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--download_data", 
        action="store_true",
        help="Download dataset from HuggingFace"
    )
    return parser.parse_args()


def main():
    args = get_args()
    config = load_config(args.config)
    
    # 数据集处理
    if args.download_data:
        datasets_dir = download_dataset(config)
    else:
        datasets_dir = Path(config['datasets']['local_dir'])
    
    # 验证数据集
    if not datasets_dir.exists():
        print(f"\n✗ Error: Datasets directory not found: {datasets_dir}")
        print("Please download the dataset first using --download_data flag")
        return
    
    print(f"\nVerifying dataset structure...")
    if not verify_dataset_structure(datasets_dir):
        print("✗ Dataset structure verification failed!")
        return
    
    # 显示配置信息
    print(f"\n{'='*60}")
    print(f"Configuration:")
    print(f"  Datasets: {datasets_dir}")
    print(f"  Model: {config['model']['name']}")
    print(f"  Task Type: {config['task']['type']}")
    print(f"  Device: {config['model']['device']}")
    print(f"{'='*60}")
    
    # 加载任务
    task_type = config['task']['type']
    
    tasks = load_all_tasks(datasets_dir, task_type)
    
    # 应用任务数量限制
    limit = config['task'].get('limit')
    if limit and limit > 0:
        tasks = tasks[:limit]
        print(f"\n⚠ Limited to {limit} tasks for testing")
    
    if not tasks:
        print("\n✗ No tasks found!")
        return
    
    # 显示任务统计
    stats = get_task_statistics(tasks)
    print(f"\nTask Statistics:")
    for category, counts in stats['by_category'].items():
        print(f"  {category}:")
        if counts['qa'] > 0:
            print(f"    - QA: {counts['qa']}")
        if counts['mask'] > 0:
            print(f"    - Mask: {counts['mask']}")
    
    # 加载模型
    print(f"\nLoading model: {config['model']['name']}...")
    model_dict = load_model(config)
    model = model_dict['model']
    model_type = model_dict['model_type']
    
    # tokenizer或processor
    if 'processor' in model_dict:
        processor = model_dict['processor']
        tokenizer = None
    else:
        tokenizer = model_dict['tokenizer']
        processor = None
    
    print(f"✓ Model loaded successfully (Type: {model_type})")
    
    # 准备结果保存目录
    results_dir = Path(config['output']['results_dir'])
    model_name = config['model']['name'].split('/')[-1]
    mask_output_dir = results_dir / model_name / "mask_details"
    
    # 运行评估
    print(f"\n{'='*60}")
    print("Starting Evaluation...")
    print(f"{'='*60}\n")
    
    results = []
    category_results = defaultdict(lambda: {"qa": [], "mask": []})
    
    for task in tqdm(tasks, desc="Evaluating"):
        frame_paths = task["frame_paths"]
        
        # 采样帧
        max_frames = config['task'].get('max_frames')
        if max_frames and len(frame_paths) > max_frames:
            indices = np.linspace(0, len(frame_paths) - 1, max_frames, dtype=int)
            frame_paths = [frame_paths[i] for i in indices]
        
        result = task.copy()
        
        if task["is_segmentation"]:
            # Mask任务 - 检查模型是否支持
            if not model_dict.get('supports_mask', False):
                print(f"\n⚠ Warning: Model {config['model']['name']} does not support mask tasks. Skipping...")
                result["prediction"] = ""
                result["J&F"] = 0.0
                category_results[task["category"]]["mask"].append(0.0)
                results.append(result)
                continue
            
            # Mask任务
            try:
                answer, pred_masks = run_mask_task(
                    model_dict=model_dict,
                    frame_paths=frame_paths,
                    question=task["question"]
                )
                result["prediction"] = answer
                
                # 保存预测掩码
                if config['output'].get('save_masks', True):
                    dataset = task["dataset"]
                    scene_name = task["scene_name"]
                    task_type = task["task_type"]
                    
                    mask_save_dir = mask_output_dir / dataset / f"{scene_name}_{task_type}_output"
                    mask_save_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 处理掩码格式
                    if isinstance(pred_masks, list) and len(pred_masks) > 0:
                        if isinstance(pred_masks[0], np.ndarray) and pred_masks[0].ndim == 3:
                            pred_masks = [pred_masks[0][i] for i in range(pred_masks[0].shape[0])]
                    
                    # 保存掩码
                    for idx, mask in enumerate(pred_masks):
                        if isinstance(mask, np.ndarray):
                            if mask.dtype != np.uint8:
                                mask_img = (mask > 0.5).astype(np.uint8) * 255
                            else:
                                mask_img = mask
                            Image.fromarray(mask_img).save(mask_save_dir / f"frame_{idx:04d}.png")
                
                # 加载GT并计算J&F
                gt_masks = load_gt_masks(
                    task["mask_dir"],
                    task["object_id"],
                    len(task["frame_paths"])
                )
                
                boundary_th = config['evaluation'].get('boundary_threshold', 2)
                jf = compute_jf_score(pred_masks, gt_masks, boundary_th)
                
                result["J"] = jf["J"]
                result["F"] = jf["F"]
                result["J&F"] = jf["J&F"]
                
                category_results[task["category"]]["mask"].append(jf["J&F"])
                
            except Exception as e:
                import traceback
                print(f"\n✗ Error in segmentation task: {e}")
                traceback.print_exc()
                result["prediction"] = ""
                result["J&F"] = 0.0
                category_results[task["category"]]["mask"].append(0.0)
        
        else:
            # QA任务
            try:
                answer = run_qa_task(
                    model_dict=model_dict,
                    frame_paths=frame_paths,
                    question=task["question"],
                    options=task["options"]
                )
                result["prediction"] = answer
                
                accuracy = compute_qa_accuracy(answer, task["answer"])
                result["accuracy"] = accuracy
                
                category_results[task["category"]]["qa"].append(accuracy)
                
            except Exception as e:
                import traceback
                print(f"\n✗ Error in QA task: {e}")
                traceback.print_exc()
                result["prediction"] = ""
                result["accuracy"] = 0.0
                category_results[task["category"]]["qa"].append(0.0)
        
        results.append(result)
    
    # 汇总结果
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    
    summary = OrderedDict()
    all_scores = []
    
    for category in ["Camera-Object", "Inter-Object", "Object-Scene"]:
        qa_scores = category_results[category]["qa"]
        mask_scores = category_results[category]["mask"]
        
        print(f"\n{category}:")
        
        if qa_scores:
            qa_acc = np.mean(qa_scores) * 100
            summary[f"{category}_QA_Accuracy"] = qa_acc
            all_scores.append(qa_acc)
            print(f"  QA Accuracy: {qa_acc:.2f}% ({len(qa_scores)} samples)")
        
        if mask_scores:
            mask_jf = np.mean(mask_scores) * 100
            summary[f"{category}_Mask_J&F"] = mask_jf
            all_scores.append(mask_jf)
            print(f"  Mask J&F: {mask_jf:.2f}% ({len(mask_scores)} samples)")
    
    if all_scores:
        overall = np.mean(all_scores)
        summary["Overall"] = overall
        print(f"\n{'='*60}")
        print(f"Overall Score: {overall:.2f}%")
        print(f"{'='*60}")
    
    # 保存结果
    save_results(results, config, summary)
    
    print(f"\n{'='*60}")
    print("✓ Evaluation completed successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__": 
    main()