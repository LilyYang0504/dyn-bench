import os
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import OrderedDict, defaultdict

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from colorama import Fore, init

from utils.load_datasets import download_dataset, verify_dataset_structure
from utils.load_model import load_model, extract_model_name_from_path, is_local_path
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
from utils.save_results import save_results, get_display_name_for_results
init(autoreset=True)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def load_config(config_path: str = "conf/config.yaml") -> Dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_args():
    parser = argparse.ArgumentParser()
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
    
    if args.download_data:
        datasets_dir = download_dataset(config)
    else:
        datasets_dir = Path(config['datasets']['local_dir'])
    
    if not datasets_dir.exists():
        print(f"{Fore.RED}ERROR: Dataset directory not found: {datasets_dir}")
        print(f"{Fore.YELLOW}Please download the dataset first using `--download`")
        return
    
    if not verify_dataset_structure(datasets_dir):
        print(f"{Fore.RED}ERROR: Dataset structure verification failed")
        return
    
    model_name = config['model']['name']
    model_alias = config['model'].get('alias', None)
    
    if model_alias:
        display_alias = model_alias
    elif is_local_path(model_name):
        display_alias = extract_model_name_from_path(model_name)
    else:
        display_alias = model_name
    
    print(f"\n{'='*60}")
    print(f"Configuration:")
    print(f"    Datasets: {datasets_dir}")
    print(f"    Model: {model_name}")
    if model_alias or (is_local_path(model_name) and display_alias != model_name):
        print(f"    Alias: {display_alias}")
    print(f"    Task Type: {config['task']['type']}")
    print(f"    Device: {config['model']['device']}")
    print(f"{'='*60}")
    
    task_type = config['task']['type']
    
    tasks = load_all_tasks(datasets_dir, task_type)
    
    limit = config['task'].get('limit')
    if limit and limit > 0:
        tasks = tasks[:limit]
        print(f"{Fore.YELLOW}WARN: Limited to {limit} tasks for testing")
    
    if not tasks:
        print(f"{Fore.RED}ERROR: No tasks found")
        return
    
    stats = get_task_statistics(tasks)
    for category, counts in stats['by_category'].items():
        print(f"{category}:")
        if counts['qa'] > 0:
            print(f"    - QA: {counts['qa']}")
        if counts['mask'] > 0:
            print(f"    - Mask: {counts['mask']}")
    
    print(f"Loading model: {config['model']['name']}")
    model_dict = load_model(config)
    model = model_dict['model']
    model_type = model_dict['model_type']
    
    if 'processor' in model_dict:
        processor = model_dict['processor']
        tokenizer = None
    else:
        tokenizer = model_dict['tokenizer']
        processor = None
    
    print(f"{Fore.GREEN}Model loaded successfully (Type: {model_type})")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"    GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"    GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    results_dir = Path(config['output']['results_dir'])
    dir_name, _ = get_display_name_for_results(config)
    mask_output_dir = results_dir / dir_name / "mask_details"
    
    
    results = []
    category_results = defaultdict(lambda: {"qa": [], "mask": []})
    
    for task in tqdm(tasks, desc="Evaluating"):
        frame_paths = task["frame_paths"]
        
        max_frames = config['task'].get('max_frames')
        if max_frames and len(frame_paths) > max_frames:
            indices = np.linspace(0, len(frame_paths) - 1, max_frames, dtype=int)
            frame_paths = [frame_paths[i] for i in indices]
        
        result = task.copy()
        
        if task["is_segmentation"]:
            if not model_dict.get('supports_mask', False):
                print(f"{Fore.YELLOW}WARN: Model {config['model']['name']} does not support mask tasks")
                result["prediction"] = ""
                result["J&F"] = 0.0
                category_results[task["category"]]["mask"].append(0.0)
                results.append(result)
                continue
            
            try:
                answer, pred_masks = run_mask_task(
                    model_dict=model_dict,
                    frame_paths=frame_paths,
                    question=task["question"],
                    crop_caption=task.get("crop_caption", ""),
                    crop_category=task.get("crop_category", "")
                )
                result["prediction"] = answer
                
                if config['output'].get('save_masks', True):
                    dataset = task["dataset"]
                    scene_name = task["scene_name"]
                    task_type = task["task_type"]
                    
                    mask_save_dir = mask_output_dir / dataset / f"{scene_name}_{task_type}_output"
                    mask_save_dir.mkdir(parents=True, exist_ok=True)
                    
                    if isinstance(pred_masks, list) and len(pred_masks) > 0:
                        if isinstance(pred_masks[0], np.ndarray) and pred_masks[0].ndim == 3:
                            pred_masks = [pred_masks[0][i] for i in range(pred_masks[0].shape[0])]
                    
                    for idx, mask in enumerate(pred_masks):
                        if isinstance(mask, np.ndarray):
                            mask = np.squeeze(mask)
                            
                            if mask.ndim != 2:
                                print(f"WARN: Mask shape is {mask.shape}, expected 2D. Skipping frame {idx}.")
                                continue
                            
                            if mask.dtype != np.uint8:
                                mask_img = (mask > 0.5).astype(np.uint8) * 255
                            else:
                                mask_img = mask
                            Image.fromarray(mask_img).save(mask_save_dir / f"frame_{idx:04d}.png")
                
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
                print(f"{Fore.RED}ERROR: Failure in segmentation task: {e}")
                traceback.print_exc()
                result["prediction"] = ""
                result["J&F"] = 0.0
                category_results[task["category"]]["mask"].append(0.0)
        
        else:
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
                print(f"{Fore.RED}ERROR: Failure in QA task: {e}")
                traceback.print_exc()
                result["prediction"] = ""
                result["accuracy"] = 0.0
                category_results[task["category"]]["qa"].append(0.0)
        
        results.append(result)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\nEvaluation Results:")
    
    summary = OrderedDict()
    all_scores = []
    
    for category in ["Camera-Object", "Inter-Object", "Object-Scene"]:
        qa_scores = category_results[category]["qa"]
        mask_scores = category_results[category]["mask"]
        
        print(f"{category}:")
        
        if qa_scores:
            qa_acc = np.mean(qa_scores) * 100
            summary[f"{category}_QA_Accuracy"] = qa_acc
            all_scores.append(qa_acc)
            print(f"    QA Accuracy: {qa_acc:.2f}% ({len(qa_scores)} samples)")
        
        if mask_scores:
            mask_jf = np.mean(mask_scores) * 100
            summary[f"{category}_Mask_J&F"] = mask_jf
            all_scores.append(mask_jf)
            print(f"    Mask J&F: {mask_jf:.2f}% ({len(mask_scores)} samples)")
    
    if all_scores:
        overall = np.mean(all_scores)
        summary["Overall"] = overall
        print(f"    Overall Score: {overall:.2f}%")
    
    save_results(results, config, summary)

if __name__ == "__main__": 
    main()