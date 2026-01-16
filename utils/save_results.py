import json
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from datetime import datetime
from colorama import Fore, init
init(autoreset=True)


def get_display_name_for_results(config: Dict) -> tuple:
    model_config = config.get('model', {})
    model_name = model_config.get('name', '')
    model_alias = model_config.get('alias', None)
    
    if model_alias:
        display_name = model_alias
        dir_name = model_alias.split('/')[-1]
    else:
        from .load_model import extract_model_name_from_path, is_local_path
        
        if is_local_path(model_name):
            extracted = extract_model_name_from_path(model_name)
            display_name = extracted
            dir_name = extracted.split('/')[-1]
        else:
            display_name = model_name
            dir_name = model_name.split('/')[-1]
    
    return dir_name, display_name


def save_results(results: List[Dict], config: Dict, summary: Dict):
    output_config = config['output']
    results_dir = Path(output_config['results_dir'])
    
    dir_name, display_name = get_display_name_for_results(config)
    
    model_results_dir = results_dir / dir_name
    
    qa_details_dir = model_results_dir / "qa_details"
    mask_details_dir = model_results_dir / "mask_details"
    qa_details_dir.mkdir(parents=True, exist_ok=True)
    mask_details_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{Fore.CYAN}Saving results to {model_results_dir}")
    
    qa_results_by_scene = defaultdict(list)
    mask_results_by_scene = defaultdict(list)
    
    for result in results:
        dataset = result['dataset']
        scene_name = result['scene_name']
        task_type = result['task_type']
        
        if result['is_segmentation']:
            mask_results_by_scene[(dataset, scene_name, task_type)].append(result)
        else:
            qa_results_by_scene[(dataset, scene_name, task_type)].append(result)
    
    for (dataset, scene_name, task_type), task_results in qa_results_by_scene.items():
        scene_dir = qa_details_dir / dataset
        scene_dir.mkdir(parents=True, exist_ok=True)
        
        details_file = scene_dir / f"{scene_name}_{task_type}_details.json"
        
        qa_details = []
        for r in task_results:
            qa_details.append({
                "question": r['question'],
                "options": r['options'],
                "answer": r['answer'],
                "model_predict": r.get('prediction', ''),
                "accuracy": r.get('accuracy', 0.0)
            })
        
        with open(details_file, 'w', encoding='utf-8') as f:
            json.dump(qa_details, f, indent=2, ensure_ascii=False)
    
    print(f"{Fore.CYAN}QA details saved to: {qa_details_dir}")
    print(f"{Fore.CYAN}Mask details saved to: {mask_details_dir}")
    
    overall_file = model_results_dir / "overall.txt"
    
    with open(overall_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"  Model: {display_name}\n")
        f.write(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        for category in ["Camera-Object", "Inter-Object", "Object-Scene"]:
            f.write(f"{category}:\n")
            
            qa_key = f"{category}_QA_Accuracy"
            mask_key = f"{category}_Mask_J&F"
            
            if qa_key in summary:
                f.write(f"  QA Accuracy: {summary[qa_key]:.2f}%\n")
            
            if mask_key in summary:
                f.write(f"  Mask J&F: {summary[mask_key]:.2f}%\n")
            
            f.write("\n")
        
        if "Overall" in summary:
            f.write("=" * 60 + "\n")
            f.write(f"  Overall Score: {summary['Overall']:.2f}%\n")
            f.write("=" * 60 + "\n")
    
    print(f"{Fore.CYAN}Overall results saved to: {overall_file}")
