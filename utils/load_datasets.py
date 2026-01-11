from pathlib import Path
from typing import Dict
from huggingface_hub import snapshot_download


def download_dataset(config: Dict) -> Path:
    """
    从HuggingFace下载数据集
    
    Args:
        config: 配置字典，包含datasets字段
            - repo_name: HF数据集仓库名称
            - local_dir: 本地存储目录
    
    Returns:
        Path: 本地数据集目录路径
    """

    repo_name = config['datasets']['repo_name']
    local_dir = Path(config['datasets']['local_dir'])
    
    print(f"Downloading dataset from {repo_name}...")
    print(f"Saving to {local_dir}...")
    
    try:
        snapshot_download(
            repo_id=repo_name,
            repo_type="dataset",
            local_dir=str(local_dir),
            ignore_patterns=[".gitattributes"]
        )
        print(f"✓ Dataset downloaded successfully to {local_dir}")
        return local_dir
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        print(f"→ Using existing local directory: {local_dir}")
        
        # 检查本地目录是否存在
        if not local_dir.exists():
            raise FileNotFoundError(
                f"Local dataset directory not found: {local_dir}\n"
                f"Please ensure the dataset is available or check the repo_name in config.yaml"
            )
        
        return local_dir


def verify_dataset_structure(datasets_dir: Path) -> bool:
    """
    验证数据集目录结构是否正确
    
    Args:
        datasets_dir: 数据集根目录
    
    Returns:
        bool: 目录结构是否正确
    """
    required_dirs = ['frames', 'masks', 'multi_json']
    
    for dir_name in required_dirs:
        dir_path = datasets_dir / dir_name
        if not dir_path.exists():
            print(f"✗ Required directory not found: {dir_path}")
            return False
        print(f"✓ Found {dir_name}/ directory")
    
    return True
