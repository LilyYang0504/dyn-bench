from pathlib import Path
from typing import Dict
from huggingface_hub import snapshot_download
from colorama import Fore, init
init(autoreset=True)


def download_dataset(config: Dict) -> Path:
    repo_name = config['datasets']['repo_name']
    local_dir = Path(config['datasets']['local_dir'])
    
    print(f"Downloading dataset from {repo_name}")
    print(f"Saving to {local_dir}")
    
    try:
        snapshot_download(
            repo_id=repo_name,
            repo_type="dataset",
            local_dir=str(local_dir),
            ignore_patterns=[".gitattributes"]
        )
        print(f"{Fore.GREEN}Dataset downloaded successfully to {local_dir}")
        return local_dir
    except Exception as e:
        print(f"{Fore.RED}ERROR: Can not download dataset: {e}")
        print(f"{Fore.YELLOW}Using existing local directory: {local_dir}")
        
        if not local_dir.exists():
            raise FileNotFoundError(
                f"Local dataset directory not found: {local_dir}\n"
                f"Please ensure the dataset is available or check the repo_name in config.yaml"
            )
        
        return local_dir


def verify_dataset_structure(datasets_dir: Path) -> bool:
    required_dirs = ['frames', 'masks', 'multi_json']
    
    for dir_name in required_dirs:
        dir_path = datasets_dir / dir_name
        if not dir_path.exists():
            print(f"{Fore.RED}ERROR: Required directory not found: {dir_path}")
            return False
        print(f"{Fore.GREEN}Found {dir_name}/ directory")
    
    return True
