import os
import argparse
from huggingface_hub import snapshot_download
from transformers import (
    AutoModel, AutoTokenizer, AutoProcessor, AutoModelForCausalLM,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)


def get_model_type(model_name: str) -> str:
    """根据模型名称判断模型类型"""
    model_name_lower = model_name.lower()
    
    # Sa2VA系列
    if "bytedance/sa2va" in model_name_lower:
        if "qwen3-vl" in model_name_lower:
            return "sa2va_qwen3"
        elif "qwen2_5-vl" in model_name_lower or "qwen2.5-vl" in model_name_lower:
            return "sa2va_qwen2_5"
        elif "internvl3" in model_name_lower:
            return "sa2va_internvl3"
        else:
            return "sa2va"
    
    # UniPixel 模型
    elif "polyu-chenlab/unipixel" in model_name_lower:
        return "unipixel"
    
    # 新增的纯QA模型
    elif "opengvlab/internvl3_5" in model_name_lower or "opengvlab/internvl3.5" in model_name_lower:
        return "internvl3_5"
    elif "opengvlab/internvl3" in model_name_lower:
        return "internvl3"
    elif "qwen/qwen3-vl-235b" in model_name_lower:
        return "qwen3_vl_moe"
    elif "qwen/qwen3-vl" in model_name_lower:
        return "qwen3_vl"
    elif "qwen/qwen2.5-vl" in model_name_lower:
        return "qwen2_5_vl"
    elif "llava-onevision" in model_name_lower:
        return "llava_onevision"
    elif "vst-7b" in model_name_lower:
        return "vst"
    elif "spatial-ssrl" in model_name_lower:
        return "spatial_ssrl"
    elif "spatialladder" in model_name_lower:
        return "spatial_ladder"
    elif "spacer-sft" in model_name_lower:
        return "spacer_sft"
    else:
        raise ValueError(f"Unknown model type for: {model_name}")


def download_model(model_name: str, cache_dir: str = None):
    """
    下载模型到本地缓存（包括所有权重文件）
    
    Args:
        model_name: HuggingFace 模型名称
        cache_dir: 缓存目录路径（可选，默认使用 HF_HOME 环境变量）
    """
    print(f"\n{'='*60}")
    print(f"开始下载模型: {model_name}")
    print(f"缓存目录: {cache_dir if cache_dir else 'HF 默认路径 (通过 HF_HOME 设置)'}")
    print(f"{'='*60}\n")
    
    # 判断模型类型
    model_type = get_model_type(model_name)
    print(f"检测到模型类型: {model_type}\n")
    
    try:
        # 使用 snapshot_download 下载完整模型（所有文件）
        print("📥 下载完整模型文件（包括权重、配置、tokenizer 等）...")
        print("   这可能需要几分钟到几小时，取决于模型大小和网络速度...\n")
        
        local_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            resume_download=True,  # 支持断点续传
            local_files_only=False
        )
        
        print(f"\n{'='*60}")
        print(f"✅ 模型下载成功: {model_name}")
        print(f"{'='*60}\n")
        
        # 打印实际保存位置
        print(f"📁 模型已保存到:")
        print(f"   {local_path}\n")
        
        
    except Exception as e:
        print(f"\n❌ 模型下载失败: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="下载 HuggingFace 模型到本地缓存以支持离线评测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 下载标准模型
  python download_model.py --model "OpenGVLab/InternVL3_5-2B"
  
  # 下载 UniPixel 模型
  python download_model.py --model "PolyU-ChenLab/UniPixel-3B"
  
  # 指定自定义缓存路径
  python download_model.py --model "OpenGVLab/InternVL3_5-2B" --cache-dir "E:/hf-download"
  
  # 批量下载多个模型
  python download_model.py --model "OpenGVLab/InternVL3_5-2B" "Qwen/Qwen2.5-VL-7B-Instruct" "PolyU-ChenLab/UniPixel-3B"

支持的模型:
  - Sa2VA 系列: ByteDance/Sa2VA-*
  - UniPixel: PolyU-ChenLab/UniPixel-{3B,7B}
  - InternVL: OpenGVLab/InternVL*
  - Qwen: Qwen/Qwen*-VL-*
  - 其他: 见 README.md

环境变量设置:
  Windows (PowerShell):  $env:HF_HOME="E:/hf-download"
  Windows (CMD):         set HF_HOME=E:/hf-download
  Linux/Mac:             export HF_HOME=/path/to/cache
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        nargs='+',
        required=True,
        help='要下载的模型名称（HuggingFace 格式，如 "OpenGVLab/InternVL3_5-2B"），支持多个模型'
    )
    
    parser.add_argument(
        '--cache-dir', '-c',
        type=str,
        default=None,
        help='模型缓存目录（可选，默认使用 HF_HOME 环境变量指定的路径）'
    )
    
    args = parser.parse_args()
    
    # 处理多个模型
    models = args.model
    cache_dir = args.cache_dir
    
    print(f"\n准备下载 {len(models)} 个模型...")
    
    for i, model_name in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] 处理模型: {model_name}")
        try:
            download_model(model_name, cache_dir)
        except Exception as e:
            print(f"⚠️  跳过模型 {model_name}，继续下一个...")
            continue
    
    print(f"\n🎉 所有任务完成！")


if __name__ == "__main__":
    main()
