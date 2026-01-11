#!/bin/bash

# 用法: bash start_eval.sh --conf ./conf/config.yaml [--download]

CONFIG_PATH="./conf/config.yaml"
DOWNLOAD_DATA=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --conf)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --download)
            DOWNLOAD_DATA=true
            shift
            ;;
        *)
            echo "错误: 未知参数 '$1'"
            echo "用法: bash start_eval.sh --conf ./conf/config.yaml [--download]"
            echo ""
            echo "选项:"
            echo "  --conf PATH    指定配置文件路径 (默认: ./conf/config.yaml)"
            echo "  --download     下载数据集 (默认: 不下载)"
            exit 1
            ;;
    esac
done

# 检查配置文件是否存在
if [ ! -f "$CONFIG_PATH" ]; then
    echo "错误: 配置文件不存在: $CONFIG_PATH"
    exit 1
fi

echo "======================================================"
echo "  配置文件: $CONFIG_PATH"
echo "  下载数据: $DOWNLOAD_DATA"
echo "======================================================"
echo ""

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "Error: Python not found!"
    echo "Please install Python 3.8 or higher."
    exit 1
fi

# 检查必要的Python包
echo "Checking Python dependencies..."
python -c "import yaml" 2>/dev/null || {
    echo "Installing PyYAML..."
    pip install pyyaml
}

python -c "import huggingface_hub" 2>/dev/null || {
    echo "Installing huggingface_hub..."
    pip install huggingface_hub
}

python -c "import transformers" 2>/dev/null || {
    echo "Warning: transformers not found. Please install it:"
    echo "  pip install transformers"
}

python -c "import torch" 2>/dev/null || {
    echo "Warning: PyTorch not found. Please install it:"
    echo "  pip install torch"
}

python -c "import numpy" 2>/dev/null || {
    echo "Warning: NumPy not found. Please install it:"
    echo "  pip install numpy"
}

python -c "import PIL" 2>/dev/null || {
    echo "Warning: Pillow not found. Please install it:"
    echo "  pip install pillow"
}

python -c "import tqdm" 2>/dev/null || {
    echo "Warning: tqdm not found. Please install it:"
    echo "  pip install tqdm"
}

python -c "import scipy" 2>/dev/null || {
    echo "Warning: scipy not found (needed for mask evaluation):"
    echo "  pip install scipy"
}

echo ""
echo "Starting evaluation..."
echo "=============================================="
echo ""

CMD="python eval.py --config \"$CONFIG_PATH\""

if [ "$DOWNLOAD_DATA" = true ]; then
    CMD="$CMD --download_data"
fi

eval $CMD
