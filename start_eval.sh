#!/bin/bash

CONFIG_PATH="./conf/config.yaml"
DOWNLOAD_DATA=false

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
            echo "Error: Unknown argument '$1'"
            echo "Usage: bash start_eval.sh --conf ./conf/config.yaml [--download]"
            echo ""
            echo "Options:"
            echo "  --conf PATH    Specify config file path (default: ./conf/config.yaml)"
            echo "  --download     Download dataset (default: do not download)"
            exit 1
            ;;
    esac
done

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

echo "======================================================"
echo "  Config: $CONFIG_PATH"
echo "  Download Data: $DOWNLOAD_DATA"
echo "======================================================"
echo ""

# Check Python environment
if ! command -v python &> /dev/null; then
    echo "Error: Python not found!"
    echo "Please install Python 3.8 or higher."
    exit 1
fi

# Check necessary Python packages
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

# Set PyTorch CUDA memory allocator to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "✓ Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo ""

CMD="python eval.py --config \"$CONFIG_PATH\""

if [ "$DOWNLOAD_DATA" = true ]; then
    CMD="$CMD --download_data"
fi

eval $CMD
