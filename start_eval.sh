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
            echo "ERROR: Unknown argument '$1'"
            echo "Usage: bash start_eval.sh --conf ./conf/config.yaml [--download]"
            echo ""
            echo "Options:"
            echo "  --conf PATH    Specify config file path (default: ./conf/config.yaml)"
            echo "  --download     Download dataset (default: do not download)"
            exit 1
            ;;
    esac
done

if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config file not found: $CONFIG_PATH"
    exit 1
fi

echo "============================================================"
echo "Config: $CONFIG_PATH"
echo "Download Datasets: $DOWNLOAD_DATA"
echo "============================================================"
echo ""

if ! command -v python &> /dev/null; then
    exit 1
fi


python -c "import yaml" 2>/dev/null || {
    echo "WARN: yaml not found"
}

python -c "import huggingface_hub" 2>/dev/null || {
    echo "WARN: huggingface_hub not found"
}

python -c "import transformers" 2>/dev/null || {
    echo "WARN: transformers not found"
}

python -c "import torch" 2>/dev/null || {
    echo "WARN: PyTorch not found"
}

python -c "import numpy" 2>/dev/null || {
    echo "WARN: NumPy not found"
}

python -c "import PIL" 2>/dev/null || {
    echo "WARN: Pillow not found"
}

python -c "import tqdm" 2>/dev/null || {
    echo "WARN: tqdm not found"
}

python -c "import scipy" 2>/dev/null || {
    echo "WARN: scipy not found (needed for mask evaluation)"
}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo ""

CMD="python eval.py --config \"$CONFIG_PATH\""

if [ "$DOWNLOAD_DATA" = true ]; then
    CMD="$CMD --download_data"
fi

eval $CMD
