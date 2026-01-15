#!/bin/bash
# Flash Attention 检查脚本

echo "==================== Flash Attention 检查 ===================="
echo ""

# 1. 检查 flash-attn 包是否安装
echo "1. 检查 flash-attn 包安装状态..."
if python -c "import flash_attn; print(f'✓ flash-attn {flash_attn.__version__} 已安装')" 2>/dev/null; then
    echo "   状态: ✓ 已安装"
else
    echo "   状态: ✗ 未安装"
    echo "   建议: pip install flash-attn --no-build-isolation"
fi
echo ""

# 2. 检查代码中的参数
echo "2. 检查代码中的 Flash Attention 参数..."
if grep -q "attn_implementation=attn_implementation" bench/utils/load_model.py; then
    echo "   ✓ Sa2VA 系列使用统一参数"
else
    echo "   ✗ Sa2VA 系列仍使用旧参数"
fi

if grep -q "is_flash_attn_available()" bench/utils/load_model.py; then
    echo "   ✓ 包含自动检测逻辑"
else
    echo "   ✗ 缺少自动检测逻辑"
fi
echo ""

# 3. 检查环境变量
echo "3. 检查内存管理环境变量..."
if grep -q "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" bench/start_eval.sh; then
    echo "   ✓ start_eval.sh 已配置环境变量"
else
    echo "   ✗ start_eval.sh 缺少环境变量"
fi

if grep -q "os.environ\['PYTORCH_CUDA_ALLOC_CONF'\]" bench/eval.py; then
    echo "   ✓ eval.py 已设置环境变量"
else
    echo "   ✗ eval.py 缺少环境变量"
fi
echo ""

# 4. 运行时检查
echo "4. 运行时检查 (需要 GPU)..."
if command -v nvidia-smi &> /dev/null; then
    echo "   GPU 信息:"
    nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader | head -1
    echo ""
    echo "   当前环境变量:"
    echo "   PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-未设置}"
else
    echo "   ⚠ 未检测到 nvidia-smi，跳过 GPU 检查"
fi
echo ""

# 5. 总结
echo "==================== 检查总结 ===================="
echo "如果所有项都显示 ✓，说明配置正确"
echo "如果有 ✗，请参考 OOM_FIX.md 进行修复"
echo ""
