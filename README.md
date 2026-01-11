## 📊 评测维度与指标
### 评测维度
| 类别 | 任务后缀 |
|------|---------|
| **Camera-Object** | `cameraqa`, `cameramask` |
| **Inter-Object** |  `qa`, `objmask` |
| **Object-Scene** |  `sceneqa`, `scenemask` |

### 评估指标

- **QA准确率**：答案匹配准确率
- **Mask J&F分数**：分割掩码的IoU (J) 和边界F值 (F) 的平均

## 🤖 现支持的模型

| 模型系列 | 模型规模 | HuggingFace模型ID |
|---------|---------|------------------|
| **Sa2VA** | 1B / 4B / 8B | `ByteDance/Sa2VA-1B` |
| **Sa2VA-InternVL3** | 2B / 8B / 14B | `ByteDance/Sa2VA-InternVL3-{2B,8B,14B}` |
| **Sa2VA-Qwen2_5-VL** | 3B / 7B | `ByteDance/Sa2VA-Qwen2_5-VL-{3B,7B}` |
| **Sa2VA-Qwen3-VL** | 2B / 4B | `ByteDance/Sa2VA-Qwen3-VL-{2B,4B}` |

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
conda create -n bench python=3.10
conda activate bench

# 安装依赖
pip install torch torchvision  # 根据CUDA版本安装(仅作示例)

pip install transformers
pip install huggingface_hub
pip install pyyaml numpy pillow tqdm scipy
pip install hf_xet
```

### 2. 配置

 `conf/config.yaml`：

```yaml
model:
  name: "ByteDance/Sa2VA-InternVL3-2B"
  device: "cuda"
  torch_dtype: "bfloat16"

task:
  type: "all"  # all / qa / mask
```

### 3. 运行评测

```bash
# 首次评测需下载数据, 通过传入download参数实现
bash start_eval.sh --conf ./conf/config.yaml [--download]
```