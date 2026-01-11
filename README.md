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

### 支持 QA + Mask 任务的模型

| 模型系列 | 模型规模 | HuggingFace模型ID |
|---------|---------|------------------|
| **Sa2VA** | 1B / 4B / 8B | `ByteDance/Sa2VA-{1B,4B,8B}` |
| **Sa2VA-InternVL3** | 2B / 8B / 14B | `ByteDance/Sa2VA-InternVL3-{2B,8B,14B}` |
| **Sa2VA-Qwen2_5-VL** | 3B / 7B | `ByteDance/Sa2VA-Qwen2_5-VL-{3B,7B}` |
| **Sa2VA-Qwen3-VL** | 2B / 4B | `ByteDance/Sa2VA-Qwen3-VL-{2B,4B}` |

### 仅支持 QA 任务的模型

| 模型系列 | 模型规模 | HuggingFace模型ID |
|---------|---------|------------------|
| **InternVL3** | 1B / 2B / 4B / 8B / 78B | `OpenGVLab/InternVL3-{1B,2B,4B,8B,78B}` |
| **InternVL3.5** | 1B / 2B / 4B / 8B / 78B | `OpenGVLab/InternVL3_5-{1B,2B,4B,8B,78B}` |
| **Qwen2.5-VL** | 2B / 7B / 72B | `Qwen/Qwen2.5-VL-{2B,7B,72B}-Instruct` |
| **Qwen3-VL** | 2B / 8B / 14B | `Qwen/Qwen3-VL-{2B,8B,14B}-Instruct` |
| **Qwen3-VL-MoE** | 235B-A22B | `Qwen/Qwen3-VL-235B-A22B-Instruct` |
| **LLaVA-OneVision** | 0.5B / 7B / 72B | `lmms-lab/LLaVA-OneVision-{0.5B,7B,72B}` |
| **VST** | 7B | `rayruiyang/VST-7B-RL` |
| **Spatial-SSRL** | 7B | `internlm/Spatial-SSRL-7B` |
| **SpatialLadder** | 3B | `hongxingli/SpatialLadder-3B` |

> **注意**: 仅支持 QA 任务的模型在运行时会自动跳过 Mask 任务。若 `task.type` 设置为 `all` 或 `mask`，系统将只执行 QA 部分并记录警告。

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
conda create -n bench python=3.10
conda activate bench

# 安装依赖
pip install torch torchvision  # 根据CUDA版本安装(仅作示例)

pip install transformers huggingface_hub hf_xet
pip install pyyaml numpy pillow tqdm scipy peft einops timm
```

### 2. 拉取仓库到本地
```bash
git clone https://github.com/LilyYang0504/Bench.git
cd Bench
```


### 3. 配置文件

 `conf/config.yaml`：

```yaml
model:
  name: "ByteDance/Sa2VA-InternVL3-2B"
  device: "cuda"
  torch_dtype: "bfloat16"
  cache_dir: null

task:
  type: "all"  # all / qa / mask
```

### 4. 运行评测

```bash
# 首次评测需下载数据, 通过传入download参数实现
bash start_eval.sh --conf ./conf/config.yaml [--download]
```