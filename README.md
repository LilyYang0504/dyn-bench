## 📊 评测维度与指标

### 评测维度

| 类别 | 任务后缀 |
|------|---------|
| **Camera-Object** | `cameraqa`, `cameramask` |
| **Inter-Object** | `qa`, `objmask` |
| **Object-Scene** | `sceneqa`, `scenemask` |

### 评估指标

- **QA 准确率**: 答案匹配准确率
- **Mask J&F 分数**: 分割掩码的 IoU (J) 和边界 F 值 (F) 的平均值

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
| **InternVL3** | 1B / 2B / 8B / 9B / 14B / 38B / 78B | `OpenGVLab/InternVL3-{1B,2B,8B,9B,78B}` |
| **InternVL3.5** | 1B / 2B / 4B / 8B / 14B / 38B | `OpenGVLab/InternVL3_5-{1B,2B,4B,8B,14B,38B}` |
| **Qwen2.5-VL** | 3B / 7B / 32B / 72B | `Qwen/Qwen2.5-VL-{3B,7B,32B,72B}-Instruct` |
| **Qwen3-VL** | 2B / 4B / 8B / 32B | `Qwen/Qwen3-VL-{2B,4B,8B,32B}-Instruct` |
| **Qwen3-VL-MoE** | 235B-A22B | `Qwen/Qwen3-VL-235B-A22B-Instruct` |
| **LLaVA-OneVision** | 4B / 8B | `lmms-lab/LLaVA-One-Vision-1.5-{4B,8B}-Instruct` |
| **VST** | 7B | `rayruiyang/VST-7B-RL` |
| **Spatial-SSRL** | 7B | `internlm/Spatial-SSRL-7B` |
| **SpatialLadder** | 3B | `hongxingli/SpatialLadder-3B` |
| **SpaceR-SFT** | 3B / 7B | `RUBBISHLIKE/SpaceR-SFT-{3B,7B}` |

## 🚀 快速开始

### 方式一：在线评测

直接从 HuggingFace 拉取模型进行评测，需要网络连接。

#### 1. 环境准备

```bash
# 创建虚拟环境
conda create -n bench python=3.10
conda activate bench

# 安装 PyTorch（根据 CUDA 版本自行安装）
pip install torch torchvision

# 安装其他依赖
pip install transformers huggingface_hub hf_xet
pip install pyyaml numpy pillow tqdm scipy peft einops timm
```

#### 2. 拉取仓库

```bash
git clone https://github.com/LilyYang0504/Bench.git
cd Bench
```

#### 3. 配置文件

编辑 `conf/config.yaml`：

```yaml
model:
  name: "ByteDance/Sa2VA-InternVL3-2B"  # HuggingFace 标准模型名称
  device: "cuda"
  torch_dtype: "bfloat16"
  cache_dir: null
  alias: null

task:
  type: "all"  # 可选: all / qa / mask
```

#### 4. 运行评测

```bash
# 首次运行需下载数据集
bash start_eval.sh --conf ./conf/config.yaml --download

# 后续运行
bash start_eval.sh --conf ./conf/config.yaml
```

---

### 方式二：离线评测

适用于无网络环境或需要匿名测试的场景，需提前下载模型。

#### 1. 环境准备

同方式一的步骤 1-2。

#### 2. 下载模型（在有网络的环境）

```bash
# 下载单个模型
python download_model.py --model "OpenGVLab/InternVL3_5-2B"

# 批量下载多个模型
python download_model.py --model "OpenGVLab/InternVL3_5-2B" "Qwen/Qwen2.5-VL-7B-Instruct"

# 指定自定义缓存目录
python download_model.py --model "OpenGVLab/InternVL3_5-2B" --cache-dir "E:/hf-download"
```

模型下载后的默认路径格式：
```
{HF_HOME}/hub/models--{org}--{model}/snapshots/{hash}/
例如: E:/hf-download/hub/models--OpenGVLab--InternVL3_5-2B/snapshots/7d7bd7b.../
```

#### 3. 配置文件（在离线环境）

**标准方式：**
```yaml
model:
  name: "E:/hf-download/hub/models--OpenGVLab--InternVL3_5-2B/snapshots/7d7bd7b..."
  alias: null  # 路径中包含模型信息，可自动识别
  device: "cuda"
```

**匿名测试方式（模型文件夹已改名）：**
```yaml
model:
  name: "E:/test/mymodel1"           # 重命名后的文件夹路径
  alias: "OpenGVLab/InternVL3_5-2B"  # 映射到标准 HF 名称
  device: "cuda"
```

#### 4. 运行评测

```bash
# 确保已下载数据集（或使用 --download）
bash start_eval.sh --conf ./conf/config.yaml
```

---

## 📝 匿名测试说明

匿名测试允许您将模型文件夹重命名为任意名称（如 `mymodel1`、`modelA`），通过 `alias` 字段映射到标准模型名称。


### 支持的标准模型名称（用于 alias）

- **Sa2VA**: `ByteDance/Sa2VA-{1B,4B,8B}`, `ByteDance/Sa2VA-InternVL3-{2B,8B,14B}`, `ByteDance/Sa2VA-Qwen2_5-VL-{3B,7B}`, `ByteDance/Sa2VA-Qwen3-VL-{2B,4B}`
- **InternVL**: `OpenGVLab/InternVL3-{1B,2B,8B,78B}`, `OpenGVLab/InternVL3_5-{1B,2B,4B,8B,14B,38B}`
- **Qwen**: `Qwen/Qwen2.5-VL-{3B,7B,32B,72B}-Instruct`, `Qwen/Qwen3-VL-{2B,4B,8B,32B}-Instruct`
- **其他**: `lmms-lab/LLaVA-OneVision-*`, `rayruiyang/VST-7B-RL`, `internlm/Spatial-SSRL-7B`, `RUBBISHLIKE/SpaceR-SFT-{3B,7B}`
