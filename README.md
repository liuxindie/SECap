# SECap - 轻量级检索增强图像描述生成系统

<div align="center">
**轻量级图像描述生成模型，通过检索增强实现高性能**
</div>

---

## 📋 目录

- [项目简介](#项目简介)
- [环境配置](#环境配置)
- [模型训练](#模型训练)
- [推理与评估](#推理与评估)

---

## 项目简介

SECap是一个创新的图像描述生成系统，通过动态门控增强语义注意力技术与语义自适应映射模块，在仅使用9.8M可训练参数的情况下，实现了与大规模模型相媲美的性能。

## 🔧 环境配置

### 系统要求
- Python 3.9
- CUDA 11.x（推荐用于GPU加速）
- 至少16GB内存（训练需要更多）

### 安装步骤

1. **创建Conda环境**
```bash
conda create -n SECap python=3.9
conda activate SECap
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **安装CLIP（用于特征提取）**
```bash
git lfs install
git clone https://huggingface.co/openai/clip-vit-large-patch14
```

4. **下载gpt2**
git clone https://huggingface.co/gpt2





## 模型训练

### 数据准备

1. **下载COCO数据集**
   - 从[Kaggle](https://www.kaggle.com/datasets/shtvkumar/karpathy-splits)下载Karpathy划分文件`dataset_coco.json`
   - 放入`data/`目录

2. **下载COCO图像**
   - 从[COCO官网](https://cocodataset.org/#download)下载2014版本图像（train, val, test）
   - 放入`data/images/`目录
   - 命名格式：12位数字+.jpg，如`000000000001.jpg`

3. **目录结构**
```
data/
├── dataset_coco.json      # Karpathy划分
└── images/
    ├── 000000000001.jpg
    ├── 000000000002.jpg
    └── ...
```
4. **准备**
mkdir datastore
```

下载以下文件并放入`datastore/`目录：
- [COCO索引文件](https://drive.google.com/file/d/1ZP5I-xbjaNU7cU48C_ctHd95SaA0jBHe/view?usp=sharing)
- [关联描述文件](https://drive.google.com/file/d/1BT0Qc6g40fvtnJ_yY0aipfCuCMgu5qaR/view?usp=sharing)

# 创建特征目录
mkdir features

# 提取训练集和验证集特征
python src/extract_fea.py
```
提取的特征将保存在H5文件中，用于训练和检索。
### 检索描述生成

```bash
python src/retrieve_captions.py
```
此步骤会为每个图像检索最相似的K个描述，用于训练时的提示构建。
---

### 启动训练

```bash
python train.py \
    --encoder_name clip-vit-large-patch14 \
    --decoder_name gpt2 \
    --features_path features/train.h5 \
    --output_dir experiments/rag_7M \
    --template_path src/template.txt
```

### 关键参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--encoder_name` | 视觉编码器名称 | `clip-vit-large-patch14` |
| `--decoder_name` | 语言解码器名称 | `gpt2` |
| `--features_path` | 预提取特征路径 | 必需 |
| `--template_path` | 提示模板路径 | `src/template.txt` |

### 训练监控

模型会自动保存检查点到`experiments/`目录：
```
experiments/rag_7M/
├── checkpoint-8856/
├── checkpoint-17712/
├── ...
└── checkpoint-{final}/
```

## 推理与评估

### 运行推理

#### 验证集推理
```bash
python infer.py \
    --model_path experiments/rag_7M \
    --checkpoint_path checkpoint-17712 \
    --features_path features/val.h5
```

#### 测试集推理
```bash
python infer.py \
    --model_path experiments/rag_7M \
    --checkpoint_path checkpoint-17712 \
    --features_path features/test.h5 \
    --infer_test
```

### 评估结果

```bash
python coco-caption/run_eval.py \
    data/dataset_coco.json \
    experiments/rag_7M/checkpoint-17712/test_preds.json
```
