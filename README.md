# SECap - Lightweight Retrieval-Augmented Image Captioning System

<div align="center">
**A lightweight image captioning model achieving high performance through retrieval augmentation**
</div>

---

## 📋 Table of Contents

- [Project Introduction](#project-introduction)
- [Environment Setup](#environment-setup)
- [Model Training](#model-training)
- [Inference & Evaluation](#inference-evaluation)

---

## 🔧 Project Introduction

SECap is an innovative image captioning system that achieves performance comparable to large-scale models with only 9.8M trainable parameters. It utilizes dynamic gating-enhanced semantic attention techniques and a semantic adaptive mapping module. This code is directly related to our submission to *The Visual Computer*. If you find our code/data/models or ideas useful for your research, please consider citing this paper. DOI URL: https://doi.org/10.5281/zenodo.18977057

## 🔧 Environment Setup

### System Requirements
- Python 3.9
- CUDA 11.x (Recommended for GPU acceleration)
- At least 16GB RAM (More memory required for training)

### Installation Steps

1. **Create a Conda Environment**
```bash
conda create -n SECap python=3.9
conda activate SECap
```
2.**Install Dependencies**
```Bash
pip install -r requirements.txt
```
3.**Install CLIP (for feature extraction)**
```Bash
git lfs install
git clone [https://huggingface.co/openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
```
4.**Download gpt2**
```Bash
git clone [https://huggingface.co/gpt2](https://huggingface.co/gpt2)
```
## 🔧 Model Training
### Data Preparation
1. **Download the COCO Dataset**
Download the Karpathy split file dataset_coco.json from [Kaggle](https://www.kaggle.com/datasets/shtvkumar/karpathy-splits).Place it in the data/ directory.
2. **Download COCO Images**
Download the 2014 version images (train, val, test) from the [COCO Official Website](https://cocodataset.org/#download).Place them in the data/images/ directory.
Naming format: 12-digit number + .jpg, e.g., 000000000001.jpg
4. **Directory Structure**
```bash
data/
├── dataset_coco.json      # Karpathy划分

└── images/
    ├── 000000000001.jpg
    ├── 000000000002.jpg
    └── ...
```
4. **Preparation**
```bash
mkdir datastore
```
Download the following files and place them in the datastore/ directory:
[COCO Index File](https://drive.google.com/file/d/1ZP5I-xbjaNU7cU48C_ctHd95SaA0jBHe/view?usp=sharing) and 
[Associated Captions File](https://drive.google.com/file/d/1BT0Qc6g40fvtnJ_yY0aipfCuCMgu5qaR/view?usp=sharing)

# Create feature directory
```bash
mkdir features
# Extract features for training and validation sets
python src/extract_fea.py
```
The extracted features will be saved in H5 files for training and retrieval.

### Retrieve Caption Generation
```bash
python src/retrieve_captions.py
```
This step will retrieve the top-K most similar captions for each image to be used for prompt construction during training.

### Start Training
```bash
python train.py \
    --encoder_name clip-vit-large-patch14 \
    --decoder_name gpt2 \
    --features_path features/train.h5 \
    --output_dir experiments/rag_7M \
    --template_path src/template.txt
```

### Key Parameters Description

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `--encoder_name` | Name of the vision encoder | `clip-vit-large-patch14` |
| `--decoder_name` | Name of the language decoder | `gpt2` |
| `--features_path` | Path to pre-extracted features | 必需 |
| `--template_path` | Path to prompt template | `src/template.txt` |

### Training Monitoring
The model will automatically save checkpoints to the experiments/ directory:
```bash
experiments/rag_7M/
├── checkpoint-8856/
├── checkpoint-17712/
├── ...
└── checkpoint-{final}/
```

## 🔧 Inference & Evaluation
#### Validation Set Inference
```Bash
python infer.py \
    --model_path experiments/rag_7M \
    --checkpoint_path checkpoint-17712 \
    --features_path features/val.h5
```

#### Test Set Inference
```bash
python infer.py \
    --model_path experiments/rag_7M \
    --checkpoint_path checkpoint-17712 \
    --features_path features/test.h5 \
    --infer_test
```
### Evaluation Results

```bash
python coco-caption/run_eval.py \
    data/dataset_coco.json \
    experiments/rag_7M/checkpoint-17712/test_preds.json
```
## Acknowledgments
This code utilizes resources from [SmallCap](https://github.com/RitaRamo/smallcap),We thank the authors for open-sourcing their wonderful project.








