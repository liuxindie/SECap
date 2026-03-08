import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageFile

import torch
import h5py

from transformers import logging
from transformers import CLIPFeatureExtractor, CLIPVisionModel
import argparse

logging.set_verbosity_error()
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_annotations(annotations_path):
    """
    从 Karpathy 的 dataset_coco.json 里读出 images 字段，
    并按 split 划分成 train / val / test。
    restval 归到 train 里。
    """
    annotations = json.load(open(annotations_path))['images']
    data = {'train': [], 'val': [], 'test': []}

    for item in annotations:
        file_name = item['filename'].split('_')[-1]
        split = item['split']
        if split == 'restval':
            split = 'train'

        rec = {'file_name': file_name, 'cocoid': item['cocoid']}

        if split == 'train':
            data['train'].append(rec)
        elif split == 'val':
            data['val'].append(rec)
        elif split == 'test':
            data['test'].append(rec)

    return data


def encode_split(
    split,
    records,
    data_dir,
    encoder_name,
    features_dir,
    device,
    batch_size=256
):
    """
    对某个 split（train/val/test）提 ViT-L/14 特征，
    做 8x8 空间池化 + CLS，总共 65 个 token，每个 1024 维，float16 存到 hdf5。
    """

    df = pd.DataFrame(records)
    if len(df) == 0:
        print(f"[{split}] 没有样本，跳过。")
        return

    os.makedirs(features_dir, exist_ok=True)
    out_path = os.path.join(features_dir, f"{split}.hdf5")
    print(f"[{split}] 样本数: {len(df)}")
    print(f"[{split}] 输出文件: {out_path}")

    # 初始化编码器
    print(f"[{split}] 加载编码器: {encoder_name}")
    feature_extractor = CLIPFeatureExtractor.from_pretrained(encoder_name)
    clip_encoder = CLIPVisionModel.from_pretrained(encoder_name).to(device)
    clip_encoder.eval()

    h5_file = h5py.File(out_path, "w")

    for idx in tqdm(range(0, len(df), batch_size), desc=f"{split} batches"):
        batch = df.iloc[idx: idx + batch_size]
        cocoids = batch['cocoid'].tolist()
        file_names = batch['file_name'].tolist()

        images = []
        for fn in file_names:
            img_path = os.path.join(data_dir, fn)
            img = Image.open(img_path).convert("RGB")
            images.append(img)

        with torch.no_grad():
            pixel_values = feature_extractor(
                images, return_tensors="pt"
            ).pixel_values.to(device)
            # last_hidden_state: [B, seq_len, hidden_dim]
            outputs = clip_encoder(pixel_values=pixel_values)
            enc = outputs.last_hidden_state  # float32, [B, L, D]

        enc = enc.cpu().numpy().astype("float16")   # 改成 float16 降空间
        B, L, D = enc.shape

        # ViT 模型：第 0 个 token 通常是 CLS，后面是 patch tokens
        cls = enc[:, 0:1, :]          # [B, 1, D]
        patch = enc[:, 1:, :]         # [B, P, D]

        # 假设 patch token 个数是 16x16 = 256
        P = patch.shape[1]
        H = W = int(P ** 0.5)
        assert H * W == P, f"patch 数 {P} 不能整成方形，H={H}, W={W}"

        patch = patch.reshape(B, H, W, D)  # [B,16,16, D]

        # 8x8 池化：每 2x2 patch 平均一次 → [B,8,8,D]
        new_H = new_W = 8
        assert H % new_H == 0 and W % new_W == 0, "H/W 不能被 8 整除"

        patch_small = patch.reshape(
            B,
            new_H, H // new_H,
            new_W, W // new_W,
            D
        ).mean(axis=(2, 4))   # [B,8,8,D]

        patch_tokens = patch_small.reshape(B, new_H * new_W, D)  # [B,64,D]

        # 拼回 CLS + pooled patch，一共 65 个 token
        enc_pooled = np.concatenate([cls, patch_tokens], axis=1)  # [B,65,D]

        # 写入 hdf5，一个 cocoid 一个 dataset
        for cocoid, feat in zip(cocoids, enc_pooled):
            h5_file.create_dataset(
                str(cocoid),
                data=feat,          # shape: (65, D)
                dtype="float16"
            )

    h5_file.close()
    print(f"[{split}] 完成！保存到 {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder_name",
        type=str,
        default="clip-vit-large-patch14",
        help="ViT-L/14 编码器在本地或 HuggingFace 上的名称/路径"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/images/",
        help="COCO 图片所在目录"
    )
    parser.add_argument(
        "--annotations_path",
        type=str,
        default="data/dataset_coco.json",
        help="Karpathy splits 的 JSON 路径"
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        default="features_vitl14/",
        help="输出 hdf5 特征文件目录"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="特征提取 batch size"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    print("加载标注:", args.annotations_path)
    data = load_annotations(args.annotations_path)

    # 依次生成 train / val / test
    for split in ["train", "val", "test"]:
        encode_split(
            split=split,
            records=data[split],
            data_dir=args.data_dir,
            encoder_name=args.encoder_name,
            features_dir=args.features_dir,
            device=device,
            batch_size=args.batch_size
        )


if __name__ == "__main__":
    main()