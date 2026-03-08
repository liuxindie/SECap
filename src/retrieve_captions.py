import json
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPProcessor, CLIPModel
import torch
import faiss
import os
import numpy as np
from PIL import Image, ImageFile
import argparse

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_coco_data(coco_data_path):
    """加载数据"""
    print(f"Loading annotations from {coco_data_path}...")
    annotations = json.load(open(coco_data_path))['images']
    images = []
    captions = []
    for item in annotations:
        if item['split'] == 'restval':
            item['split'] = 'train'

        # 只有 Train 的文本进入数据库
        if item['split'] == 'train':
            for sentence in item['sentences']:
                captions.append({
                    'image_id': item['cocoid'],
                    'caption': ' '.join(sentence['tokens'])
                })

        # 所有图片（包括 Test）都作为查询
        images.append({
            'image_id': item['cocoid'],
            'file_name': item['filename'].split('_')[-1]
        })

    print(f"Loaded {len(images)} images and {len(captions)} database captions.")
    return images, captions


def filter_captions(data):
    """
    过滤掉过短的 caption。
    FIX: 修复了返回类型，确保返回的是字典列表，而不是字符串列表。
    """
    print("Filtering captions...")
    decoder_name = 'gpt2'

    # 尝试加载 tokenizer，增加容错
    try:
        tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    except:
        tokenizer = AutoTokenizer.from_pretrained(decoder_name, local_files_only=True)

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    bs = 512

    image_ids = [d['image_id'] for d in data]
    caps = [d['caption'] for d in data]
    encodings = []

    for idx in tqdm(range(0, len(data), bs)):
        encodings += tokenizer.batch_encode_plus(
            caps[idx:idx + bs],
            return_tensors='np',
            padding=True
        )['input_ids'].tolist()

    filtered_image_ids, filtered_captions = [], []

    assert len(image_ids) == len(caps) and len(caps) == len(encodings)
    for image_id, cap, encoding in zip(image_ids, caps, encodings):
        if len(encoding) <= 25:
            filtered_image_ids.append(image_id)
            # === 核心修复在这里 ===
            # 原来是: filtered_captions.append(cap) -> 导致后面报错
            # 现在改为: 保持字典结构
            filtered_captions.append({'image_id': image_id, 'caption': cap})

    return filtered_image_ids, filtered_captions


def encode_captions(captions, model, processor, device):
    """使用 HF Processor 编码文本"""
    bs = 256
    encoded_captions = []
    print("Encoding captions...")

    # 现在 captions 是字典列表，这行代码就不会报错了
    text_list = [c['caption'] for c in captions]

    for idx in tqdm(range(0, len(text_list), bs)):
        batch_texts = text_list[idx:idx + bs]
        with torch.no_grad():
            inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model.get_text_features(**inputs)
            # 归一化
            outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            encoded_captions.append(outputs.cpu().numpy())

    encoded_captions = np.concatenate(encoded_captions)
    return encoded_captions


def encode_images(images, image_path, model, processor, device):
    """使用 HF Processor 编码图片"""
    image_ids = [i['image_id'] for i in images]
    bs = 64
    image_features = []
    print("Encoding images...")

    for idx in tqdm(range(0, len(images), bs)):
        batch_imgs = []
        for i in images[idx:idx + bs]:
            path = os.path.join(image_path, i['file_name'])
            try:
                # 如果没有去前缀，尝试加上前缀寻找
                if not os.path.exists(path):
                    # 备选方案：尝试拼接 "COCO_val2014_" 等前缀，根据你的实际情况调整
                    pass

                image = Image.open(path).convert("RGB")
                batch_imgs.append(image)
            except Exception as e:
                print(f"Error reading {path}: {e}")
                # 填充黑图防止崩溃
                batch_imgs.append(Image.new('RGB', (224, 224)))

        if not batch_imgs:
            continue

        with torch.no_grad():
            # 使用 HF 标准预处理 (Resize + CenterCrop)
            # 既然你决定跳过手动 Resize，这里直接用 processor 即可
            inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
            outputs = model.get_image_features(**inputs)
            # 归一化
            outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            image_features.append(outputs.cpu().numpy())

    image_features = np.concatenate(image_features)
    return image_ids, image_features


def get_nns(captions, images, k=15):
    print("Building FAISS index...")
    xq = images.astype(np.float32)
    xb = captions.astype(np.float32)

    faiss.normalize_L2(xb)
    faiss.normalize_L2(xq)

    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)

    print(f"Searching index (k={k})...")
    D, I = index.search(xq, k)

    return index, I

def filter_nns(nns, xb_image_ids, captions, xq_image_ids):
    """过滤逻辑"""
    print("Filtering results...")
    retrieved_captions = {}

    # 构建查找表
    # captions 是字典列表，这里取出来存入 lookup
    caption_lookup = {i: c for i, c in enumerate(captions)}
    caption_img_lookup = {i: xb_image_ids[i] for i in range(len(xb_image_ids))}

    for i, (nns_row, query_img_id) in enumerate(zip(nns, xq_image_ids)):
        good_nns = []
        for nn_idx in nns_row:
            if caption_img_lookup[nn_idx] == query_img_id:
                continue
            # 从字典中取出 caption 字符串
            good_nns.append(caption_lookup[nn_idx]['caption'])
            if len(good_nns) == 4:
                break

        retrieved_captions[str(query_img_id)] = good_nns

    return retrieved_captions


def main():
    parser = argparse.ArgumentParser()
    # 填你的本地 HuggingFace 文件夹路径
    parser.add_argument("--retrieval_encoder", type=str, required=True, help="Path to local HF model folder")
    parser.add_argument("--coco_data_path", type=str, default='data/dataset_coco.json')
    parser.add_argument("--image_path", type=str, default='data/images/')
    parser.add_argument("--out_path", type=str, default='data/retrieved_caps_vitl14.json')
    args = parser.parse_args()

    print('Loading data')
    images, captions = load_coco_data(args.coco_data_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载本地模型
    print(f"Loading local model from: {args.retrieval_encoder}")
    model = CLIPModel.from_pretrained(args.retrieval_encoder, local_files_only=True).to(device)
    processor = CLIPProcessor.from_pretrained(args.retrieval_encoder, local_files_only=True)
    model.eval()

    print('Filtering captions')
    xb_image_ids, filtered_captions = filter_captions(captions)

    print('Encoding captions')
    encoded_captions = encode_captions(filtered_captions, model, processor, device)

    print('Encoding images')
    xq_image_ids, encoded_images = encode_images(images, args.image_path, model, processor, device)

    print('Retrieving neighbors')
    index, nns = get_nns(encoded_captions, encoded_images)
    retrieved_caps = filter_nns(nns, xb_image_ids, filtered_captions, xq_image_ids)

    print(f'Writing files to {args.out_path}')
    with open(args.out_path, 'w') as f:
        json.dump(retrieved_caps, f)

    print("Done!")


if __name__ == '__main__':
    main()