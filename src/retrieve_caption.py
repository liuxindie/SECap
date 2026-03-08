import json
from tqdm import tqdm
from transformers import AutoTokenizer
import clip
import torch
import faiss
import os
import numpy as np
from PIL import Image
from PIL import ImageFile
import argparse

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_coco_data(coco_data_path):
    """We load in all images and only the train captions."""

    annotations = json.load(open(coco_data_path))['images']
    images = []
    captions = []
    for item in annotations:
        if item['split'] == 'restval':
            item['split'] = 'train'
        if item['split'] == 'train':
            for sentence in item['sentences']:
                captions.append(
                    {
                        'image_id': item['cocoid'],
                        'caption': ' '.join(sentence['tokens'])
                    }
                )
        images.append(
            {
                'image_id': item['cocoid'],
                'file_name': item['filename'].split('_')[-1]
            }
        )

    return images, captions


def filter_captions(data):

    decoder_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    bs = 512

    image_ids = [d['image_id'] for d in data]
    caps = [d['caption'] for d in data]
    encodings = []
    for idx in range(0, len(data), bs):
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
            filtered_captions.append(cap)

    return filtered_image_ids, filtered_captions


def encode_captions(captions, model, device):

    bs = 256
    encoded_captions = []

    for idx in tqdm(range(0, len(captions), bs)):
        with torch.no_grad():
            input_ids = clip.tokenize(captions[idx:idx + bs]).to(device)
            encoded_captions.append(model.encode_text(input_ids).cpu().numpy())

    encoded_captions = np.concatenate(encoded_captions)

    return encoded_captions


def encode_images(images, image_path, model, feature_extractor, device):

    image_ids = [i['image_id'] for i in images]

    bs = 64
    image_features = []

    for idx in tqdm(range(0, len(images), bs)):
        image_input = [
            feature_extractor(Image.open(os.path.join(image_path, i['file_name'])))
            for i in images[idx:idx + bs]
        ]
        with torch.no_grad():
            image_features.append(
                model.encode_image(torch.tensor(np.stack(image_input)).to(device)).cpu().numpy()
            )

    image_features = np.concatenate(image_features)

    return image_ids, image_features


def get_nns(captions, images, k=15):
    xq = images.astype(np.float32)
    xb = captions.astype(np.float32)
    faiss.normalize_L2(xb)
    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)
    faiss.normalize_L2(xq)
    D, I = index.search(xq, k)

    return index, I


def filter_nns(nns, xb_image_ids, captions, xq_image_ids):
    """ We filter out nearest neighbors which are actual captions for the query image, keeping 7 neighbors per image."""
    retrieved_captions = {}
    for nns_list, image_id in zip(nns, xq_image_ids):
        good_nns = []
        for nn in nns_list:
            if xb_image_ids[nn] == image_id:
                continue
            good_nns.append(captions[nn])
            if len(good_nns) == 7:
                break
        assert len(good_nns) == 7
        retrieved_captions[image_id] = good_nns
    return retrieved_captions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieval_encoder",
        type=str,
        default="ViT-L/14",
        help="CLIP encoder used for retrieval, e.g. RN50x64, ViT-L/14"
    )
    parser.add_argument(
        "--coco_data_path",
        type=str,
        default="data/dataset_coco.json"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="data/images/"
    )
    parser.add_argument(
        "--out_index",
        type=str,
        default=None,
        help="Path to save FAISS index"
    )
    parser.add_argument(
        "--out_caps",
        type=str,
        default=None,
        help="Path to save captions list (for index)"
    )
    parser.add_argument(
        "--out_retrieved",
        type=str,
        default=None,
        help="Path to save retrieved captions JSON"
    )
    args = parser.parse_args()

    coco_data_path = args.coco_data_path
    image_path = args.image_path

    print("Loading data")
    images, captions = load_coco_data(coco_data_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model: {args.retrieval_encoder}")
    clip_model, feature_extractor = clip.load(args.retrieval_encoder, device=device)

    print("Filtering captions")
    xb_image_ids, captions = filter_captions(captions)

    print("Encoding captions")
    encoded_captions = encode_captions(captions, clip_model, device)

    print("Encoding images")
    xq_image_ids, encoded_images = encode_images(images, image_path, clip_model, feature_extractor, device)

    print("Retrieving neighbors")
    index, nns = get_nns(encoded_captions, encoded_images)
    retrieved_caps = filter_nns(nns, xb_image_ids, captions, xq_image_ids)

    # --------- 输出文件名根据 encoder 自动命名 ---------
    enc_tag = args.retrieval_encoder.replace("/", "_").replace("@", "").replace("-", "").lower()
    # 简单区分：RN50x64 -> resnet50x64, ViT-L/14 -> vitl14
    if "rn50x64" in enc_tag:
        short = "resnet50x64"
    elif "vitl14" in enc_tag:
        short = "vitl14"
    else:
        short = enc_tag

    out_index = args.out_index or f"datastore/coco_index_{short}"
    out_caps = args.out_caps or f"datastore/coco_index_captions_{short}.json"
    out_retrieved = args.out_retrieved or f"data/retrieved_caps_{short}.json"

    print("Writing files")
    faiss.write_index(index, out_index)
    json.dump(captions, open(out_caps, 'w'))
    json.dump(retrieved_caps, open(out_retrieved, 'w'))

    print("Done.")
    print(f"FAISS index: {out_index}")
    print(f"Index captions: {out_caps}")
    print(f"Retrieved caps: {out_retrieved}")


if __name__ == '__main__':
    main()
