import pandas as pd
import argparse
import os
from tqdm import tqdm
import json
from PIL import Image
import h5py
from PIL import ImageFile
import torch
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.modeling_outputs import BaseModelOutput

from src.utils import load_data_for_inference, prep_strings, postprocess_preds

ImageFile.LOAD_TRUNCATED_IMAGES = True

PAD_TOKEN = '!'
EOS_TOKEN = '.'
CAPTION_LENGTH = 25


def strip_prefix(filename: str):
    """去掉 COCO_train2014_ / COCO_val2014_ 等前缀，只保留文件名部分"""
    return filename.split('_')[-1] if '_' in filename else filename


def get_target_resolution(encoder_name):
    """根据编码器名称确定目标分辨率"""
    if "336" in encoder_name:
        return 336
    return 224


def evaluate_rag_model(args, feature_extractor, tokenizer, model, eval_df):
    """RAG models can only be evaluated with a batch of length 1."""
    template = open(args.template_path).read().strip() + ' '

    features = None
    if args.features_path is not None:
        try:
            features = h5py.File(args.features_path, 'r')
        except OSError:
            print(f"Warning: Could not open hdf5 file: {args.features_path}. "
                  f"Will fallback to online extraction if possible.")
            features = None

    target_size = get_target_resolution(args.encoder_name)

    out = []
    for idx in tqdm(range(len(eval_df))):
        file_name = eval_df['filename'][idx]
        image_id = eval_df['cocoid'][idx]
        caps = eval_df['caps'][idx]

        decoder_input_ids = prep_strings(
            '',
            tokenizer,
            template=template,
            retrieved_caps=caps,
            k=int(args.k),
            is_test=True
        )

        # 1）优先使用预提取的特征 (h5)
        if features is not None and str(image_id) in features:
            encoder_last_hidden_state = torch.FloatTensor(
                [features[str(image_id)][()]]
            )
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_last_hidden_state.to(args.device)
            )

            with torch.no_grad():
                pred = model.generate(
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=torch.tensor([decoder_input_ids]).to(args.device),
                    **args.generation_kwargs
                )

        # 2）否则在线提取特征
        else:
            real_name = strip_prefix(file_name)
            img_path = os.path.join(args.images_dir, real_name)

            if not os.path.exists(img_path):
                id_name = f"{int(image_id):012d}.jpg"
                alt_path = os.path.join(args.images_dir, id_name)
                if os.path.exists(alt_path):
                    img_path = alt_path
                else:
                    print(f"Skipping missing image: {file_name} / {real_name}")
                    continue

            image = Image.open(img_path).convert("RGB")
            image = image.resize((target_size, target_size), resample=Image.BICUBIC)

            processed = feature_extractor(
                images=image,
                return_tensors="pt",
                do_center_crop=False
            )
            pixel_values = processed["pixel_values"]

            with torch.no_grad():
                pred = model.generate(
                    pixel_values.to(args.device),
                    decoder_input_ids=torch.tensor([decoder_input_ids]).to(args.device),
                    **args.generation_kwargs
                )

        pred = tokenizer.decode(pred[0])
        pred = postprocess_preds(pred, tokenizer)
        out.append({"image_id": int(image_id), "caption": pred})

    if features is not None:
        features.close()

    return out


def evaluate_norag_model(args, feature_extractor, tokenizer, model, eval_df):
    """Models without retrival augmentation can be evaluated with a batch of length >1."""
    out = []
    bs = args.batch_size

    target_size = get_target_resolution(args.encoder_name)

    for idx in tqdm(range(0, len(eval_df), bs)):
        file_names = eval_df['filename'][idx:idx + bs]
        image_ids = eval_df['cocoid'][idx:idx + bs]
        decoder_input_ids = [prep_strings('', tokenizer, is_test=True)
                             for _ in range(len(image_ids))]

        images = []
        for file_name in file_names:
            stripped = strip_prefix(file_name)
            img_path = os.path.join(args.images_dir, stripped)

            if not os.path.exists(img_path):
                alt_path = os.path.join(args.images_dir, file_name)
                if os.path.exists(alt_path):
                    img_path = alt_path
                else:
                    raise FileNotFoundError(f"[Error] Image not found: {img_path}")

            img = Image.open(img_path).convert("RGB")
            img = img.resize((target_size, target_size), resample=Image.BICUBIC)
            images.append(img)

        processed = feature_extractor(
            images=images,
            return_tensors="pt",
            do_center_crop=False
        )
        pixel_values = processed["pixel_values"]

        with torch.no_grad():
            preds = model.generate(
                pixel_values.to(args.device),
                decoder_input_ids=torch.tensor(decoder_input_ids).to(args.device),
                **args.generation_kwargs
            )

        preds = tokenizer.batch_decode(preds)

        for image_id, pred in zip(image_ids, preds):
            pred = postprocess_preds(pred, tokenizer)
            out.append({"image_id": int(image_id), "caption": pred})

    return out


def load_model(args, checkpoint_path):
    config = AutoConfig.from_pretrained(checkpoint_path + '/config.json')
    model = AutoModel.from_pretrained(checkpoint_path)
    model.config = config
    model.eval()
    model.to(args.device)
    return model


def infer_one_checkpoint(args, feature_extractor, tokenizer, checkpoint_path, eval_df, infer_fn):
    model = load_model(args, checkpoint_path)
    preds = infer_fn(args, feature_extractor, tokenizer, model, eval_df)
    with open(os.path.join(checkpoint_path, args.outfile_name), 'w') as outfile:
        json.dump(preds, outfile)


def register_model_and_config():
    from transformers import AutoModelForCausalLM
    from src.vision_encoder_decoder import SmallCap, SmallCapConfig
    from src.gpt2 import ThisGPT2Config, ThisGPT2LMHeadModel
    from src.opt import ThisOPTConfig, ThisOPTForCausalLM
    from src.xglm import ThisXGLMConfig, ThisXGLMForCausalLM

    AutoConfig.register("this_xglm", ThisXGLMConfig)
    AutoModel.register(ThisXGLMConfig, ThisXGLMForCausalLM)
    AutoModelForCausalLM.register(ThisXGLMConfig, ThisXGLMForCausalLM)

    AutoConfig.register("this_opt", ThisOPTConfig)
    AutoModel.register(ThisOPTConfig, ThisOPTForCausalLM)
    AutoModelForCausalLM.register(ThisOPTConfig, ThisOPTForCausalLM)

    AutoConfig.register("this_gpt2", ThisGPT2Config)
    AutoModel.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    AutoModelForCausalLM.register(ThisGPT2Config, ThisGPT2LMHeadModel)

    AutoConfig.register("smallcap", SmallCapConfig)
    AutoModel.register(SmallCapConfig, SmallCap)


def main(args):
    register_model_and_config()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 只在 disable_rag 时强制不用特征
    if args.disable_rag:
        args.features_path = None

    # 如果有 hdf5 特征，就不需要 feature_extractor
    if args.features_path is not None:
        feature_extractor = None
    else:
        feature_extractor = AutoProcessor.from_pretrained(args.encoder_name)

    if args.disable_rag:
        args.k = 0
        infer_fn = evaluate_norag_model
    else:
        infer_fn = evaluate_rag_model

    # 选择 split
    if args.infer_train:
        split = ['train', 'restval']
    elif args.infer_test:
        split = ['test']
    else:
        split = ['val']

    print(f"Loading annotations from: {args.annotations_path}")
    full_data = json.load(open(args.annotations_path, 'r'))['images']

    data = {'train': [], 'restval': [], 'val': [], 'test': [], 'train_restval': []}
    for img_data in full_data:
        if img_data['split'] in ['train', 'restval']:
            data['train_restval'].append(img_data)
        if img_data['split'] in data:
            data[img_data['split']].append(img_data)

    train_count = len(data['train'])
    restval_count = len([x for x in full_data if x['split'] == 'restval'])
    print(
        f"Loaded {train_count} train, {restval_count} restval, "
        f"{len(data['val'])} val, {len(data['test'])} test images."
    )

    if args.infer_train:
        print(f"✅ 合并 train+restval，共 {train_count + restval_count} 张图像")

    # 加载检索字幕（只在 RAG 下）
    if not args.disable_rag:
        print(f"Loading retrieved captions from: {args.captions_path}")
        retrieved_caps_dict = json.load(open(args.captions_path))

        from collections import defaultdict
        img_id2caps = defaultdict(list)
        for image_id_str, captions in retrieved_caps_dict.items():
            img_id2caps[int(image_id_str)] = captions

        for split_name in data:
            for img_data in data[split_name]:
                img_data['caps'] = img_id2caps[img_data['cocoid']]
    else:
        for split_name in data:
            for img_data in data[split_name]:
                img_data['caps'] = []

    # === 生成 eval_df（★ 这是之前缺失导致 NameError 的部分） ===
    if isinstance(split, list):
        combined = []
        for sp in split:
            combined += data[sp]
        eval_df = pd.DataFrame(combined)
    else:
        eval_df = pd.DataFrame(data[split])

    # 再计算 split_name 用于输出文件名
    if isinstance(split, list):
        if 'test' in split:
            split_name = 'test'
        elif 'train' in split and 'restval' in split:
            split_name = 'train_restval'
        else:
            split_name = '_'.join(split)
    else:
        split_name = split

    args.outfile_name = f'{split_name}_preds.json'

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_name)
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.eos_token = EOS_TOKEN

    # generation 配置
    args.generation_kwargs = {
        'max_new_tokens': CAPTION_LENGTH,
        'no_repeat_ngram_size': 0,    #从0改到2，防止重复
        'length_penalty': 0.,  #从0.改成1.0
        'num_beams': 3,    #从3改到5
        'early_stopping': True,
        'eos_token_id': tokenizer.eos_token_id
    }

    # 推理
    if args.checkpoint_path is not None:
        checkpoint_path = os.path.join(args.model_path, args.checkpoint_path)
        infer_one_checkpoint(args, feature_extractor, tokenizer, checkpoint_path, eval_df, infer_fn)
    else:
        for checkpoint_path in os.listdir(args.model_path):
            if 'runs' in checkpoint_path:
                continue
            checkpoint_path = os.path.join(args.model_path, checkpoint_path)
            # if os.path.exists(os.path.join(checkpoint_path, args.outfile_name)):
            #     print(f'Found existing file for', checkpoint_path)
            # else:
            infer_one_checkpoint(args, feature_extractor, tokenizer, checkpoint_path, eval_df, infer_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Inference')
    parser.add_argument("--images_dir", type=str, default="data/images/",
                        help="Directory where input image features are stored")
    parser.add_argument("--features_path", type=str, default="features_vitl14/val.hdf5",
                        help="H5 file with cached input image features. "
                             "Set to None to use raw images.")
    parser.add_argument("--annotations_path", type=str, default="data/dataset_coco.json",
                        help="JSON file with annotations in Karpathy splits")

    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model to use for inference")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to checkpoint to use for inference; "
                             "If not specified, will infer with all checkpoints")

    parser.add_argument("--infer_test", action="store_true", default=False,
                        help="Use test data instead of val data")
    parser.add_argument("--infer_train", action="store_true", default=False,
                        help="Use train data instead of val data")

    parser.add_argument("--encoder_name", type=str,
                        default="clip-vit-large-patch14",
                        help="Encoder name as found of HuggingFace or stored locally")
    parser.add_argument("--decoder_name", type=str, default="gpt2",
                        help="Decoder name as found of HuggingFace or stored locally")

    parser.add_argument("--disable_rag", action="store_true", default=False,
                        help="Disable retrieval augmentation or not")
    parser.add_argument("--k", type=int, default=4,
                        help="Number of retrieved captions to use in prefix")
    parser.add_argument("--retrieval_encoder", type=str, default="vit-L/14",
                        help="Visual encoder used for retieving captions")
    parser.add_argument("--captions_path", type=str,
                        default="data/retrieved_caps_vitl14.json",
                        help="JSON file with retrieved captions")
    parser.add_argument("--template_path", type=str, default="src/template.txt",
                        help="TXT file with template")

    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size; only matter if evaluating a norag model")

    args = parser.parse_args()
    main(args)
