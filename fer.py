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

# 假设你的项目结构中有这些模块，保持原样导入
from src.utils import load_data_for_inference, prep_strings, postprocess_preds

ImageFile.LOAD_TRUNCATED_IMAGES = True

PAD_TOKEN = '!'
EOS_TOKEN = '.'
CAPTION_LENGTH = 25


def get_target_resolution(encoder_name):
    if "336" in encoder_name:
        return 336
    return 224


def load_nocaps_data(annotations_path):
    """读取 nocaps validation json"""
    print(f"Loading annotations from: {annotations_path}")
    with open(annotations_path, 'r') as f:
        content = json.load(f)

    # 提取 images 列表
    images = content.get('images', [])
    data = []
    for item in images:
        data.append({
            'file_name': item['file_name'],
            'image_id': item['id']  # nocaps 使用 id
        })

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} images from nocaps validation set.")
    return df


def evaluate_rag_model(args, feature_extractor, tokenizer, model, eval_df):
    """RAG 模式：Batch size 只能为 1"""
    template = open(args.template_path).read().strip() + ' '

    # 加载 H5 特征
    features = None
    if args.features_path is not None:
        try:
            print(f"Loading features from {args.features_path} ...")
            features = h5py.File(args.features_path, 'r')
        except OSError:
            print(f"Warning: Could not open hdf5 file: {args.features_path}. Fallback to raw images.")
            features = None

    target_size = get_target_resolution(args.encoder_name)
    out = []

    for idx in tqdm(range(len(eval_df)), desc="Inference (RAG)"):
        file_name = eval_df['file_name'][idx]
        image_id = eval_df['image_id'][idx]
        caps = eval_df['caps'][idx]  # 检索到的 caption

        # 准备输入 Prompt
        decoder_input_ids = prep_strings(
            '',
            tokenizer,
            template=template,
            retrieved_caps=caps,
            k=int(args.k),
            is_test=True
        )

        # 1. 优先尝试从 H5 读取特征
        if features is not None and str(image_id) in features:
            # H5 中存的是 float16, 转回 float32 用于推理
            feat_data = features[str(image_id)][()]
            encoder_last_hidden_state = torch.FloatTensor([feat_data])  # [1, 65, D]

            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_last_hidden_state.to(args.device)
            )

            with torch.no_grad():
                pred = model.generate(
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=torch.tensor([decoder_input_ids]).to(args.device),
                    **args.generation_kwargs
                )

        # 2. 如果 H5 中没有，则读取原图
        else:
            img_path = os.path.join(args.images_dir, file_name)
            try:
                image = Image.open(img_path).convert("RGB")
                image = image.resize((target_size, target_size), resample=Image.BICUBIC)

                processed = feature_extractor(
                    images=image,
                    return_tensors="pt",
                    do_center_crop=False
                )
                pixel_values = processed["pixel_values"].to(args.device)

                with torch.no_grad():
                    pred = model.generate(
                        pixel_values=pixel_values,
                        decoder_input_ids=torch.tensor([decoder_input_ids]).to(args.device),
                        **args.generation_kwargs
                    )
            except Exception as e:
                print(f"Error processing image {file_name}: {e}")
                continue

        pred_str = tokenizer.decode(pred[0])
        pred_str = postprocess_preds(pred_str, tokenizer)
        out.append({"image_id": int(image_id), "caption": pred_str})

    if features is not None:
        features.close()

    return out


def evaluate_norag_model(args, feature_extractor, tokenizer, model, eval_df):
    """非 RAG 模式：支持 batch 推理，但为了配合 H5 特征，这里演示逐个或 batch 读取 H5"""
    # 注意：如果使用 H5 特征，通常 batch 处理比较麻烦（因为需要手动拼 tensor），
    # 为了简化逻辑，这里针对 H5 场景写一个简单的 batch 循环，或者沿用 batch=1
    # 下面代码保留了原始 batch 图片读取逻辑，并在 H5 存在时优先使用 H5。

    out = []
    bs = args.batch_size
    target_size = get_target_resolution(args.encoder_name)

    # 加载 H5
    features = None
    if args.features_path is not None:
        try:
            features = h5py.File(args.features_path, 'r')
        except:
            features = None

    for idx in tqdm(range(0, len(eval_df), bs), desc="Inference (No-RAG)"):
        batch_df = eval_df.iloc[idx:idx + bs]
        file_names = batch_df['file_name'].tolist()
        image_ids = batch_df['image_id'].tolist()

        decoder_input_ids = [prep_strings('', tokenizer, is_test=True) for _ in range(len(image_ids))]

        # 检查是否所有图片都在 H5 中
        use_h5_batch = (features is not None) and all(str(iid) in features for iid in image_ids)

        if use_h5_batch:
            # 从 H5 读取 batch 特征
            batch_feats = []
            for iid in image_ids:
                batch_feats.append(features[str(iid)][()])

            # Stack 成 [B, L, D]
            encoder_last_hidden_state = torch.FloatTensor(batch_feats).to(args.device)
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_last_hidden_state)

            with torch.no_grad():
                preds = model.generate(
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=torch.tensor(decoder_input_ids).to(args.device),
                    **args.generation_kwargs
                )
        else:
            # 这里的 fallback 逻辑：如果 batch 中有一个不在 H5，就全读图片（简化处理）
            images = []
            valid_ids = []
            for i, fn in enumerate(file_names):
                img_path = os.path.join(args.images_dir, fn)
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize((target_size, target_size), resample=Image.BICUBIC)
                    images.append(img)
                    valid_ids.append(image_ids[i])
                except:
                    pass

            if not images: continue

            processed = feature_extractor(images=images, return_tensors="pt", do_center_crop=False)
            pixel_values = processed["pixel_values"].to(args.device)

            with torch.no_grad():
                preds = model.generate(
                    pixel_values,
                    decoder_input_ids=torch.tensor(decoder_input_ids).to(args.device),
                    **args.generation_kwargs
                )

        preds_str = tokenizer.batch_decode(preds)
        for iid, p in zip(image_ids, preds_str):
            p = postprocess_preds(p, tokenizer)
            out.append({"image_id": int(iid), "caption": p})

    if features: features.close()
    return out


def load_model(args, checkpoint_path):
    # 使用 trust_remote_code=True 或 use_safetensors=True 防止某些加载错误
    config = AutoConfig.from_pretrained(checkpoint_path + '/config.json')
    model = AutoModel.from_pretrained(checkpoint_path, use_safetensors=True)  # 尝试强制使用 safetensors
    model.config = config
    model.eval()
    model.to(args.device)
    return model


def infer_one_checkpoint(args, feature_extractor, tokenizer, checkpoint_path, eval_df, infer_fn):
    print(f"Inferring with checkpoint: {checkpoint_path}")
    model = load_model(args, checkpoint_path)
    preds = infer_fn(args, feature_extractor, tokenizer, model, eval_df)

    out_file = os.path.join(checkpoint_path, args.outfile_name)
    print(f"Saving predictions to {out_file}")
    with open(out_file, 'w') as outfile:
        json.dump(preds, outfile)


def register_model_and_config():
    # 注册你的自定义模型类
    from transformers import AutoModelForCausalLM
    try:
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
    except ImportError as e:
        print(f"Warning: Model registration failed. Check src imports. {e}")


def main(args):
    register_model_and_config()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 逻辑处理：如果有 H5 路径且不禁用 RAG，则用 RAG 模式
    # 如果禁用 RAG，features_path 依然可以用来加速纯 ViT 编码
    if args.disable_rag:
        # 纯 SmallCap/Captioning 模式
        infer_fn = evaluate_norag_model
        args.k = 0
    else:
        # RAG 模式
        infer_fn = evaluate_rag_model

    # 如果提供了 h5 特征，就不一定要 feature_extractor，但在 fallback 时需要
    # 这里初始化一个以防万一
    try:
        feature_extractor = AutoProcessor.from_pretrained(args.encoder_name)
    except:
        feature_extractor = None

        # 1. 加载数据
    eval_df = load_nocaps_data(args.annotations_path)

    # 2. 如果是 RAG 模式，需要加载检索到的 Captions
    if not args.disable_rag:
        if not os.path.exists(args.captions_path):
            raise FileNotFoundError(f"RAG enabled but captions file not found: {args.captions_path}")

        print(f"Loading retrieved captions from: {args.captions_path}")
        retrieved_caps_dict = json.load(open(args.captions_path))

        # 将检索结果映射到 dataframe
        # 假设 json key 是 string 类型的 image_id
        caps_map = {}
        for k, v in retrieved_caps_dict.items():
            caps_map[str(k)] = v  # 确保 key 是 str

        # 填充 caps 列
        # 注意：nocaps 的 id 在 df 里可能是 int，需要转 str 匹配
        eval_df['caps'] = eval_df['image_id'].apply(lambda x: caps_map.get(str(x), []))
    else:
        eval_df['caps'] = [[] for _ in range(len(eval_df))]

    args.outfile_name = 'nocaps_val_preds.json'

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_name)
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.eos_token = EOS_TOKEN

    args.generation_kwargs = {
        'max_new_tokens': CAPTION_LENGTH,
        'no_repeat_ngram_size': 0,
        'length_penalty': 0.,
        'num_beams': 3,
        'early_stopping': True,
        'eos_token_id': tokenizer.eos_token_id
    }

    if args.checkpoint_path is not None:
        full_ckpt_path = os.path.join(args.model_path, args.checkpoint_path)
        infer_one_checkpoint(args, feature_extractor, tokenizer, full_ckpt_path, eval_df, infer_fn)
    else:
        # 遍历目录下所有 checkpoint
        for ckpt in os.listdir(args.model_path):
            if 'checkpoint-' in ckpt:
                full_ckpt_path = os.path.join(args.model_path, ckpt)
                infer_one_checkpoint(args, feature_extractor, tokenizer, full_ckpt_path, eval_df, infer_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nocaps Inference')

    # 路径参数
    parser.add_argument("--images_dir", type=str, default="datastore/nocaps_val_images",
                        help="Nocaps 图片目录 (用于 fallback)")
    parser.add_argument("--features_path", type=str, default="features__vitl14/nocaps_val.hdf5",
                        help="之前生成的 HDF5 特征文件路径")
    parser.add_argument("--annotations_path", type=str, default="datastore/nocaps_val_4500_captions.json",
                        help="Nocaps 验证集 JSON 路径")

    # 模型参数
    parser.add_argument("--model_path", type=str, required=True,
                        help="训练好的模型根目录 (包含 checkpoint 文件夹)")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="指定某个 checkpoint 名字，不指定则跑所有")

    parser.add_argument("--encoder_name", type=str, default="clip-vit-large-patch14")
    parser.add_argument("--decoder_name", type=str, default="gpt2")

    # RAG 参数
    parser.add_argument("--disable_rag", action="store_true", help="是否禁用 RAG")
    parser.add_argument("--k", type=int, default=4, help="检索数量")
    parser.add_argument("--captions_path", type=str, default="data/retrieved_caps_vitl14.json",
                        help="Nocaps 的检索结果 JSON (如果 disable_rag=False 必填)")
    parser.add_argument("--template_path", type=str, default="src/template.txt")

    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    main(args)