import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer
import logging

# ==========================================
# 配置部分
# ==========================================

# [关键要求] 本地模型路径，绝不从 HuggingFace Hub 下载
VICUNA_PATH = '/path/to/your/local/vicuna-13b'

# 获取 logger
logger = logging.getLogger(__name__)

class VicunaDecoder(nn.Module):
    """
    Vicuna-13b 解码器封装类 (基于 LLaMA 架构)
    用于替换原有的 GPT-2 解码器。
    
    架构变更说明:
    原 GPT-2: 使用 Cross-Attention 注入视觉特征
    新 Vicuna: 使用 Linear Projection + Embedding Concatenation (Soft Prompt) 方式
              这是大语言模型(LLM)处理多模态输入的标准范式 (如 Llava, MiniGPT-4)。
    """
    
    def __init__(self, encoder_hidden_size=1024, device='cuda', freeze_llm=True):
        """
        初始化 Vicuna 解码器
        
        Args:
            encoder_hidden_size (int): 视觉编码器的输出维度 (例如 CLIP ViT-L/14 为 1024)
            device (str): 运行设备
            freeze_llm (bool): 是否冻结 LLM 参数 (建议为 True，仅训练投影层，因为 13B 模型显存开销大)
        """
        super().__init__()
        self.device = device
        
        logger.info(f"正在从本地路径加载 Vicuna: {VICUNA_PATH}")
        
        # 1. 加载 Tokenizer
        # use_fast=False 是为了避免某些 sentencepiece 版本的兼容性问题
        try:
            self.tokenizer = LlamaTokenizer.from_pretrained(VICUNA_PATH, local_files_only=True, use_fast=False)
        except Exception as e:
            logger.warning(f"LlamaTokenizer 加载失败，尝试 AutoTokenizer: {e}")
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(VICUNA_PATH, local_files_only=True, use_fast=False)

        # [关键要求] Tokenizer 处理
        # LLaMA/Vicuna 原生没有 pad_token，必须显式设置以避免 batch 生成时报错
        # 优先使用 unk_token，如果没有则使用 eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token if self.tokenizer.unk_token else self.tokenizer.eos_token
            # 确保 pad_token_id 也被正确更新
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        
        logger.info(f"Pad Token 设置为: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")

        # 2. 加载模型 (显存优化)
        # [关键要求] 默认添加 torch_dtype=torch.float16 或 load_in_8bit=True
        
        # 检测是否安装了 bitsandbytes 以支持 8bit 量化
        try:
            import bitsandbytes
            load_in_8bit = True
            logger.info("检测到 bitsandbytes，启用 8-bit 量化加载模式。")
        except ImportError:
            load_in_8bit = False
            logger.info("未检测到 bitsandbytes，使用 float16 加载模式。")

        self.model = LlamaForCausalLM.from_pretrained(
            VICUNA_PATH,
            local_files_only=True,
            torch_dtype=torch.float16,  # 默认使用半精度，节省显存
            load_in_8bit=load_in_8bit,  # 如果环境支持，使用 8bit
            device_map="auto"           # 自动分配显存 (CPU/GPU)
        )

        # 3. 维度对齐
        # Vicuna-13b (LLaMA-13b) 的 hidden_size 通常为 5120
        self.vicuna_hidden_size = self.model.config.hidden_size
        if self.vicuna_hidden_size != 5120:
            logger.warning(f"注意: 检测到模型 hidden_size 为 {self.vicuna_hidden_size}，而非预期的 5120。")

        # [关键要求] 重写投影层 (Projection Layer)
        # 作用: 将 Visual Encoder 的特征维度 (encoder_hidden_size) 映射到 Vicuna 的 Embedding 维度
        self.feature_projection = nn.Linear(encoder_hidden_size, self.vicuna_hidden_size)
        
        # 初始化投影层 (有助于训练收敛)
        self.feature_projection.weight.data.normal_(mean=0.0, std=0.02)
        if self.feature_projection.bias is not None:
            self.feature_projection.bias.data.zero_()

        # 4. 可选：冻结 LLM 参数
        if freeze_llm:
            for name, param in self.model.named_parameters():
                param.requires_grad = False
            logger.info("Vicuna 主干参数已冻结，仅训练 Feature Projection 层。")

    def forward(self, input_ids, encoder_hidden_states, attention_mask=None, labels=None):
        """
        Args:
            input_ids (torch.LongTensor): 文本输入 ID, shape [batch_size, seq_len]
            encoder_hidden_states (torch.FloatTensor): 视觉特征, shape [batch_size, image_len, encoder_dim]
            attention_mask (torch.LongTensor): 文本 Attention Mask
            labels (torch.LongTensor): 标签 (用于计算 loss)
            
        Returns:
            CausalLMOutputWithPast: 模型输出，包含 loss (如果提供了 labels) 和 logits
        """
        
        # 1. 文本 Embedding
        # [batch_size, seq_len, vicuna_dim]
        text_embeds = self.model.get_input_embeddings()(input_ids)
        
        # 2. 视觉特征投影
        # [batch_size, image_len, encoder_dim] -> [batch_size, image_len, vicuna_dim]
        image_embeds = self.feature_projection(encoder_hidden_states)
        
        # 3. 特征拼接 (Visual Tokens + Text Tokens)
        # [batch_size, image_len + seq_len, vicuna_dim]
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        
        # 4. 处理 Attention Mask
        if attention_mask is not None:
            # 这里的 attention_mask 对应 text 部分
            # 我们需要给 image 部分创建一个全是 1 的 mask
            batch_size = attention_mask.shape[0]
            image_len = image_embeds.shape[1]
            
            image_mask = torch.ones((batch_size, image_len), dtype=attention_mask.dtype, device=attention_mask.device)
            
            # 拼接 Mask: [image_mask, text_mask]
            attention_mask = torch.cat([image_mask, attention_mask], dim=1)
            
        # 5. 处理 Labels
        if labels is not None:
            # 图像部分不计算 loss，用 -100 填充
            batch_size = labels.shape[0]
            image_len = image_embeds.shape[1]
            
            ignore_labels = torch.full((batch_size, image_len), -100, dtype=labels.dtype, device=labels.device)
            
            # 拼接 Labels: [ignore_labels, text_labels]
            labels = torch.cat([ignore_labels, labels], dim=1)

        # 6. 前向传播
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs

    def generate(self, encoder_hidden_states, max_new_tokens=20, num_beams=1, **kwargs):
        """
        生成函数封装
        由于直接拼接 embedding 导致 input_ids 长度与 inputs_embeds 不一致，
        这里使用 inputs_embeds 进行生成的简化逻辑。
        """
        # 1. 投影视觉特征
        image_embeds = self.feature_projection(encoder_hidden_states) # [B, ImgLen, D]
        
        # 2. 准备起始 Token (BOS)
        batch_size = image_embeds.shape[0]
        bos_token_id = self.tokenizer.bos_token_id
        
        # 构造 BOS Embedding
        bos_embeds = self.model.get_input_embeddings()(
            torch.tensor([[bos_token_id]], device=self.device)
        ).repeat(batch_size, 1, 1) # [B, 1, D]
        
        # 拼接: [Image, BOS]
        inputs_embeds = torch.cat([image_embeds, bos_embeds], dim=1)
        
        # 注意: 使用 inputs_embeds 进行 generate 在 HF 中较为复杂，
        # 通常需要配合 input_ids (即便是 dummy 的) 或者使用 model.generate 的 inputs_embeds 参数(如果支持)
        # 下面展示标准调用方式：
        
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
        
        return outputs