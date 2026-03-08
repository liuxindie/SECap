# 文件路径: src/mini_qformer.py

import torch
import torch.nn as nn

class MiniQFormer(nn.Module):
    """
    Mini Q-Former 去噪模块
    作用：作为信息瓶颈，利用 Learnable Queries 从 CLIP 特征中提取关键信息，过滤噪声。
    """
    def __init__(self, hidden_size, num_heads=4, num_layers=2):
        super().__init__()
        # 这里的 hidden_size 应该对应 Decoder (GPT-2) 的 embedding 维度
        self.hidden_size = hidden_size
        
        # 定义 Transformer Decoder Layer
        # batch_first=True 确保输入输出格式为 [Batch, Seq, Dim]
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, 
                                                   dim_feedforward=hidden_size*4, 
                                                   batch_first=True)
        
        # 堆叠层数 (Mini版建议2层，保持轻量)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 输出归一化，有助于训练稳定性
        self.ln_post = nn.LayerNorm(hidden_size)

    def forward(self, queries, visual_features):
        """
        Args:
            queries: [Batch, Num_Queries, Dim] -> 你的“过滤器”
            visual_features: [Batch, Seq_Len, Dim] -> 原始 CLIP 特征 (含噪声)
        Returns:
            refined_features: [Batch, Num_Queries, Dim] -> 去噪后的特征
        """
        # 注意：TransformerDecoder 的输入顺序是 tgt=queries, memory=visual_features
        # 这里的 Cross-Attention 也就是：Query 查 Image，Key/Value 来自 Image
        out = self.decoder(tgt=queries, memory=visual_features)
        return self.ln_post(out)