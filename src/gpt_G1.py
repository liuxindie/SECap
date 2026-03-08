# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

# Global counter for debugging gate_score
GATE_SCORE_PRINT_COUNTER = 0

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.gpt2.modeling_gpt2 import load_tf_weights_in_gpt2, GPT2LMHeadModel, GPT2MLP, GPT2Attention, \
    GPT2Block, GPT2Model

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel, SequenceSummary
from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers.utils import (
    ModelOutput,
    logging,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

if version.parse(torch.__version__) >= version.parse("1.6"):
    is_amp_available = True
    from torch.cuda.amp import autocast
else:
    is_amp_available = False


class ThisGPT2Config(GPT2Config):
    model_type = "this_gpt2"

    def __init__(
            self,
            cross_attention_reduce_factor=1,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.cross_attention_reduce_factor = cross_attention_reduce_factor
        # 新增：如果外面没显式设，就默认为 hidden_size（768）
        # 之后 from_pretrained / vision_encoder_decoder 会把它改成 1024
        if not hasattr(self, "encoder_hidden_size"):
            self.encoder_hidden_size = self.hidden_size

class SemanticAdaptiveMappingModule(nn.Module):
    """
    Semantic Adaptive Mapping Module (SAMM)
    Based on Squeeze-and-Excitation (SE) block structure adapted for sequence data.
    Enhances key semantic features from the encoder before cross-attention.
    """
    def __init__(self, hidden_dim, reduction=16):
        super().__init__()
        # Ensure reduction doesn't make hidden dimension 0
        reduced_dim = max(1, hidden_dim // reduction)
        self.fc1 = nn.Linear(hidden_dim, reduced_dim, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(reduced_dim, hidden_dim, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [batch_size, seq_len, hidden_dim]
        
        # Squeeze: Global Average Pooling over sequence dimension (dim=1)
        # y shape: [batch_size, hidden_dim]
        y = x.mean(dim=1)
        
        # Excitation
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        
        # Reweight: Element-wise multiplication
        # Reshape y to [batch_size, 1, hidden_dim] for broadcasting
        y = y.unsqueeze(1)
        
        # Residual Connection: Output = (x * y) + x
        return x * y + x

class ThisGPT2Attention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention, layer_idx)

        # print("this gpt2")

        # print("self.is_cross_attention = is_cross_attention", self.is_cross_attention, is_cross_attention)

        self.cross_attention_reduce_factor = config.cross_attention_reduce_factor
        # 新增：编码器输出的维度（ViT-L/14 是 1024）
        self.encoder_hidden_size = getattr(config, "encoder_hidden_size", self.embed_dim)

        if self.is_cross_attention:
            self.c_attn = Conv1D(int(2 / self.cross_attention_reduce_factor * self.embed_dim),
                                 self.encoder_hidden_size)
            self.q_attn = Conv1D(int(self.embed_dim / self.cross_attention_reduce_factor), self.embed_dim)
            self.c_proj = Conv1D(self.embed_dim, int(self.embed_dim / self.cross_attention_reduce_factor))

            # Semantic Adaptive Mapping Module
            self.semantic_adaptive_mapping = SemanticAdaptiveMappingModule(
                self.encoder_hidden_size, reduction=16
            )
            
            # Head-Specific Gating Mechanism: Applied after SDPA, before output projection
            # Each head has independent gate parameters for fine-grained control
            self.head_gate_proj = nn.ModuleList([
                Conv1D(int(self.head_dim / self.cross_attention_reduce_factor),
                       int(self.head_dim / self.cross_attention_reduce_factor))
                for _ in range(self.num_heads)
            ])
            
            # [Initialization Strategy: Start Open]
            # Initialize each head's gate with small random noise and bias=2.0
            for gate in self.head_gate_proj:
                gate.weight.data.normal_(mean=0.0, std=0.02)
                gate.bias.data.fill_(2.0)  # Sigmoid(2.0) ≈ 0.88 (Gate mostly open)
            
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
            self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            # Apply Semantic Adaptive Mapping Module to encoder hidden states
            # before they enter cross-attention key/value projection
            encoder_hidden_states = self.semantic_adaptive_mapping(encoder_hidden_states)
            
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            split_size = int(self.split_size / self.cross_attention_reduce_factor)
            head_dim = int(self.head_dim / self.cross_attention_reduce_factor)

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(split_size, dim=2)

            attention_mask = encoder_attention_mask

            query = self._split_heads(query, self.num_heads, head_dim)
            key = self._split_heads(key, self.num_heads, head_dim)
            value = self._split_heads(value, self.num_heads, head_dim)
        else:
            # -----------------------------------------
            head_dim = self.head_dim
            # ------------------------------------------
            
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

            query = self._split_heads(query, self.num_heads, self.head_dim)
            key = self._split_heads(key, self.num_heads, self.head_dim)
            value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # Head-Specific Gating Mechanism: Applied after SDPA, before merging heads
        # Y' = Y ⊙ σ(Q W_θ) where Y is SDPA output, Q is query before splitting
        if encoder_hidden_states is not None:
            # Compute gate scores for each head using the original query (before splitting)
            # Query shape: [batch_size, num_heads, seq_len, head_dim]
            batch_size, num_heads, seq_len, head_dim = query.size()
            
            # Process each head independently with head-specific gate parameters
            gated_attn_output = []
            for head_idx in range(num_heads):
                # Extract current head's attention output
                head_attn_output = attn_output[:, head_idx, :, :]  # [batch_size, seq_len, head_dim]
                
                # Extract current head's query for gate computation
                head_query = query[:, head_idx, :, :]  # [batch_size, seq_len, head_dim]
                
                # Compute gate score for this head: σ(head_query W_θ_head)
                # Reshape for linear layer: [batch_size * seq_len, head_dim]
                head_query_flat = head_query.reshape(-1, head_dim)
                gate_score = torch.sigmoid(self.head_gate_proj[head_idx](head_query_flat))
                
                # Reshape gate score back: [batch_size, seq_len, head_dim]
                gate_score = gate_score.reshape(batch_size, seq_len, head_dim)
                
                # Apply gating: Y' = Y ⊙ gate_score
                gated_head_output = head_attn_output + gate_score
                
                gated_attn_output.append(gated_head_output)
            
            # Stack gated outputs back together
            attn_output = torch.stack(gated_attn_output, dim=1)  # [batch_size, num_heads, seq_len, head_dim]
            
            # Debug: Print gate statistics for first layer
            if self.layer_idx == 0:
                global GATE_SCORE_PRINT_COUNTER
                GATE_SCORE_PRINT_COUNTER += 1
                if GATE_SCORE_PRINT_COUNTER % 500 == 0:
                    avg_gate_scores = []
                    for head_idx in range(num_heads):
                        head_query_flat = query[:, head_idx, :, :].reshape(-1, head_dim)
                        gate_score = torch.sigmoid(self.head_gate_proj[head_idx](head_query_flat))
                        avg_gate_scores.append(gate_score.mean().item())
                    
                    print(f"[DEBUG] Step {GATE_SCORE_PRINT_COUNTER} | Head-specific gate scores (Layer 0): "
                          f"Mean across heads={torch.tensor(avg_gate_scores).mean():.4f}, "
                          f"Min={min(avg_gate_scores):.4f}, "
                          f"Max={max(avg_gate_scores):.4f}")

        # Merge heads and apply output projection
        attn_output = self._merge_heads(attn_output, self.num_heads, head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class ThisGPT2Block(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        hidden_size = config.hidden_size

        if config.add_cross_attention:
            self.crossattention = ThisGPT2Attention(config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)


class ThisGPT2Model(GPT2Model):

    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList([ThisGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])


class ThisGPT2LMHeadModel(GPT2LMHeadModel):
    config_class = ThisGPT2Config

    def __init__(self, config):
        super().__init__(config)
        self.transformer = ThisGPT2Model(config)

