import torch
import torch.nn as nn

# --- Monkey Patch for optimum.gptq.quantizer NameError ---
# Issue: 'QuantizeConfig' is not defined in some versions of optimum/gptq/quantizer.py
try:
    import optimum.gptq.quantizer
    # Check if QuantizeConfig is missing in the module
    if not hasattr(optimum.gptq.quantizer, "QuantizeConfig"):
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Applying Monkey Patch for optimum.gptq.quantizer.QuantizeConfig...")
        
        try:
            # Try to import BaseQuantizeConfig from auto_gptq
            from auto_gptq import BaseQuantizeConfig
            optimum.gptq.quantizer.QuantizeConfig = BaseQuantizeConfig
        except ImportError:
            # Fallback if auto_gptq is not installed or structure is different
            logger.warning("Could not import BaseQuantizeConfig from auto_gptq. GPTQ loading might fail.")
except ImportError:
    # optimum not installed, ignore
    pass
except Exception as e:
    print(f"Warning: Failed to apply optimum monkey patch: {e}")
# ---------------------------------------------------------

# Monkey patch for transformers.utils.cached_file if missing (for older versions compatibility with peft)
try:
    from transformers.utils import cached_file
except ImportError:
    try:
        from transformers.utils import get_file_from_repo
        import transformers.utils
        transformers.utils.cached_file = get_file_from_repo
    except ImportError:
         pass # If this fails too, we can't do much, let the error propagate normally later if needed

from transformers import (
    CLIPVisionModel,
    AutoModelForCausalLM,
    AutoConfig,
    Blip2Config
)

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

try:
    from transformers import Blip2QFormerModel
except ImportError:
    try:
        from transformers.models.blip_2.modeling_blip_2 import Blip2QFormerModel
    except ImportError:
        # Fallback for older transformers versions or if not available
        import logging
        logging.warning("Blip2QFormerModel not found in transformers. Please ensure transformers>=4.26.0 is installed.")
        Blip2QFormerModel = None
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class GatedProjector(nn.Module):
    """
    Custom Gated Projector
    Architecture: Input (B, 32, 768) -> Channel Attention (SE-Block) -> Learnable Gating (Residual) -> Linear -> Output
    """
    def __init__(self, input_dim, output_dim, reduction=16):
        super().__init__()
        
        # 1. Channel Attention (SE-Block style for Sequence)
        # Reduction ratio for the bottleneck in SE block
        self.reduction_dim = max(input_dim // reduction, 1)
        
        self.channel_attention = nn.Sequential(
            nn.Linear(input_dim, self.reduction_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.reduction_dim, input_dim, bias=False),
            nn.Sigmoid()
        )
        
        # 2. Learnable Gate Parameter for Residual Connection
        # Initialize to 0 so the module starts as an identity mapping (plus noise from random Linear init)
        # Equation: x_gated = x + gate * attention(x)
        self.gate = nn.Parameter(torch.tensor([0.0]))
        
        # 3. Linear Projection to LLM dimension
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Seq_Len, Input_Dim) -> (B, 32, 768)
        """
        b, n, c = x.shape
        
        # --- Channel Attention ---
        # Global Average Pooling across the sequence dimension (dim=1) -> (B, C)
        y = x.mean(dim=1)
        
        # Excitation -> (B, C)
        weights = self.channel_attention(y)
        
        # Reweight: (B, N, C) * (B, 1, C) broadcast
        # This highlights important channels across all tokens
        x_attended = x * weights.unsqueeze(1)
        
        # --- Learnable Gating with Residual Connection ---
        # x_gated = x + alpha * f(x)
        x_gated = x + self.gate * x_attended
        
        # --- Projection ---
        # Map to LLM hidden dimension -> (B, 32, 4096/5120)
        out = self.proj(x_gated)
        
        return out


class RetrievalAugmentedCaptionModel(nn.Module):
    def __init__(
        self, 
        qformer_path, 
        vicuna_path, 
        vit_path="clip-vit-large-patch14", # Default CLIP ViT-L/14
        num_query_tokens=32,
        lora_r=8,
        lora_alpha=32,
        device="cuda"
    ):
        super().__init__()
        self.device = device
        self.num_query_tokens = num_query_tokens
        
        logger.info("Initializing RetrievalAugmentedCaptionModel...")

        # =================================================================
        # 1. Visual Dimension Adapter (Input: Cached CLIP Features)
        # =================================================================
        # We assume input features come from pre-computed HDF5 (CLIP ViT-L/14)
        # Standard CLIP ViT-L hidden size is 1024
        self.visual_dim_in = 1024
        # BLIP-2 Q-Former input (EVA-CLIP based): 1408
        self.visual_dim_out = 1408
        
        logger.info(f"Creating Visual Projection Layer: {self.visual_dim_in} -> {self.visual_dim_out}")
        self.visual_projection = nn.Linear(self.visual_dim_in, self.visual_dim_out)
        # This layer is trainable by default
        
        # =================================================================
        # 2. Load Q-Former & Unfreeze Only Query Embeddings
        # =================================================================
        logger.info(f"Loading Q-Former from {qformer_path}...")
        try:
            # 尝试先作为 Blip2Config 加载，以提取 qformer_config
            # 如果 qformer_path 指向完整的 Blip2 模型目录，这是必须的
            config = Blip2Config.from_pretrained(qformer_path)
            if hasattr(config, "qformer_config"):
                logger.info("Detected Blip2Config, extracting qformer_config for Q-Former initialization...")
                self.qformer = Blip2QFormerModel.from_pretrained(qformer_path, config=config.qformer_config)
            else:
                self.qformer = Blip2QFormerModel.from_pretrained(qformer_path)
        except Exception as e:
            logger.warning(f"Could not load path as Blip2Config or extract qformer_config: {e}. Falling back to default loading.")
            self.qformer = Blip2QFormerModel.from_pretrained(qformer_path)
        
        # [Trainable Queries]
        # Initialize learned query tokens (B, 32, 768)
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_tokens, self.qformer.config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=0.02)
        
        # [Freezing Strategy]
        # Freeze ALL Q-Former parameters
        for param in self.qformer.parameters():
            param.requires_grad = False
        
        # Ensure query_tokens are trainable
        self.query_tokens.requires_grad = True
        logger.info("Q-Former frozen. Query tokens initialized and set to trainable.")

        # =================================================================
        # 3. Gated Projector
        # =================================================================
        qformer_dim = self.qformer.config.hidden_size
        
        # Determine LLM hidden size
        vicuna_config = AutoConfig.from_pretrained(vicuna_path, local_files_only=True)
        llm_dim = vicuna_config.hidden_size
        
        logger.info(f"Initializing GatedProjector: {qformer_dim} -> {llm_dim}")
        self.projector = GatedProjector(input_dim=qformer_dim, output_dim=llm_dim)

        # =================================================================
        # 4. Load Vicuna (LLM) with 4-bit Quantization & LoRA
        # =================================================================
        logger.info(f"Loading Vicuna (4-bit) from {vicuna_path}...")
        
        # Check if the model config already has a quantization configuration (e.g., GPTQ)
        # If it does, we should NOT pass BitsAndBytesConfig, as they are incompatible.
        if hasattr(vicuna_config, "quantization_config") and vicuna_config.quantization_config:
            logger.info(f"Detected existing quantization config in {vicuna_path} (likely GPTQ/AWQ). Skipping BitsAndBytes 4-bit quantization.")
            quantization_kwargs = {}
        elif BitsAndBytesConfig is not None:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            quantization_kwargs = {"quantization_config": bnb_config}
        else:
            logger.warning("BitsAndBytesConfig not found (transformers version too old?). Loading model without 4-bit quantization.")
            quantization_kwargs = {}
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            vicuna_path,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
            **quantization_kwargs
        )
        
        # Prepare for k-bit training (freezes LLM, casts layer norm to fp32, etc.)
        self.llm = prepare_model_for_kbit_training(self.llm)
        
        # Apply LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=lora_r, 
            lora_alpha=lora_alpha, 
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]
        )
        
        self.llm = get_peft_model(self.llm, peft_config)
        self.llm.print_trainable_parameters()
        
        # Move components to device
        # self.vision_model is removed
        self.visual_projection.to(self.device)
        self.qformer.to(self.device)
        self.projector.to(self.device)

    def forward(
        self,
        encoder_outputs,
        retrieved_input_ids,
        input_ids=None,
        attention_mask=None,
        labels=None,
        pixel_values=None # Ignored, kept for compatibility signatures
    ):
        """
        Args:
            encoder_outputs: (B, Seq_Len, Dim) Pre-computed visual features (Required)
            retrieved_input_ids: (B, N_Retrieved) Token IDs of retrieved captions
            input_ids: (B, Seq_Len) Token IDs of target caption/instruction (optional for inference)
            attention_mask: Mask for input_ids
            labels: Labels for calculating loss
        """
        batch_size = encoder_outputs.shape[0]
        
        # ---------------------------------------------------------
        # 1. Visual Branch (From Cached Features)
        # ---------------------------------------------------------
        # Use pre-computed features (e.g., from HDF5)
        # Ensure they are on the correct device and dtype
        image_embeds = encoder_outputs.to(self.device).to(self.visual_projection.weight.dtype)
            
        # [Visual Projection] 1024 -> 1408
        image_embeds = self.visual_projection(image_embeds)
        
        # Create attention mask for Q-Former
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)

        # ---------------------------------------------------------
        # 2. Q-Former Branch
        # ---------------------------------------------------------
        # Expand queries: (B, 32, 768)
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
        )
        q_hidden_state = query_outputs.last_hidden_state # (B, 32, 768)
        
        # ---------------------------------------------------------
        # 3. Gated Projection
        # ---------------------------------------------------------
        # (B, 32, 768) -> (B, 32, LLM_Dim)
        inputs_visual = self.projector(q_hidden_state)
        
        # ---------------------------------------------------------
        # 4. Retrieval Fusion (Embed -> Concatenate)
        # ---------------------------------------------------------
        # Embed retrieved text: (B, N_ret, LLM_Dim)
        inputs_retrieved = self.llm.get_input_embeddings()(retrieved_input_ids)
        
        # [Fusion Strategy]
        # Concatenate: [Visual Embeddings, Retrieved Text Embeddings]
        # Shape: (B, 32 + N_ret, LLM_Dim)
        context_embeds = torch.cat([inputs_visual, inputs_retrieved], dim=1)
        
        # Create context attention mask (1s for both visual and retrieved)
        # Assuming retrieved_input_ids are padded properly externally or we treat all as valid for now
        context_atts = torch.ones(context_embeds.size()[:-1], dtype=torch.long).to(self.device)
        
        # ---------------------------------------------------------
        # 5. Construct Final LLM Input
        # ---------------------------------------------------------
        if input_ids is not None:
            # Embed target/instruction text
            inputs_text = self.llm.get_input_embeddings()(input_ids)
            
            # Final Concatenation: [Context (Visual+Retrieved), Text]
            inputs_embeds = torch.cat([context_embeds, inputs_text], dim=1)
            
            # Handle Attention Mask
            if attention_mask is not None:
                # Concatenate: [Context_Mask, Text_Mask]
                full_attention_mask = torch.cat([context_atts, attention_mask], dim=1)
            else:
                full_attention_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long).to(self.device)
        else:
            inputs_embeds = context_embeds
            full_attention_mask = context_atts

        # ---------------------------------------------------------
        # 6. Handle Labels for Training
        # ---------------------------------------------------------
        if labels is not None:
            # We must mask out the loss for the visual + retrieved context
            # Create a tensor of -100 with shape of context_embeds
            context_len = context_embeds.shape[1]
            ignored_labels = torch.full(
                (batch_size, context_len), 
                -100, 
                dtype=labels.dtype
            ).to(self.device)
            
            # Concatenate: [-100...-100, text_labels...]
            full_labels = torch.cat([ignored_labels, labels], dim=1)
        else:
            full_labels = None
            
        # ---------------------------------------------------------
        # 7. LLM Forward
        # ---------------------------------------------------------
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
            return_dict=True
        )
        
        return outputs

    @torch.no_grad()
    def generate(self, encoder_outputs, retrieved_input_ids, input_ids=None, pixel_values=None, **generate_kwargs):
        """
        Inference generation method using cached features
        """
        # 1. Encode Images (Projection)
        image_embeds = encoder_outputs.to(self.device).to(self.visual_projection.weight.dtype)
        image_embeds = self.visual_projection(image_embeds)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
        
        # 2. Q-Former
        batch_size = image_embeds.shape[0]
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        query_out = self.qformer(
            query_embeds=query_tokens, 
            encoder_hidden_states=image_embeds, 
            encoder_attention_mask=image_atts
        )
        inputs_visual = self.projector(query_out.last_hidden_state)
        
        # 3. Retrieval
        inputs_retrieved = self.llm.get_input_embeddings()(retrieved_input_ids)
        
        # 4. Context
        context_embeds = torch.cat([inputs_visual, inputs_retrieved], dim=1)
        
        # 5. Prompts
        if input_ids is not None:
            inputs_text = self.llm.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([context_embeds, inputs_text], dim=1)
        else:
            inputs_embeds = context_embeds
            
        # 6. Generate
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            **generate_kwargs
        )
        return outputs