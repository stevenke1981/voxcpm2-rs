"""Run VoxCPM2 Python inference to get reference latent."""
import sys, os
sys.path.insert(0, r"D:\VoxCPM\src")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import numpy as np
from transformers import LlamaTokenizerFast

device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)
model_dir = r"E:\voxcpm2_rust_candle_pack\models\VoxCPM2"

# Load model
from voxcpm.model.voxcpm2 import VoxCPM2Model
print("Loading model...")
# from_local already loads tokenizer internally
# We need the tokenizer separately
tokenizer = LlamaTokenizerFast.from_pretrained(model_dir)
model = VoxCPM2Model.from_local(model_dir, device=device_str)
print(f"Model loaded on {device}")

# Prepare input: tokenize text
text = "hello world"
messages = [
    {"role": "user", "content": text},
]
# Apply chat template
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"Prompt: {prompt}")

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
input_ids = inputs["input_ids"].to(device)
# Create text_mask - all ones since all tokens are text
seq_len = input_ids.shape[1]
text_mask = torch.ones(1, seq_len, device=device, dtype=torch.bool)
print(f"Input shape: {input_ids.shape}, tokens: {list(input_ids[0].cpu().numpy())}")

# For zero-shot, we need feat and feat_mask to have the same sequence length as text
# because the code does combined_embed = text_mask * text_embed + feat_mask * feat_embed
B = 1
T = seq_len  # Must match text length for the interleaved embedding
P = model.patch_size
D = model.config.feat_dim

feat = torch.zeros(B, T, P, D, device=device, dtype=torch.bfloat16)
feat_mask = torch.zeros(B, T, device=device, dtype=torch.bool)  # All False = all positions are text

# The text has 11 tokens. For zero-shot, we need text_mask and feat_mask to sum to the same T.
# With T=1, the code should work since feat has 1 patch and feat_mask is all False.
# But text has 11 tokens... let's check the internal concatenation.
# Actually, the code does: combined_embed = text_mask * text_embed + feat_mask * feat_embed
# Where text_embed = [B, text_len, H] and feat_embed = [B, T, H]
# These MUST have the same T dimension.
# So either text_len == T or there's some alignment.
# 
# Actually, looking at the code again:
# B, T, P, D = feat.shape  # T from feat dimension
# feat_embed = prefill_encoder(feat)  # [b, t, H]
# feat_embed = self.enc_to_lm_proj(feat_embed)  # [b, t, 2048]
# text_embed = self.base_lm.embed_tokens(text) * scale_emb  # [b, text_len, 2048]
# combined_embed = text_mask.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed
#
# If text_len != T, this will fail. But the code is written for aligned text+feat.
# Let me try T = input_ids.shape[1] (11, matching text length)
T = input_ids.shape[1]  # Match text length
feat = torch.zeros(B, T, P, D, device=device, dtype=torch.bfloat16)
feat_mask = torch.zeros(B, T, device=device, dtype=torch.bool)  # All False (no audio)

# Run inference
print("Running inference...")
with torch.no_grad():
    gen = model._inference(
        text=input_ids,
        text_mask=text_mask,
        feat=feat,
        feat_mask=feat_mask,
        min_len=2,
        max_len=5,  # match Rust's max_len for comparison
        inference_timesteps=10,
        cfg_value=2.0,
        streaming=False,
    )
    for result in gen:
        if isinstance(result, tuple) and len(result) == 3:
            feat_pred, generated_feat, context_len = result
        elif isinstance(result, tuple) and len(result) == 2:
            feat_pred, generated_feat = result
            context_len = 0
        else:
            print(f"Unexpected result type: {type(result)}")
            continue
        print(f"feat_pred shape: {feat_pred.shape}")
        if hasattr(generated_feat, 'shape'):
            print(f"generated_feat shape: {generated_feat.shape}")
        else:
            print(f"generated_feat type: {type(generated_feat)}")
        
        # Save generated_feat for comparison
        generated_feat_np = generated_feat.float().cpu().numpy().astype(np.float32)
        np.save(r"E:\voxcpm2_rust_candle_pack\output\python_latent.npy", generated_feat_np)
        generated_feat_np.flatten().tofile(
            r"E:\voxcpm2_rust_candle_pack\output\python_latent.f32"
        )
        
        print(f"Latent saved: shape={generated_feat_np.shape}")
        print(f"Latent peak: {np.abs(generated_feat_np).max():.6f}")
        print(f"Latent mean: {generated_feat_np.mean():.6f}")
        print(f"Latent std: {generated_feat_np.std():.6f}")

print("Done!")
