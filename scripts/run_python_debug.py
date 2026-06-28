"""Run VoxCPM2 Python inference with detailed step debug."""
import sys, os
sys.path.insert(0, r"D:\VoxCPM\src")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
from transformers import LlamaTokenizerFast

device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)
model_dir = r"E:\voxcpm2_rust_candle_pack\models\VoxCPM2"

from voxcpm.model.voxcpm2 import VoxCPM2Model
print("Loading model...")
tokenizer = LlamaTokenizerFast.from_pretrained(model_dir)
model = VoxCPM2Model.from_local(model_dir, device=device_str)
print(f"Model loaded on {device}")

text = "hello world"
messages = [{"role": "user", "content": text}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
input_ids = inputs["input_ids"].to(device)
seq_len = input_ids.shape[1]
text_mask = torch.ones(1, seq_len, device=device, dtype=torch.bool)
print(f"Input shape: {input_ids.shape}")

feat = torch.zeros(1, seq_len, model.patch_size, model.config.feat_dim, device=device, dtype=torch.bfloat16)
feat_mask = torch.zeros(1, seq_len, device=device, dtype=torch.bool)

# Monkey-patch to capture intermediate values
orig_base_lm_forward = model.base_lm.forward

# Hook into forward_step
orig_forward_step = model.base_lm.forward_step
step_counters = [0]  # mutable counter
def debug_forward_step(inputs_embeds, position_id):
    out = orig_forward_step(inputs_embeds, position_id)
    step = step_counters[0]
    # Save lm_hidden
    out_np = out.float().cpu().numpy().astype(np.float32)
    out_np.flatten().tofile(
        rf"E:\voxcpm2_rust_candle_pack\output\py_hlm_step{step}.f32"
    )
    print(f"  [py debug] forward_step {step}: lm_hidden mean={out_np.mean():.4f} std={out_np.std():.4f} peak={np.abs(out_np).max():.4f}")
    step_counters[0] += 1
    return out

model.base_lm.forward_step = debug_forward_step

# Also hook the RALM forward_step
orig_ralm_forward_step = model.residual_lm.forward_step
ralm_step_counters = [0]
def debug_ralm_forward_step(inputs_embeds, position_id):
    out = orig_ralm_forward_step(inputs_embeds, position_id)
    step = ralm_step_counters[0]
    out_np = out.float().cpu().numpy().astype(np.float32)
    out_np.flatten().tofile(
        rf"E:\voxcpm2_rust_candle_pack\output\py_hres_step{step}.f32"
    )
    print(f"  [py debug] ralm forward_step {step}: hres mean={out_np.mean():.4f} std={out_np.std():.4f} peak={np.abs(out_np).max():.4f}")
    ralm_step_counters[0] += 1
    return out

model.residual_lm.forward_step = debug_ralm_forward_step

# Hook the feat_encoder
orig_feat_enc = model.feat_encoder.forward
feat_step_counters = [0]
def debug_feat_enc(x):
    out = orig_feat_enc(x)
    step = feat_step_counters[0]
    out_np = out.float().cpu().numpy().astype(np.float32)
    out_np.flatten().tofile(
        rf"E:\voxcpm2_rust_candle_pack\output\py_currembed_step{step}.f32"
    )
    print(f"  [py debug] feat_enc step {step}: mean={out_np.mean():.4f} std={out_np.std():.4f}")
    
    # Also save the enc_to_lm_proj output
    lm_proj_np = model.enc_to_lm_proj(out).float().cpu().numpy().astype(np.float32)
    lm_proj_np.flatten().tofile(
        rf"E:\voxcpm2_rust_candle_pack\output\py_currembed_proj{step}.f32"
    )
    print(f"  [py debug] enc_to_lm_proj step {step}: mean={lm_proj_np.mean():.4f} std={lm_proj_np.std():.4f}")
    
    feat_step_counters[0] += 1
    return out

model.feat_encoder.forward = debug_feat_enc

print("Running inference with debug...")
with torch.no_grad():
    gen = model._inference(
        text=input_ids, text_mask=text_mask, feat=feat, feat_mask=feat_mask,
        min_len=2, max_len=1,  # single step to isolate
        inference_timesteps=10, cfg_value=2.0, streaming=False,
    )
    for result in gen:
        if isinstance(result, tuple):
            feat_pred, generated_feat = result[0], result[1]
        break

generated_feat_np = generated_feat.float().cpu().numpy().astype(np.float32)
generated_feat_np.tofile(r"E:\voxcpm2_rust_candle_pack\output\python_latent.f32")
print(f"Latent saved: shape={generated_feat_np.shape}")
print(f"Latent peak: {np.abs(generated_feat_np).max():.6f}")
print(f"Latent mean: {generated_feat_np.mean():.6f}")
print(f"Latent std: {generated_feat_np.std():.6f}")
print("Done!")
