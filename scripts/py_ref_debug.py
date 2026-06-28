"""
Python reference debug script.
Uses VoxCPM2Model.from_local() for proper loading.
Runs zero-shot generation with seed=100, saves per-step AR states.
"""
import sys, os, json, torch, numpy as np
sys.path.insert(0, 'D:\\VoxCPM\\src')
from voxcpm.model.voxcpm2 import VoxCPM2Model

MODEL_DIR = 'models/VoxCPM2'
OUT_DIR = 'output/py_ref'
os.makedirs(OUT_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

torch.manual_seed(100)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(100)

# Load model using the proper from_local method
model = VoxCPM2Model.from_local(MODEL_DIR, optimize=False, device=device)
model = model.eval()
print(f'Model loaded')

# ── Zero-shot inference ──
text = "这是修正后的语音现在应该更清楚"
print(f'Text: {text}')

tokens = model.text_tokenizer(text)
tokens = torch.LongTensor(tokens).to(device)
audio_start = torch.tensor([model.audio_start_token], dtype=torch.int32, device=device)
tokens = torch.cat([tokens, audio_start])
print(f'Tokens: {tokens.shape} = {tokens.tolist()}')

# Match model dtype for all tensors
model_dtype = next(model.parameters()).dtype
print(f'Model dtype: {model_dtype}')

T = len(tokens)
text_mask = torch.ones(1, T, dtype=torch.int32, device=device)
feat_mask = torch.zeros(1, T, dtype=torch.int32, device=device)
feat = torch.zeros(1, T, model.patch_size, model.feat_dim, device=device, dtype=model_dtype)

with torch.inference_mode():
    # ── Prefill ──
    prefill_encoder = getattr(model, '_feat_encoder_raw', model.feat_encoder)
    feat_embed = prefill_encoder(feat)
    feat_embed = model.enc_to_lm_proj(feat_embed)

    scale_emb = model.config.lm_config.scale_emb if model.config.lm_config.use_mup else 1.0
    text_embed = model.base_lm.embed_tokens(tokens) * scale_emb
    
    combined_embed = text_mask.unsqueeze(-1).to(model_dtype) * text_embed.to(model_dtype) + feat_mask.unsqueeze(-1).to(model_dtype) * feat_embed

    enc_outputs, kv_cache_tuple = model.base_lm(inputs_embeds=combined_embed, is_causal=True)
    model.base_lm.kv_cache.fill_caches(kv_cache_tuple)

    # Python: FSQ only on feat positions (zero-shot: none)
    enc_outputs = model.fsq_layer(enc_outputs) * feat_mask.unsqueeze(-1).float() + enc_outputs * text_mask.unsqueeze(-1).float()
    enc_outputs = enc_outputs.to(model_dtype)
    h_lm = enc_outputs[:, -1, :]  # [1, 2048]
    print(f'Prefill h_lm: mean={h_lm.mean():.4f} std={h_lm.std():.4f} peak={h_lm.abs().max():.4f}')

    residual_enc_inputs = model.fusion_concat_proj(
        torch.cat((enc_outputs, feat_mask.unsqueeze(-1).to(model_dtype) * feat_embed.to(model_dtype)), dim=-1)
    )
    residual_enc_outputs, rkv_tuple = model.residual_lm(inputs_embeds=residual_enc_inputs, is_causal=True)
    model.residual_lm.kv_cache.fill_caches(rkv_tuple)
    h_res = residual_enc_outputs[:, -1, :]
    print(f'RALM init: mean={h_res.mean():.4f} std={h_res.std():.4f} peak={h_res.abs().max():.4f}')

    # Save prefill states for direct comparison with Rust
    np.fromiter(enc_outputs[0].flatten().cpu().float().numpy(), dtype=np.float32).tofile(f'{OUT_DIR}/debug_tslm_init.f32')
    np.fromiter(residual_enc_outputs[0].flatten().cpu().float().numpy(), dtype=np.float32).tofile(f'{OUT_DIR}/debug_ralm_init.f32')

    # ── AR Loop ──
    prefix_feat_cond = feat[:, -1, ...].to(model_dtype)  # [1, 4, 64]
    pred_feat_seq = []
    max_len = 100
    min_len = 2

    for step in range(max_len):
        # a. mu
        dit_h = torch.cat([
            model.lm_to_dit_proj(h_lm),
            model.res_to_dit_proj(h_res),
        ], dim=-1)  # [1, 2048]
        
        np.fromiter(h_lm[0].cpu().float().numpy(), dtype=np.float32).tofile(f'{OUT_DIR}/debug_step{step}_hlm_before.f32')
        np.fromiter(h_res[0].cpu().float().numpy(), dtype=np.float32).tofile(f'{OUT_DIR}/debug_step{step}_hres_before.f32')
        np.fromiter(dit_h[0].cpu().float().numpy(), dtype=np.float32).tofile(f'{OUT_DIR}/debug_step{step}_mu_input.f32')
        
        # b. CFM
        pred_feat = model.feat_decoder(
            mu=dit_h, patch_size=model.patch_size,
            cond=prefix_feat_cond.transpose(1, 2).contiguous(),
            n_timesteps=30, cfg_value=2.0,
        ).transpose(1, 2)  # [1, 4, 64]
        
        np.fromiter(pred_feat[0].flatten().cpu().float().numpy(), dtype=np.float32).tofile(f'{OUT_DIR}/debug_step{step}_pred_feat.f32')
        
        # c. curr_embed
        ce = model.feat_encoder(pred_feat.unsqueeze(1).to(model_dtype))  # [1, 1, 1024]
        ce = model.enc_to_lm_proj(ce)  # [1, 1, 2048]
        np.fromiter(ce[0,0].cpu().float().numpy(), dtype=np.float32).tofile(f'{OUT_DIR}/debug_step{step}_currembed.f32')
        
        pred_feat_seq.append(pred_feat.unsqueeze(1))
        prefix_feat_cond = pred_feat
        
        # d. stop head (uses h_lm BEFORE step)
        stop_logits = model.stop_head(model.stop_actn(model.stop_proj(h_lm)))
        stop_flag = stop_logits.argmax(dim=-1)[0].item()
        if step > min_len and stop_flag == 1:
            print(f'  stop_head triggered at step {step}')
            break
        
        if step < 5 or step % 10 == 0:
            pf_peak = pred_feat.abs().max().item()
            pf_abs = pred_feat.abs().mean().item()
            print(f'  step {step}: pred_feat peak={pf_peak:.6f} mean_abs={pf_abs:.6f}')
        
        # e. TSLM step
        lm_hidden = model.base_lm.forward_step(
            ce[:, 0, :],
            torch.tensor([model.base_lm.kv_cache.step()], device=device)
        ).clone()
        
        # f. FSQ
        lm_hidden = model.fsq_layer(lm_hidden)
        h_lm = lm_hidden
        
        # g. fusion
        curr_residual_input = model.fusion_concat_proj(
            torch.cat((lm_hidden, ce[:, 0, :].to(model_dtype)), dim=-1)
        )
        
        # h. RALM step
        h_res = model.residual_lm.forward_step(
            curr_residual_input,
            torch.tensor([model.residual_lm.kv_cache.step()], device=device)
        ).clone()
    
    if pred_feat_seq:
        pred_feat_seq_t = torch.cat(pred_feat_seq, dim=1)
        feat_pred = pred_feat_seq_t.permute(0, 3, 1, 2).reshape(1, model.feat_dim, -1)
        print(f'\nFinal latent: {feat_pred.shape} mean={feat_pred.mean():.4f} std={feat_pred.std():.4f} peak={feat_pred.abs().max():.4f}')
        np.fromiter(feat_pred[0].flatten().cpu().float().numpy(), dtype=np.float32).tofile(f'{OUT_DIR}/latent_py.f32')
    
    print(f'\nTotal steps: {len(pred_feat_seq)}')
    print(f'Debug tensors saved to {OUT_DIR}/')
