#!/usr/bin/env python3
"""
Golden test: Run first TSLM layer in PyTorch, save reference output.

Uses model.safetensors from models/VoxCPM2/ to extract TSLM layer 0 weights,
runs a forward pass with deterministic input, and saves all intermediate
activations as a .safetensors file for Rust comparison.

Usage: python scripts/golden_tslm.py [--model-dir models/VoxCPM2] [--out golden/tslm_layer0.safetensors]
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file


# ─── Helpers ────────────────────────────────────────────────────────────────


def load_tensor(d, key: str) -> torch.Tensor:
    """Load a torch tensor from safetensors dict reader, convert to F32."""
    if key not in d.keys():
        raise KeyError(f"Key {key!r} not found in safetensors")
    return d.get_tensor(key).float()


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMSNorm: x * rsqrt(mean(x^2) + eps) * weight"""
    dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight.float() * x).to(dtype)


def precompute_rope_freqs(dim: int, max_len: int, theta: float = 10000.0,
                          device: torch.device = torch.device("cpu")) -> tuple:
    """Precompute RoPE frequencies (same as Candle RoPE)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to x (with real-imag rotation)."""
    # x: [batch, heads, seq, head_dim]
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    cos = cos[:x.shape[2], :half].unsqueeze(0).unsqueeze(0)  # [1, 1, seq, half]
    sin = sin[:x.shape[2], :half].unsqueeze(0).unsqueeze(0)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1)


def swiglu_mlp(gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor,
               x: torch.Tensor, gate_b=None, up_b=None, down_b=None) -> torch.Tensor:
    """SwiGLU MLP: down(silu(gate(x)) * up(x))"""
    gate_h = F.linear(x, gate_w, gate_b)
    up_h = F.linear(x, up_w, up_b)
    return F.linear(F.silu(gate_h) * up_h, down_w, down_b)


# ─── TSLM Layer forward ─────────────────────────────────────────────────────


def tslm_layer_forward(
    x: torch.Tensor,
    input_layernorm_weight: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int = 16,
    num_kv_heads: int = 4,
    head_dim: int = 128,
    eps: float = 1e-5,
    # optional bias
    q_proj_bias=None,
    k_proj_bias=None,
    v_proj_bias=None,
    o_proj_bias=None,
) -> dict:
    """Run one TSLM layer forward; return dict of intermediate activations."""
    results = {}

    # ── Pre-attention norm ──
    h = rms_norm(x, input_layernorm_weight, eps)
    results["after_input_layernorm"] = h.detach().clone()

    # ── QKV projections ──
    q = F.linear(h, q_proj_weight, q_proj_bias)  # [B, T, num_heads * head_dim]
    k = F.linear(h, k_proj_weight, k_proj_bias)  # [B, T, num_kv_heads * head_dim]
    v = F.linear(h, v_proj_weight, v_proj_bias)  # [B, T, num_kv_heads * head_dim]
    results["q_proj_out"] = q.detach().clone()
    results["k_proj_out"] = k.detach().clone()
    results["v_proj_out"] = v.detach().clone()

    B, T, _ = q.shape

    # Reshape for attention
    q = q.view(B, T, num_heads, head_dim).transpose(1, 2)       # [B, H, T, D]
    k = k.view(B, T, num_kv_heads, head_dim).transpose(1, 2)    # [B, KV, T, D]
    v = v.view(B, T, num_kv_heads, head_dim).transpose(1, 2)    # [B, KV, T, D]

    # RoPE
    q = apply_rotary_emb(q, cos, sin)
    k = apply_rotary_emb(k, cos, sin)
    results["q_rope_out"] = q.detach().clone()
    results["k_rope_out"] = k.detach().clone()

    # GQA: expand KV heads (insert new dim after kv_heads, before T)
    group_size = num_heads // num_kv_heads
    k = k.unsqueeze(2).expand(B, num_kv_heads, group_size, T, head_dim).reshape(
        B, num_heads, T, head_dim
    )
    v = v.unsqueeze(2).expand(B, num_kv_heads, group_size, T, head_dim).reshape(
        B, num_heads, T, head_dim
    )

    # Scaled dot-product attention
    scale = head_dim ** -0.5
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, T, T]
    results["attn_weights"] = attn_weights.detach().clone()
    attn_weights = F.softmax(attn_weights, dim=-1)
    results["attn_probs"] = attn_weights.detach().clone()

    attn_out = torch.matmul(attn_weights, v)  # [B, H, T, D]
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)  # [B, T, H*D]
    attn_out = F.linear(attn_out, o_proj_weight, o_proj_bias)
    results["attn_output"] = attn_out.detach().clone()

    # Residual
    h = x + attn_out
    results["after_attention_residual"] = h.detach().clone()

    # ── Post-attention norm & MLP ──
    h_norm = rms_norm(h, post_attention_layernorm_weight, eps)
    results["after_post_layernorm"] = h_norm.detach().clone()

    mlp_out = swiglu_mlp(gate_proj_weight, up_proj_weight, down_proj_weight, h_norm,
                          gate_b=None, up_b=None, down_b=None)
    results["mlp_output"] = mlp_out.detach().clone()

    # Final residual
    h = h + mlp_out
    results["layer_output"] = h.detach().clone()

    return results


# ─── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/VoxCPM2", help="Model directory")
    parser.add_argument("--out", default="golden/tslm_layer0.safetensors",
                        help="Output golden file")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    safetensors_path = model_dir / "model.safetensors"
    if not safetensors_path.exists():
        print(f"ERROR: {safetensors_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {safetensors_path} ...")
    ref = {}
    with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
        # Get config
        config_path = model_dir / "config.json"
        import json
        with open(config_path) as cf:
            config = json.load(cf)

        lm_cfg = config.get("lm_config", config)
        num_heads = lm_cfg.get("num_attention_heads", 16)
        num_kv_heads = lm_cfg.get("num_key_value_heads", 4)
        hidden_size = lm_cfg.get("hidden_size", 2048)
        num_layers = lm_cfg.get("num_hidden_layers", 24)
        intermediate_size = lm_cfg.get("intermediate_size", 8192)
        head_dim = lm_cfg.get("kv_channels", 128)
        rms_norm_eps = lm_cfg.get("rms_norm_eps", 1e-5)
        vocab_size = lm_cfg.get("vocab_size", 73448)
        rope_theta = lm_cfg.get("rope_theta", 10000.0)
        max_seq_len = lm_cfg.get("max_position_embeddings", 8192)

        # Load embed_tokens
        embed_weight = load_tensor(f, "base_lm.embed_tokens.weight")
        ref["embed_tokens.weight"] = embed_weight

        # Load first layer weights
        layer_prefix = "base_lm.layers.0"
        ref[f"{layer_prefix}.input_layernorm.weight"] = load_tensor(
            f, f"{layer_prefix}.input_layernorm.weight")
        ref[f"{layer_prefix}.self_attn.q_proj.weight"] = load_tensor(
            f, f"{layer_prefix}.self_attn.q_proj.weight")
        ref[f"{layer_prefix}.self_attn.k_proj.weight"] = load_tensor(
            f, f"{layer_prefix}.self_attn.k_proj.weight")
        ref[f"{layer_prefix}.self_attn.v_proj.weight"] = load_tensor(
            f, f"{layer_prefix}.self_attn.v_proj.weight")
        ref[f"{layer_prefix}.self_attn.o_proj.weight"] = load_tensor(
            f, f"{layer_prefix}.self_attn.o_proj.weight")
        # Bias: check if exists (some projections have optional bias)
        for bias_key in [f"{layer_prefix}.self_attn.q_proj.bias",
                         f"{layer_prefix}.self_attn.k_proj.bias",
                         f"{layer_prefix}.self_attn.v_proj.bias",
                         f"{layer_prefix}.self_attn.o_proj.bias",
                         f"{layer_prefix}.mlp.gate_proj.bias",
                         f"{layer_prefix}.mlp.up_proj.bias",
                         f"{layer_prefix}.mlp.down_proj.bias"]:
            if bias_key in f.keys():
                ref[bias_key] = load_tensor(f, bias_key)
        ref[f"{layer_prefix}.post_attention_layernorm.weight"] = load_tensor(
            f, f"{layer_prefix}.post_attention_layernorm.weight")
        ref[f"{layer_prefix}.mlp.gate_proj.weight"] = load_tensor(
            f, f"{layer_prefix}.mlp.gate_proj.weight")
        ref[f"{layer_prefix}.mlp.up_proj.weight"] = load_tensor(
            f, f"{layer_prefix}.mlp.up_proj.weight")
        ref[f"{layer_prefix}.mlp.down_proj.weight"] = load_tensor(
            f, f"{layer_prefix}.mlp.down_proj.weight")

    print(f"Loaded weights: embed [{tuple(embed_weight.shape)}], "
          f"{num_heads} heads, {num_kv_heads} KV heads, "
          f"{hidden_size} hidden, {num_layers} layers")

    # ── Create sample input (short deterministic token sequence) ──
    # Use token IDs [1, 100, 200] (BOS and two regular tokens)
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_ids = torch.tensor([[1, 100, 200]], dtype=torch.long)  # [1, 3]
    B, T = input_ids.shape
    ref["input_ids"] = input_ids

    # ── Embed lookup ──
    x = F.embedding(input_ids, embed_weight)  # [1, 3, 2048]
    ref["embed_output"] = x.detach().clone()
    print(f"Embed output shape: {tuple(x.shape)}")

    # ── Precompute RoPE ──
    cos, sin = precompute_rope_freqs(head_dim, T, rope_theta)
    ref["rope_cos"] = cos[:T, :head_dim//2]
    ref["rope_sin"] = sin[:T, :head_dim//2]

    # ── Layer 0 forward ──
    results = tslm_layer_forward(
        x,
        input_layernorm_weight=ref[f"{layer_prefix}.input_layernorm.weight"],
        q_proj_weight=ref[f"{layer_prefix}.self_attn.q_proj.weight"],
        k_proj_weight=ref[f"{layer_prefix}.self_attn.k_proj.weight"],
        v_proj_weight=ref[f"{layer_prefix}.self_attn.v_proj.weight"],
        o_proj_weight=ref[f"{layer_prefix}.self_attn.o_proj.weight"],
        post_attention_layernorm_weight=ref[
            f"{layer_prefix}.post_attention_layernorm.weight"],
        gate_proj_weight=ref[f"{layer_prefix}.mlp.gate_proj.weight"],
        up_proj_weight=ref[f"{layer_prefix}.mlp.up_proj.weight"],
        down_proj_weight=ref[f"{layer_prefix}.mlp.down_proj.weight"],
        cos=cos, sin=sin,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        eps=rms_norm_eps,
        # Optional bias (None if not in weights)
        q_proj_bias=ref.get(f"{layer_prefix}.self_attn.q_proj.bias"),
        k_proj_bias=ref.get(f"{layer_prefix}.self_attn.k_proj.bias"),
        v_proj_bias=ref.get(f"{layer_prefix}.self_attn.v_proj.bias"),
        o_proj_bias=ref.get(f"{layer_prefix}.self_attn.o_proj.bias"),
    )

    # Add intermediate results to ref dict
    for k, v in results.items():
        ref[f"golden_{k}"] = v

    # ── Also save the full TSLM output (all 24 layers) ──
    # Run all layers
    h = x
    for layer_idx in range(num_layers):
        lp = f"base_lm.layers.{layer_idx}"
        # Load weights (only first load is cached; for all layers we re-load)
        # Actually we need to reload from safetensors for each layer or cache all
        # Let's cache all layer weights first
    print("Note: full 24-layer forward requires caching all layer weights.")

    # For now, just save the single layer test
    ref["config_vocab_size"] = torch.tensor([vocab_size], dtype=torch.int32)
    ref["config_hidden_size"] = torch.tensor([hidden_size], dtype=torch.int32)
    ref["config_num_heads"] = torch.tensor([num_heads], dtype=torch.int32)
    ref["config_num_kv_heads"] = torch.tensor([num_kv_heads], dtype=torch.int32)
    ref["config_head_dim"] = torch.tensor([head_dim], dtype=torch.int32)
    ref["config_num_layers"] = torch.tensor([num_layers], dtype=torch.int32)
    ref["config_rms_norm_eps"] = torch.tensor([rms_norm_eps], dtype=torch.float64)

    # ── Save ──
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(ref, str(out_path))
    print(f"Golden data saved to {out_path}")
    print(f"  Total tensors: {len(ref)}")
    print(f"  Layer0 output shape: {tuple(results['layer_output'].shape)}")
    print("DONE")


if __name__ == "__main__":
    main()
