#!/usr/bin/env python3
"""Generate golden tokenizer test cases from VoxCPM2 official tokenizer.

Usage:
    python tests/golden/make_tokenizer_golden.py --model-dir models/VoxCPM2 --out tests/golden/tokenizer_cases.json
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/VoxCPM2")
    ap.add_argument("--out", default="tests/golden/tokenizer_cases.json")
    args = ap.parse_args()

    model_dir = Path(args.model_dir).resolve()
    sys.path.insert(0, str(model_dir))
    # Import VoxCPM2's custom tokenizer
    from tokenization_voxcpm2 import VoxCPM2Tokenizer

    tok = VoxCPM2Tokenizer.from_pretrained(str(model_dir))

    # Test cases covering:
    # - Plain English
    # - Chinese (short and long)
    # - Japanese
    # - Korean
    # - Voice Design format
    # - Chat messages
    # - Mixed content
    test_cases = [
        # English
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        # Chinese
        "你好",
        "你好，世界！",
        "这是一个测试句子，用來驗證中文分詞。",
        # Japanese
        "こんにちは",
        "こんにちは、世界！",
        # Korean
        "안녕하세요",
        "안녕하세요, 세계!",
        # CJK mixed
        "你好こんにちは안녕하세요",
        # Voice Design format (with special tokens)
        "<|background|>安静的图书馆<|/background|><|characters|>温柔的女声<|/characters|>",
        # Numbers and punctuation
        "2024年6月27日，温度25.5°C。",
        # Longer sentence
        "基于 VoxCPM2 的 Rust 推理引擎，使用 Candle 框架實作 GPU 加速。",
    ]

    results = []

    for text in test_cases:
        # Encode WITHOUT special tokens (add_special_tokens=False)
        ids_no_special = tok.encode(text, add_special_tokens=False)
        # Encode WITH special tokens (default: add_special_tokens=True)
        ids_with_special = tok.encode(text, add_special_tokens=True)
        # Decode back to verify roundtrip
        decoded = tok.decode(ids_with_special, skip_special_tokens=True)

        results.append({
            "text": text,
            "ids_no_special": ids_no_special,
            "ids_with_special": ids_with_special,
            "decoded": decoded,
        })

    # Chat template test
    chat_messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
        {"role": "user", "content": "用中文說一句話。"},
    ]

    chat_text = tok.apply_chat_template(chat_messages, add_generation_prompt=True, tokenize=False)
    chat_ids = tok.encode(chat_text, add_special_tokens=True)

    results.append({
        "chat": True,
        "messages": chat_messages,
        "chat_text": chat_text,
        "ids": chat_ids,
    })

    # Special token IDs
    special_ids = {
        "bos_token": str(tok.bos_token),
        "bos_token_id": tok.bos_token_id,
        "eos_token": str(tok.eos_token),
        "eos_token_id": tok.eos_token_id,
        "unk_token_id": tok.unk_token_id,
        "pad_token_id": tok.pad_token_id,
        "im_start_id": tok.convert_tokens_to_ids("<|im_start|>"),
        "im_end_id": tok.convert_tokens_to_ids("<|im_end|>"),
        "audio_start_id": tok.convert_tokens_to_ids("<|audio_start|>"),
        "audio_end_id": tok.convert_tokens_to_ids("<|audio_end|>"),
        "vocab_size": len(tok),
    }

    output = {
        "special_ids": special_ids,
        "cases": results,
        "meta": {
            "generated_by": "make_tokenizer_golden.py",
            "model": "openbmb/VoxCPM2",
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path} with {len(results)} cases")


if __name__ == "__main__":
    main()
