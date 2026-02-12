#!/usr/bin/env python3
"""
generation.py

Minimal text generation utilities for the standalone LLaMA-style model in model_modern.py.
This replaces the legacy Meta/FairScale generation stack with a simple, single-process setup.

Usage (CLI):
  python generation.py --ckpt path/to/checkpoint.pt --prompt "Hello" --device cpu

Notes:
- Expects checkpoints saved by the patched training scripts (dict with "model_state_dict").
- Uses tiktoken encodings (default: cl100k_base to match train_openwebtext_mps_patched.py).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn.functional as F
import tiktoken

from .model import ModelArgs, LlamaForCausalLM


def load_checkpoint(ckpt_path: str, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt
    # Fallback: allow passing raw state_dict
    return {"model_state_dict": ckpt}


@torch.no_grad()
def generate(
    model: LlamaForCausalLM,
    enc,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.95,
    eos_token_id: Optional[int] = None,
) -> str:
    model.eval()
    device = next(model.parameters()).device

    # Encode prompt
    ids = enc.encode(prompt)
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]  # (1, T)

    for _ in range(max_new_tokens):
        logits, _ = model(x)
        logits = logits[:, -1, :]  # (1, vocab)

        if temperature <= 0:
            next_id = torch.argmax(logits, dim=-1)
        else:
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)

            # nucleus sampling
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)

            cutoff = cum > top_p
            cutoff[..., 0] = False
            sorted_probs[cutoff] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            sampled = torch.multinomial(sorted_probs, num_samples=1)
            next_id = sorted_idx.gather(-1, sampled).squeeze(-1)

        x = torch.cat([x, next_id[:, None]], dim=1)

        if eos_token_id is not None and int(next_id.item()) == int(eos_token_id):
            break

    return enc.decode(x[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt/.pth")
    ap.add_argument("--device", type=str, default="cpu", help="cpu, cuda, mps")
    ap.add_argument("--tokenizer", type=str, default="cl100k_base", help="tiktoken encoding name")
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)

    # Model args override (must match training config used for ckpt)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--n_layers", type=int, default=12)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_kv_heads", type=int, default=4)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.0)

    args = ap.parse_args()

    device = torch.device(args.device)

    enc = tiktoken.get_encoding(args.tokenizer)

    params = ModelArgs(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        vocab_size=enc.n_vocab,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    )

    model = LlamaForCausalLM(params).to(device)

    ckpt = load_checkpoint(args.ckpt, device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    out = generate(
        model=model,
        enc=enc,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=None,  # tiktoken encodings don't have a universal EOS id
    )
    print(out)


if __name__ == "__main__":
    main()
