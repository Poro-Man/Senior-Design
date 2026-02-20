#!/usr/bin/env python3
"""
shard_dataset.py

ONE sharding script for:
  - Raw text datasets (OpenWebText, etc.)           --format text
  - Alpaca-style instruction datasets              --format alpaca
  - Chat/ShareGPT-style datasets (UltraChat, etc.) --format chat

Writes shards:
  tokens_000000.bin  (uint16/uint32)
  mask_000000.bin    (uint8) 1=train loss, 0=ignore
  meta.json

Mask policy:
  text:   all tokens masked (train on everything)
  alpaca: mask only the response span (prompt ignored)
  chat:   mask only assistant spans
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import tiktoken
from datasets import load_dataset


# -----------------------------
# Alpaca formatting
# -----------------------------

def alpaca_prompt(instruction: str, inp: str) -> str:
    instruction = (instruction or "").strip()
    inp = (inp or "").strip()
    if inp:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def format_alpaca(ex: Dict, instr_key: str, input_key: str, output_key: str) -> Tuple[str, str]:
    instr = (ex.get(instr_key) or "").strip()
    inp = (ex.get(input_key) or "").strip()
    out = (ex.get(output_key) or "").strip()
    prompt = alpaca_prompt(instr, inp)
    resp = out + "\n"
    full = prompt + resp
    return full, resp


# -----------------------------
# Chat formatting
# -----------------------------

def normalize_role(role: str) -> str:
    r = (role or "").strip().lower()
    if r in ("user", "human"):
        return "user"
    if r in ("assistant", "gpt", "bot"):
        return "assistant"
    return r


def extract_messages(ex: Dict, path: str) -> Optional[List[Dict]]:
    """
    Supports dot-path traversal:
      messages
      conversation
      conversations
      data.messages
      etc.
    """
    cur = ex
    for key in path.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return None
    return cur if isinstance(cur, list) else None


def build_chat_tokens_and_mask(
    enc,
    messages: List[Dict],
    role_key: str,
    content_key: str,
    user_prefix: str,
    assistant_prefix: str,
) -> Tuple[List[int], List[int]]:
    tokens: List[int] = []
    mask: List[int] = []

    for m in messages:
        role = normalize_role(m.get(role_key, ""))
        content = (m.get(content_key) or "").strip()
        if not content:
            continue

        if role == "user":
            txt = f"{user_prefix}{content}\n"
            ids = enc.encode(txt, disallowed_special=())
            tokens.extend(ids)
            mask.extend([0] * len(ids))

        elif role == "assistant":
            txt = f"{assistant_prefix}{content}\n"
            ids = enc.encode(txt, disallowed_special=())
            tokens.extend(ids)
            mask.extend([1] * len(ids))

        else:
            # unknown roles -> include but don't train
            txt = f"{content}\n"
            ids = enc.encode(txt, disallowed_special=())
            tokens.extend(ids)
            mask.extend([0] * len(ids))

    return tokens, mask


# -----------------------------
# Text formatting
# -----------------------------

def format_text(ex: Dict, text_field: str) -> Optional[str]:
    t = ex.get(text_field)
    if not isinstance(t, str):
        return None
    t = t.strip()
    if not t:
        return None
    # add newline separator so adjacent samples don't glue together
    return t + "\n"


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dataset", required=True, help="HF dataset id (e.g. openwebtext, HuggingFaceH4/ultrachat_200k)")
    ap.add_argument("--subset", default=None, help="Optional HF config/subset name")
    ap.add_argument("--split", default="train")

    ap.add_argument("--format", choices=["text", "alpaca", "chat"], required=True)

    # text mode
    ap.add_argument("--text_field", default="text")

    # alpaca mode
    ap.add_argument("--instr_key", default="instruction")
    ap.add_argument("--input_key", default="input")
    ap.add_argument("--output_key", default="output")

    # chat mode
    ap.add_argument("--messages_path", default="messages", help="Dot-path to list of messages (e.g. messages, conversation)")
    ap.add_argument("--role_key", default="role")
    ap.add_argument("--content_key", default="content")
    ap.add_argument("--user_prefix", default="User: ")
    ap.add_argument("--assistant_prefix", default="Assistant: ")

    # tokenization / sharding
    ap.add_argument("--tokenizer", choices=["cl100k_base", "gpt2"], default="cl100k_base")
    ap.add_argument("--dtype", choices=["uint16", "uint32"], default="uint32")
    ap.add_argument("--tokens_per_shard", type=int, default=25_000_000)
    ap.add_argument("--max_tokens", type=int, default=50_000_000)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    enc = tiktoken.get_encoding(args.tokenizer)
    vocab_size = enc.n_vocab
    dtype_np = np.uint16 if args.dtype == "uint16" else np.uint32

    if args.subset:
        ds = load_dataset(args.dataset, args.subset, split=args.split, streaming=True)
    else:
        ds = load_dataset(args.dataset, split=args.split, streaming=True)

    shard_idx = 0
    buf_tokens: List[int] = []
    buf_mask: List[int] = []
    total = 0

    def flush():
        nonlocal shard_idx, buf_tokens, buf_mask
        if not buf_tokens:
            return
        tok_path = os.path.join(args.out_dir, f"tokens_{shard_idx:06d}.bin")
        msk_path = os.path.join(args.out_dir, f"mask_{shard_idx:06d}.bin")
        np.array(buf_tokens, dtype=dtype_np).tofile(tok_path)
        np.array(buf_mask, dtype=np.uint8).tofile(msk_path)
        print(f"[write] shard {shard_idx} tokens={len(buf_tokens):,} -> {tok_path}")
        shard_idx += 1
        buf_tokens = []
        buf_mask = []

    for ex in ds:
        if args.format == "text":
            txt = format_text(ex, args.text_field)
            if not txt:
                continue
            ids = enc.encode(txt, disallowed_special=())
            msk = [1] * len(ids)

        elif args.format == "alpaca":
            full, resp = format_alpaca(ex, args.instr_key, args.input_key, args.output_key)
            full_ids = enc.encode(full, disallowed_special=())
            prompt_text = full[: -len(resp)]
            prompt_ids = enc.encode(prompt_text, disallowed_special=())
            prompt_len = min(len(prompt_ids), len(full_ids))
            ids = full_ids
            msk = [0] * prompt_len + [1] * (len(ids) - prompt_len)

        else:  # chat
            messages = extract_messages(ex, args.messages_path)
            if not messages:
                continue
            ids, msk = build_chat_tokens_and_mask(
                enc,
                messages,
                role_key=args.role_key,
                content_key=args.content_key,
                user_prefix=args.user_prefix,
                assistant_prefix=args.assistant_prefix,
            )
            if not ids:
                continue

        buf_tokens.extend(ids)
        buf_mask.extend(msk)
        total += len(ids)

        if len(buf_tokens) >= args.tokens_per_shard:
            flush()

        if total >= args.max_tokens:
            break

    flush()

    meta = {
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "format": args.format,
        "tokenizer": args.tokenizer,
        "vocab_size": vocab_size,
        "dtype": args.dtype,
        "text_field": args.text_field if args.format == "text" else None,
        "alpaca_keys": {"instruction": args.instr_key, "input": args.input_key, "output": args.output_key}
        if args.format == "alpaca" else None,
        "chat_keys": {
            "messages_path": args.messages_path,
            "role_key": args.role_key,
            "content_key": args.content_key,
            "user_prefix": args.user_prefix,
            "assistant_prefix": args.assistant_prefix,
        } if args.format == "chat" else None,
        "mask": "uint8 (1=train loss, 0=ignore)",
        "max_tokens_written": total,
        "tokens_per_shard": args.tokens_per_shard,
    }

    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[done] wrote meta.json")


if __name__ == "__main__":
    main()
