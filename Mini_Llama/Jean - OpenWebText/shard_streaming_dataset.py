#!/usr/bin/env python3
"""
shard_streaming_dataset.py

Stream a Hugging Face dataset (default: OpenWebText), tokenize, and write fixed-size
token shards to disk. Designed for Tier-2 (~1B params) scale, but reusable for other datasets.

Why this script is useful when you plan to use multiple datasets:
- Same sharding format across datasets (so your training loader can mix/alternate shards)
- Deterministic-ish chunking (fixed tokens per shard)
- Resume-friendly (auto-detect next shard index; optional skip_docs)
- Metadata saved per dataset folder

Recommended Tier-2 token budget:
- total tokens across ALL datasets: ~1–2B tokens
So you might shard:
  owt: 800M tokens
  wiki: 300M tokens
  books: 400M tokens
  total: 1.5B tokens

Examples:
  # OpenWebText ~1B tokens, 50M per shard (20 shards)
  python shard_streaming_dataset.py \
    --out_root ./shards \
    --name owt \
    --dataset openwebtext \
    --split train \
    --tokenizer gpt2 \
    --tokens_per_shard 50000000 \
    --max_tokens 1000000000 \
    --dtype uint16

  # Wikipedia (example dataset) 300M tokens into ./shards/wiki
  python shard_streaming_dataset.py \
    --out_root ./shards \
    --name wiki \
    --dataset wikipedia \
    --subset 20220301.en \
    --split train \
    --text_field text \
    --tokenizer gpt2 \
    --tokens_per_shard 50000000 \
    --max_tokens 300000000 \
    --dtype uint16

Resume:
  - It auto-picks the next shard index if shards already exist.
  - If you must skip documents, set --skip_docs (best-effort; streaming isn't perfectly reproducible).

Notes:
- uint16 is safe if vocab < 65536 (GPT-2 tokenizer is 50257).
- If your tokenizer vocab >= 65536, use --dtype uint32.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from datasets import load_dataset

try:
    import tiktoken
except ImportError as e:
    raise SystemExit("tiktoken is required. Install with: pip install tiktoken") from e


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Output layout: out_root/name/shard_XXXXX.bin + meta.json
    p.add_argument("--out_root", type=str, required=True, help="Root directory for all dataset shards")
    p.add_argument("--name", type=str, required=True, help="Subfolder name under out_root (e.g., owt, wiki, books)")

    # HF dataset controls
    p.add_argument("--dataset", type=str, default="openwebtext", help="HF dataset name (default: openwebtext)")
    p.add_argument("--subset", type=str, default=None, help="HF dataset subset/config name (optional)")
    p.add_argument("--split", type=str, default="train", help="Split (default: train)")
    p.add_argument("--text_field", type=str, default="text", help="Field containing text (default: text)")

    # Tokenization / sharding
    p.add_argument("--tokenizer", type=str, default="gpt2", help="tiktoken encoding name (default: gpt2)")
    p.add_argument("--tokens_per_shard", type=int, default=50_000_000, help="Tokens per shard file (default: 50M)")
    p.add_argument("--max_tokens", type=int, default=1_000_000_000, help="Stop after this many tokens (default: 1B)")
    p.add_argument("--dtype", choices=["uint16", "uint32"], default="uint16", help="Token dtype on disk")
    p.add_argument("--doc_boundary", type=str, default="\n\n", help="Text appended after each doc (default: \\n\\n)")

    # Resume / progress
    p.add_argument("--skip_docs", type=int, default=0, help="Best-effort resume: skip this many docs/rows")
    p.add_argument("--start_shard_idx", type=int, default=-1, help="Override start shard index (-1 auto-detect)")
    p.add_argument("--log_every_docs", type=int, default=2000, help="Log progress every N docs")
    return p.parse_args()


def np_dtype(dtype_str: str):
    return np.uint16 if dtype_str == "uint16" else np.uint32


def safe_write_bin(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    arr.tofile(tmp)
    tmp.replace(path)


def detect_next_shard_idx(out_dir: Path) -> int:
    # Looks for shard_00000.bin, shard_00001.bin, ...
    pat = re.compile(r"shard_(\d{5})\.bin$")
    mx = -1
    for p in out_dir.glob("shard_*.bin"):
        m = pat.search(p.name)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx + 1


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_root) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding(args.tokenizer)
    vocab_size = getattr(enc, "n_vocab", None)

    # Safety: dtype must hold vocab IDs
    if vocab_size is not None and args.dtype == "uint16" and vocab_size >= 65536:
        raise SystemExit(f"Tokenizer vocab_size={vocab_size} won't fit in uint16. Use --dtype uint32.")

    shard_idx = args.start_shard_idx if args.start_shard_idx >= 0 else detect_next_shard_idx(out_dir)

    # Metadata (append/update friendly)
    meta_path = out_dir / "meta.json"
    meta = {
        "name": args.name,
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "text_field": args.text_field,
        "tokenizer": args.tokenizer,
        "vocab_size": vocab_size,
        "dtype": args.dtype,
        "tokens_per_shard": args.tokens_per_shard,
        "doc_boundary": args.doc_boundary,
        "max_tokens_target": args.max_tokens,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    # Streaming dataset (no full download required)
    if args.subset:
        ds = load_dataset(args.dataset, args.subset, split=args.split, streaming=True, trust_remote_code=True)  # CHANGE
    else:
        ds = load_dataset(args.dataset, split=args.split, streaming=True, trust_remote_code=True)  # CHANGE

    token_buffer: List[int] = []
    total_tokens_written = 0
    docs_seen = 0

    start_time = time.time()

    def flush_shard(tokens: List[int], idx: int) -> Tuple[Path, int]:
        arr = np.asarray(tokens, dtype=np_dtype(args.dtype))
        shard_path = out_dir / f"shard_{idx:05d}.bin"
        safe_write_bin(shard_path, arr)
        return shard_path, int(arr.size)

    print(f"[config] out_dir={out_dir}")
    print(f"[config] dataset={args.dataset} subset={args.subset} split={args.split} streaming=True")
    print(f"[config] text_field={args.text_field}")
    print(f"[config] tokenizer={args.tokenizer} vocab_size={vocab_size} dtype={args.dtype}")
    print(f"[config] tokens_per_shard={args.tokens_per_shard:,} max_tokens={args.max_tokens:,}")
    print(f"[config] skip_docs={args.skip_docs:,} start_shard_idx={shard_idx}")
    print("")

    for row in ds:
        # Best-effort skip (streaming isn't perfectly reproducible, but this works well enough)
        if docs_seen < args.skip_docs:
            docs_seen += 1
            continue

        text = row.get(args.text_field, "")
        if not isinstance(text, str):
            text = str(text)

        # Add boundary to avoid silently blending docs
        text = text + args.doc_boundary

        toks = enc.encode(text, allowed_special="all")
        token_buffer.extend(toks)
        docs_seen += 1

        # Write full shards
        while len(token_buffer) >= args.tokens_per_shard:
            shard_tokens = token_buffer[: args.tokens_per_shard]
            token_buffer = token_buffer[args.tokens_per_shard :]

            shard_path, n = flush_shard(shard_tokens, shard_idx)
            shard_idx += 1
            total_tokens_written += n

            elapsed = time.time() - start_time
            tok_per_s = total_tokens_written / max(elapsed, 1e-9)
            print(
                f"[shard] wrote={shard_path.name} "
                f"tokens={n:,} total={total_tokens_written:,} docs_seen={docs_seen:,} "
                f"({tok_per_s:,.0f} tok/s)"
            )

            if total_tokens_written >= args.max_tokens:
                print("[done] reached max_tokens cap")
                return

        # Progress log
        if args.log_every_docs and (docs_seen % args.log_every_docs == 0):
            elapsed = time.time() - start_time
            tok_per_s = total_tokens_written / max(elapsed, 1e-9)
            print(
                f"[progress] docs_seen={docs_seen:,} buffer_tokens={len(token_buffer):,} "
                f"total_tokens_written={total_tokens_written:,} ({tok_per_s:,.0f} tok/s)"
            )

        if total_tokens_written >= args.max_tokens:
            break

    # Final partial shard (optional)
    if token_buffer:
        shard_path, n = flush_shard(token_buffer, shard_idx)
        total_tokens_written += n
        print(f"[final] wrote={shard_path.name} tokens={n:,} total={total_tokens_written:,} docs_seen={docs_seen:,}")

    print("[done] stream exhausted")


if __name__ == "__main__":
    main()
