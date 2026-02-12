#!/usr/bin/env python3
"""
train_shards_cuda.py

Train from local .bin token shards created by shard_openwebtext.py.
- Fast random sampling from shards via memmap
- Saves checkpoint with model_args as plain dict (no pickle issues)

Assumes model.py exports:
  ModelArgs, LlamaForCausalLM
"""

import os
import json
import math
import time
import glob
import argparse
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from src.model import ModelArgs, LlamaForCausalLM

import tiktoken


def load_meta(shard_dir: str):
    meta_path = os.path.join(shard_dir, "meta.json")
    with open(meta_path, "r") as f:
        return json.load(f)


class ShardDataset:
    def __init__(self, shard_dir: str):
        self.meta = load_meta(shard_dir)
        self.shard_dir = shard_dir

        dtype = np.uint16 if self.meta.get("dtype", "uint32") == "uint16" else np.uint32
        self.dtype = dtype

        # Shard list compatibility:
        # - Older sharders write meta["shards"] = ["shard_00000.bin", ...]
        # - Newer/other sharders may omit the list; in that case, discover shards from disk.
        if "shards" in self.meta and self.meta["shards"]:
            shard_names = list(self.meta["shards"])
            self.shard_paths = [os.path.join(shard_dir, s) for s in shard_names]
        else:
            # Prefer common pretrain shard patterns
            candidates = sorted(
                glob.glob(os.path.join(shard_dir, "shard_*.bin"))
                + glob.glob(os.path.join(shard_dir, "tokens_*.bin"))
            )
            # Last resort: any .bin except masks/meta
            if not candidates:
                candidates = sorted(
                    p
                    for p in glob.glob(os.path.join(shard_dir, "*.bin"))
                    if "mask_" not in os.path.basename(p)
                )
            self.shard_paths = candidates

        if not self.shard_paths:
            raise RuntimeError(
                f"No shard .bin files found in {shard_dir}. "
                "Expected meta.json with a 'shards' list or files like shard_*.bin."
            )

        # memmap each shard
        self.shards = []
        self.lengths = []
        for p in self.shard_paths:
            arr = np.memmap(p, mode="r", dtype=dtype)
            self.shards.append(arr)
            self.lengths.append(arr.shape[0])

        self.total_tokens = int(sum(self.lengths))

    def sample_batch(self, batch_size: int, seq_len: int, device: torch.device):
        # Sample random (x,y) pairs from random shards
        # x: (B,T) tokens, y: (B,T) next-token targets
        xs = torch.empty((batch_size, seq_len), dtype=torch.long, device=device)
        ys = torch.empty((batch_size, seq_len), dtype=torch.long, device=device)

        for i in range(batch_size):
            shard_idx = np.random.randint(0, len(self.shards))
            shard = self.shards[shard_idx]
            L = self.lengths[shard_idx]

            # Need seq_len + 1 tokens to make targets
            if L <= seq_len + 1:
                raise RuntimeError(
                    f"Shard {self.shard_paths[shard_idx]} too small ({L}) for seq_len={seq_len}"
                )

            start = np.random.randint(0, L - (seq_len + 1))
            chunk = np.array(shard[start : start + seq_len + 1], dtype=np.int64)

            x = torch.from_numpy(chunk[:-1]).to(device=device, dtype=torch.long)
            y = torch.from_numpy(chunk[1:]).to(device=device, dtype=torch.long)

            xs[i] = x
            ys[i] = y

        return xs, ys


def save_checkpoint(
    path: str,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_args: ModelArgs,
):
    ckpt = {
        "step": step,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "model_args": asdict(model_args),  # dict, not pickled dataclass
    }
    torch.save(ckpt, path)


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt


def cosine_lr(step: int, max_steps: int, base_lr: float, min_lr: float = 0.0):
    if step >= max_steps:
        return min_lr
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * step / max_steps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--resume", type=str, default=None)

    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--n_layers", type=int, default=12)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_kv_heads", type=int, default=4)
    ap.add_argument("--max_seq_len", type=int, default=512)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=61036)

    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--device", type=str, default="cuda")

    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    ds = ShardDataset(args.shard_dir)

    # Use vocab_size from meta if present
    vocab_size = int(ds.meta.get("vocab_size", 0))
    if vocab_size <= 0:
        # cl100k_base default size
        vocab_size = 100277

    model_args = ModelArgs(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        vocab_size=vocab_size,
        max_seq_len=args.max_seq_len,
    )

    model = LlamaForCausalLM(model_args).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    step = 0
    if args.resume is not None:
        ckpt = load_checkpoint(args.resume, device)
        step = int(ckpt["step"])
        # model_args from checkpoint if present (ensures exact config)
        if "model_args" in ckpt:
            loaded_args = ckpt["model_args"]
            # Keep tokenizer vocab from ckpt if any
            model_args = ModelArgs(**loaded_args)
            model = LlamaForCausalLM(model_args).to(device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        optimizer.load_state_dict(ckpt["optim_state"])
        print(f"[resume] step={step} from {args.resume}")

    tokens_per_step = args.batch_size * args.grad_accum * args.max_seq_len
    print(f"[data] shards={len(ds.shard_paths)} total_tokens≈{ds.total_tokens:,}")
    print(f"[train] tokens/step={tokens_per_step:,} max_steps={args.max_steps:,}")
    print(f"[train] effective tokens≈{tokens_per_step * args.max_steps:,}")

    model.train()
    t0 = time.time()

    pbar = tqdm(range(step, args.max_steps), dynamic_ncols=True)
    for step in pbar:
        # LR schedule (warmup + cosine)
        if step < args.warmup_steps:
            lr = args.lr * (step + 1) / max(1, args.warmup_steps)
        else:
            lr = cosine_lr(step - args.warmup_steps, args.max_steps - args.warmup_steps, args.lr, min_lr=0.0)

        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        loss_accum = 0.0
        for _ in range(args.grad_accum):
            x, y = ds.sample_batch(args.batch_size, args.max_seq_len, device=device)
            out = model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out  # unwrap (logits, ...)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B * T, V), y.reshape(B * T))
            (loss / args.grad_accum).backward()
            loss_accum += float(loss.item())

        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        dt = time.time() - t0
        toks = (step + 1) * tokens_per_step
        toks_per_s = toks / max(1e-9, dt)

        pbar.set_description(
            f"step {step+1}/{args.max_steps} loss {loss_accum/args.grad_accum:.4f} lr {lr:.2e} tok/s {toks_per_s:,.0f}"
        )

        if (step + 1) % args.save_every == 0 or (step + 1) == args.max_steps:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_step_{step+1}.pt")
            save_checkpoint(ckpt_path, step + 1, model, optimizer, model_args)
            # also write a "latest" pointer
            latest_path = os.path.join(args.output_dir, "checkpoint_latest.pt")
            save_checkpoint(latest_path, step + 1, model, optimizer, model_args)
            print(f"[save] {ckpt_path}")

    print("[done]")


if __name__ == "__main__":
    main()