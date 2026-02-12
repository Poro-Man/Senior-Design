#!/usr/bin/env python3
"""
train_sft_shards_cuda.py

Supervised fine-tuning (SFT) on instruction shards produced by shard_sft_alpaca.py.
Uses loss mask so we ONLY train on assistant response tokens.
"""

import argparse
import glob
import json
import os
import time
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from src.model import ModelArgs, LlamaForCausalLM

def load_meta(shard_dir: str) -> Dict:
    p = os.path.join(shard_dir, "meta.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def list_pairs(shard_dir: str) -> List[Tuple[str, str]]:
    toks = sorted(glob.glob(os.path.join(shard_dir, "tokens_*.bin")))
    pairs = []
    for t in toks:
        m = t.replace("tokens_", "mask_")
        if not os.path.exists(m):
            raise FileNotFoundError(f"Missing mask for {t}: {m}")
        pairs.append((t, m))
    if not pairs:
        raise RuntimeError(f"No shards found in {shard_dir}")
    return pairs


def iter_batches_from_shards(pairs, seq_len, batch_size, dtype_np):
    """
    Yields (tokens, mask) where:
      tokens: (B, T) int64
      mask:   (B, T) uint8   (1=response token)
    """
    for tok_path, msk_path in pairs:
        toks = np.fromfile(tok_path, dtype=dtype_np)
        msk = np.fromfile(msk_path, dtype=np.uint8)
        assert toks.shape[0] == msk.shape[0], "token/mask length mismatch"

        # chop into sequences
        n_seq = toks.shape[0] // seq_len
        toks = toks[: n_seq * seq_len].reshape(n_seq, seq_len)
        msk = msk[: n_seq * seq_len].reshape(n_seq, seq_len)

        # batch
        for i in range(0, n_seq, batch_size):
            x = toks[i : i + batch_size]
            y = msk[i : i + batch_size]
            if x.shape[0] < batch_size:
                continue
            yield x, y


def save_ckpt(out_dir, step, model, optim, args_dict, meta):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"checkpoint_step_{step}.pt")
    ckpt = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "model_args": args_dict,
        "shard_meta": meta,
    }
    torch.save(ckpt, path)
    print(f"[save] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--resume", default=None)

    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--n_layers", type=int, default=12)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_kv_heads", type=int, default=4)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)

    ap.add_argument("--lr", type=float, default=1e-4)          # lower LR for SFT
    ap.add_argument("--weight_decay", type=float, default=0.0) # often 0 for SFT
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--init_from", default=None, help="Load model weights from a checkpoint but start SFT from step 0.")

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    meta = load_meta(args.shard_dir)
    pairs = list_pairs(args.shard_dir)

    dtype_np = np.uint16 if meta["dtype"] == "uint16" else np.uint32
    vocab_size = int(meta["vocab_size"])

    cfg = ModelArgs(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        vocab_size=vocab_size,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    )
    model = LlamaForCausalLM(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    start_step = 0
    start_step = 0
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        print("[init_from] loaded model weights; starting step=0")

    elif args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        optim.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = int(ckpt.get("step", 0))
        print(f"[resume] step={start_step}")


    model.train()
    step = start_step
    t0 = time.time()

    data_iter = iter_batches_from_shards(pairs, args.max_seq_len, args.batch_size, dtype_np)

    optim.zero_grad(set_to_none=True)

    while step < args.max_steps:
        # Gradient accumulation
        for _ in range(args.grad_accum):
            try:
                x_np, m_np = next(data_iter)
            except StopIteration:
                data_iter = iter_batches_from_shards(pairs, args.max_seq_len, args.batch_size, dtype_np)
                x_np, m_np = next(data_iter)

            x = torch.from_numpy(x_np.astype(np.int64)).to(device)  # (B, T)
            m = torch.from_numpy(m_np.astype(np.uint8)).to(device)  # (B, T)

            # next-token prediction
            inp = x[:, :-1]
            tgt = x[:, 1:]
            tgt_mask = m[:, 1:]  # align with targets

            logits, _ = model(inp, targets=None)  # (B, T-1, V)
            logits = logits.reshape(-1, logits.size(-1))
            tgt = tgt.reshape(-1)
            tgt_mask = tgt_mask.reshape(-1)

            # ignore everything except response tokens
            IGN = -100
            tgt = torch.where(tgt_mask.bool(), tgt, torch.tensor(IGN, device=device, dtype=tgt.dtype))

            loss = F.cross_entropy(logits, tgt, ignore_index=IGN)
            (loss / args.grad_accum).backward()

        # update
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()
        optim.zero_grad(set_to_none=True)
        step += 1

        if step % 10 == 0:
            dt = time.time() - t0
            print(f"[step {step}] loss={loss.item():.4f} ({dt:.1f}s)")
            t0 = time.time()

        if step % args.save_every == 0 or step == args.max_steps:
            save_ckpt(args.output_dir, step, model, optim, asdict(cfg), meta)

    print("[done]")


if __name__ == "__main__":
    main()
