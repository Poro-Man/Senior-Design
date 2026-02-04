#!/usr/bin/env python3
"""
train_openwebtext_mps.py — Train LLaMA on OpenWebText using Mac GPU (MPS)

Optimized for Apple Silicon Macs with Metal Performance Shaders.
Note: MPS doesn't support all operations in float16, so we use float32 by default.
"""

import os
import math
import time
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

import tiktoken
from datasets import load_dataset
from tqdm import tqdm


# =============================================================================
# Model Components (Same as base version)
# =============================================================================

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    vocab_size: int = 100277
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_seq_len: int = 1024
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.dropout(scores)
        output = torch.matmul(scores, xv)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float], dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier, dropout=args.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None):
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class LlamaForCausalLM(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = nn.ModuleList([TransformerBlock(i, params) for i in range(params.n_layers)])
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len * 2)
        self.tok_embeddings.weight = self.output.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None):
        bsz, seqlen = tokens.shape
        device = tokens.device
        h = self.dropout(self.tok_embeddings(tokens))
        self.freqs_cis = self.freqs_cis.to(device)
        freqs_cis = self.freqs_cis[:seqlen]
        mask = torch.full((seqlen, seqlen), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)

        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
        h = self.norm(h)
        logits = self.output(h).float()

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=-100)
        return logits, loss

    def get_num_params(self, non_embedding: bool = True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.tok_embeddings.weight.numel()
        return n_params


# =============================================================================
# Dataset (Same as base version)
# =============================================================================

class StreamingOpenWebTextDataset(IterableDataset):
    def __init__(self, tokenizer, max_seq_len: int = 1024, skip_docs: int = 0):  # CHANGE: skip is doc-based for correct resume
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.skip_docs = skip_docs  # CHANGE
        self.docs_seen = 0  # CHANGE: global docs index
        self.sequences_yielded = 0  # CHANGE: packed sequences yielded

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True, trust_remote_code=True)
        token_buffer = []
        docs_processed = 0  # CHANGE: raw docs processed (global)

        for sample in dataset:
            self.docs_seen = docs_processed  # CHANGE: track doc index for resume
            if docs_processed < self.skip_docs:
                docs_processed += 1
                continue

            tokens = self.tokenizer.encode(sample["text"] + "\n\n", allowed_special="all")  # CHANGE: add doc boundary
            token_buffer.extend(tokens)

            while len(token_buffer) >= self.max_seq_len + 1:
                chunk = token_buffer[: self.max_seq_len + 1]
                token_buffer = token_buffer[self.max_seq_len:]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield input_ids, labels
                self.sequences_yielded += 1  # CHANGE: track yielded sequences

            docs_processed += 1


# =============================================================================
# MPS-Specific Utilities
# =============================================================================

def get_mps_device():
    """Get MPS device if available, otherwise fall back to CPU."""
    if torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            return torch.device("mps")
        else:
            print("[warning] MPS available but not built, using CPU")
    return torch.device("cpu")


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def save_checkpoint(model, optimizer, step, docs_seen, args, output_dir, train_args):  # CHANGE: docs_seen for correct resume
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "docs_seen": docs_seen,  # CHANGE
        "model_args": args,
        "train_args": vars(train_args),
    }
    path = os.path.join(output_dir, f"checkpoint_step_{step}.pt")
    torch.save(checkpoint, path)
    print(f"[checkpoint] Saved to {path}")
    return path


def load_checkpoint(path: str, model, optimizer):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["step"], checkpoint.get("docs_seen", checkpoint.get("samples_seen", 0))  # CHANGE: backward compatible


# =============================================================================
# Main Training Loop (MPS Optimized)
# =============================================================================

def train(args):
    print(f"[config] {args}")

    # MPS Device
    device = get_mps_device()
    print(f"[device] {device}")

    if device.type == "mps":
        # MPS-specific optimizations
        print("[mps] Using float32 (MPS has limited float16 support)")
        # Disable MPS fallback warnings for cleaner output
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    # Tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")
    vocab_size = tokenizer.n_vocab
    print(f"[tokenizer] cl100k_base, vocab_size={vocab_size}")

    # Model
    model_args = ModelArgs(
        dim=args.dim, n_layers=args.n_layers, n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads, vocab_size=vocab_size,
        max_seq_len=args.max_seq_len, dropout=args.dropout,
    )
    model = LlamaForCausalLM(model_args).to(device)
    print(f"[model] {model.get_num_params() / 1e6:.2f}M parameters")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    # Resume
    start_step, docs_seen = 0, 0  # CHANGE: docs_seen counts raw docs in stream
    if args.resume:
        print(f"[resume] Loading from {args.resume}")
        start_step, docs_seen = load_checkpoint(args.resume, model, optimizer)  # CHANGE
        # Move optimizer state to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print(f"[resume] Resuming from step {start_step}")

    # Dataloader
    dataset = StreamingOpenWebTextDataset(tokenizer, args.max_seq_len, docs_seen)  # CHANGE
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, pin_memory=False)

    # Training
    model.train()
    optimizer.zero_grad()
    step, accum_loss, accum_count = start_step, 0.0, 0
    min_lr = args.lr / 10

    pbar = tqdm(total=args.max_steps - start_step, desc="Training (MPS)")
    start_time = time.time()

    for input_ids, labels in dataloader:
        if step >= args.max_steps:
            break

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # Forward (no AMP on MPS, use float32)
        _, loss = model(input_ids, labels)
        loss = loss / args.grad_accum
        loss.backward()

        accum_loss += loss.item()
        accum_count += 1
        docs_seen = dataset.docs_seen  # CHANGE: doc-based counter

        if accum_count >= args.grad_accum:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.step()
            optimizer.zero_grad()

            # Sync MPS to get accurate timing
            if device.type == "mps":
                torch.mps.synchronize()

            pbar.set_postfix({"loss": f"{accum_loss * args.grad_accum:.4f}", "lr": f"{lr:.2e}", "docs": docs_seen})
            pbar.update(1)

            accum_loss, accum_count = 0.0, 0
            step += 1

            if step > 0 and step % args.save_every == 0:
                save_checkpoint(model, optimizer, step, docs_seen, model_args, args.output_dir, args)  # CHANGE

    save_checkpoint(model, optimizer, step, docs_seen, model_args, args.output_dir, args)  # CHANGE
    print(f"\n[done] Training completed in {(time.time() - start_time) / 60:.2f} min, step: {step}")


def main():
    parser = argparse.ArgumentParser(description="Train LLaMA on OpenWebText (Mac MPS)")
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_kv_heads", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints_mps")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
