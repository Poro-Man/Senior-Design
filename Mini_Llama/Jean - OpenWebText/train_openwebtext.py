#!/usr/bin/env python3
"""
train_openwebtext.py — Train LLaMA on OpenWebText with streaming shards
Features:
- Streaming OpenWebText dataset (no full download required)
- Mixed precision training with AMP
- Gradient accumulation
- Cosine LR schedule with warmup
- Checkpointing with resume support
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
from torch.cuda.amp import autocast, GradScaler
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
# =============================================================================
# Model Components (Single-GPU version without fairscale)
# =============================================================================
@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: Optional[int] = None  # None = same as n_heads (MHA)
    vocab_size: int = 100277  # tiktoken cl100k_base vocab size
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
    """Precompute RoPE frequencies."""
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
    """Apply rotary embeddings to Q and K."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads for GQA."""
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
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        # GQA: repeat KV heads
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)
        # (bsz, n_heads, seqlen, head_dim)
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
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        dropout: float = 0.0,
    ):
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
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
class LlamaForCausalLM(nn.Module):
    """LLaMA model for causal language modeling (training)."""
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads, params.max_seq_len * 2
        )
        # Weight tying (optional but common)
        self.tok_embeddings.weight = self.output.weight
        # Initialize weights
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        Forward pass for training.
        
        Args:
            tokens: (batch, seq_len) input token ids
            targets: (batch, seq_len) target token ids (shifted by 1)
        
        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalar loss if targets provided
        """
        bsz, seqlen = tokens.shape
        device = tokens.device
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        self.freqs_cis = self.freqs_cis.to(device)
        freqs_cis = self.freqs_cis[:seqlen]
        # Causal mask
        mask = torch.full((seqlen, seqlen), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
        h = self.norm(h)
        logits = self.output(h).float()
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss
    def get_num_params(self, non_embedding: bool = True):
        """Return number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.tok_embeddings.weight.numel()
        return n_params
# =============================================================================
# Streaming Dataset
# =============================================================================
class StreamingOpenWebTextDataset(IterableDataset):
    """
    Streaming dataset for OpenWebText.
    
    Tokenizes text on-the-fly and packs sequences to max_seq_len.
    """
    def __init__(
        self,
        tokenizer,
        max_seq_len: int = 1024,
        skip_docs: int = 0,  # CHANGE: resume skips by *documents*, not batches
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.skip_docs = skip_docs  # CHANGE: track skipped documents for resume
        self.docs_seen = 0  # CHANGE: documents processed (for checkpoints/resume)
        self.sequences_yielded = 0  # CHANGE: packed sequences yielded (debug/metrics)
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        # Load streaming dataset
        dataset = load_dataset(
            "Skylion007/openwebtext",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        # Buffer for packing sequences
        token_buffer = []
        docs_processed = 0  # CHANGE: counts raw dataset documents
        # NOTE: If you set DataLoader num_workers>0, each worker has its own counter.
        # For deterministic resume, keep num_workers=0 when using skip_docs.
        for sample in dataset:
            # CHANGE: Skip *documents* if resuming (skip_docs counts raw dataset rows/docs)
            if docs_processed < self.skip_docs:
                docs_processed += 1
                continue
            text = sample["text"]
            # CHANGE: add a lightweight document boundary so we don't blend docs silently
            tokens = self.tokenizer.encode(text + "\n\n", allowed_special="all")
            token_buffer.extend(tokens)
            # Yield packed sequences
            while len(token_buffer) >= self.max_seq_len + 1:
                chunk = token_buffer[: self.max_seq_len + 1]
                token_buffer = token_buffer[self.max_seq_len:]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                self.sequences_yielded += 1  # CHANGE: track yielded sequences
                yield input_ids, labels
            docs_processed += 1
            self.docs_seen = docs_processed  # CHANGE: expose progress to training loop/checkpointing
def create_dataloader(
    tokenizer,
    batch_size: int,
    max_seq_len: int,
    skip_docs: int = 0,  # CHANGE: resume skips by *documents*, not batches
    num_workers: int = 0,
):
    """Create a streaming dataloader."""
    dataset = StreamingOpenWebTextDataset(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        skip_docs=skip_docs,  # CHANGE
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
# =============================================================================
# Training Utilities
# =============================================================================
def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float):
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    step: int,
    docs_seen: int,  # CHANGE: resume counter is docs_seen
    args: ModelArgs,
    output_dir: str,
    train_args: argparse.Namespace,
):
    """Save training checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "step": step,
        "docs_seen": docs_seen,  # CHANGE
        "model_args": args,
        "train_args": vars(train_args),
    }
    path = os.path.join(output_dir, f"checkpoint_step_{step}.pt")
    torch.save(checkpoint, path)
    print(f"[checkpoint] Saved to {path}")
    return path
def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, scaler: GradScaler):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint["step"], checkpoint["docs_seen"]  # CHANGE
# =============================================================================
# Main Training Loop
# =============================================================================
def train(args):
    print(f"[config] {args}")
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    # Tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")
    vocab_size = tokenizer.n_vocab
    print(f"[tokenizer] cl100k_base, vocab_size={vocab_size}")
    # Model args
    model_args = ModelArgs(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        vocab_size=vocab_size,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    )
    # Model
    model = LlamaForCausalLM(model_args)
    model = model.to(device)
    num_params = model.get_num_params()
    print(f"[model] {num_params / 1e6:.2f}M parameters")
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    # Mixed precision scaler
    scaler = GradScaler(enabled=args.use_amp)  # AMP
    # Resume from checkpoint
    start_step = 0
    docs_seen = 0  # CHANGE: documents processed for resume
    if args.resume:
        print(f"[resume] Loading from {args.resume}")
        start_step, docs_seen = load_checkpoint(args.resume, model, optimizer, scaler)  # CHANGE
        print(f"[resume] Resuming from step {start_step}, docs_seen={docs_seen}")  # CHANGE
    # Dataloader
    dataloader = create_dataloader(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        skip_docs=docs_seen,  # CHANGE: skip docs processed
    )
    # Training loop
    model.train()
    optimizer.zero_grad()
    step = start_step
    accum_loss = 0.0
    accum_count = 0
    min_lr = args.lr / 10
    pbar = tqdm(total=args.max_steps - start_step, desc="Training", initial=0)
    start_time = time.time()
    for batch_idx, (input_ids, labels) in enumerate(dataloader):
        if step >= args.max_steps:
            break
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        # Forward with AMP
        with autocast(enabled=args.use_amp):
            _, loss = model(input_ids, labels)
            loss = loss / args.grad_accum
        # Backward
        scaler.scale(loss).backward()
        accum_loss += loss.item()
        accum_count += 1
        # CHANGE: docs_seen is tracked inside the dataset (documents processed)
        docs_seen = getattr(dataloader.dataset, 'docs_seen', docs_seen)
        # Gradient accumulation step
        if accum_count >= args.grad_accum:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # Update learning rate
            lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            # Logging
            avg_loss = accum_loss * args.grad_accum  # Undo the division
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "lr": f"{lr:.2e}",
                "docs": docs_seen,  # CHANGE: log docs
            })
            pbar.update(1)
            # Reset accumulators
            accum_loss = 0.0
            accum_count = 0
            step += 1
            # Checkpoint
            if step > 0 and step % args.save_every == 0:
                save_checkpoint(
                    model, optimizer, scaler, step, docs_seen,  # CHANGE
                    model_args, args.output_dir, args
                )
    # Final save
    save_checkpoint(
        model, optimizer, scaler, step, docs_seen,  # CHANGE
        model_args, args.output_dir, args
    )
    elapsed = time.time() - start_time
    print(f"\n[done] Training completed in {elapsed / 60:.2f} min")
    print(f"[done] Final step: {step}, docs seen: {docs_seen}")  # CHANGE  # CHANGE
def main():
    parser = argparse.ArgumentParser(description="Train LLaMA on OpenWebText")
    # Model args
    parser.add_argument("--dim", type=int, default=512, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=8, help="Number of layers")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n_kv_heads", type=int, default=None, help="Number of KV heads (GQA)")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    # Training args
    parser.add_argument("--batch_size", type=int, default=4, help="Micro batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--max_steps", type=int, default=100000, help="Max training steps")
    parser.add_argument(
        "--use_amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use mixed precision (use --no-use_amp to disable)",
    )  # CHANGE: allow disabling AMP
    # Checkpoint args
    parser.add_argument("--save_every", type=int, default=1000, help="Save every N steps")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    args = parser.parse_args()
    train(args)
if __name__ == "__main__":
    main()