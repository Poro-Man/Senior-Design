
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
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import tiktoken
from datasets import load_dataset
from tqdm import tqdm


# =============================================================================
# Distributed Utilities
# =============================================================================

def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ:
        # Launched via torchrun
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    elif "SLURM_PROCID" in os.environ:
        # Launched via SLURM
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "localhost")
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
    else:
        # Single GPU fallback
        rank, local_rank, world_size = 0, 0, 1

    if world_size > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def print_main(*args, **kwargs):
    """Print only from main process."""
    if is_main_process():
        print(*args, **kwargs)


# =============================================================================
# Model Components
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


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
    return x[:, :, :, None, :].expand(bs, slen, n_kv_heads, n_rep, head_dim).reshape(bs, slen, n_kv_heads * n_rep, head_dim)


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
        xk, xv = repeat_kv(xk, self.n_rep), repeat_kv(xv, self.n_rep)

        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
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
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            dropout=args.dropout,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None):
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        return h + self.feed_forward(self.ffn_norm(h))


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
        mask = torch.triu(torch.full((seqlen, seqlen), float("-inf"), device=device), diagonal=1)

        for layer in self.layers:
            h = layer(h, freqs_cis, mask)

        logits = self.output(self.norm(h)).float()

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
# Dataset (Streaming + DDP sharding) — RedPajama V2 sample + Alpaca wrapper
# =============================================================================

def format_redpajama_as_alpaca(text: str) -> str:
    """
    Simple Alpaca format wrapper for raw text documents.
    Not true instruction-response, but matches team's requested prompt structure.
    """
    t = (text or "").strip()
    if not t:
        t = " "
    return (
        "### Instruction:\n"
        "Continue the following text.\n\n"
        "### Response:\n"
        f"{t}\n"
    )


class StreamingRedPajamaDataset(IterableDataset):
    """
    Streams Together RedPajama V2 sample-10B.
    - Shards documents across ranks
    - Packs tokens into fixed-length training sequences
    - doc-based skip counter for deterministic resume (with num_workers=0)
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len: int = 1024,
        skip_docs: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.skip_docs = skip_docs
        self.rank = rank
        self.world_size = world_size
        self.docs_seen = 0
        self.sequences_yielded = 0

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        # Stream RedPajama V2 sample-10B
        ds = load_dataset(
            "togethercomputer/RedPajama-Data-V2",
            name="sample-10B",
            streaming=True,
            trust_remote_code=True,
        )["train"]

        token_buffer = []
        docs_processed = 0

        for sample in ds:
            self.docs_seen = docs_processed

            # resume skip
            if docs_processed < self.skip_docs:
                docs_processed += 1
                continue

            # shard by doc index
            if docs_processed % self.world_size != self.rank:
                docs_processed += 1
                continue

            # RedPajama samples contain "text"
            raw_text = sample.get("text", "")
            alpaca_text = format_redpajama_as_alpaca(raw_text)

            tokens = self.tokenizer.encode(alpaca_text + "\n\n", allowed_special="all")
            token_buffer.extend(tokens)

            while len(token_buffer) >= self.max_seq_len + 1:
                chunk = token_buffer[: self.max_seq_len + 1]
                token_buffer = token_buffer[self.max_seq_len:]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)

                yield input_ids, labels
                self.sequences_yielded += 1

            docs_processed += 1


# =============================================================================
# Training Utilities
# =============================================================================

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) * (max_lr - min_lr)


def save_checkpoint(model, optimizer, scaler, step, docs_seen, model_args, output_dir, train_args, rank):
    if rank != 0:
        return
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    checkpoint = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "step": step,
        "docs_seen": docs_seen,
        "model_args": model_args,
        "train_args": vars(train_args),
    }
    path = os.path.join(output_dir, f"checkpoint_step_{step}.pt")
    torch.save(checkpoint, path)
    print(f"[checkpoint] Saved to {path}")


def load_checkpoint(path: str, model, optimizer, scaler):
    checkpoint = torch.load(path, map_location="cpu")
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint["step"], checkpoint.get("docs_seen", 0)


# =============================================================================
# Main Training Loop
# =============================================================================

def train(args):
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    print_main(f"[config] {args}")
    print_main(f"[distributed] world_size={world_size}")
    print_main(f"[device] {device}")

    tokenizer = tiktoken.get_encoding("cl100k_base")
    vocab_size = tokenizer.n_vocab
    print_main(f"[tokenizer] cl100k_base, vocab_size={vocab_size}")

    model_args = ModelArgs(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        vocab_size=vocab_size,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    )
    model = LlamaForCausalLM(model_args).to(device)
    print_main(f"[model] {model.get_num_params() / 1e6:.2f}M parameters")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.use_amp)

    start_step, docs_seen = 0, 0
    if args.resume:
        print_main(f"[resume] Loading from {args.resume}")
        start_step, docs_seen = load_checkpoint(args.resume, model, optimizer, scaler)
        print_main(f"[resume] Resuming from step {start_step}, docs_seen={docs_seen}")

    # Deterministic streaming resume:
    if args.num_workers != 0:
        print_main("[warning] Streaming datasets should use --num_workers 0 for deterministic resume. Overriding to 0.")
        args.num_workers = 0

    dataset = StreamingRedPajamaDataset(
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        skip_docs=docs_seen,
        rank=rank,
        world_size=world_size,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    model.train()
    optimizer.zero_grad()
    step, accum_loss, accum_count = start_step, 0.0, 0
    min_lr = args.lr / 10

    pbar = tqdm(total=args.max_steps - start_step, desc="Training (RedPajama/DDP)", disable=rank != 0)
    start_time = time.time()

    for input_ids, labels in dataloader:
        if step >= args.max_steps:
            break

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        with autocast(enabled=args.use_amp):
            _, loss = model(input_ids, labels)
            loss = loss / args.grad_accum

        scaler.scale(loss).backward()
        accum_loss += loss.item()
        accum_count += 1
        docs_seen = dataset.docs_seen

        if accum_count >= args.grad_accum:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if rank == 0:
                pbar.set_postfix({"loss": f"{accum_loss * args.grad_accum:.4f}", "lr": f"{lr:.2e}", "docs": docs_seen})
                pbar.update(1)

            accum_loss, accum_count = 0.0, 0
            step += 1

            if step > 0 and step % args.save_every == 0:
                if world_size > 1:
                    dist.barrier()
                save_checkpoint(model, optimizer, scaler, step, docs_seen, model_args, args.output_dir, args, rank)

    if world_size > 1:
        dist.barrier()
    save_checkpoint(model, optimizer, scaler, step, docs_seen, model_args, args.output_dir, args, rank)

    print_main(f"\n[done] Training completed in {(time.time() - start_time) / 60:.2f} min, step: {step}")
    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="Train LLaMA-style model on streamed RedPajama V2 (Alpaca wrapper) with DDP")

    # Model
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_kv_heads", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Training
    parser.add_argument("--batch_size", type=int, default=4, help="Per-GPU batch size")
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_workers", type=int, default=0)

    # Checkpoint
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints_redpajama_ddp")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()