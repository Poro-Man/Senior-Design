# scripts/train_shards_ddp.py
import os, json, math, time, glob, argparse, random
import numpy as np
from dataclasses import asdict

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.model import ModelArgs, LlamaForCausalLM

# -------------------------
# DDP helpers
# -------------------------
def ddp_is_enabled():
    return int(os.environ.get("RANK", "-1")) != -1 or int(os.environ.get("SLURM_PROCID", "-1")) != -1

def ddp_init(backend="nccl"):
    """
    Supports torchrun env (RANK/WORLD_SIZE/LOCAL_RANK) and SLURM env (SLURM_PROCID/SLURM_NTASKS).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world = int(os.environ.get("SLURM_NTASKS", "1"))
        # SLURM_LOCALID exists on many clusters
        local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
        os.environ.setdefault("RANK", str(rank))
        os.environ.setdefault("WORLD_SIZE", str(world))
        os.environ.setdefault("LOCAL_RANK", str(local_rank))
    else:
        rank, world, local_rank = 0, 1, 0

    dist.init_process_group(backend=backend, rank=rank, world_size=world)
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank

def is_rank0():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def ddp_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

# -------------------------
# Shard dataset (random windows)
# -------------------------
class ShardDataset:
    """
    Reads shards created by your sharder.
    Supports two meta formats:
      - old: {"shards":[...], "tokens_per_shard":..., ...}
      - new: {"files":[...], "tokens_per_shard":..., ...}
    Files can be named shard_*.bin or tokens_*.bin.
    """
    def __init__(self, shard_dir: str):
        meta_path = os.path.join(shard_dir, "meta.json")
        self.shard_dir = shard_dir
        self.meta = json.load(open(meta_path, "r", encoding="utf-8"))
        self.dtype = np.uint32 if self.meta.get("dtype") == "uint32" else np.uint16

        files = self.meta.get("shards", None) or self.meta.get("files", None)
        if files is None:
            # fallback: scan directory
            cand = sorted(glob.glob(os.path.join(shard_dir, "shard_*.bin")))
            if not cand:
                cand = sorted(glob.glob(os.path.join(shard_dir, "tokens_*.bin")))
            if not cand:
                raise FileNotFoundError(f"No shard files found in {shard_dir}")
            self.shard_paths = cand
        else:
            self.shard_paths = [os.path.join(shard_dir, f) for f in files]

        # estimate tokens
        total = 0
        for p in self.shard_paths:
            total += os.path.getsize(p) // np.dtype(self.dtype).itemsize
        self.total_tokens = total

    def sample_batch(self, batch_size: int, seq_len: int, rng: random.Random):
        """
        Randomly samples (B, T+1) tokens from random shard windows.
        Returns x,y of shape (B, T)
        """
        T = seq_len
        x = torch.empty((batch_size, T), dtype=torch.long)
        y = torch.empty((batch_size, T), dtype=torch.long)

        for i in range(batch_size):
            sp = rng.choice(self.shard_paths)
            arr = np.memmap(sp, dtype=self.dtype, mode="r")
            # need T+1 tokens to form x->y
            if arr.shape[0] < (T + 1):
                raise ValueError(f"Shard too small for seq_len={T}: {sp}")
            start = rng.randint(0, arr.shape[0] - (T + 1))
            chunk = np.asarray(arr[start:start + T + 1], dtype=np.int64)
            x[i] = torch.from_numpy(chunk[:-1])
            y[i] = torch.from_numpy(chunk[1:])

        return x, y

# -------------------------
# Utils
# -------------------------
def save_checkpoint(path, step, model, optimizer, args: ModelArgs, extra=None):
    if not is_rank0():
        return
    ckpt = {
        "step": step,
        "model_args": asdict(args),
        "model": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
        "optimizer": optimizer.state_dict(),
        "extra": extra or {},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)

def load_checkpoint(path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    return ckpt

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_dir", type=str, required=True)

    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--n_layers", type=int, default=12)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_kv_heads", type=int, default=4)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--batch_size", type=int, default=8)     # per-GPU
    ap.add_argument("--grad_accum", type=int, default=4)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=10000)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--resume", type=str, default=None)

    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=1337)

    args = ap.parse_args()

    # DDP init
    rank, world, local_rank = (0, 1, 0)
    if args.device == "cuda" and ddp_is_enabled():
        rank, world, local_rank = ddp_init("nccl")

    device = torch.device("cuda", local_rank) if args.device == "cuda" else torch.device("cpu")

    # deterministic-ish per rank
    seed = args.seed + rank * 1000
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # dataset
    ds = ShardDataset(args.shard_dir)
    if is_rank0():
        tokens_per_step = args.batch_size * args.max_seq_len * args.grad_accum * world
        eff_tokens = tokens_per_step * args.max_steps
        print(f"[data] shards={len(ds.shard_paths)} total_tokens≈{ds.total_tokens:,}")
        print(f"[train] tokens/step={tokens_per_step:,} max_steps={args.max_steps:,}")
        print(f"[train] effective tokens≈{eff_tokens:,}")

    # model args
    margs = ModelArgs(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        vocab_size=100277,         # your cl100k_base setting
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    )
    model = LlamaForCausalLM(margs).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )

    start_step = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location="cpu")
        start_step = int(ckpt.get("step", 0))
        # enforce same arch
        ckpt_args = ckpt.get("model_args", {})
        for k in ["dim","n_layers","n_heads","n_kv_heads","vocab_size","max_seq_len"]:
            if str(ckpt_args.get(k)) != str(getattr(margs, k)):
                raise ValueError(f"Resume arch mismatch on {k}: ckpt={ckpt_args.get(k)} vs current={getattr(margs,k)}")
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        if is_rank0():
            print(f"[resume] step={start_step}")

    if args.device == "cuda" and ddp_is_enabled():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    # LR schedule (simple warmup + cosine)
    def get_lr(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        t = (step - args.warmup_steps) / max(1, (args.max_steps - args.warmup_steps))
        t = min(max(t, 0.0), 1.0)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * t))

    # train
    model.train()
    rng = random.Random(seed + 999)

    for step in range(start_step, args.max_steps):
        # update LR
        lr_now = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        optimizer.zero_grad(set_to_none=True)

        for micro in range(args.grad_accum):
            x, y = ds.sample_batch(args.batch_size, args.max_seq_len, rng)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            out = model(x)  # may return logits or (logits, extra)
            logits = out[0] if isinstance(out, (tuple, list)) else out  # FIX: handle tuple

            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss / args.grad_accum
            loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        if is_rank0() and (step % 50 == 0):
            print(f"step={step} loss={loss.item()*args.grad_accum:.4f} lr={lr_now:.2e}")

        if (step + 1) % args.save_every == 0 or (step + 1) == args.max_steps:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_step_{step+1}.pt")
            save_checkpoint(ckpt_path, step + 1, model, optimizer, margs)

    ddp_barrier()
    if is_rank0():
        print("[done]")

if __name__ == "__main__":
    main()
    