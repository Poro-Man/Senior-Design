# scripts/train_sft_shards_ddp.py
import os, json, math, glob, argparse, random
import numpy as np
from dataclasses import asdict

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.model import ModelArgs, LlamaForCausalLM

def ddp_is_enabled():
    return int(os.environ.get("RANK", "-1")) != -1 or int(os.environ.get("SLURM_PROCID", "-1")) != -1

def ddp_init(backend="nccl"):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world = int(os.environ.get("SLURM_NTASKS", "1"))
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

class SFTShardDataset:
    """
    Expects:
      tokens_000000.bin, mask_000000.bin, meta.json
    mask: uint8 (1=train loss, 0=ignore)
    """
    def __init__(self, shard_dir: str):
        self.shard_dir = shard_dir
        self.meta = json.load(open(os.path.join(shard_dir, "meta.json"), "r", encoding="utf-8"))
        self.dtype = np.uint32 if self.meta.get("dtype") == "uint32" else np.uint16

        token_files = sorted(glob.glob(os.path.join(shard_dir, "tokens_*.bin")))
        if not token_files:
            raise FileNotFoundError(f"No tokens_*.bin in {shard_dir}")
        self.token_paths = token_files

        self.mask_paths = []
        for t in self.token_paths:
            m = t.replace("tokens_", "mask_")
            if not os.path.exists(m):
                raise FileNotFoundError(f"Missing mask for {t}: {m}")
            self.mask_paths.append(m)

        # estimate total tokens
        total = 0
        for p in self.token_paths:
            total += os.path.getsize(p) // np.dtype(self.dtype).itemsize
        self.total_tokens = total

    def sample_batch(self, batch_size: int, seq_len: int, rng: random.Random):
        """
        Returns x,y,mask aligned to targets:
          x: (B,T)
          y: (B,T)
          tgt_mask: (B,T) where 1 means include token loss
        """
        T = seq_len
        x = torch.empty((batch_size, T), dtype=torch.long)
        y = torch.empty((batch_size, T), dtype=torch.long)
        m = torch.empty((batch_size, T), dtype=torch.uint8)

        for i in range(batch_size):
            idx = rng.randrange(len(self.token_paths))
            tp = self.token_paths[idx]
            mp = self.mask_paths[idx]

            toks = np.memmap(tp, dtype=self.dtype, mode="r")
            msk  = np.memmap(mp, dtype=np.uint8, mode="r")

            if toks.shape[0] != msk.shape[0]:
                raise ValueError(f"token/mask length mismatch: {tp}")

            if toks.shape[0] < (T + 1):
                raise ValueError(f"Shard too small for seq_len={T}: {tp}")

            start = rng.randint(0, toks.shape[0] - (T + 1))
            chunk_t = np.asarray(toks[start:start + T + 1], dtype=np.int64)
            chunk_m = np.asarray(msk[start:start + T + 1], dtype=np.uint8)

            x[i] = torch.from_numpy(chunk_t[:-1])
            y[i] = torch.from_numpy(chunk_t[1:])
            # align mask to y targets
            m[i] = torch.from_numpy(chunk_m[1:])

        return x, y, m

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
    return torch.load(path, map_location=map_location, weights_only=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_dir", type=str, required=True)
    ap.add_argument("--resume", type=str, required=True)

    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--n_layers", type=int, default=12)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_kv_heads", type=int, default=4)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--batch_size", type=int, default=8)  # per-GPU
    ap.add_argument("--grad_accum", type=int, default=4)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--save_every", type=int, default=500)

    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    # DDP
    rank, world, local_rank = (0, 1, 0)
    if args.device == "cuda" and ddp_is_enabled():
        rank, world, local_rank = ddp_init("nccl")

    device = torch.device("cuda", local_rank) if args.device == "cuda" else torch.device("cpu")

    seed = args.seed + rank * 1000
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    ds = SFTShardDataset(args.shard_dir)
    if is_rank0():
        tokens_per_step = args.batch_size * args.max_seq_len * args.grad_accum * world
        print(f"[data] token_files={len(ds.token_paths)} total_tokens≈{ds.total_tokens:,}")
        print(f"[train] tokens/step={tokens_per_step:,} max_steps={args.max_steps:,}")

    margs = ModelArgs(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        vocab_size=100277,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    )
    model = LlamaForCausalLM(margs).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    # resume base checkpoint (pretrain)
    ckpt = load_checkpoint(args.resume, map_location="cpu")
    ckpt_args = ckpt.get("model_args", {})
    for k in ["dim","n_layers","n_heads","n_kv_heads","vocab_size","max_seq_len"]:
        if str(ckpt_args.get(k)) != str(getattr(margs, k)):
            raise ValueError(f"Resume arch mismatch on {k}: ckpt={ckpt_args.get(k)} vs current={getattr(margs,k)}")

    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    start_step = 0
    if is_rank0():
        print(f"[resume] base={args.resume} (ckpt_step={ckpt.get('step')})")

    if args.device == "cuda" and ddp_is_enabled():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    def get_lr(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        t = (step - args.warmup_steps) / max(1, (args.max_steps - args.warmup_steps))
        t = min(max(t, 0.0), 1.0)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * t))

    IGN = -100
    model.train()
    rng = random.Random(seed + 999)

    for step in range(start_step, args.max_steps):
        lr_now = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        optimizer.zero_grad(set_to_none=True)

        for micro in range(args.grad_accum):
            x, y, m = ds.sample_batch(args.batch_size, args.max_seq_len, rng)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)

            out = model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out

            # apply mask to targets
            tgt = y.reshape(-1)
            tgt_mask = m.reshape(-1).bool()
            tgt = torch.where(tgt_mask, tgt, torch.tensor(IGN, device=device, dtype=tgt.dtype))

            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt, ignore_index=IGN)
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
    