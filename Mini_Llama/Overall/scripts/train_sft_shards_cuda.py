# scripts/train_sft_shards_cuda.py
"""
Single-GPU SFT trainer using (tokens_*.bin, mask_*.bin) shards.
- Trains ONLY on assistant tokens via mask.
- Can start from a pretrain checkpoint (--init_from) or resume SFT (--resume).
- Loads multiple checkpoint formats (old/new).
"""

import os, json, math, time, glob, argparse, random
import numpy as np
from dataclasses import asdict

import torch
import torch.nn.functional as F

from src.model import ModelArgs, LlamaForCausalLM


# -------------------------
# Shard dataset (random windows)
# -------------------------
class SFTShardDataset:
    """
    Expects shard_dir with:
      tokens_*.bin (uint16/uint32)
      mask_*.bin   (uint8)  1=train loss, 0=ignore
      meta.json
    """
    def __init__(self, shard_dir: str):
        self.shard_dir = shard_dir
        meta_path = os.path.join(shard_dir, "meta.json")
        self.meta = json.load(open(meta_path, "r", encoding="utf-8"))

        dtype = self.meta.get("dtype", "uint16")
        self.dtype = np.uint32 if dtype == "uint32" else np.uint16

        self.token_paths = sorted(glob.glob(os.path.join(shard_dir, "tokens_*.bin")))
        self.mask_paths  = [p.replace("tokens_", "mask_") for p in self.token_paths]

        if not self.token_paths:
            raise FileNotFoundError(f"No tokens_*.bin found in {shard_dir}")
        for t, m in zip(self.token_paths, self.mask_paths):
            if not os.path.exists(m):
                raise FileNotFoundError(f"Missing mask for {t}: {m}")

        # estimate tokens
        total = 0
        for p in self.token_paths:
            total += os.path.getsize(p) // np.dtype(self.dtype).itemsize
        self.total_tokens = total

    def sample_batch(self, batch_size: int, seq_len: int, rng: random.Random):
        """
        Returns:
          x: (B,T) long
          y: (B,T) long
          m: (B,T) uint8  (1=train on that target token, 0=ignore)
        Note: mask is aligned with y (targets). We build (T+1) window then shift.
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
                raise ValueError(f"token/mask length mismatch: {tp} vs {mp}")

            if toks.shape[0] < (T + 1):
                raise ValueError(f"Shard too small for seq_len={T}: {tp}")

            start = rng.randint(0, toks.shape[0] - (T + 1))

            chunk_t = np.asarray(toks[start:start + T + 1], dtype=np.int64)
            chunk_m = np.asarray(msk[start:start + T + 1], dtype=np.uint8)

            x[i] = torch.from_numpy(chunk_t[:-1])
            y[i] = torch.from_numpy(chunk_t[1:])
            # align mask with targets (y): mask for token t+1 is chunk_m[1:]
            m[i] = torch.from_numpy(chunk_m[1:])

        return x, y, m


# -------------------------
# Checkpoint IO (multi-format)
# -------------------------
def _extract_model_state(ckpt: dict):
    # common possibilities
    for k in ["model_state_dict", "model_state", "model"]:
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    raise KeyError("No model state found. Expected one of: model_state_dict / model_state / model")

def _extract_optim_state(ckpt: dict):
    for k in ["optimizer_state_dict", "optim_state", "optimizer"]:
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    return None

def save_checkpoint(path, step, model, optimizer, args: ModelArgs, extra=None):
    ckpt = {
        "step": step,
        "model_args": asdict(args),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "extra": extra or {},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)

def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)

    # resume vs init
    ap.add_argument("--resume", type=str, default=None, help="Resume an SFT run checkpoint")
    ap.add_argument("--init_from", type=str, default=None, help="Initialize from a pretrain checkpoint")

    # model
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--n_layers", type=int, default=12)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_kv_heads", type=int, default=4)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.0)

    # train
    ap.add_argument("--batch_size", type=int, default=8)   # per step (single GPU)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1337)

    args = ap.parse_args()

    if args.resume and args.init_from:
        raise ValueError("Use only one of --resume or --init_from (not both).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ds = SFTShardDataset(args.shard_dir)

    tokens_per_step = args.batch_size * args.max_seq_len * args.grad_accum
    eff_tokens = tokens_per_step * args.max_steps
    print(f"[data] shards={len(ds.token_paths)} total_tokens≈{ds.total_tokens:,}")
    print(f"[train] tokens/step={tokens_per_step:,} max_steps={args.max_steps:,}")
    print(f"[train] effective tokens≈{eff_tokens:,}")
    print(f"[device] {device}")

    # model args (vocab fixed for cl100k_base in your project)
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )

    start_step = 0

    # init_from: load ONLY model weights (optionally also optimizer if present, but usually no)
    if args.init_from:
        ckpt = load_checkpoint(args.init_from, map_location="cpu")
        state = _extract_model_state(ckpt)
        model.load_state_dict(state, strict=True)
        print(f"[init_from] loaded model weights from {args.init_from}")

    # resume: load model + optimizer + step
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location="cpu")
        state = _extract_model_state(ckpt)
        model.load_state_dict(state, strict=True)

        opt_state = _extract_optim_state(ckpt)
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
        start_step = int(ckpt.get("step", 0))
        print(f"[resume] step={start_step} from {args.resume}")

    # LR schedule (warmup + cosine)
    def get_lr(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        t = (step - args.warmup_steps) / max(1, (args.max_steps - args.warmup_steps))
        t = min(max(t, 0.0), 1.0)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * t))

    model.train()
    rng = random.Random(args.seed + 999)

    t0 = time.time()
    for step in range(start_step, args.max_steps):
        lr_now = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        for micro in range(args.grad_accum):
            x, y, m = ds.sample_batch(args.batch_size, args.max_seq_len, rng)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)

            out = model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out  # (B,T,V)

            # mask targets: where m==0, set to IGN
            IGN = -100
            tgt = y.clone()
            tgt = torch.where(m.bool(), tgt, torch.tensor(IGN, device=device, dtype=tgt.dtype))

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                ignore_index=IGN
            )
            loss = loss / args.grad_accum
            loss.backward()
            running_loss += float(loss.item())

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        if step % 50 == 0:
            dt = time.time() - t0
            print(f"step={step} loss={running_loss*args.grad_accum:.4f} lr={lr_now:.2e} ({dt:.1f}s)")
            t0 = time.time()

        if (step + 1) % args.save_every == 0 or (step + 1) == args.max_steps:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_step_{step+1}.pt")
            save_checkpoint(ckpt_path, step + 1, model, optimizer, margs)

    print("[done]")


if __name__ == "__main__":
    main()
