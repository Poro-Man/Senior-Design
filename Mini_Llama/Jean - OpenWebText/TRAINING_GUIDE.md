# LLaMA Training Script for OpenWebText - Walkthrough

## What Was Created

Created [train_openwebtext.py](file:///Users/jean/Documents/sdp1_/Mini_Llama/Jean%20-%20OpenWebText/train_openwebtext.py) — a complete training script for training LLaMA on OpenWebText with streaming shards.

## Features

| Feature | Description |
|---------|-------------|
| **Streaming Dataset** | Uses HuggingFace `datasets` to stream OpenWebText without full download |
| **Single-GPU Model** | LLaMA architecture without fairscale dependencies |
| **Mixed Precision** | AMP for faster training and lower memory usage |
| **Gradient Accumulation** | Simulate larger batch sizes |
| **Cosine LR Schedule** | With linear warmup |
| **Checkpointing** | Resume training from any checkpoint |

---

## Installation

```bash
pip install torch tiktoken datasets tqdm
```

---

## Usage

### Basic Training
```bash
cd /Users/jean/Documents/sdp1_/Mini_Llama/Jean\ -\ OpenWebText/

python3 train_openwebtext.py \
    --dim 512 \
    --n_layers 8 \
    --n_heads 8 \
    --max_seq_len 1024 \
    --batch_size 4 \
    --grad_accum 8 \
    --lr 3e-4 \
    --max_steps 100000 \
    --output_dir ./checkpoints
```

### Small Model Smoke Test
```bash
python3 train_openwebtext.py \
    --dim 128 --n_layers 2 --n_heads 4 \
    --batch_size 2 --grad_accum 2 \
    --max_steps 20 --output_dir ./test_run
```

### Resume Training
```bash
python3 train_openwebtext.py \
    --resume ./checkpoints/checkpoint_step_1000.pt \
    --output_dir ./checkpoints
```

---

## Command-Line Arguments

```
Model:
  --dim           Model dimension (default: 512)
  --n_layers      Number of layers (default: 8)
  --n_heads       Number of attention heads (default: 8)
  --n_kv_heads    Number of KV heads for GQA (default: same as n_heads)
  --max_seq_len   Maximum sequence length (default: 1024)
  --dropout       Dropout rate (default: 0.0)

Training:
  --batch_size    Micro batch size (default: 4)
  --grad_accum    Gradient accumulation steps (default: 8)
  --lr            Learning rate (default: 3e-4)
  --weight_decay  Weight decay (default: 0.1)
  --grad_clip     Gradient clipping (default: 1.0)
  --warmup_steps  Warmup steps (default: 1000)
  --max_steps     Max training steps (default: 100000)
  --use_amp       Use mixed precision (default: True)

Checkpointing:
  --save_every    Save every N steps (default: 1000)
  --resume        Resume from checkpoint path
  --output_dir    Output directory (default: ./checkpoints)
```

---

## Model Size Reference

| Config | Parameters | Memory (approx) |
|--------|------------|-----------------|
| dim=256, layers=4, heads=4 | ~8M | ~200MB |
| dim=512, layers=8, heads=8 | ~45M | ~500MB |
| dim=1024, layers=12, heads=16 | ~300M | ~3GB |

---

## GPU Variants

### Mac MPS (Apple Silicon)

Use `train_openwebtext_mps.py` for Mac GPU acceleration:

```bash
python3 train_openwebtext_mps.py \
    --dim 512 --n_layers 8 --n_heads 8 \
    --batch_size 4 --output_dir ./checkpoints_mps
```

> [!NOTE]
> MPS uses float32 (limited float16 support). Expect ~2-3x speedup over CPU.

---

### Multi-GPU (HPC / SLURM)

Use `train_openwebtext_ddp.py` with torchrun or SLURM:

**Single node, 4 GPUs:**
```bash
torchrun --standalone --nproc_per_node=4 train_openwebtext_ddp_patched.py \
  --dim 1024 --n_layers 12 --n_heads 8 --n_kv_heads 4 \
  --batch_size 4 --output_dir ./checkpoints_ddp

```

**Multi-node (2 nodes × 4 GPUs):**
```bash
# Node 0
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 \
  --master_addr=$MASTER_ADDR --master_port=29500 \
  train_openwebtext_ddp_patched.py --output_dir ./checkpoints_ddp

# Node 1
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=4 \
  --master_addr=$MASTER_ADDR --master_port=29500 \
  train_openwebtext_ddp_patched.py --output_dir ./checkpoints_ddp

```

**SLURM example:**
```bash
#!/bin/bash
#SBATCH --job-name=owt-ddp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # IMPORTANT: 1 torchrun per node
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Pick master as the first node in the allocation
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  train_openwebtext_ddp_patched.py \
    --dim 1024 --n_layers 12 --n_heads 8 --n_kv_heads 4 \
    --batch_size 4 --output_dir ./checkpoints_ddp

```

> [!TIP]
> Effective batch size = `batch_size × grad_accum × num_gpus`
