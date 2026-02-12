# Mini_Llama (Overall) — Quick Commands README

This README assumes you run everything from the **Overall/** folder.

---

## 0) (Optional but recommended) Prevent Windows “Roaming site-packages” conflicts

```bash
$env:PYTHONNOUSERSITE="1"
```

---

## 1) Project layout (expected)

- `src/` → model + inference + server + UI
- `scripts/` → sharding + training entrypoints
- `data/` → token shards
- `checkpoints/` → saved checkpoints

---

## 2) Sharding (ONE script)

All sharding uses: `scripts/shard_dataset.py`

### A) OpenWebText (Pretrain shards) — raw text
```bash
python .\scripts\shard_dataset.py `
  --out_dir .\data\owt_shards_200m `
  --dataset openwebtext `
  --split train `
  --format text `
  --text_field text `
  --tokenizer cl100k_base `
  --dtype uint32 `
  --tokens_per_shard 25000000 `
  --max_tokens 200000000
```

### B) Alpaca (SFT shards) — instruction format
```bash
python .\scripts\shard_dataset.py `
  --out_dir .\data\alpaca_sft_10m `
  --dataset tatsu-lab/alpaca `
  --split train `
  --format alpaca `
  --tokenizer cl100k_base `
  --dtype uint32 `
  --tokens_per_shard 5000000 `
  --max_tokens 10000000
```

### C) UltraChat 200k (SFT shards) — chat/messages format (recommended)
```bash
python .\scripts\shard_dataset.py `
  --out_dir .\data\ultrachat_sft_50m `
  --dataset HuggingFaceH4/ultrachat_200k `
  --split train `
  --format chat `
  --messages_path messages `
  --role_key role `
  --content_key content `
  --tokenizer cl100k_base `
  --dtype uint32 `
  --tokens_per_shard 25000000 `
  --max_tokens 50000000
```

---

## 3) Training

### A) Pretrain on OpenWebText shards (CUDA single-node)
(Outputs to `checkpoints/pretrain_owt/`)

```bash
python -m scripts.train_shards_cuda `
  --shard_dir .\data\owt_shards_200m `
  --dim 1024 --n_layers 12 --n_heads 8 --n_kv_heads 4 `
  --max_seq_len 512 `
  --batch_size 8 --grad_accum 4 `
  --lr 3e-4 --weight_decay 0.1 --grad_clip 1.0 `
  --warmup_steps 100 --max_steps 2000 `
  --save_every 500 `
  --output_dir .\checkpoints\pretrain_owt
```

> Notes:
> - `batch_size * grad_accum` controls effective batch.
> - Increase `max_seq_len` later when stable.

### B) SFT on Alpaca shards (initialize from pretrain checkpoint)
This assumes a pretrain checkpoint exists at:
`.\checkpoints\pretrain_owt\checkpoint_step_2000.pt`

```bash
python -m scripts.train_sft_shards_cuda `
  --shard_dir .\data\alpaca_sft_10m `
  --resume .\checkpoints\pretrain_owt\checkpoint_step_2000.pt `
  --dim 1024 --n_layers 12 --n_heads 8 --n_kv_heads 4 `
  --max_seq_len 512 `
  --batch_size 8 --grad_accum 4 `
  --lr 5e-5 --weight_decay 0.0 `
  --warmup_steps 100 --max_steps 5000 `
  --save_every 500 `
  --output_dir .\checkpoints\sft_alpaca
```

> Important:
> - `--resume` carries the step counter from pretrain.  
> - Example above runs SFT from step 2000 → 5000 (≈ 3000 SFT steps).

### C) SFT on UltraChat shards (recommended)
```bash
python -m scripts.train_sft_shards_cuda `
  --shard_dir .\data\ultrachat_sft_50m `
  --resume .\checkpoints\pretrain_owt\checkpoint_step_2000.pt `
  --dim 1024 --n_layers 12 --n_heads 8 --n_kv_heads 4 `
  --max_seq_len 512 `
  --batch_size 8 --grad_accum 4 `
  --lr 5e-5 --weight_decay 0.0 `
  --warmup_steps 100 --max_steps 5000 `
  --save_every 500 `
  --output_dir .\checkpoints\sft_ultrachat
```

---

## 4) Quick generation test (CLI)

Replace CKPT with the checkpoint you want to test.

### A) Test a pretrain checkpoint
```bash
python .\src\generation.py `
  --ckpt <CKPT_PATH_HERE> `
  --device cuda `
  --prompt "Hello! Briefly explain attention.`n" `
  --max_new_tokens 80
```

### B) Test an SFT checkpoint (Alpaca-style prompt)
```bash
python .\src\generation.py `
  --ckpt <CKPT_PATH_HERE> `
  --device cuda `
  --prompt "### Instruction:`nSay hi and explain attention in 2 sentences.`n`n### Response:`n" `
  --max_new_tokens 120
```

---

## 5) Start the FastAPI server

Set environment variables first, then start Uvicorn.

```bash
$env:LLM_CKPT="<CKPT_PATH_HERE>"
$env:LLM_DEVICE="cuda"
uvicorn src.server:app --host 127.0.0.1 --port 8000
```

### Test the server (PowerShell curl)
```bash
curl -s -X POST http://127.0.0.1:8000/generate `
  -H "Content-Type: application/json" `
  -d '{"prompt":"### Instruction:\nExplain attention in 2 sentences.\n\n### Response:\n","max_new_tokens":120,"temperature":0.7,"top_p":0.9}'
```

---

## 6) Start the Gradio UI

```bash
python .\src\ui.py
```

> UI expects the server to be running (FastAPI on `127.0.0.1:8000`) unless your `ui.py` is configured for direct local inference.

---

## 7) Common fixes

### “No module named src”
Run training as a module:
```bash
python -m scripts.train_shards_cuda <args...>
python -m scripts.train_sft_shards_cuda <args...>
```

### Roaming Python packages interfering (datasets/gradio/fastapi issues)
```bash
$env:PYTHONNOUSERSITE="1"
```

---

## 8) Useful paths

- Pretrain shards: `.\data\owt_shards_200m\`
- Alpaca SFT shards: `.\data\alpaca_sft_10m\`
- UltraChat SFT shards: `.\data\ultrachat_sft_50m\`
- Pretrain checkpoints: `.\checkpoints\pretrain_owt\`
- SFT checkpoints: `.\checkpoints\sft_alpaca\` / `.\checkpoints\sft_ultrachat\`

---
