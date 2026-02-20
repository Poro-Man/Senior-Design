# Mini LLaMA — Technical Documentation

This document provides an in-depth technical reference for the **Mini LLaMA** project, a custom-built, LLaMA-style decoder-only Transformer designed for efficient training and inference. It covers all source modules under `src/`, including the model architecture, optimizations, inference engine, serving layer, and chat UI.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Model — `src/model.py`](#model--srcmodelpy)
   - [Configuration (`ModelArgs`)](#configuration-modelargs)
   - [RMSNorm](#rmsnorm)
   - [Rotary Position Embeddings (RoPE)](#rotary-position-embeddings-rope)
   - [Attention (with GQA + SDPA)](#attention-with-gqa--sdpa)
   - [FeedForward (SwiGLU)](#feedforward-swiglu)
   - [TransformerBlock](#transformerblock)
   - [LlamaForCausalLM (Full Model)](#llamaforcausallm-full-model)
3. [Optimizations](#optimizations)
   - [FlashAttention via SDPA](#1-flashattention-via-pytorch-sdpa)
   - [Grouped-Query Attention (GQA)](#2-grouped-query-attention-gqa)
   - [Precomputed RoPE Frequencies](#3-precomputed-rope-frequency-cache)
   - [SwiGLU Activation](#4-swiglu-activation-in-feedforward)
   - [Weight Tying](#5-weight-tying)
   - [Causal Mask Elision](#6-causal-mask-elision)
4. [Inference Engine — `src/llm_infer.py`](#inference-engine--srcllm_inferpy)
   - [Device Resolution](#device-resolution)
   - [Checkpoint Format Support](#checkpoint-format-support)
   - [`LLMInfer` Class](#llminfer-class)
   - [Text Generation Pipeline](#text-generation-pipeline)
5. [CLI Generation — `src/generation.py`](#cli-generation--srcgenerationpy)
6. [REST API Server — `src/server.py`](#rest-api-server--srcserverpy)
7. [Chat UI — `src/ui.py`](#chat-ui--srcuipy)
8. [End-to-End System Architecture](#end-to-end-system-architecture)
9. [Retraining Guidelines](#retraining-guidelines)

---

## Architecture Overview

The Mini LLaMA model is a **decoder-only causal Transformer** following the LLaMA family design:

```
Input Token IDs
      │
      ▼
┌─────────────┐
│ tok_embeddings│  (nn.Embedding, vocab_size × dim)
└──────┬──────┘
       │ + dropout
       ▼
┌──────────────────────────────────┐
│  TransformerBlock × n_layers     │
│  ┌────────────────────────────┐  │
│  │ RMSNorm → Attention → +x  │  │  (pre-norm residual)
│  └────────────────────────────┘  │
│  ┌────────────────────────────┐  │
│  │ RMSNorm → FeedForward → +h│  │  (pre-norm residual)
│  └────────────────────────────┘  │
└──────────────────────────────────┘
       │
       ▼
┌─────────────┐
│   RMSNorm   │  (final normalization)
└──────┬──────┘
       ▼
┌─────────────┐
│   output    │  (nn.Linear → vocab logits, weight-tied with embeddings)
└─────────────┘
```

---

## Model — `src/model.py`

This file (305 lines) implements the full model architecture. All classes and functions are detailed below.

### Configuration (`ModelArgs`)

`ModelArgs` is a `@dataclass` that holds every hyperparameter for the model. It enables building models of different sizes from the same code.

| Parameter          | Type             | Default    | Description                                                                 |
|--------------------|------------------|------------|-----------------------------------------------------------------------------|
| `dim`              | `int`            | `512`      | Hidden dimension of the model (embedding size)                              |
| `n_layers`         | `int`            | `8`        | Number of stacked Transformer blocks                                        |
| `n_heads`          | `int`            | `8`        | Number of query attention heads                                             |
| `n_kv_heads`       | `Optional[int]`  | `None`     | Number of key/value heads for GQA. `None` defaults to `n_heads` (MHA)       |
| `vocab_size`       | `int`            | `100277`   | Vocabulary size (matches `cl100k_base` tiktoken encoding)                   |
| `multiple_of`      | `int`            | `256`      | FFN hidden dim is rounded up to the nearest multiple of this value          |
| `ffn_dim_multiplier`| `Optional[float]`| `None`    | Optional scalar applied to FFN hidden dim before rounding                   |
| `norm_eps`         | `float`          | `1e-5`     | Epsilon for RMSNorm to avoid division by zero                               |
| `max_seq_len`      | `int`            | `1024`     | Maximum sequence length the model supports                                  |
| `dropout`          | `float`          | `0.0`      | Dropout rate applied in attention and FFN (0.0 = no dropout during inference)|

---

### RMSNorm

**Root Mean Square Layer Normalization** replaces the traditional `LayerNorm` used in vanilla Transformers. It is faster and more stable because it normalizes using only the RMS (no mean subtraction, no bias).

```python
class RMSNorm(nn.Module):
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        out = self._norm(x.float()).type_as(x)   # upcasts to float32 for numerical stability
        return out * self.weight                  # learnable per-dimension scale
```

**Key details:**
- The input is **cast to `float32`** before normalization to prevent overflow/underflow, then cast back to the original dtype. This is critical for `float16` / `bfloat16` training.
- A learnable `weight` parameter (initialized to all-ones) provides per-channel scaling.
- **No bias term** — unlike LayerNorm, RMSNorm omits the learnable bias and mean centering, reducing parameters and computation.

---

### Rotary Position Embeddings (RoPE)

RoPE encodes positional information by rotating query and key vectors in pairs, using complex-number multiplication. This produces position-dependent attention scores without adding any learnable parameters.

#### `precompute_freqs_cis(dim, end, theta=10000.0)`

Pre-computes the complex rotation factors for every position up to `end`:

```python
freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))   # frequency per pair
t     = torch.arange(end).float()                                   # position indices
freqs = torch.outer(t, freqs)                                       # [end, dim/2]
freqs_cis = torch.polar(torch.ones_like(freqs), freqs)              # e^{iθ} = cos(θ) + i·sin(θ)
```

- `theta = 10000.0` is the RoPE base frequency, controlling the wavelength range.
- The result is a `[end, head_dim/2]` complex tensor. The model pre-computes this **once** at construction for `max_seq_len * 2` positions, eliminating repeated trig calls.

#### `apply_rotary_emb(xq, xk, freqs_cis)`

Applies the rotation to query and key tensors:

1. Reshape `xq` and `xk` into pairs: `[bs, seqlen, heads, head_dim]` → complex `[bs, seqlen, heads, head_dim/2]`
2. Multiply by the precomputed `freqs_cis` (broadcasting across batch and heads)
3. Convert back to real: flatten the last two dims to restore `head_dim`

This applies a **different rotation angle per position**, giving the model relative position awareness — tokens closer together share similar rotation, yielding higher dot-product scores.

---

### Attention (with GQA + SDPA)

The `Attention` module is the most complex component and includes two major optimizations: **Grouped-Query Attention** and **SDPA dispatch**.

#### Initialization

```python
self.n_heads   = args.n_heads        # query heads (e.g., 8)
self.n_kv_heads = args.n_kv_heads    # key/value heads (e.g., 4), defaults to n_heads
self.n_rep     = n_heads // n_kv_heads  # replication factor (e.g., 2)
self.head_dim  = args.dim // args.n_heads

self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)   # fewer KV params
self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)
```

- **All projections are bias-free**, following the LLaMA convention.
- The key/value projections have **fewer parameters** when `n_kv_heads < n_heads`, which is the GQA benefit.
- A validation check ensures `n_heads % n_kv_heads == 0`.

#### Forward Pass Flow

```
Input x: [batch, seqlen, dim]
    │
    ├──→ wq(x) → [batch, seqlen, n_heads, head_dim]      ──┐
    ├──→ wk(x) → [batch, seqlen, n_kv_heads, head_dim]   ──┼──→ apply_rotary_emb
    └──→ wv(x) → [batch, seqlen, n_kv_heads, head_dim]     │
                                                             │
    After RoPE:                                              │
    q, k, v  (k and v are expanded via repeat_kv)           │
         │                                                   │
         ▼                                                   │
    ┌──────────────────────────────────┐                     │
    │ SDPA fast path (if available)    │                     │
    │  · is_causal=True (no mask alloc)│                     │
    │  · dropout during training only  │                     │
    ├──────────────────────────────────┤                     │
    │ Fallback: manual attention       │                     │
    │  · scores = Q·Kᵀ / √d           │                     │
    │  · additive causal mask [-inf]   │                     │
    │  · softmax → dropout → V matmul │                     │
    └──────────────────────────────────┘
         │
         ▼
    wo(out) → [batch, seqlen, dim]
```

#### `repeat_kv(x, n_rep)`

When using GQA (`n_kv_heads < n_heads`), key and value tensors have fewer heads than queries. `repeat_kv` replicates each KV head `n_rep` times to match the query head count:

```python
# [bs, seqlen, n_kv_heads, head_dim] → [bs, seqlen, n_kv_heads, n_rep, head_dim] → [bs, seqlen, n_heads, head_dim]
x[:, :, :, None, :].expand(...).reshape(...)
```

If `n_rep == 1` (standard multi-head attention), it returns the tensor unchanged (zero-cost).

---

### FeedForward (SwiGLU)

The feed-forward network uses the **SwiGLU** activation, a gated variant of SiLU/Swish that LLaMA popularized. It uses three linear projections instead of two:

```python
def forward(self, x):
    return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
```

- `w1(x)` produces the **gate input**, which is passed through `SiLU(x) = x · σ(x)`
- `w3(x)` produces the **value path**
- These are multiplied element-wise (the "gating"), then projected down by `w2`

#### Hidden Dimension Sizing

The hidden dimension follows the LLaMA recipe:

```python
hidden_dim = int(2 * hidden_dim / 3)                                    # ~ 2/3 of 4×dim
hidden_dim = int(ffn_dim_multiplier * hidden_dim)                        # optional scaling
hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)  # round up to 256
```

With default `dim=512`, the base FFN hidden dim is `4 × 512 = 2048`, reduced to `~1365`, then rounded up to the nearest `256` → **`1536`**. This rounding ensures memory-aligned tensor shapes for efficient GPU computation.

---

### TransformerBlock

Each block applies the standard LLaMA **pre-norm residual** pattern:

```python
def forward(self, x, freqs_cis, mask=None):
    h   = x + self.attention(self.attention_norm(x), freqs_cis, mask)   # attention sub-layer
    out = h + self.feed_forward(self.ffn_norm(h))                        # FFN sub-layer
    return out
```

- **Pre-norm**: The normalization is applied _before_ the sub-layer, not after. This improves training stability, especially at scale.
- **Residual connection**: Input is added directly to the sub-layer output, enabling gradient flow through deep networks.

---

### LlamaForCausalLM (Full Model)

The top-level module wires everything together:

```python
self.tok_embeddings = nn.Embedding(vocab_size, dim)
self.dropout        = nn.Dropout(dropout)
self.layers         = nn.ModuleList([TransformerBlock(i, params) for i in range(n_layers)])
self.norm           = RMSNorm(dim)
self.output         = nn.Linear(dim, vocab_size, bias=False)
```

#### Forward Pass

1. **Embed tokens**: `h = dropout(tok_embeddings(tokens))`
2. **Slice precomputed RoPE** to the sequence length: `freqs_cis[:seqlen]`
3. **Determine masking strategy**: If SDPA is available → `mask = None` (SDPA handles causal masking internally). If not → explicitly build an upper-triangular `[-inf]` mask.
4. **Run through all transformer blocks**: `for layer in self.layers: h = layer(h, freqs_cis, mask)`
5. **Final norm + output head**: `logits = self.output(self.norm(h)).float()`
6. **Optional training loss**: `cross_entropy(logits, targets, ignore_index=-100)`

#### Weight Initialization

All `nn.Linear` and `nn.Embedding` layers are initialized with **normal distribution** (`mean=0.0, std=0.02`). Biases (where present) are zeroed.

#### `get_num_params(non_embedding=True)`

Returns the total parameter count, optionally excluding embedding parameters (useful since embeddings are weight-tied with the output head).

---

## Optimizations

### 1) FlashAttention via PyTorch SDPA

The model uses `torch.nn.functional.scaled_dot_product_attention` when available (PyTorch ≥ 2.0):

```python
if hasattr(F, "scaled_dot_product_attention"):
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True)
```

**Why this matters:**
- SDPA dispatches to **FlashAttention** or **Memory-Efficient Attention** kernels on supported GPUs
- **O(N) memory** instead of O(N²) for the attention matrix — critical for long sequences
- **Fused CUDA kernels** avoid materializing the full `[heads, seqlen, seqlen]` attention matrix
- Automatic fallback to classic attention if SDPA is unavailable (e.g., older PyTorch)

**Implementation detail:** Dropout probability is set to `0.0` during evaluation (`self.training` check) and the configured rate during training.

---

### 2) Grouped-Query Attention (GQA)

GQA reduces memory bandwidth and KV-cache size by using fewer key/value heads than query heads:

| Setting | `n_heads` | `n_kv_heads` | `n_rep` | Type |
|---------|-----------|--------------|---------|------|
| Standard MHA | 8 | 8 | 1 | Multi-Head Attention |
| GQA | 8 | 4 | 2 | Grouped-Query (default) |
| GQA | 8 | 2 | 4 | More aggressive grouping |
| MQA | 8 | 1 | 8 | Multi-Query Attention |

**Benefits:**
- KV projection parameters reduced by a factor of `n_heads / n_kv_heads`
- KV-cache memory during inference reduced proportionally
- Minimal quality degradation compared to full MHA at typical ratios (2:1 or 4:1)

---

### 3) Precomputed RoPE Frequency Cache

RoPE frequencies are computed **once** during model construction:

```python
self.freqs_cis = precompute_freqs_cis(head_dim, max_seq_len * 2)
```

- Precomputed for `2x max_seq_len` positions to provide headroom
- Sliced to `[:seqlen]` on each forward pass (cheap tensor view)
- Moved to the correct device on first use via `.to(device)`
- Eliminates per-forward-pass `sin`/`cos` computation

---

### 4) SwiGLU Activation in FeedForward

Instead of the standard `ReLU` FFN:
```
FFN(x) = W₂ · ReLU(W₁ · x)       ← standard Transformer
FFN(x) = W₂ · (SiLU(W₁ · x) ⊙ W₃ · x)  ← SwiGLU (this model)
```

SwiGLU has been shown to improve language modeling quality at equivalent parameter counts. The trade-off is an extra linear projection (`w3`), but the hidden dim is reduced by ~1/3 to compensate, keeping total FLOPs comparable.

---

### 5) Weight Tying

The embedding layer and the output projection layer **share the same weight matrix**:

```python
self.tok_embeddings.weight = self.output.weight
```

- Reduces total parameters by `vocab_size × dim` (e.g., 100277 × 512 ≈ **51M parameters** saved)
- Maintains checkpoint compatibility (`strict=True` loading works correctly)
- Well-studied technique that often improves quality by forcing a shared token representation space

---

### 6) Causal Mask Elision

When SDPA is available, the model **skips allocating the causal mask** entirely:

```python
need_explicit_mask = not hasattr(F, "scaled_dot_product_attention")
mask = None   # SDPA uses is_causal=True internally
```

This avoids creating a `[seqlen × seqlen]` float tensor on every forward pass, saving both memory and time.

---

## Inference Engine — `src/llm_infer.py`

This module (139 lines) provides a high-level `LLMInfer` class that handles checkpoint loading, tokenization, and autoregressive text generation.

### Device Resolution

The `resolve_device()` function intelligently selects hardware:

| Input        | Result                                      |
|-------------|---------------------------------------------|
| `"cuda"` / `"gpu"` | CUDA if available, else CPU fallback    |
| `"mps"`      | Apple Metal if available, else CPU fallback |
| `"cpu"` / `""` | CPU                                       |
| Anything else | Passed directly to `torch.device()`       |

---

### Checkpoint Format Support

The `_extract_state_dict()` function supports **four checkpoint formats**, making the inference engine robust to different training scripts:

| Format | Structure | Detection |
|--------|-----------|-----------|
| A) Raw state dict | `{"tok_embeddings.weight": ..., ...}` | Checks for `"tok_embeddings.weight"` key |
| B) Wrapped with model_args | `{"model": state_dict, "model_args": {...}}` | Looks for `"model"` key |
| C) Standard PyTorch | `{"model_state_dict": state_dict, ...}` | Looks for `"model_state_dict"` key |
| D) Legacy | `{"model_state": state_dict, ...}` or `{"state_dict": ...}` | Tries remaining keys |

If `model_args` is present in the checkpoint, those hyperparameters are used to reconstruct the model. Otherwise, it falls back to reasonable defaults (`dim=1024, n_layers=12, n_heads=8, n_kv_heads=4`).

---

### `LLMInfer` Class

#### Constructor

```python
LLMInfer(ckpt_path: str, device: str = "cpu", tokenizer_name: str = "cl100k_base")
```

1. Resolves the device
2. Loads the **tiktoken** tokenizer (`cl100k_base` — the same encoding used by GPT-4 / ChatGPT). This gives a vocabulary of **100,277 tokens** with strong multilingual and code coverage.
3. Loads the checkpoint from disk (`weights_only=False` for compatibility with complex save formats)
4. Extracts the state dict and model args
5. Instantiates `LlamaForCausalLM`, loads weights with `strict=True`, and puts the model in `.eval()` mode

---

### Text Generation Pipeline

The `generate()` method implements **autoregressive decoding** with temperature scaling and top-p (nucleus) sampling:

```python
@torch.no_grad()
def generate(self, prompt, max_new_tokens=128, temperature=0.8, top_p=0.95) -> str:
```

#### Step-by-step:

1. **Encode** the prompt string to token IDs using tiktoken
2. **Loop** for `max_new_tokens` iterations:
   - Forward pass through the model → get logits `[batch, seqlen, vocab_size]`
   - Take the **last position's logits** `[:, -1, :]` (next-token prediction)
   - **Temperature scaling**: `logits / temperature` (lower = more deterministic, higher = more random)
   - **Top-p (nucleus) sampling**:
     - Sort tokens by probability (descending)
     - Compute cumulative sum; keep only tokens within the top `p` cumulative probability mass
     - Always keep at least 1 token (prevents empty distribution)
     - Renormalize the surviving probabilities
   - **Sample** from the filtered distribution (`torch.multinomial`)
   - **Append** the sampled token to the sequence
   - **Context window enforcement**: If the sequence exceeds `max_seq_len`, truncate from the left to stay within bounds
3. **Decode** the full token sequence back to text

**Greedy decoding** (when `temperature ≤ 0`): Skips sampling entirely and uses `argmax`.

---

## CLI Generation — `src/generation.py`

A minimal command-line tool for interactive text generation:

```bash
python -m src.generation \
  --ckpt ./checkpoints/model.pt \
  --device cuda \
  --prompt "Once upon a time" \
  --max_new_tokens 128 \
  --temperature 0.8 \
  --top_p 0.95
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--ckpt` | Yes | — | Path to checkpoint file |
| `--device` | No | `cuda` | Device to run on |
| `--prompt` | Yes | — | Input text prompt |
| `--max_new_tokens` | No | `128` | Maximum tokens to generate |
| `--temperature` | No | `0.8` | Sampling temperature |
| `--top_p` | No | `0.95` | Nucleus sampling threshold |

---

## REST API Server — `src/server.py`

A **FastAPI** server that wraps `LLMInfer` into a REST API for production deployment:

### Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `LLM_CKPT` | `./checkpoints/pretrain_owt_1b/checkpoint_step_61036.pt` | Checkpoint path |
| `LLM_DEVICE` | `cuda` | Device to run on |

### Endpoints

#### `GET /health`

Health check endpoint. Returns:
```json
{"status": "ok", "device": "cuda", "ckpt": "./checkpoints/..."}
```

#### `POST /generate`

Text generation endpoint. Request body:
```json
{
  "prompt": "Once upon a time",
  "max_new_tokens": 128,
  "temperature": 0.8,
  "top_p": 0.95
}
```

Response:
```json
{"text": "Once upon a time, in a land far away..."}
```

The `GenRequest` Pydantic model validates inputs, providing automatic type checking and default values.

---

## Chat UI — `src/ui.py`

A **Gradio** ChatInterface that provides a web-based conversational frontend connected to the FastAPI server.

### Architecture
```
User (Browser) ↔ Gradio ChatInterface ↔ FastAPI Server ↔ LLMInfer ↔ Model
```

### Features

- **Multi-turn conversation**: The `build_prompt()` function constructs a conversation prompt from the chat history by formatting all prior messages:
  ```
  User: Hello
  Assistant: Hi there!
  User: Tell me a story
  Assistant:
  ```
- **Adjustable parameters** via Gradio sliders:
  - `max_new_tokens`: 1–512 (default: 128)
  - `temperature`: 0.0–2.0 (default: 0.8)
  - `top_p`: 0.1–1.0 (default: 0.95)
- **5-minute timeout** for long generations (`timeout=300`)
- Connects to the FastAPI backend at `http://127.0.0.1:8000/generate`

---

## End-to-End System Architecture

```
┌────────────────────────────────────────────────────┐
│                   User Interfaces                   │
│  ┌──────────────┐    ┌───────────────────────────┐ │
│  │  CLI Tool     │    │  Gradio Chat UI (ui.py)   │ │
│  │ (generation.py)│    │  Port 7860               │ │
│  └──────┬───────┘    └───────────┬───────────────┘ │
│         │                        │ HTTP POST       │
│         │ Direct Python call     │ /generate       │
│         ▼                        ▼                 │
│  ┌────────────────────────────────────────────┐    │
│  │        Inference Engine (llm_infer.py)      │    │
│  │  ┌────────────────────────────────────┐    │    │
│  │  │ LLMInfer                           │    │    │
│  │  │  · tiktoken tokenizer              │    │    │
│  │  │  · checkpoint loader               │    │    │
│  │  │  · autoregressive decode loop      │    │    │
│  │  │  · top-p sampling                  │    │    │
│  │  └─────────────┬──────────────────────┘    │    │
│  │                │                           │    │
│  │  ┌─────────────▼──────────────────────┐    │    │
│  │  │ LlamaForCausalLM (model.py)        │    │    │
│  │  │  · Token Embeddings (weight-tied)  │    │    │
│  │  │  · TransformerBlock × n_layers     │    │    │
│  │  │    ├ RMSNorm + Attention (GQA+SDPA)│    │    │
│  │  │    └ RMSNorm + FeedForward (SwiGLU)│    │    │
│  │  │  · Final RMSNorm + Output Head     │    │    │
│  │  └────────────────────────────────────┘    │    │
│  └────────────────────────────────────────────┘    │
│                        │                           │
│  ┌─────────────────────▼──────────────────────┐    │
│  │           FastAPI Server (server.py)        │    │
│  │  Port 8000  ·  /health  ·  /generate       │    │
│  └────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────┘
```

---

## Retraining Guidelines

| Scenario | Retrain Required? |
|----------|-------------------|
| Switch attention implementation to SDPA (same head config) | **No** — same weights, different kernel |
| Change `n_kv_heads` (e.g., 8 → 4) | **Yes** — KV projection shapes change |
| Change `dim`, `n_heads`, or `n_layers` | **Yes** — incompatible weight shapes |
| Change `vocab_size` | **Yes** — embedding/output head shape changes |
| Change `max_seq_len` | **No** — RoPE is recomputed; context window changes |
| Change `dropout` rate | **No** — dropout is not a learned parameter |
| Change `ffn_dim_multiplier` | **Yes** — FFN weight shapes change |

> **Rule of thumb:** If the change affects the *shape* of any weight tensor, you must retrain. If it only changes the *computation path* or *runtime behavior*, existing checkpoints remain compatible.