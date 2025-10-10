## Chapter 4.5 — Transformer Block

### What is the block?
A **Transformer block** is the repeatable unit in GPT-style models. It enriches token embeddings with context and keeps the same shape so blocks can be stacked. One pass through the block:

1) **Pre-Norm → Multi-Head Self-Attention** (from §4.3)  
2) **Residual Add (+ Dropout)**  
3) **Pre-Norm → Position-wise Feed-Forward MLP** (from §3.6)  
4) **Residual Add (+ Dropout)**  

> Shape invariant: input and output are both `(batch, seq_len, emb_dim)`.

---

### Why this design works (deeper view)
- **Pre-LayerNorm (LN before each sublayer)**  
  Stabilizes training when stacking many blocks; gradients flow better than post-norm.
- **Residual (shortcut) connections**  
  Preserve pathways for information/gradients; if a sublayer learns little, the identity path still passes signal.
- **Attention then MLP**  
  *Attention* mixes information across positions (context); *MLP* deepens the per-token transformation.
- **Dropout positions**  
  Applied on the sublayer outputs **before** residual add to regularize the learned update.
- **Causality (for LMs)**  
  The attention from §4.3 uses a **causal mask** so each token only attends to current/past tokens.

---

### Interface & invariants
- **Input/Output**: `(B, T, C)` with `C = emb_dim`.  
- **Attention**: Internally splits `C` across `n_heads` → each head has `head_dim = C / n_heads`.  
- **MLP**: Two linear layers with GELU in between. Expansion factor is typically **4×C**.  
- **Residual compatibility**: Both sublayers return `(B, T, C)` so residual adds are shape-safe.

---

### Parameter intuition (GPT-2-124M-style, C=768, H=12)
- Q, K, V projections: `3 * C * C ≈ 3 * 768² ≈ 1.77M` params  
- Attention out-proj: `C * C ≈ 0.59M`  
- MLP (4× expansion): `C*4C + 4C*C ≈ 2 * 768 * 3072 ≈ 4.72M` (≈ 6.29M if you include both mats; depends on counting biases)  
- Total per block is a few million params; stacking L blocks scales linearly.

*(Exact numbers vary with bias flags and implementations.)*

---

### Code — our Transformer block (`AutoBlocks`)
Imports: `MultiHeadAttn` (from §4.3) and `ConveyorBelt` (from §3.6).  
*(Adjust import paths to your project tree.)*

```python
from torch.nn.modules import LayerNorm
from resources.Multihead import MultiHeadAttn   # §4.3
from resources.Glue import ConveyorBelt        # §3.6
import torch
import torch.nn as nn

class AutoBlocks(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttn(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"], num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = ConveyorBelt(cfg)             # 2-layer MLP with GELU, 4x expansion
        self.norm1 = LayerNorm(cfg["emb_dim"])  # pre-norms
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # --- attention sublayer (pre-LN) ---
        short = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.shortcut(x)
        x = x + short            # residual

        # --- feed-forward sublayer (pre-LN) ---
        short = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.shortcut(x)
        x = x + short            # residual
        return x
```

---

### Run example — shape check & quick gradient sanity
Verifies `(B, T, C)` → `(B, T, C)` and that both sublayers contribute gradients.

```python
if __name__ == "__main__":
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,     # not used here, but typical stack size
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    torch.manual_seed(123)
    B, T, C = 2, 4, GPT_CONFIG_124M["emb_dim"]
    x = torch.randn(B, T, C)

    block = AutoBlocks(GPT_CONFIG_124M)
    y = block(x)
    print("Input :", x.shape)    # torch.Size([2, 4, 768])
    print("Output:", y.shape)    # torch.Size([2, 4, 768])

    # tiny gradient probe
    y.sum().backward()
    has_grad = {n: (p.grad is not None and p.grad.abs().mean().item()) for n,p in block.named_parameters() if p.requires_grad}
    print("Mean|grad| (nonzero-ish expected):")
    for k,v in list(has_grad.items())[:6]:
        print(f"  {k:30s} -> {v}")
```

Expected:
- Output shape equals input shape (residual compatibility).  
- Non-zero gradient stats for attention and MLP params (training signal flows).

---

### Practical notes
- **Pre-Norm vs Post-Norm**: we use **pre-norm** (LN before sublayers), which typically trains more stably for deep stacks without special warmups.  
- **Dropout**: the same rate is used for both sublayers here; you can tune them separately if needed.  
- **Causal mask**: lives inside your §4.3 attention; ensure `context_length` is ≥ the longest sequence you’ll feed.  
- **Scaling up**: stack `AutoBlocks` in an `nn.ModuleList`. Keep `emb_dim` constant across blocks to preserve residuals.  
- **Numerics**: start in `float32`; consider gradient clipping if you crank up depth or learning rate.

---

### Key insight
A Transformer block **preserves shape** but **adds context**. The combination of **pre-norm + residuals** keeps training stable as you stack many blocks, while attention and the MLP provide complementary transformations (global mixing + per-position depth). This is the engine room of GPT-style LLMs.
