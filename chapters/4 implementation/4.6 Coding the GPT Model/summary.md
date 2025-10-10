## Chapter 4.6 — Coding the GPT Model

### Big picture (what the full GPT does)
We assemble a complete GPT-style model that turns **token indices** into **context-aware vector states** (via a stack of Transformer blocks) and finally into **logits over the vocabulary** for next-token prediction. The flow mirrors Fig. 4.15:

Token IDs → **Token Embedding** + **Positional Embedding** → Dropout →  
[ **(× N) Transformer Block**: Pre-LN → Self-Attention (§4.3) → Residual → Pre-LN → Feed-Forward (§3.6) → Residual ] →  
Final LayerNorm → **Linear output head (to vocab)** → logits.

Shape stays **(B, T, C)** the whole way inside the stack; only the final head maps **C → V**.

---

### File map (how your repo composes the model)
- **`toilet.py`** — the *main GPT module*: embeddings, dropout, block stack, final norm, output head, test harness.  
- **`optimus.py`** — defines **`AutoBlocks`** (a single Transformer block) used N times (§4.5).  
- **`Multihead.py`** — **`MultiHeadAttn`** (masked multi-head self-attention, §4.3).  
- **`Glue.py`** — **`ConveyorBelt`** (position-wise 2-layer MLP with GELU, §3.6).  
- **`gpt_config.py`** — config dict (vocab size, context length, model width, heads, layers, dropout, etc.).

> In §4.6 we **don’t re-implement** attention or the MLP; we **import** them and wire everything together.

---

### Model skeleton (from `toilet.py`)
Core modules created in `__init__`: token/positional embeddings, dropout, N stacked `AutoBlocks`, final LayerNorm, and the vocab projection head.

```python
class Toilet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 1) embeddings
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 2) stack of Transformer blocks (see §4.5 AutoBlocks)
        self.trf_blocks = nn.Sequential(
            *[AutoBlocks(cfg) for _ in range(cfg["n_layers"])])
        
        # 3) final normalization and output projection
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head   = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
```

**Notes**
- Positional embeddings indexed by `[0..T-1]` are *added* to token embeddings — gives the model order information.
- `nn.Sequential(*[AutoBlocks(...) for _ in range(n_layers)])` repeats the block **N** times (e.g., 12).
- Final **LayerNorm** (pre-output) often improves stability and perplexity.
- Output head maps `(B, T, C)` → `(B, T, V)` where `V = vocab_size`.

---

### Forward pass (end-to-end)
How input token ids become logits for next-token prediction.

```python
def forward(self, in_idx):  # in_idx: (B, T) token ids
    B, T = in_idx.size()

    # 1) token + positional embedding
    tok = self.tok_emb(in_idx)                     # (B, T, C)
    pos = self.pos_emb(torch.arange(T, device=in_idx.device))  # (T, C)
    x = tok + pos                                  # (B, T, C)
    x = self.drop_emb(x)

    # 2) transformer stack
    x = self.trf_blocks(x)                         # (B, T, C)

    # 3) final norm + vocab projection
    x = self.final_norm(x)                         # (B, T, C)
    logits = self.out_head(x)                      # (B, T, V)
    return logits
```

**Shape invariants**  
- Inside the stack we always keep `(B, T, C)` so **residual adds** stay valid.  
- Only the last step changes `C` → `V`.

---

### What’s inside one block (imported)
- `AutoBlocks` (from **§4.5**) performs:
  - **Pre-LN → MultiHeadAttn (§4.3)** → Dropout → Residual add.  
  - **Pre-LN → ConveyorBelt MLP (§3.6)** → Dropout → Residual add.  
- `MultiHeadAttn` applies a **causal mask** so position *t* cannot attend to *future* positions.  
- `ConveyorBelt` expands to `4×C`, uses **GELU**, and projects back to `C`.

---

### Example: quick smoke test (bottom of `toilet.py`)
You include a harness that builds GPT-2-small-like dims and verifies shapes + parameter counts.

```python
if __name__ == "__main__":
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    torch.manual_seed(123)
    model = Toilet(GPT_CONFIG_124M)
    batch = torch.randint(0, GPT_CONFIG_124M["vocab_size"], (2, 4))  # (B=2, T=4)

    out = model(batch)
    print("Output shape:", out.shape)          # -> torch.Size([2, 4, 50257])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # compare with hypothetical weight tying (tying tok_emb and out_head)
    total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
    print(f"Params if output tied to token-embed: {total_params_gpt2:,}")

    # estimate model size assuming float32
    print(f"Model size (MB): {total_params * 4 / (1024*1024):.2f}")
```

**Why this is useful**
- Confirms end-to-end wiring and that logits dimension is `(B, T, V)`.  
- Parameter print helps sanity-check scale when you change `C`, `H`, or `N`.

---

### Training loop sketch (how §4.6 is typically trained)
- **Objective:** next-token prediction via **cross-entropy** over logits `(B, T, V)` vs. targets `(B, T)`.  
- **Teacher forcing:** targets are the input sequence shifted by one position.  
- **Masking:** causal mask already handled inside attention; no separate padding mask shown here.

*(Pseudocode — integrate with your dataloader & optimizer.)*

```python
model = Toilet(cfg).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
criterion = nn.CrossEntropyLoss()

for batch_tokens in loader:            # (B, T)
    batch_tokens = batch_tokens.to(device)
    logits = model(batch_tokens)       # (B, T, V)
    targets = batch_tokens             # next-token task: shift inside loss indexing

    loss = criterion(logits.view(-1, cfg["vocab_size"]), targets.view(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
```

---

### Practical knobs that matter
- **Depth/width:** `n_layers` and `emb_dim` dominate compute/params; increase carefully.  
- **Heads:** `n_heads` must divide `emb_dim`; more heads → finer attention patterns.  
- **Dropout:** keep around `0.1` for GPT-2-small scale; tune with dataset size.  
- **Final LayerNorm:** small but consistent perplexity gain; keep it.  
- **(Optional) Weight tying:** tie `out_head.weight = tok_emb.weight` to save params and sometimes help training.

---

### Key takeaways
- `toilet.py` is the **orchestrator**: embeddings → stacked `AutoBlocks` → final norm → vocab head → logits.  
- Attention and MLP are **imported** from earlier chapters; §4.6 focuses on wiring them into a full GPT.  
- The model preserves `(B, T, C)` through the stack and only projects to `(B, T, V)` at the end, matching Fig. 4.15.  
- Your test block in `toilet.py` is a great template for sanity-checking shapes, parameter counts, and scaling behavior.
