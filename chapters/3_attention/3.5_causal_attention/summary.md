# Section 3.5 – Causal Attention

## 🔑 Core Idea
In earlier sections (§3.4), we built **scaled dot-product attention**, where each token in a sequence can look at (attend to) every other token.  

In §3.5, the book introduces **causal attention** (sometimes also called *masked self-attention*).  
The key idea is simple:  
- **Future information must be hidden** during training, because a language model predicts the *next* word one at a time.  
- For example, when predicting the 3rd word in a sentence, the model must not see the 4th or 5th word.  

This is enforced with a **causal mask**, a triangular matrix that blocks attention to future positions.

---

## 🛠 What Terms Mean
- **Token:** a chunk of text (a word or subword) that has been converted to a numeric ID.  
- **Embedding:** a dense vector representation of a token, learned by the model.  
- **Query, Key, Value (Q/K/V):** three different projections of the same input embeddings.  
  - *Queries* ask questions like: "what am I looking for?"  
  - *Keys* are the “labels” or “addresses” of each token.  
  - *Values* are the actual information content carried forward.  
- **Dot-product attention:** measure similarity by multiplying queries and keys.  
- **Softmax:** a mathematical function that turns raw scores into probabilities that sum to 1.  
- **Causal mask:** a tool that sets impossible connections to negative infinity (`-inf`) so they become zero after softmax.  
- **Dropout:** a regularization technique that randomly sets some values to zero during training to reduce overfitting.  

---

## ⚙️ How Causal Attention Works
1. Each token embedding is projected into Q, K, and V vectors with trainable linear layers.  
2. Attention scores are computed with `QKᵀ / √d_k`. This says: “compare my query with every key.”  
3. The **causal mask** is applied to block all positions *after* the current token. Those positions get `-inf` so they vanish after softmax.  
4. Apply softmax: the remaining scores are converted into valid probability weights.  
5. Multiply those weights by the Value vectors → get a **context vector** for each token that only depends on past tokens.  

---

## 📂 Implementation Files
For clarity, the code for §3.5 is split into multiple files inside the `3.5/` directory:

- `casual_attention.py` → Compact causal attention class (Listing 3.3 style)  
- `dropout.py` → Demonstrates dropout applied to attention weights  
- `masked_attention_scores.py` → Shows attention scores before applying causal mask  
- `masked_attention_weights.py` → Shows attention weights after masking and softmax  

---

## 🔒 Causal Attention Example (from `casual_attention.py`)

```python
import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        # Upper-triangular mask: blocks attending to future positions
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor):
        # x: (batch, num_tokens, d_in)
        b, num_tokens, d_in = x.shape
        keys    = self.W_key(x)        # (b, T, d_out)
        queries = self.W_query(x)      # (b, T, d_out)
        values  = self.W_value(x)      # (b, T, d_out)

        # Scaled dot-product attention
        attn_scores = queries @ keys.transpose(1, 2)   # (b, T, T)

        # Apply mask (future tokens get -inf)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens],
            -torch.inf
        )

        attn_weights = torch.softmax(attn_scores / (keys.shape[-1]**0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values            # (b, T, d_out)
        return context_vec


if __name__ == "__main__":
    torch.manual_seed(123)
    T, d_in, d_out = 6, 3, 2
    inputs = torch.randn(T, d_in)
    batch = torch.stack((inputs, inputs), dim=0)   # (batch=2, T, d_in)

    context_length = batch.shape[1]
    ca = CausalAttention(d_in, d_out, context_length, dropout=0.0)
    context_vecs = ca(batch)
    print("context_vecs.shape:", context_vecs.shape)
    # -> torch.Size([2, 6, 2])
```

---

## 🎲 Dropout in Attention (from `dropout.py`)
Dropout is applied to the **attention weights** after the softmax step to prevent overfitting.

```python
import torch
import torch.nn as nn

drop = nn.Dropout(p=0.5)
x = torch.ones(10)
print("before dropout:", x)
print("after dropout:", drop(x))
```

---

## 🎭 Masked Attention Scores (from `masked_attention_scores.py`)
Here we inspect the attention **logits** (raw scores before softmax) after applying the causal mask.

```python
import torch

torch.manual_seed(123)
T = 5
scores = torch.randn(T, T)
mask = torch.triu(torch.ones(T, T), diagonal=1)
masked = scores.masked_fill(mask.bool(), -torch.inf)
print(masked)
```

---

## 🎭 Masked Attention Weights (from `masked_attention_weights.py`)
Now we softmax the masked scores to get valid probability distributions.

```python
import torch

torch.manual_seed(123)
T = 5
scores = torch.randn(T, T)
mask = torch.triu(torch.ones(T, T), diagonal=1)
masked = scores.masked_fill(mask.bool(), -torch.inf)
weights = torch.softmax(masked, dim=-1)
print(weights)
print("row sums:", weights.sum(dim=-1))
```

---

## 🌟 Reasoning from the Textbook
- Transformers rely entirely on attention — no recurrence (like RNNs) or convolution (like CNNs).  
- Without masking, a model could “peek into the future,” which would make training unrealistic.  
- Causal masking ensures the model learns to predict one token at a time, autoregressively.  
- Dropout on attention weights adds randomness during training so the model doesn’t overfit to patterns in small datasets.  

---

## ✅ Takeaways
- **Causal attention** makes sure a token can only see the past and present, not the future.  
- **Queries, Keys, Values** are just different linear projections of embeddings that let tokens “ask,” “address,” and “carry” information.  
- **Masks** are matrices of 1’s and 0’s used to block future tokens; applying them with `-inf` ensures those connections get probability 0 after softmax.  
- **Dropout** introduces noise during training for better generalization.  

---

### 🔗 References
- **Files:**  
  - `3.5/casual_attention.py`  
  - `3.5/dropout.py`  
  - `3.5/masked_attention_scores.py`  
  - `3.5/masked_attention_weights.py`  
- **Textbook:** *Build a Large Language Model (From Scratch)*, Chapter 3, Section 3.5 (pp. ~67–72 in MEAP V08).
