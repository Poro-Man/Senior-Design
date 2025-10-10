# Section 3.6 – Multi-Head Attention

## 🔑 Core Idea
In the last section (§3.5), we saw **causal attention**, where each token could only look backward at itself and earlier tokens. That gave us a *single attention head*.  

But a single head has a limitation: it can only capture one kind of relationship at a time. For instance, one head might learn to focus on **short-range dependencies** like subject–verb agreement (“the dog … runs”), while another might be better at **long-range dependencies** like linking pronouns back to nouns (“the dog … it”). If we only have one head, the model has to choose.  

**Multi-Head Attention (MHA)** solves this by splitting the embedding into several **heads**.  
> *A head is just a separate attention mechanism with its own Q/K/V weights. You can think of each head as a different “specialist” focusing on a unique type of relationship.*  

Each head has its own learnable **query (Q), key (K), and value (V) projections**.  
> *Queries ask “what am I looking for?”, keys act like “labels” or “addresses” for tokens, and values carry the information content forward.*  

The heads process the sequence in parallel and their results are **concatenated**.  
> *Concatenation means gluing the results along the feature dimension so that the output has all heads’ information combined.*  

Finally, an **output projection** (a linear layer) blends them back into the model’s embedding space.  
> *This layer allows the model to mix and re-balance what each head has learned.*  

---

## 📖 Structure of Multi-Head Attention

The diagram below shows how heads work in parallel:

![Multihead Structure](./images/multihead%20structure%20example%20.png)

- Each colored box represents one **head**, which is just a copy of attention with its own Q/K/V weights.  
- The **head dimension** is smaller (`d_out / num_heads`).  
  > *This ensures that splitting into multiple heads doesn’t blow up the total size of the embeddings.*  
- After concatenation, the **output projection** mixes the heads back together.  

---

## 📂 Implementation Files
- `casual_attention.py` – single-head causal attention (from §3.5)  
- `stacking_heads_of_casual_attention.py` – wrapper-based multi-head attention using multiple `CausalAttention` modules  
- `Multihead.py` – efficient vectorized multi-head attention implementation  

---

## 📦 Naïve Wrapper Implementation

The first way to build multi-head attention is conceptually simple:  
- Reuse the **CausalAttention** class from §3.5 for each head.  
- Run them separately and then **concatenate** their outputs.  

```python
import torch
import torch.nn as nn
from casual_attention import CausalAttention 

class MultiHeadedAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)
```

This wrapper is **easy to understand** because it directly mirrors the diagram above.  
> *One `CausalAttention` per head → very intuitive for learning and debugging.*  

But it’s also **inefficient**:  
> *Each head duplicates weights and runs a separate forward pass, which wastes memory and ignores GPU parallelism.*  

---

## ⚡ Efficient Class-Based Implementation

In practice, libraries like PyTorch don’t build MHA as “many little modules.” Instead, they **vectorize the whole thing**:  

- Use **one big linear layer** for queries, one for keys, and one for values.  
  > *Each produces `[batch, seq_len, d_out]`, which we then reshape into heads.*  
- Reshape into `[batch, num_heads, seq_len, head_dim]`.  
  > *This way, all heads are handled in parallel, like running many specialists at once inside one big computation.*  
- Compute attention for all heads at once.  
- Concatenate and pass through an **output projection**.  

Here’s the version from `Multihead.py`:

![MultiHead Class](./images/Multiheaded%20Attenion%20Class.png)

```python
import torch
import torch.nn as nn

class MultiHeadAttn(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # size of each head
        
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, T, d_in = x.shape

        keys    = self.W_key(x).view(b, T, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.W_query(x).view(b, T, self.num_heads, self.head_dim).transpose(1, 2)
        values  = self.W_value(x).view(b, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask = self.mask[:T, :T]
        attn_scores.masked_fill_(mask.bool(), -torch.inf)

        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vecs = (attn_weights @ values).transpose(1, 2).contiguous().view(b, T, self.d_out)
        return self.out_proj(context_vecs)
```

---

## ⚖️ Why the Class is More Efficient
- **Wrapper:**  
  > *Each head is its own `CausalAttention`, with separate weights and computations. Easy to grasp, but inefficient.*  

- **Class-based MHA:**  
  > *One set of Q/K/V weights is shared. All heads are processed in parallel by reshaping, making it faster and lighter.*  

This design matches how real Transformers are implemented in frameworks like PyTorch.  

---

## ✅ Takeaways
- Multi-Head Attention = many specialists working in parallel.  
- The wrapper = intuitive but slow and memory-heavy.  
- The class = efficient, production-ready, and closer to how real GPT-style models are built.  
- Combining causal masks with multi-heads is the **backbone of modern language models**.  

---

### 🔗 References
- `casual_attention.py`  
- `stacking_heads_of_casual_attention.py`  
- `Multihead.py`  
- Textbook: *Build a Large Language Model (From Scratch)*, Chapter 3, Section 3.6 (MEAP V08).
