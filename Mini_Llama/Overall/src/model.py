# model.py
"""
Standalone LLaMA-style decoder-only Transformer (training-friendly).

Drop-in improvements (no retrain required):
- Uses torch.nn.functional.scaled_dot_product_attention (SDPA) when available
  (FlashAttention-style fastpath on CUDA), with safe fallback.
- Avoids allocating a full causal mask every forward when SDPA is used.
- Keeps identical parameter/module names so existing checkpoints load strict=True.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Config
# =============================================================================

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    vocab_size: int = 100277
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_seq_len: int = 1024
    dropout: float = 0.0


# =============================================================================
# Norm
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._norm(x.float()).type_as(x)
        return out * self.weight


# =============================================================================
# RoPE (complex formulation, matches your current checkpoints)
# =============================================================================

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=freqs.device).float()
    freqs = torch.outer(t, freqs)  # [end, dim/2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # e^{i*freqs}
    return freqs_cis


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # x is complex: [bs, seqlen, heads, head_dim/2]
    return freqs_cis[None, :, None, :]


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    xq: [bs, seqlen, n_heads, head_dim]
    xk: [bs, seqlen, n_kv_heads, head_dim]
    freqs_cis: [seqlen, head_dim/2] complex
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs = _reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    x: [bs, seqlen, n_kv_heads, head_dim]
    -> [bs, seqlen, n_kv_heads*n_rep, head_dim]
    """
    if n_rep == 1:
        return x
    bs, slen, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


# =============================================================================
# Blocks
# =============================================================================

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # [bs, heads, seqlen, head_dim]
        q = xq.transpose(1, 2)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)

        # ---- Fast path: SDPA (FlashAttention-like on CUDA) ----
        if hasattr(F, "scaled_dot_product_attention"):
            # SDPA expects (B, H, T, D)
            dropout_p = self.dropout.p if self.training else 0.0

            if mask is None:
                # Let SDPA build the causal mask internally (no big allocation)
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=dropout_p,
                    is_causal=True,
                )
            else:
                # Your mask is additive [-inf] upper-tri, shape [T, T]
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=mask,
                    dropout_p=dropout_p,
                    is_causal=False,
                )

            out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
            return self.wo(out)

        # ---- Fallback: classic attention (your original logic) ----
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores.float(), dim=-1).type_as(q)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [bs, heads, seqlen, head_dim]
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        dropout: float = 0.0,
    ):
        super().__init__()
        # SwiGLU hidden dim following LLaMA recipe
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            dropout=args.dropout,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


# =============================================================================
# Model
# =============================================================================

class LlamaForCausalLM(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = nn.ModuleList([TransformerBlock(i, params) for i in range(params.n_layers)])
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # RoPE precompute (moved to device on first forward)
        self.freqs_cis = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len * 2)

        # Weight tying (keeps same checkpoint structure)
        self.tok_embeddings.weight = self.output.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        tokens: [bs, seqlen]
        returns: logits [bs, seqlen, vocab], loss (optional)
        """
        bsz, seqlen = tokens.shape
        device = tokens.device

        h = self.dropout(self.tok_embeddings(tokens))

        freqs_cis = self.freqs_cis.to(device)[:seqlen]

        # IMPORTANT CHANGE:
        # - If Attention uses SDPA, it will do causal masking internally (mask=None).
        # - If SDPA is unavailable, Attention will fall back to classic attention;
        #   in that case we provide the explicit triangular mask.
        need_explicit_mask = not hasattr(F, "scaled_dot_product_attention")
        mask = None
        if need_explicit_mask:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=device)
            mask = torch.triu(mask, diagonal=1)

        for layer in self.layers:
            h = layer(h, freqs_cis, mask)

        h = self.norm(h)
        logits = self.output(h).float()

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.tok_embeddings.weight.numel()
        return n_params
