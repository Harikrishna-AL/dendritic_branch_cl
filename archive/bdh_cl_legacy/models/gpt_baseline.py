"""nanoGPT-style Transformer baseline, parameter-matched to BDH.

Deliberate design choices, so that a BDH-vs-GPT difference is about the
architecture and not about incidentals:

* **Byte vocabulary (256)** -- the same input representation as BDH.
* **RoPE, not learned position embeddings.** BDH uses rotary phases inside its
  attention, and learned ``wpe`` would make the parameter count depend on
  ``block_size``, which would break parameter matching whenever we change the
  context length (Step 4 evaluates beyond the training length).
* **No weight tying** between ``embed`` and ``lm_head``, because upstream BDH
  keeps them separate too.
* **Causal mask ``diagonal=0``** -- i.e. a token may attend to itself. This is
  the standard Transformer and differs from BDH, whose ``tril(diagonal=-1)``
  excludes the current token. We keep each model's native convention and note
  the asymmetry in NOTES.md rather than crippling either one.

The same ``forward(idx, targets, loss_mask=..., return_activations=...)``
signature as :class:`~bdh_cl.models.bdh_wrapper.BDHWrapper`, so the continual-
learning runner and the probes treat the two models interchangeably. For GPT,
"activations" are the post-GELU MLP hidden states, which are the closest
analogue to BDH's latent neurons -- but they are *dense and signed*, which is
exactly the difference the Step 3 hypothesis is about.
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn

from ..config import GPTSpec
from .bdh_wrapper import masked_cross_entropy


def build_rope_cache(head_dim: int, max_T: int, device, dtype, theta: float = 10000.0):
    """Standard RoPE cos/sin tables, shape ``(1, 1, T, head_dim)``."""
    half = head_dim // 2
    inv = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(max_T, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)  # T, half
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
    return cos.view(1, 1, max_T, head_dim).to(dtype), sin.view(1, 1, max_T, head_dim).to(dtype)


def apply_rope(v: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
    return v * cos + v_rot * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, spec: GPTSpec) -> None:
        super().__init__()
        assert spec.n_embd % spec.n_head == 0
        self.n_head = spec.n_head
        self.head_dim = spec.n_embd // spec.n_head
        self.c_attn = nn.Linear(spec.n_embd, 3 * spec.n_embd, bias=spec.bias)
        self.c_proj = nn.Linear(spec.n_embd, spec.n_embd, bias=spec.bias)
        self.dropout = spec.dropout
        self.resid_drop = nn.Dropout(spec.dropout)

    def forward(self, x, cos, sin):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos[:, :, :T], sin[:, :, :T])
        k = apply_rope(k, cos[:, :, :T], sin[:, :, :T])
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, spec: GPTSpec) -> None:
        super().__init__()
        self.c_fc = nn.Linear(spec.n_embd, 4 * spec.n_embd, bias=spec.bias)
        self.c_proj = nn.Linear(4 * spec.n_embd, spec.n_embd, bias=spec.bias)
        self.drop = nn.Dropout(spec.dropout)

    def forward(self, x, return_hidden: bool = False):
        h = F.gelu(self.c_fc(x))
        out = self.drop(self.c_proj(h))
        if return_hidden:
            return out, h
        return out, None


class Block(nn.Module):
    def __init__(self, spec: GPTSpec) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(spec.n_embd, bias=spec.bias)
        self.attn = CausalSelfAttention(spec)
        self.ln_2 = nn.LayerNorm(spec.n_embd, bias=spec.bias)
        self.mlp = MLP(spec)

    def forward(self, x, cos, sin, return_hidden: bool = False):
        x = x + self.attn(self.ln_1(x), cos, sin)
        out, hidden = self.mlp(self.ln_2(x), return_hidden=return_hidden)
        x = x + out
        return x, hidden


@dataclasses.dataclass
class GPTActivations:
    """Post-GELU MLP hidden states per layer, each ``(B, T, 4*n_embd)``.

    Unlike BDH's ``x_sparse``, these are dense and signed. ``probes.py``
    handles both by thresholding at zero.
    """

    hidden: list[torch.Tensor]


class GPTBaseline(nn.Module):
    def __init__(self, spec: GPTSpec) -> None:
        super().__init__()
        self.spec = spec
        self.embed = nn.Embedding(spec.vocab_size, spec.n_embd)
        self.drop = nn.Dropout(spec.dropout)
        self.blocks = nn.ModuleList([Block(spec) for _ in range(spec.n_layer)])
        self.ln_f = nn.LayerNorm(spec.n_embd, bias=spec.bias)
        self.lm_head = nn.Linear(spec.n_embd, spec.vocab_size, bias=False)

        self.apply(self._init_weights)
        # nanoGPT's scaled init for residual projections.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * spec.n_layer))

        self._rope_T = 0
        self.register_buffer("rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("rope_sin", torch.empty(0), persistent=False)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _rope(self, T: int, device, dtype):
        """Grow the RoPE cache on demand -- Step 4 evaluates past training
        length, so this must not be capped at block_size."""
        if self._rope_T < T or self.rope_cos.device != device:
            cos, sin = build_rope_cache(
                self.spec.n_embd // self.spec.n_head, max(T, 1), device, dtype
            )
            self.rope_cos, self.rope_sin = cos, sin
            self._rope_T = T
        return self.rope_cos, self.rope_sin

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        return_activations: bool = False,
        return_hidden: bool = False,
        loss_mask: torch.Tensor | None = None,
    ):
        # ``return_hidden`` gives the post-ln_f representation the head reads,
        # so a prototype readout can classify without the learned head. Same
        # contract as BDHWrapper, so one evaluation path covers both models.
        B, T = idx.size()
        x = self.drop(self.embed(idx))
        cos, sin = self._rope(T, idx.device, x.dtype)

        hiddens: list[torch.Tensor] = []
        for block in self.blocks:
            x, h = block(x, cos, sin, return_hidden=return_activations)
            if return_activations:
                hiddens.append(h.detach())

        hidden = self.ln_f(x)
        logits = self.lm_head(hidden)

        loss = None
        if targets is not None:
            loss = masked_cross_entropy(logits, targets, loss_mask)

        if return_activations:
            return logits, loss, GPTActivations(hidden=hiddens)
        if return_hidden:
            return logits, loss, hidden.detach()
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx
