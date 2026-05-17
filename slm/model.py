"""60M-target SLM: Llama-style decoder with RoPE + SwiGLU + RMSNorm, no biases."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from slm.config import GPTConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms).to(x.dtype) * self.weight


def precompute_rope(head_dim: int, max_seq: int, base: float = 10000.0):
    """Return cos/sin tables of shape (max_seq, head_dim) for RoPE half-rotation variant."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    return torch.cat([-x[..., d // 2:], x[..., : d // 2]], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: (B, H, T, head_dim).  cos/sin: (T, head_dim) broadcast to (1, 1, T, head_dim).
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout_p = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos, sin):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos[:T], sin[:T])
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class SwiGLU(nn.Module):
    """SwiGLU MLP: down(silu(gate(x)) * up(x)). Same FLOP budget as 4x GELU MLP when h_inner ≈ 8/3 * d."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.w_gate = nn.Linear(config.n_embd, config.swiglu_inner, bias=False)
        self.w_up   = nn.Linear(config.n_embd, config.swiglu_inner, bias=False)
        self.w_down = nn.Linear(config.swiglu_inner, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight  # weight tying

        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_rope(head_dim, config.block_size, base=config.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        scaled_std = 0.02 / math.sqrt(2 * config.n_layer)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=scaled_std)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Subtract once for the tied embedding (counted in both tok_emb and lm_head)
        return n - self.tok_emb.weight.numel()

    def forward(self, idx, targets=None, z_loss_weight: float = 0.0):
        B, T = idx.size()
        assert T <= self.config.block_size, f"seq {T} > block_size {self.config.block_size}"
        x = self.drop(self.tok_emb(idx))
        cos = self.rope_cos.to(dtype=x.dtype)
        sin = self.rope_sin.to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            if z_loss_weight > 0.0:
                lse = torch.logsumexp(logits.float(), dim=-1)
                z = (lse ** 2).mean()
                return logits, ce + z_loss_weight * z
            return logits, ce
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size:]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx