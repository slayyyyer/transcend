"""Encoder-decoder Transformer built from PyTorch primitives.

Deliberately explicit rather than a pretrained-model class: every parameter is
created here and initialized by :func:`build_model`, which makes the
"trained from scratch" claim auditable and makes the ONNX cache layout ours to
document.

Architecture decisions (recorded in the initialization manifest):
  * Pre-LayerNorm residual blocks, GELU FFN.
  * Sinusoidal (fixed, non-learned) position encoding, selected by explicit
    ``position_ids`` so the exported graph stays fully dynamic in sequence
    length (no shapes are baked in at export time).
  * Scaled dot-product attention written with matmul/softmax so the same module
    exports as an ONNX graph with an explicit KV cache.
  * Encoder/decoder input embeddings tied with the output projection.
  * Weight init N(0, 0.02), all biases 0, LayerNorm weight 1 / bias 0.

Masks are explicit additive tensors supplied by the caller. Training builds them
from shapes; the exported decoder graphs receive them as inputs.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def build_causal_mask(q_len: int, k_len: int, offset: int, device, dtype) -> torch.Tensor:
    """Additive [1, 1, q_len, k_len] mask blocking positions after ``offset + i``."""
    key_positions = torch.arange(k_len, device=device)
    query_positions = torch.arange(offset, offset + q_len, device=device)
    blocked = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
    additive = torch.zeros((1, 1, q_len, k_len), dtype=dtype, device=device)
    return additive.masked_fill(blocked.unsqueeze(0).unsqueeze(0), torch.finfo(dtype).min)


def build_padding_mask(pad_mask: torch.Tensor, dtype) -> torch.Tensor:
    """Bool [B, L] (True = pad) to additive [B, 1, 1, L]."""
    additive = torch.zeros((pad_mask.size(0), 1, 1, pad_mask.size(1)), dtype=dtype, device=pad_mask.device)
    return additive.masked_fill(
        pad_mask.unsqueeze(1).unsqueeze(2), torch.finfo(dtype).min
    )


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_positions: int):
        super().__init__()
        position = torch.arange(max_positions, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_positions, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: (d_model + 1) // 2])
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        table = self.pe.to(dtype=x.dtype)
        return x + F.embedding(position_ids, table)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.d_head = d_model // heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.heads, self.d_head).transpose(1, 2)

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        training: bool,
    ) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        if mask is not None:
            scores = scores + mask.to(dtype=scores.dtype)
        probs = torch.softmax(scores, dim=-1)
        if training and self.dropout > 0:
            probs = F.dropout(probs, p=self.dropout)
        return torch.matmul(probs, v)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor | None,
        mask: torch.Tensor | None,
        *,
        past: dict | None = None,
        is_cross: bool = False,
        training: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        q = self._split(self.q_proj(query))
        if is_cross and past is not None:
            # Cross-attention keys/values never change; reuse them so the cached
            # decoder step does not need the encoder memory as an input at all.
            k, v = past["k"], past["v"]
        else:
            source = key_value if key_value is not None else query
            k_new = self._split(self.k_proj(source))
            v_new = self._split(self.v_proj(source))
            if past is None:
                k, v = k_new, v_new
            else:
                k = torch.cat([past["k"], k_new], dim=2)
                v = torch.cat([past["v"], v_new], dim=2)
        new_cache = {"k": k, "v": v}

        attended = self._attend(q, k, v, mask, training=training)
        batch, heads, length, d_head = attended.shape
        merged = attended.transpose(1, 2).contiguous().view(batch, length, heads * d_head)
        return self.out_proj(merged), new_cache


class EncoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.attention_heads, cfg.dropout)
        self.norm1 = nn.LayerNorm(cfg.d_model, eps=cfg.layernorm_eps)
        self.norm2 = nn.LayerNorm(cfg.d_model, eps=cfg.layernorm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn_dim),
            nn.GELU(),
            nn.Linear(cfg.ffn_dim, cfg.d_model),
        )
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None, training: bool) -> torch.Tensor:
        attended, _ = self.self_attn(self.norm1(x), None, mask, training=training)
        x = x + F.dropout(attended, p=self.dropout, training=training)
        feed = self.ffn(self.norm2(x))
        return x + F.dropout(feed, p=self.dropout, training=training)


class Encoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.encoder_layers)])
        self.norm = nn.LayerNorm(cfg.d_model, eps=cfg.layernorm_eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None, training: bool = False) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask, training)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.attention_heads, cfg.dropout)
        self.cross_attn = MultiHeadAttention(cfg.d_model, cfg.attention_heads, cfg.dropout)
        self.norm1 = nn.LayerNorm(cfg.d_model, eps=cfg.layernorm_eps)
        self.norm2 = nn.LayerNorm(cfg.d_model, eps=cfg.layernorm_eps)
        self.norm3 = nn.LayerNorm(cfg.d_model, eps=cfg.layernorm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn_dim),
            nn.GELU(),
            nn.Linear(cfg.ffn_dim, cfg.d_model),
        )
        self.dropout = cfg.dropout

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None,
        self_mask: torch.Tensor | None,
        cross_mask: torch.Tensor | None,
        past: dict | None,
        training: bool,
    ) -> tuple[torch.Tensor, dict]:
        normalised = self.norm1(x)
        attended, self_cache = self.self_attn(normalised, None, self_mask, past=past, training=training)
        x = x + F.dropout(attended, p=self.dropout, training=training)
        cross_past = None if past is None else {"k": past["cross_k"], "v": past["cross_v"]}
        cross, cross_cache = self.cross_attn(
            self.norm2(x), memory, cross_mask, past=cross_past, is_cross=True, training=training
        )
        x = x + F.dropout(cross, p=self.dropout, training=training)
        feed = self.ffn(self.norm3(x))
        x = x + F.dropout(feed, p=self.dropout, training=training)
        cache = {
            "k": self_cache["k"],
            "v": self_cache["v"],
            "cross_k": cross_cache["k"],
            "cross_v": cross_cache["v"],
        }
        return x, cache


class Decoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.decoder_layers)])
        self.norm = nn.LayerNorm(cfg.d_model, eps=cfg.layernorm_eps)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None,
        self_mask: torch.Tensor | None,
        cross_mask: torch.Tensor | None,
        past: list[dict] | None = None,
        training: bool = False,
    ) -> tuple[torch.Tensor, list[dict]]:
        caches: list[dict] = []
        for index, layer in enumerate(self.layers):
            x, cache = layer(
                x,
                memory,
                self_mask,
                cross_mask,
                None if past is None else past[index],
                training,
            )
            caches.append(cache)
        return self.norm(x), caches


def _default_positions(batch: int, length: int, device) -> torch.Tensor:
    return torch.arange(length, device=device, dtype=torch.long).unsqueeze(0).expand(batch, length)


class TransformerNMT(nn.Module):
    """Shared English<->French model; direction comes from the control token."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_positions)
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.output_projection = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.output_projection.weight = self.embed_tokens.weight

    # ------------------------------------------------------------------ core
    def encode(
        self,
        src_ids: torch.Tensor,
        src_pad_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        training: bool = False,
    ) -> torch.Tensor:
        if position_ids is None:
            position_ids = _default_positions(src_ids.size(0), src_ids.size(1), src_ids.device)
        x = self.embed_tokens(src_ids) * math.sqrt(self.cfg.d_model)
        x = self.position(x, position_ids)
        return self.encoder(x, build_padding_mask(src_pad_mask, x.dtype), training)

    def decode(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor | None,
        self_mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past: list[dict] | None = None,
        training: bool = False,
    ):
        if past is not None and int(tgt_ids.size(1)) != 1:
            raise ValueError("cached decoding supports exactly one new query token per step")
        if position_ids is None:
            offset = 0 if past is None else int(past[0]["k"].size(2))
            position_ids = _default_positions(
                tgt_ids.size(0), tgt_ids.size(1), tgt_ids.device
            ) + offset
        x = self.embed_tokens(tgt_ids) * math.sqrt(self.cfg.d_model)
        x = self.position(x, position_ids)
        hidden, caches = self.decoder(x, memory, self_mask, cross_mask, past=past, training=training)
        return self.output_projection(hidden), caches

    def forward(
        self,
        src_ids: torch.Tensor,
        src_pad_mask: torch.Tensor,
        tgt_in_ids: torch.Tensor,
        tgt_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src_ids, src_pad_mask, training=True)
        length = int(tgt_in_ids.size(1))
        self_mask = build_causal_mask(length, length, 0, tgt_in_ids.device, memory.dtype)
        self_mask = self_mask + build_padding_mask(tgt_pad_mask, memory.dtype)
        cross_mask = build_padding_mask(src_pad_mask, memory.dtype)
        logits, _ = self.decode(
            tgt_in_ids, memory, self_mask=self_mask, cross_mask=cross_mask, training=True
        )
        return logits

    # ------------------------------------------------------------- inspection
    def parameter_count(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        per_module = {name: sum(p.numel() for p in module.parameters()) for name, module in self.named_children()}
        return {"total": total, "trainable": trainable, "per_module": per_module}

    def config_dict(self) -> dict:
        return asdict(self.cfg)

    def parameter_shapes(self) -> Iterable[tuple[str, tuple[int, ...]]]:
        for name, parameter in self.named_parameters():
            yield name, tuple(parameter.shape)


def build_model(cfg: ModelConfig, seed: int) -> TransformerNMT:
    """Create a model and initialize every learned parameter from a local RNG.

    Using a dedicated ``torch.Generator`` makes initialization independent of any
    RNG consumption that happened earlier in the process, so a checkpoint's
    parameters are reproducible from ``(config, seed)`` alone.
    """
    model = TransformerNMT(cfg)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("weight") and parameter.dim() >= 2:
                parameter.normal_(mean=0.0, std=0.02, generator=generator)
            elif name.endswith("weight"):
                parameter.fill_(1.0)
            else:
                parameter.zero_()
    return model


INITIALIZATION_MANIFEST = {
    "kind": "explicit-random-initialization",
    "distribution": "normal(mean=0.0, std=0.02) for matrices; LayerNorm weight=1, bias=0; all other biases=0",
    "independent_of_global_rng": True,
    "generator": "torch.Generator(device='cpu') seeded with the experiment seed",
    "no_pretrained_weights_loaded": True,
}
