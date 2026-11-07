"""Deterministic greedy decoding and beam search with an incremental KV cache.

Every decoder call consumes exactly one new token (q_len = 1), so no causal mask
is needed during generation: all cached keys are in the past.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from .model import TransformerNMT, build_padding_mask
from .tokenizer import BOS_ID, EOS_ID, LANG_IDS, PAD_ID, UNK_ID

SUPPRESSED_IDS = (PAD_ID, BOS_ID, UNK_ID, LANG_IDS["en"], LANG_IDS["fr"])


@dataclass
class DecodeOptions:
    max_new_tokens: int = 128
    beam_size: int = 1
    length_penalty: float = 1.0
    min_new_tokens: int = 0
    suppress_ids: Sequence[int] = SUPPRESSED_IDS


@dataclass
class Decoded:
    ids: list[int]
    termination: str  # "eos" | "max_length"
    score: float = 0.0
    warnings: list[str] = field(default_factory=list)


def reorder_cache(past: list[dict], indices: torch.Tensor) -> list[dict]:
    """Reorder the batch dimension of a KV cache (needed by beam search)."""
    return [
        {key: value.index_select(0, indices) for key, value in layer.items()} for layer in past
    ]


def _suppress(logits: torch.Tensor, ids: Sequence[int]) -> torch.Tensor:
    if ids:
        logits = logits.clone()
        logits[:, list(ids)] = torch.finfo(logits.dtype).min
    return logits


def _pad_sources(src_list: Sequence[Sequence[int]], pad_id: int, device: torch.device):
    length = max(len(item) for item in src_list)
    src = torch.full((len(src_list), length), pad_id, dtype=torch.long, device=device)
    mask = torch.ones((len(src_list), length), dtype=torch.bool, device=device)
    for row, item in enumerate(src_list):
        src[row, : len(item)] = torch.tensor(item, dtype=torch.long, device=device)
        mask[row, : len(item)] = False
    return src, mask


@torch.no_grad()
def greedy_decode_batch(
    model: TransformerNMT,
    src_list: Sequence[Sequence[int]],
    *,
    max_new_tokens: int,
    pad_id: int = PAD_ID,
    eos_id: int = EOS_ID,
    device: torch.device | None = None,
    suppress_ids: Sequence[int] = SUPPRESSED_IDS,
) -> list[Decoded]:
    if not src_list:
        return []
    device = device or next(model.parameters()).device
    model.eval()
    src, src_mask = _pad_sources(src_list, pad_id, device)
    memory = model.encode(src, src_mask)
    cross_mask = build_padding_mask(src_mask, memory.dtype)
    batch = src.size(0)
    current = torch.full((batch, 1), BOS_ID, dtype=torch.long, device=device)
    past: list[dict] | None = None
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    outputs: list[list[int]] = [[] for _ in range(batch)]
    reasons = ["max_length"] * batch

    for _ in range(max_new_tokens):
        logits, past = model.decode(current, memory, cross_mask=cross_mask, past=past)
        next_ids = torch.argmax(_suppress(logits[:, -1], suppress_ids), dim=-1)
        for row in range(batch):
            if finished[row]:
                continue
            token = int(next_ids[row])
            if token == eos_id:
                finished[row] = True
                reasons[row] = "eos"
            else:
                outputs[row].append(token)
        if bool(finished.all()):
            break
        current = next_ids.unsqueeze(1)
    return [Decoded(ids=ids, termination=reason) for ids, reason in zip(outputs, reasons)]


@torch.no_grad()
def beam_search(
    model: TransformerNMT,
    src_ids: Sequence[int],
    *,
    options: DecodeOptions,
    device: torch.device | None = None,
    pad_id: int = PAD_ID,
    eos_id: int = EOS_ID,
) -> Decoded:
    beam_size = max(1, options.beam_size)
    if beam_size == 1:
        return greedy_decode_batch(
            model,
            [src_ids],
            max_new_tokens=options.max_new_tokens,
            pad_id=pad_id,
            eos_id=eos_id,
            device=device,
            suppress_ids=options.suppress_ids,
        )[0]

    device = device or next(model.parameters()).device
    model.eval()
    src = torch.tensor([list(src_ids)], dtype=torch.long, device=device)
    src_mask = torch.zeros_like(src, dtype=torch.bool)
    memory = model.encode(src, src_mask)
    cross_mask = build_padding_mask(src_mask, memory.dtype)

    scores = torch.zeros(1, device=device)
    hypotheses: list[list[int]] = [[]]
    past: list[dict] | None = None
    current = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)
    completed: list[tuple[float, list[int]]] = []

    for step in range(options.max_new_tokens):
        logits, past = model.decode(current, memory, cross_mask=cross_mask, past=past)
        log_probs = _suppress(torch.log_softmax(logits[:, -1], dim=-1), options.suppress_ids)
        if options.min_new_tokens > 0 and step < options.min_new_tokens:
            log_probs[:, eos_id] = torch.finfo(log_probs.dtype).min
        vocab = log_probs.size(-1)
        active = log_probs.size(0)
        candidates = scores.unsqueeze(1) + log_probs
        k = min(beam_size * 2, active * vocab)
        top_scores, top_indices = torch.topk(candidates.reshape(-1), k=k)
        beam_indices = torch.div(top_indices, vocab, rounding_mode="floor")
        token_indices = top_indices % vocab

        new_past = reorder_cache(past, beam_indices)
        active_hypotheses: list[list[int]] = []
        active_scores: list[float] = []
        active_rows: list[int] = []
        active_tokens: list[int] = []
        for rank in range(k):
            source_beam = int(beam_indices[rank])
            token = int(token_indices[rank])
            score = float(top_scores[rank])
            hypothesis = hypotheses[source_beam]
            if token == eos_id:
                length = len(hypothesis) + 1
                penalty = ((5.0 + length) / 6.0) ** options.length_penalty
                completed.append((score / penalty, hypothesis))
            elif len(active_hypotheses) < beam_size:
                active_hypotheses.append(hypothesis + [token])
                active_scores.append(score)
                active_rows.append(rank)
                active_tokens.append(token)

        if not active_rows:
            break
        selected = torch.tensor(active_rows, dtype=torch.long, device=device)
        source_rows = beam_indices[selected]
        memory = memory[source_rows]
        cross_mask = cross_mask[source_rows]
        past = [
            {key: value.index_select(0, selected) for key, value in layer.items()}
            for layer in new_past
        ]
        hypotheses = active_hypotheses
        scores = torch.tensor(active_scores, dtype=torch.float32, device=device)
        current = torch.tensor(active_tokens, dtype=torch.long, device=device).unsqueeze(1)

    if completed:
        best_score, best_ids = max(completed, key=lambda item: item[0])
        return Decoded(ids=best_ids, termination="eos", score=best_score)
    best_index = int(torch.argmax(scores))
    return Decoded(ids=hypotheses[best_index], termination="max_length", score=float(scores[best_index]))


@torch.no_grad()
def beam_search_batch(
    model: TransformerNMT,
    src_list: Sequence[Sequence[int]],
    *,
    options: DecodeOptions,
    device: torch.device | None = None,
    pad_id: int = PAD_ID,
    eos_id: int = EOS_ID,
) -> list[Decoded]:
    """Beam-search a batch while keeping one fixed beam block per sentence."""
    if not src_list:
        return []
    beam_size = max(1, options.beam_size)
    if beam_size == 1:
        return greedy_decode_batch(
            model,
            src_list,
            max_new_tokens=options.max_new_tokens,
            pad_id=pad_id,
            eos_id=eos_id,
            device=device,
            suppress_ids=options.suppress_ids,
        )

    device = device or next(model.parameters()).device
    model.eval()
    src, src_mask = _pad_sources(src_list, pad_id, device)
    batch = src.size(0)
    memory = model.encode(src, src_mask)
    cross_mask = build_padding_mask(src_mask, memory.dtype)
    flat = batch * beam_size
    vocab = model.cfg.vocab_size

    hypotheses: list[list[int]] = [[] for _ in range(flat)]
    scores = torch.full((flat,), float("-inf"), dtype=torch.float32, device=device)
    current_slots = [eos_id] * flat
    completed: list[list[tuple[float, list[int]]]] = [[] for _ in range(batch)]
    current = torch.full((batch, 1), BOS_ID, dtype=torch.long, device=device)
    past: list[dict] | None = None

    # First token: one BOS row per sentence, then distinct tokens per beam.
    logits, past = model.decode(current, memory, cross_mask=cross_mask, past=past)
    log_probs = _suppress(torch.log_softmax(logits[:, -1], dim=-1), options.suppress_ids)
    if options.min_new_tokens > 0:
        log_probs[:, eos_id] = torch.finfo(log_probs.dtype).min
    expand = torch.arange(batch, device=device).repeat_interleave(beam_size)
    memory = memory.index_select(0, expand)
    cross_mask = cross_mask.index_select(0, expand)
    past = [
        {key: value.index_select(0, expand) for key, value in layer.items()}
        for layer in past
    ]
    for row in range(batch):
        top_scores, top_indices = torch.topk(
            log_probs[row], k=min(beam_size * 2, log_probs.size(1))
        )
        active = 0
        for rank in range(top_indices.numel()):
            token = int(top_indices[rank])
            score = float(top_scores[rank])
            if token == eos_id:
                completed[row].append((score, []))
            elif active < beam_size:
                slot = row * beam_size + active
                hypotheses[slot] = [token]
                scores[slot] = score
                current_slots[slot] = token
                active += 1

    current = torch.tensor(current_slots, dtype=torch.long, device=device).unsqueeze(1)
    for step in range(1, options.max_new_tokens):
        logits, past = model.decode(current, memory, cross_mask=cross_mask, past=past)
        log_probs = _suppress(torch.log_softmax(logits[:, -1], dim=-1), options.suppress_ids)
        if options.min_new_tokens > 0 and step < options.min_new_tokens:
            log_probs[:, eos_id] = torch.finfo(log_probs.dtype).min

        new_hypotheses: list[list[int]] = [[] for _ in range(flat)]
        new_scores = torch.full((flat,), float("-inf"), dtype=torch.float32, device=device)
        new_current = [eos_id] * flat
        new_source = torch.zeros((flat,), dtype=torch.long, device=device)
        for row in range(batch):
            start = row * beam_size
            candidates = scores[start : start + beam_size].unsqueeze(1) + log_probs[start : start + beam_size]
            k = min(beam_size * 2, beam_size * vocab)
            top_scores, top_indices = torch.topk(candidates.reshape(-1), k=k)
            active = 0
            for rank in range(k):
                source_local = int(top_indices[rank]) // vocab
                token = int(top_indices[rank]) % vocab
                score = float(top_scores[rank])
                source = start + source_local
                hypothesis = hypotheses[source]
                if token == eos_id:
                    length = len(hypothesis) + 1
                    penalty = ((5.0 + length) / 6.0) ** options.length_penalty
                    completed[row].append((score / penalty, hypothesis))
                elif active < beam_size:
                    slot = start + active
                    new_hypotheses[slot] = hypothesis + [token]
                    new_scores[slot] = score
                    new_current[slot] = token
                    new_source[slot] = source
                    active += 1

        past = reorder_cache(past, new_source)
        memory = memory.index_select(0, new_source)
        cross_mask = cross_mask.index_select(0, new_source)
        hypotheses = new_hypotheses
        scores = new_scores
        current = torch.tensor(new_current, dtype=torch.long, device=device).unsqueeze(1)

    results: list[Decoded] = []
    for row in range(batch):
        if completed[row]:
            best_score, best_ids = max(completed[row], key=lambda item: item[0])
            results.append(Decoded(ids=best_ids, termination="eos", score=best_score))
        else:
            start = row * beam_size
            best_local = int(torch.argmax(scores[start : start + beam_size]))
            best_index = start + best_local
            results.append(
                Decoded(
                    ids=hypotheses[best_index],
                    termination="max_length",
                    score=float(scores[best_index]),
                )
            )
    return results
