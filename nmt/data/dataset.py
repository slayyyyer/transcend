"""Directional dataset construction, token-budgeted batching and bucketing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from ..tokenizer import BOS_ID, EOS_ID, PAD_ID, ByteBPETokenizer
from .prepare import Pair

IGNORE_INDEX = -100
DIRECTIONS = ("en-fr", "fr-en")


@dataclass
class Example:
    pair_id: str
    direction: str
    src_ids: np.ndarray
    tgt_in_ids: np.ndarray
    tgt_out_ids: np.ndarray

    @property
    def src_len(self) -> int:
        return len(self.src_ids)

    @property
    def tgt_len(self) -> int:
        return len(self.tgt_out_ids)


@dataclass
class DatasetStats:
    pairs_considered: int = 0
    pairs_used: int = 0
    examples: dict[str, int] = field(default_factory=dict)
    dropped_source_too_long: dict[str, int] = field(default_factory=dict)
    dropped_target_too_long: dict[str, int] = field(default_factory=dict)
    source_tokens: dict[str, int] = field(default_factory=dict)
    target_tokens: dict[str, int] = field(default_factory=dict)
    max_source_tokens: int = 0
    max_target_tokens: int = 0

    def as_dict(self) -> dict:
        return {
            "pairs_considered": self.pairs_considered,
            "pairs_used": self.pairs_used,
            "examples": self.examples,
            "dropped_source_too_long": self.dropped_source_too_long,
            "dropped_target_too_long": self.dropped_target_too_long,
            "source_tokens": self.source_tokens,
            "target_tokens": self.target_tokens,
            "max_source_tokens": self.max_source_tokens,
            "max_target_tokens": self.max_target_tokens,
        }


def build_examples(
    pairs: Sequence[Pair],
    tokenizer: ByteBPETokenizer,
    *,
    max_source_tokens: int,
    max_target_tokens: int,
) -> tuple[list[Example], DatasetStats]:
    """Create both directional examples for each pair.

    A pair is assigned to one split beforehand; both directions inherit it. An
    over-long direction is dropped independently, and the counts are reported.
    """
    stats = DatasetStats(
        pairs_considered=len(pairs),
        examples={direction: 0 for direction in DIRECTIONS},
        dropped_source_too_long={direction: 0 for direction in DIRECTIONS},
        dropped_target_too_long={direction: 0 for direction in DIRECTIONS},
        source_tokens={direction: 0 for direction in DIRECTIONS},
        target_tokens={direction: 0 for direction in DIRECTIONS},
        max_source_tokens=max_source_tokens,
        max_target_tokens=max_target_tokens,
    )
    examples: list[Example] = []
    used_pairs = 0
    for pair in pairs:
        produced = False
        for direction, source_text, target_text in (
            ("en-fr", pair.en, pair.fr),
            ("fr-en", pair.fr, pair.en),
        ):
            source_lang, target_lang = direction.split("-")
            src_ids = tokenizer.encode(source_text, lang=target_lang, add_eos=True)
            if len(src_ids) > max_source_tokens:
                stats.dropped_source_too_long[direction] += 1
                continue
            tgt_out_ids = tokenizer.encode(target_text, add_eos=True)
            if len(tgt_out_ids) > max_target_tokens:
                stats.dropped_target_too_long[direction] += 1
                continue
            tgt_in_ids = [BOS_ID] + tgt_out_ids[:-1]
            examples.append(
                Example(
                    pair_id=pair.pair_id,
                    direction=direction,
                    src_ids=np.asarray(src_ids, dtype=np.int32),
                    tgt_in_ids=np.asarray(tgt_in_ids, dtype=np.int32),
                    tgt_out_ids=np.asarray(tgt_out_ids, dtype=np.int32),
                )
            )
            stats.examples[direction] += 1
            stats.source_tokens[direction] += len(src_ids)
            stats.target_tokens[direction] += len(tgt_out_ids)
            produced = True
        if produced:
            used_pairs += 1
    stats.pairs_used = used_pairs
    return examples, stats


def chunk_by_length(examples: Sequence[Example], batch_size: int) -> list[list[int]]:
    """Sort indices by source length, then cut into near-equal-length chunks."""
    order = sorted(range(len(examples)), key=lambda index: (examples[index].src_len, index))
    return [order[start : start + batch_size] for start in range(0, len(order), batch_size)]


def make_batch(examples: Sequence[Example], indices: Sequence[int], pad_id: int = PAD_ID) -> dict[str, torch.Tensor]:
    selected = [examples[index] for index in indices]
    src_max = max(len(item.src_ids) for item in selected)
    tgt_max = max(len(item.tgt_in_ids) for item in selected)

    src = torch.full((len(selected), src_max), pad_id, dtype=torch.long)
    src_mask = torch.ones((len(selected), src_max), dtype=torch.bool)
    tgt_in = torch.full((len(selected), tgt_max), pad_id, dtype=torch.long)
    tgt_mask = torch.ones((len(selected), tgt_max), dtype=torch.bool)
    tgt_out = torch.full((len(selected), tgt_max), IGNORE_INDEX, dtype=torch.long)

    for row, item in enumerate(selected):
        src[row, : len(item.src_ids)] = torch.tensor(item.src_ids, dtype=torch.long)
        src_mask[row, : len(item.src_ids)] = False
        tgt_in[row, : len(item.tgt_in_ids)] = torch.tensor(item.tgt_in_ids, dtype=torch.long)
        tgt_mask[row, : len(item.tgt_in_ids)] = False
        tgt_out[row, : len(item.tgt_out_ids)] = torch.tensor(item.tgt_out_ids, dtype=torch.long)

    return {
        "src_ids": src,
        "src_pad_mask": src_mask,
        "tgt_in_ids": tgt_in,
        "tgt_pad_mask": tgt_mask,
        "tgt_out_ids": tgt_out,
        "padded_source_tokens": torch.tensor(src.numel(), dtype=torch.long),
        "padded_target_tokens": torch.tensor(tgt_in.numel(), dtype=torch.long),
        "target_tokens": torch.tensor(sum(len(item.tgt_out_ids) for item in selected), dtype=torch.long),
    }


class TokenBudgetSampler:
    """Deterministic, resumable walk over length-bucketed microbatches.

    State is ``(epoch, chunk_cursor)``; the chunk order is a deterministic
    permutation derived from ``(seed, epoch)`` so any pause resumes exactly.
    ``chunk_tokens[i]`` is the true non-padding target-token count of chunk ``i``
    and drives the accumulation budget.
    """

    def __init__(
        self,
        chunk_tokens: Sequence[int],
        *,
        seed: int,
        epoch: int = 0,
        chunk_cursor: int = 0,
    ):
        if not chunk_tokens:
            raise ValueError("sampler requires at least one chunk")
        self.chunk_tokens = list(chunk_tokens)
        self.chunk_count = len(self.chunk_tokens)
        self.seed = seed
        self.epoch = epoch
        self.chunk_cursor = chunk_cursor
        self._order = self._permutation(epoch)
        # A checkpoint may be written exactly at an epoch boundary; advance
        # explicitly instead of silently replaying the finished epoch.
        while self.chunk_cursor >= self.chunk_count:
            self.chunk_cursor -= self.chunk_count
            self.epoch += 1
            self._order = self._permutation(self.epoch)

    def _permutation(self, epoch: int) -> list[int]:
        digest = hashlib.sha256(f"{self.seed}:{epoch}".encode()).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "big") % (2**63))
        return torch.randperm(self.chunk_count, generator=generator).tolist()

    def groups(self, budget: int) -> Iterable[list[int]]:
        """Yield groups of chunk indices whose non-padding target tokens reach ``budget``."""
        while True:
            while self.chunk_cursor < self.chunk_count:
                group: list[int] = []
                tokens = 0
                while self.chunk_cursor < self.chunk_count and tokens < budget:
                    chunk_index = self._order[self.chunk_cursor]
                    group.append(chunk_index)
                    self.chunk_cursor += 1
                    tokens += self.chunk_tokens[chunk_index]
                yield group
            self.epoch += 1
            self.chunk_cursor = 0
            self._order = self._permutation(self.epoch)

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "chunk_cursor": self.chunk_cursor, "seed": self.seed}

    @classmethod
    def from_state(cls, chunk_tokens: Sequence[int], state: dict) -> "TokenBudgetSampler":
        return cls(
            chunk_tokens,
            seed=int(state.get("seed", 1337)),
            epoch=int(state.get("epoch", 0)),
            chunk_cursor=int(state.get("chunk_cursor", 0)),
        )


def dataset_hash(examples: Sequence[Example]) -> str:
    digest = hashlib.sha256()
    for example in examples:
        digest.update(example.pair_id.encode())
        digest.update(example.direction.encode())
        digest.update(bytes(str(len(example.src_ids)), "ascii"))
        digest.update(bytes(str(len(example.tgt_out_ids)), "ascii"))
    return digest.hexdigest()


def write_stats(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
