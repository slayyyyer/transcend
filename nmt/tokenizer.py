"""Joint English/French byte-level BPE tokenizer, trained from scratch.

Design decision (documented in docs/DECISIONS.md): instead of SentencePiece we
train a byte-level BPE over *words*. Reasons:

* the deployed Java library must reproduce token ids exactly without a JNI
  bridge to a C++ tokenizer; a byte-level BPE over explicit merge ranks is
  implementable in both languages with integer-only arithmetic and therefore
  bit-identical by construction;
* every UTF-8 byte is in the base vocabulary, so there is no UNK path and no
  byte-fallback divergence to prove;
* word boundaries are an explicit token, so BPE never merges across spaces and
  encoding is cacheable per word (fast enough for CPU-only workflows).

The tradeoff versus SentencePiece: no learned linguistic normalization
(we apply NFC ourselves) and somewhat longer sequences at equal vocabulary.

Normalization policy: Unicode NFC, then ASCII whitespace runs collapse to a
single space, then strip ASCII whitespace. Non-ASCII spaces (e.g. NBSP) are
preserved as content. Both runtimes implement exactly this.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

# Base id layout. Byte ids are 0..255; 256 is the word-boundary (space) token.
SPACE_ID = 256
PAD_ID = 257
UNK_ID = 258
BOS_ID = 259
EOS_ID = 260
LANG_IDS = {"en": 261, "fr": 262}
FIRST_MERGED_ID = 263

_ASCII_WS = re.compile(r"[ \t\n\r\f\v]+")
_ASCII_TRIM = " \t\n\r\f\v"

NORMALIZATION_DESCRIPTION = "Unicode NFC; ASCII whitespace runs -> U+0020; strip ASCII whitespace"


def normalize_text(text: str) -> str:
    """Shared normalization. Java applies the same steps in Preprocessor.java."""
    return _ASCII_WS.sub(" ", unicodedata.normalize("NFC", text)).strip(_ASCII_TRIM)


def _split_words(normalized: str) -> list[str]:
    if not normalized:
        return []
    return normalized.split(" ")


@dataclass
class TokenizerTrainingStats:
    lines: int
    en_lines: int
    fr_lines: int
    unique_words: int
    bytes_seen: int
    vocab_size: int
    merge_count: int
    text_sha256: str


class ByteBPETokenizer:
    """Deterministic byte-level BPE with explicit word boundaries."""

    def __init__(self, merges: Sequence[tuple[int, int]], meta: dict | None = None):
        self.merges: list[tuple[int, int]] = [tuple(m) for m in merges]  # type: ignore[misc]
        self.ranks: dict[tuple[int, int], int] = {pair: i for i, pair in enumerate(self.merges)}
        self.merge_to_id: dict[tuple[int, int], int] = {
            pair: FIRST_MERGED_ID + i for i, pair in enumerate(self.merges)
        }
        # id -> bytes for merged pieces (byte ids handled directly on decode)
        self.pieces: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for pair, new_id in self.merge_to_id.items():
            self.pieces[new_id] = self.pieces[pair[0]] + self.pieces[pair[1]]
        self.meta = meta or {}
        self._word_cache: dict[str, tuple[int, ...]] = {}

    # ---------------------------------------------------------------- encoding
    @property
    def vocab_size(self) -> int:
        return FIRST_MERGED_ID + len(self.merges)

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    def lang_id(self, language: str) -> int:
        try:
            return LANG_IDS[language]
        except KeyError as exc:  # pragma: no cover - guarded by callers
            raise ValueError(f"unsupported language: {language}") from exc

    def encode_word(self, word: str) -> list[int]:
        cached = self._word_cache.get(word)
        if cached is not None:
            return list(cached)
        symbols = list(word.encode("utf-8"))
        if len(symbols) > 1:
            while True:
                best_pair = None
                best_rank = None
                for index in range(len(symbols) - 1):
                    rank = self.ranks.get((symbols[index], symbols[index + 1]))
                    if rank is not None and (best_rank is None or rank < best_rank):
                        best_rank = rank
                        best_pair = (symbols[index], symbols[index + 1])
                if best_pair is None:
                    break
                new_id = self.merge_to_id[best_pair]
                merged: list[int] = []
                index = 0
                while index < len(symbols):
                    if (
                        index + 1 < len(symbols)
                        and symbols[index] == best_pair[0]
                        and symbols[index + 1] == best_pair[1]
                    ):
                        merged.append(new_id)
                        index += 2
                    else:
                        merged.append(symbols[index])
                        index += 1
                symbols = merged
        result = tuple(symbols)
        if len(self._word_cache) < 250_000:
            self._word_cache[word] = result
        return list(result)

    def encode(
        self,
        text: str,
        *,
        lang: str | None = None,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode one segment.

        Direction is expressed only by prepending the target-language control id;
        user text can never produce a control id because ids >= 256 are never
        emitted by byte-level encoding.
        """
        ids: list[int] = []
        if lang is not None:
            ids.append(self.lang_id(lang))
        if add_bos:
            ids.append(BOS_ID)
        for word_index, word in enumerate(_split_words(normalize_text(text))):
            if word_index:
                ids.append(SPACE_ID)
            ids.extend(self.encode_word(word))
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def count_tokens(self, text: str) -> int:
        """Content token count (no controls/EOS)."""
        total = 0
        for word_index, word in enumerate(_split_words(normalize_text(text))):
            if word_index:
                total += 1
            total += len(self.encode_word(word))
        return total

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        buffer = bytearray()
        for token_id in ids:
            if token_id < 256:
                buffer.append(token_id)
            elif token_id == SPACE_ID:
                buffer.append(0x20)
            elif token_id in (PAD_ID, UNK_ID, BOS_ID, EOS_ID) or token_id in LANG_IDS.values():
                if not skip_special and token_id == EOS_ID:
                    buffer.extend(b"</s>")
                continue
            elif token_id in self.pieces:
                buffer.extend(self.pieces[token_id])
            else:
                raise ValueError(f"token id {token_id} is outside the trained vocabulary")
        return buffer.decode("utf-8", errors="replace")

    # ---------------------------------------------------------------- training
    @classmethod
    def train(
        cls,
        texts: Sequence[str],
        *,
        vocab_size: int = 8000,
        min_pair_frequency: int = 2,
        max_words: int = 400_000,
        meta: dict | None = None,
    ) -> "ByteBPETokenizer":
        """Learn merges from the supplied (train-split only) text."""
        if vocab_size <= FIRST_MERGED_ID:
            raise ValueError(f"vocab_size must exceed {FIRST_MERGED_ID}")
        merge_budget = vocab_size - FIRST_MERGED_ID

        word_freq: dict[str, int] = {}
        bytes_seen = 0
        for text in texts:
            normalized = normalize_text(text)
            if not normalized:
                continue
            bytes_seen += len(normalized.encode("utf-8"))
            for word in _split_words(normalized):
                word_freq[word] = word_freq.get(word, 0) + 1

        # Deterministic ordering, and a hard cap to keep memory bounded on the
        # actual machine: keep the most frequent words first, then sort by text.
        ranked = sorted(word_freq.items(), key=lambda kv: (-kv[1], kv[0]))
        if len(ranked) > max_words:
            ranked = ranked[:max_words]
        ranked.sort(key=lambda kv: kv[0])

        words: list[list[int]] = [list(word.encode("utf-8")) for word, _ in ranked]
        freqs: list[int] = [count for _, count in ranked]

        pair_counts: dict[tuple[int, int], int] = {}
        pair_where: dict[tuple[int, int], set[int]] = {}
        for word_index, symbols in enumerate(words):
            frequency = freqs[word_index]
            for pair in zip(symbols, symbols[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + frequency
                pair_where.setdefault(pair, set()).add(word_index)

        import heapq

        heap: list[tuple[int, int, int]] = [(-count, a, b) for (a, b), count in pair_counts.items()]
        heapq.heapify(heap)

        merges: list[tuple[int, int]] = []
        while len(merges) < merge_budget and heap:
            neg_count, a, b = heapq.heappop(heap)
            pair = (a, b)
            current = pair_counts.get(pair, 0)
            if current != -neg_count or current < min_pair_frequency:
                continue
            new_id = FIRST_MERGED_ID + len(merges)
            affected = sorted(pair_where.get(pair, ()))
            touched: set[tuple[int, int]] = set()
            for word_index in affected:
                symbols = words[word_index]
                frequency = freqs[word_index]
                for old_pair in zip(symbols, symbols[1:]):
                    pair_counts[old_pair] = pair_counts.get(old_pair, 0) - frequency
                    touched.add(old_pair)
                    where = pair_where.get(old_pair)
                    if where is not None:
                        where.discard(word_index)
                merged: list[int] = []
                index = 0
                while index < len(symbols):
                    if index + 1 < len(symbols) and symbols[index] == a and symbols[index + 1] == b:
                        merged.append(new_id)
                        index += 2
                    else:
                        merged.append(symbols[index])
                        index += 1
                words[word_index] = merged
                for new_pair in zip(merged, merged[1:]):
                    pair_counts[new_pair] = pair_counts.get(new_pair, 0) + frequency
                    pair_where.setdefault(new_pair, set()).add(word_index)
                    touched.add(new_pair)
            merges.append(pair)
            pair_counts.pop(pair, None)
            pair_where.pop(pair, None)
            for candidate in touched:
                count = pair_counts.get(candidate, 0)
                if count > 0:
                    heapq.heappush(heap, (-count, candidate[0], candidate[1]))
                else:
                    pair_counts.pop(candidate, None)
                    pair_where.pop(candidate, None)

        stats = TokenizerTrainingStats(
            lines=len(texts),
            en_lines=int((meta or {}).get("en_lines", 0)),
            fr_lines=int((meta or {}).get("fr_lines", 0)),
            unique_words=len(ranked),
            bytes_seen=bytes_seen,
            vocab_size=FIRST_MERGED_ID + len(merges),
            merge_count=len(merges),
            text_sha256=(meta or {}).get("text_sha256", ""),
        )
        merged_meta = dict(meta or {})
        merged_meta["stats"] = stats.__dict__
        merged_meta["min_pair_frequency"] = min_pair_frequency
        merged_meta["merge_budget"] = merge_budget
        return cls(merges, meta=merged_meta)

    # ------------------------------------------------------------- persistence
    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "merges.txt").write_text(
            "".join(f"{a} {b}\n" for a, b in self.merges), encoding="utf-8"
        )
        payload = {
            "format": "transcend-byte-bpe",
            "version": 1,
            "normalization": NORMALIZATION_DESCRIPTION,
            "special_ids": {
                "space": SPACE_ID,
                "pad": PAD_ID,
                "unk": UNK_ID,
                "bos": BOS_ID,
                "eos": EOS_ID,
                "lang_en": LANG_IDS["en"],
                "lang_fr": LANG_IDS["fr"],
            },
            "first_merged_id": FIRST_MERGED_ID,
            "num_merges": len(self.merges),
            "vocab_size": self.vocab_size,
            "meta": self.meta,
            "pieces": [
                {"id": piece_id, "bytes_b64": base64.b64encode(piece).decode("ascii")}
                for piece_id, piece in sorted(self.pieces.items())
                if piece_id >= FIRST_MERGED_ID
            ],
        }
        (directory / "tokenizer.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (directory / "merges.sha256").write_text(
            hashlib.sha256((directory / "merges.txt").read_bytes()).hexdigest() + "\n", encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "ByteBPETokenizer":
        directory = Path(directory)
        meta_path = directory / "tokenizer.json"
        merges_path = directory / "merges.txt"
        if not meta_path.exists() or not merges_path.exists():
            raise FileNotFoundError(f"tokenizer artifacts missing in {directory}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("format") != "transcend-byte-bpe":
            raise ValueError(f"unsupported tokenizer format: {meta.get('format')}")
        merges = []
        for line in merges_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            a, b = line.split()
            merges.append((int(a), int(b)))
        if int(meta["num_merges"]) != len(merges):
            raise ValueError("tokenizer merge count does not match tokenizer.json")
        return cls(merges, meta=meta)
