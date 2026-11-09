"""Parsers for the two everyday-language sources added in this iteration.

Tatoeba is read from its weekly per-language sentence files plus the EN-FR
link file. ParaCrawl is read from the official v9 two-column TXT gzip stream,
with a bounded compressed-byte limit so the 21 GB release is never required in
full for a first experiment.
"""

from __future__ import annotations

import bz2
import hashlib
import zlib
from dataclasses import dataclass
from pathlib import Path

from .prepare import Pair, cap_pairs, normalize_text


@dataclass
class ExternalReadStats:
    source: str
    raw_lines: int = 0
    decoded_lines: int = 0
    malformed_lines: int = 0
    missing_ids: int = 0
    links: int = 0
    pairs: int = 0
    pairs_after_cap: int = 0
    groups: int = 0
    compressed_bytes: int = 0
    partial: bool = False
    encoding_fallbacks: int = 0
    score_rows: int = 0
    dropped_low_score: int = 0
    score_column: bool = False
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "raw_lines": self.raw_lines,
            "decoded_lines": self.decoded_lines,
            "malformed_lines": self.malformed_lines,
            "missing_ids": self.missing_ids,
            "links": self.links,
            "pairs": self.pairs,
            "pairs_after_cap": self.pairs_after_cap,
            "groups": self.groups,
            "compressed_bytes": self.compressed_bytes,
            "partial": self.partial,
            "encoding_fallbacks": self.encoding_fallbacks,
            "score_rows": self.score_rows,
            "dropped_low_score": self.dropped_low_score,
            "score_column": self.score_column,
            "notes": self.notes,
        }


def decode_line(raw: bytes) -> tuple[str, bool]:
    """Decode a source row as UTF-8, falling back to CP1252 for old web pages."""
    try:
        return raw.decode("utf-8"), False
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace"), True


def _tatoeba_block(sentence_id: str) -> int:
    return int(hashlib.sha256(f"tatoeba:{sentence_id}".encode()).hexdigest()[:16], 16)


def read_tatoeba(
    eng_sentences: Path,
    fra_sentences: Path,
    links_path: Path,
    *,
    max_pairs: int = 0,
    seed: int = 1337,
) -> tuple[list[Pair], ExternalReadStats]:
    """Build EN-FR pairs from Tatoeba's sentence-id link export.

    All translations attached to one English sentence share a split block; the
    block is a related-sentence group, not a claim that the rows are documents.
    """
    stats = ExternalReadStats(source="tatoeba")
    links: list[tuple[str, str]] = []
    wanted_eng: set[str] = set()
    wanted_fra: set[str] = set()
    with bz2.open(links_path, "rb") as handle:
        for raw in handle:
            stats.raw_lines += 1
            line, fallback = decode_line(raw.rstrip(b"\r\n"))
            stats.encoding_fallbacks += int(fallback)
            parts = line.split("\t")
            if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
                stats.malformed_lines += 1
                continue
            eng_id, fra_id = parts
            links.append((eng_id, fra_id))
            wanted_eng.add(eng_id)
            wanted_fra.add(fra_id)
            stats.decoded_lines += 1
    stats.links = len(links)

    def load_sentences(path: Path, language: str, wanted: set[str]) -> dict[str, str]:
        found: dict[str, str] = {}
        with bz2.open(path, "rb") as handle:
            for raw in handle:
                stats.raw_lines += 1
                line, fallback = decode_line(raw.rstrip(b"\r\n"))
                stats.encoding_fallbacks += int(fallback)
                parts = line.split("\t", 2)
                if len(parts) != 3:
                    stats.malformed_lines += 1
                    continue
                sentence_id, tagged_language, text = parts
                if tagged_language != language or sentence_id not in wanted:
                    continue
                found[sentence_id] = normalize_text(text)
                stats.decoded_lines += 1
        return found

    eng = load_sentences(eng_sentences, "eng", wanted_eng)
    fra = load_sentences(fra_sentences, "fra", wanted_fra)
    pairs: list[Pair] = []
    for eng_id, fra_id in links:
        en = eng.get(eng_id)
        fr = fra.get(fra_id)
        if not en or not fr:
            stats.missing_ids += 1
            continue
        pairs.append(
            Pair(
                pair_id=f"tatoeba:{eng_id}:{fra_id}",
                block=_tatoeba_block(eng_id),
                en=en,
                fr=fr,
                source="tatoeba",
            )
        )
    stats.pairs = len(pairs)
    stats.groups = len({pair.block for pair in pairs})
    if max_pairs > 0:
        pairs, cap = cap_pairs(pairs, max_pairs, seed=seed, by_block=True)
        stats.pairs_after_cap = cap["kept"]
    else:
        stats.pairs_after_cap = len(pairs)
    stats.notes = (
        "Weekly per-language sentences + eng-fra links. All translations of one "
        "English sentence stay in the same split block. The weekly files do not "
        "carry a per-row license or review column."
    )
    return pairs, stats


def read_paracrawl(
    path: Path,
    *,
    max_bytes: int = 64 * 1024 * 1024,
    block_size: int = 200,
    block_offset: int = 20_000_000,
    min_score: float = 0.0,
) -> tuple[list[Pair], ExternalReadStats]:
    """Parse a bounded prefix of ParaCrawl v9's two-column EN-FR TXT gzip.

    The official v9 TXT has no score column. If a later/re-release adds one,
    a third numeric column is honored and filtered by ``min_score``.
    """
    if max_bytes < 1:
        raise ValueError("ParaCrawl byte limit must be at least one")
    compressed_size = path.stat().st_size
    stats = ExternalReadStats(
        source="paracrawl",
        compressed_bytes=min(compressed_size, max_bytes),
        partial=compressed_size >= max_bytes,
    )
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    pending = b""
    pairs: list[Pair] = []
    line_number = 0
    remaining = min(compressed_size, max_bytes)

    def consume(raw: bytes) -> None:
        nonlocal line_number
        line, fallback = decode_line(raw.rstrip(b"\r\n"))
        stats.encoding_fallbacks += int(fallback)
        stats.raw_lines += 1
        if not line.strip():
            return
        parts = line.split("\t")
        score = None
        if len(parts) == 2:
            en, fr = parts
        elif len(parts) == 3:
            en, fr, raw_score = parts
            try:
                score = float(raw_score)
            except ValueError:
                stats.malformed_lines += 1
                return
            stats.score_rows += 1
            stats.score_column = True
            if score < min_score:
                stats.dropped_low_score += 1
                return
        else:
            stats.malformed_lines += 1
            return
        en = normalize_text(en)
        fr = normalize_text(fr)
        if not en or not fr:
            stats.malformed_lines += 1
            return
        pairs.append(
            Pair(
                pair_id=f"paracrawl:{line_number:08d}",
                block=block_offset + line_number // block_size,
                en=en,
                fr=fr,
                source="paracrawl",
            )
        )
        stats.decoded_lines += 1
        line_number += 1

    with path.open("rb") as handle:
        while remaining > 0:
            chunk = handle.read(min(1 << 20, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            try:
                data = decompressor.decompress(chunk)
            except zlib.error:
                stats.malformed_lines += 1
                break
            pending += data
            rows = pending.split(b"\n")
            pending = rows.pop()
            for row in rows:
                consume(row)
            if decompressor.eof:
                break
    # A bounded prefix usually ends mid-member; the incomplete trailing row is
    # deliberately discarded rather than guessed at.
    if decompressor.eof and pending.strip():
        consume(pending)
    stats.pairs = len(pairs)
    stats.pairs_after_cap = len(pairs)
    stats.notes = (
        "ParaCrawl v9 TXT is two-column UTF-8 and has no per-row Bicleaner score. "
        "The sample is a deterministic hash cap over a bounded prefix of the "
        "release; it is not a random sample of the whole 21 GB corpus."
        if not stats.score_column
        else "A numeric score column was present and filtered."
    )
    return pairs, stats