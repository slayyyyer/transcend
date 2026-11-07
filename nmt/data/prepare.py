"""Cleaning, split construction and contamination auditing.

Splits are assigned to *bilingual pairs* before either direction is derived, so
both directions always inherit the same split. Europarl v7 and News Commentary
v9 are distributed as flat, line-aligned text without document identifiers; we
therefore group pairs into opaque contiguous blocks and split by block. That is
a conservative grouping for locality, not a claim that blocks are documents.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from ..tokenizer import normalize_text

SGM_SEG = re.compile(r'<seg id="(\d+)">(.*?)</seg>', re.DOTALL)
SGM_DOC = re.compile(r'<doc docid="([^"]+)"[^>]*>')
MARKUP_MARKERS = ("<seg ", "<doc ", "</doc>", "</seg>")


@dataclass
class Pair:
    pair_id: str
    block: int
    en: str
    fr: str


@dataclass
class CleanStats:
    input_pairs: int = 0
    kept: int = 0
    dropped_empty: int = 0
    dropped_identical: int = 0
    dropped_too_short: int = 0
    dropped_ratio: int = 0
    dropped_markup: int = 0
    dropped_too_long: int = 0
    dropped_duplicate: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def read_tsv(path: Path, *, has_header: bool = False) -> list[Pair]:
    pairs: list[Pair] = []
    # Split on "\n" only: corpus text may legitimately contain Unicode line
    # separators (U+2028, U+0085) that str.splitlines() would treat as new rows.
    for index, line in enumerate(path.read_text(encoding="utf-8").split("\n")):
        if not line.strip() or line.startswith("#"):
            continue
        if has_header and index == 0:
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise ValueError(f"{path}: expected 3 tab-separated columns, got {len(parts)}")
        pair_id, en, fr = parts
        pairs.append(Pair(pair_id=pair_id, block=int(hashlib.sha256(pair_id.encode()).hexdigest()[:8], 16), en=en, fr=fr))
    return pairs


def read_parallel(
    en_path: Path,
    fr_path: Path,
    *,
    source: str,
    block_size: int,
    limit: int | None = None,
) -> list[Pair]:
    en_lines: list[str] = []
    fr_lines: list[str] = []
    with en_path.open("r", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            en_lines.append(line.rstrip("\n"))
    with fr_path.open("r", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            fr_lines.append(line.rstrip("\n"))
    if len(en_lines) != len(fr_lines):
        raise ValueError(f"aligned corpora differ in length: {len(en_lines)} vs {len(fr_lines)}")
    pairs: list[Pair] = []
    for index, (en, fr) in enumerate(zip(en_lines, fr_lines)):
        if limit is not None and index >= limit:
            break
        pairs.append(
            Pair(
                pair_id=f"{source}:{index:07d}",
                block=index // block_size,
                en=en,
                fr=fr,
            )
        )
    return pairs


def read_sgm(path: Path) -> tuple[list[str], list[str], list[str]]:
    """Parse a WMT SGM file into (doc_ids, seg_ids, segments)."""
    text = path.read_text(encoding="utf-8", errors="strict")
    doc_ids: list[str] = []
    seg_ids: list[str] = []
    segments: list[str] = []
    current_doc = "unknown"
    position = 0
    for match in SGM_SEG.finditer(text):
        between = text[position : match.start()]
        doc_match = None
        for doc_match in SGM_DOC.finditer(between):
            pass
        if doc_match is not None:
            current_doc = doc_match.group(1)
        doc_ids.append(current_doc)
        seg_ids.append(match.group(1))
        segments.append(_unescape(match.group(2)).strip())
        position = match.end()
    if not segments:
        raise ValueError(f"no <seg> elements found in {path}")
    return doc_ids, seg_ids, segments


def _unescape(text: str) -> str:
    return (
        text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&apos;", "'")
        .replace("&amp;", "&")
    )


def clean_pairs(
    pairs: list[Pair],
    *,
    min_chars: int = 1,
    max_chars: int = 4000,
    max_char_ratio: float = 3.0,
) -> tuple[list[Pair], CleanStats]:
    stats = CleanStats(input_pairs=len(pairs))
    seen: set[tuple[str, str]] = set()
    kept: list[Pair] = []
    for pair in pairs:
        en = normalize_text(pair.en)
        fr = normalize_text(pair.fr)
        if not en or not fr:
            stats.dropped_empty += 1
            continue
        if any(marker in en or marker in fr for marker in MARKUP_MARKERS):
            stats.dropped_markup += 1
            continue
        if len(en) > max_chars or len(fr) > max_chars:
            stats.dropped_too_long += 1
            continue
        if len(en) < min_chars or len(fr) < min_chars:
            stats.dropped_too_short += 1
            continue
        if en == fr:
            stats.dropped_identical += 1
            continue
        ratio = len(en) / len(fr)
        if ratio > max_char_ratio or ratio < 1.0 / max_char_ratio:
            stats.dropped_ratio += 1
            continue
        key = (en, fr)
        if key in seen:
            stats.dropped_duplicate += 1
            continue
        seen.add(key)
        kept.append(Pair(pair_id=pair.pair_id, block=pair.block, en=en, fr=fr))
    stats.kept = len(kept)
    return kept, stats


def assign_split(block: int, salt: str, *, dev_permyriad: int = 100, test_permyriad: int = 100) -> str:
    digest = hashlib.sha256(f"{salt}:block:{block}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 10_000
    if bucket < test_permyriad:
        return "test"
    if bucket < test_permyriad + dev_permyriad:
        return "dev"
    return "train"


def split_pairs(
    pairs: list[Pair], *, salt: str, dev_permyriad: int = 100, test_permyriad: int = 100
) -> dict[str, list[Pair]]:
    splits: dict[str, list[Pair]] = {"train": [], "dev": [], "test": []}
    for pair in pairs:
        splits[assign_split(pair.block, salt, dev_permyriad=dev_permyriad, test_permyriad=test_permyriad)].append(pair)
    return splits


def remove_cross_split_leakage(splits: dict[str, list[Pair]]) -> tuple[dict[str, list[Pair]], dict]:
    """Drop train pairs whose English or French side also occurs in dev/test."""
    held_out = splits["dev"] + splits["test"]
    held_en = {pair.en for pair in held_out}
    held_fr = {pair.fr for pair in held_out}
    kept_train: list[Pair] = []
    removed_en = removed_fr = 0
    for pair in splits["train"]:
        overlap_en = pair.en in held_en
        overlap_fr = pair.fr in held_fr
        if overlap_en or overlap_fr:
            if overlap_en:
                removed_en += 1
            if overlap_fr:
                removed_fr += 1
            continue
        kept_train.append(pair)
    # dev material that is also in test belongs only to test.
    test_en = {pair.en for pair in splits["test"]}
    test_fr = {pair.fr for pair in splits["test"]}
    kept_dev = [pair for pair in splits["dev"] if pair.en not in test_en and pair.fr not in test_fr]
    removed_dev = len(splits["dev"]) - len(kept_dev)
    result = {"train": kept_train, "dev": kept_dev, "test": splits["test"]}
    report = {
        "train_pairs_removed_for_en_overlap": removed_en,
        "train_pairs_removed_for_fr_overlap": removed_fr,
        "dev_pairs_removed_for_test_overlap": removed_dev,
    }
    return result, report


def assert_disjoint(splits: dict[str, list[Pair]]) -> dict:
    ids = {name: {pair.pair_id for pair in pairs} for name, pairs in splits.items()}
    en = {name: {pair.en for pair in pairs} for name, pairs in splits.items()}
    fr = {name: {pair.fr for pair in pairs} for name, pairs in splits.items()}
    report = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if ids[left] & ids[right]:
            raise AssertionError(f"pair ids overlap between {left} and {right}")
        report[f"{left}_{right}_pair_id_overlap"] = 0
        report[f"{left}_{right}_en_overlap"] = len(en[left] & en[right])
        report[f"{left}_{right}_fr_overlap"] = len(fr[left] & fr[right])
        if en[left] & en[right] or fr[left] & fr[right]:
            raise AssertionError(f"held-out text leaks between {left} and {right}")
    return report


def write_splits(directory: Path, splits: dict[str, list[Pair]]) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name, pairs in splits.items():
        path = directory / f"{name}.tsv"
        path.write_text(
            "".join(f"{p.pair_id}\t{p.en}\t{p.fr}\n" for p in pairs), encoding="utf-8"
        )
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def write_benchmark(directory: Path, rows: list[dict], *, name: str) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(rows)}


def length_profile(pairs: list[Pair], field: str) -> dict:
    lengths = sorted(len(getattr(pair, field)) for pair in pairs)
    if not lengths:
        return {"count": 0}
    def percentile(q: float) -> int:
        index = min(len(lengths) - 1, max(0, int(round(q * (len(lengths) - 1)))))
        return lengths[index]
    return {
        "count": len(lengths),
        "mean_chars": round(sum(lengths) / len(lengths), 1),
        "p50_chars": percentile(0.5),
        "p90_chars": percentile(0.9),
        "p99_chars": percentile(0.99),
        "max_chars": lengths[-1],
    }


def duplicate_statistics(pairs: list[Pair]) -> Counter:
    counter: Counter = Counter()
    for pair in pairs:
        counter[pair.en] += 1
    return counter
