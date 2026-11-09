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
from collections import defaultdict
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from ..tokenizer import normalize_text

SGM_SEG = re.compile(r'<seg id="(\d+)">(.*?)</seg>', re.DOTALL)
SGM_DOC = re.compile(r'<doc docid="([^"]+)"[^>]*>')
MARKUP_MARKERS = ("<seg ", "<doc ", "</doc>", "</seg>")

MALFORMED_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd]|"
    r"</?(?:html|body|div|p|br|span|a|script|style)\b|"
    r"https?://|www\.|\{\{|\}\}|\[\[|\]\]|&#\d+;|&[a-z][a-z0-9]+;",
    re.IGNORECASE,
)
REPEATED_CHAR_RE = re.compile(r"([^\W\d_])\1{5,}", re.UNICODE)
WORD_RE = re.compile(r"[\w'’‑-]+", re.UNICODE)

EN_STOP = frozenset(
    "a an and are as at be been but by can could did do does for from had has have he her his how i if in into is it its me more most my no not of on one only or other our out she should so some than that the their them there these they this those to two up us was we were what when where which who why will with would you your".split()
)
FR_STOP = frozenset(
    "a ai au aux avec ce ces cette comme comment dans de des du elle en est et eu fait faire faites fois il ils je la le les leur lui ma mais me mes moi mon ne nos notre nous on ou où par pas peu plus pour que quel quelle qui sa se ses si son sur ta te tes toi ton tu un une vos votre vous y été être".split()
)


@dataclass
class Pair:
    pair_id: str
    block: int
    en: str
    fr: str
    source: str = ""


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
    dropped_language: int = 0
    dropped_malformed: int = 0
    dropped_repetition: int = 0
    dropped_length_mismatch: int = 0

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
        pairs.append(
            Pair(
                pair_id=pair_id,
                block=int(hashlib.sha256(pair_id.encode()).hexdigest()[:8], 16),
                en=en,
                fr=fr,
                source=pair_id.split(":", 1)[0],
            )
        )
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
                source=source,
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
    max_word_ratio: float = 5.0,
    quality_filter: bool = False,
    language_filter: bool = False,
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
        if quality_filter:
            if MALFORMED_RE.search(en) or MALFORMED_RE.search(fr):
                stats.dropped_malformed += 1
                continue
            if _is_repetitive(en) or _is_repetitive(fr):
                stats.dropped_repetition += 1
                continue
            en_words = WORD_RE.findall(en)
            fr_words = WORD_RE.findall(fr)
            if not en_words or not fr_words:
                stats.dropped_malformed += 1
                continue
            word_ratio = len(en_words) / len(fr_words)
            if word_ratio > max_word_ratio or word_ratio < 1.0 / max_word_ratio:
                stats.dropped_length_mismatch += 1
                continue
            if max(len(en_words), len(fr_words)) > 40 and min(len(en_words), len(fr_words)) < 3:
                stats.dropped_length_mismatch += 1
                continue
        if language_filter:
            if not language_ok(en, "en") or not language_ok(fr, "fr"):
                stats.dropped_language += 1
                continue
        key = (en, fr)
        if key in seen:
            stats.dropped_duplicate += 1
            continue
        seen.add(key)
        kept.append(Pair(pair_id=pair.pair_id, block=pair.block, en=en, fr=fr, source=pair.source))
    stats.kept = len(kept)
    return kept, stats


def _is_repetitive(text: str) -> bool:
    if REPEATED_CHAR_RE.search(text):
        return True
    words = [word.casefold() for word in WORD_RE.findall(text)]
    if len(words) >= 4 and max(Counter(words).values()) >= 4:
        return True
    for n in (2, 3):
        if len(words) >= 3 * n:
            grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
            if any(grams.count(gram) >= 3 for gram in set(grams)):
                return True
    return False


def language_scores(text: str) -> tuple[int, int]:
    """Return rough English/French marker counts; a guard, not a language ID model."""
    words = [word.casefold() for word in WORD_RE.findall(normalize_text(text))]
    return sum(word in EN_STOP for word in words), sum(word in FR_STOP for word in words)


def language_ok(text: str, language: str) -> bool:
    words = WORD_RE.findall(normalize_text(text))
    if len(words) < 3:
        # Short strings cannot be identified reliably. Tatoeba's language tags
        # are authoritative; this helper is only used for ParaCrawl.
        return True
    en, fr = language_scores(text)
    if language == "en":
        return en >= 1 and en >= fr
    return fr >= 1 and fr >= en


def dedup_text(text: str, *, aggressive: bool = True) -> str:
    key = normalize_text(text).casefold()
    if aggressive:
        key = re.sub(r"[^\w]+", " ", key, flags=re.UNICODE).strip()
    return key


def deduplicate_pairs(pairs: list[Pair], *, aggressive: bool = True) -> tuple[list[Pair], dict]:
    """Keep the first occurrence of each pair, including reversed duplicates."""
    seen: set[tuple[str, str]] = set()
    kept: list[Pair] = []
    removed_by_source: Counter = Counter()
    for pair in pairs:
        en_key = dedup_text(pair.en, aggressive=aggressive)
        fr_key = dedup_text(pair.fr, aggressive=aggressive)
        key = tuple(sorted((en_key, fr_key)))
        if key in seen:
            removed_by_source[pair.source or "unknown"] += 1
            continue
        seen.add(key)
        kept.append(pair)
    return kept, {
        "removed_duplicate": len(pairs) - len(kept),
        "removed_by_source": dict(sorted(removed_by_source.items())),
        "key": "casefolded normalized sides, punctuation removed, reversed pairs unified"
        if aggressive
        else "exact normalized pair",
    }


def cap_pairs(
    pairs: list[Pair],
    max_pairs: int,
    *,
    seed: int = 1337,
    by_block: bool = False,
) -> tuple[list[Pair], dict]:
    """Deterministically cap a source. Grouped caps never split a Tatoeba group."""
    if max_pairs <= 0 or len(pairs) <= max_pairs:
        return list(pairs), {
            "requested": max_pairs,
            "available": len(pairs),
            "kept": len(pairs),
            "by_block": by_block,
        }
    if not by_block:
        ranked = sorted(
            pairs, key=lambda pair: hashlib.sha256(f"{seed}:{pair.pair_id}".encode()).hexdigest()
        )
        kept = ranked[:max_pairs]
        return kept, {
            "requested": max_pairs,
            "available": len(pairs),
            "kept": len(kept),
            "by_block": False,
        }
    groups: dict[int, list[Pair]] = defaultdict(list)
    for pair in pairs:
        groups[pair.block].append(pair)
    order = sorted(
        groups, key=lambda block: hashlib.sha256(f"{seed}:block:{block}".encode()).hexdigest()
    )
    kept: list[Pair] = []
    for block in order:
        group = groups[block]
        if kept and len(kept) + len(group) > max_pairs:
            continue
        kept.extend(group)
        if len(kept) >= max_pairs:
            break
    return kept, {
        "requested": max_pairs,
        "available": len(pairs),
        "kept": len(kept),
        "by_block": True,
        "blocks_available": len(groups),
    }


def _near_shingles(text: str, n: int = 3) -> frozenset[str]:
    normalized = re.sub(r"[^\w]+", " ", normalize_text(text).casefold(), flags=re.UNICODE).strip()
    tokens = normalized.split()
    if len(tokens) >= n:
        return frozenset(" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return frozenset([normalized]) if normalized else frozenset()


class NearDuplicateIndex:
    """Small word-shingle index for approximate held-out contamination checks."""

    def __init__(self, texts: list[str], *, max_postings: int = 200):
        self.shingles = [_near_shingles(text) for text in texts]
        postings: dict[str, list[int]] = defaultdict(list)
        for index, shingles in enumerate(self.shingles):
            for shingle in shingles:
                postings[shingle].append(index)
        self.postings = {
            shingle: indices
            for shingle, indices in postings.items()
            if len(indices) <= max_postings
        }

    def match(self, text: str, *, threshold: float = 0.85, max_candidates: int = 100) -> tuple[int, float] | None:
        shingles = _near_shingles(text)
        if not shingles:
            return None
        candidates: Counter = Counter()
        for shingle in shingles:
            for index in self.postings.get(shingle, ()):
                candidates[index] += 1
        for index, _ in candidates.most_common(max_candidates):
            other = self.shingles[index]
            union = len(shingles | other)
            if union and len(shingles & other) / union >= threshold:
                return index, len(shingles & other) / union
        return None


def filter_near_duplicates(
    pairs: list[Pair],
    held_out: list[Pair],
    *,
    threshold: float = 0.85,
    max_postings: int = 200,
) -> tuple[list[Pair], dict]:
    if threshold <= 0 or threshold > 1:
        raise ValueError("near-duplicate threshold must be in (0, 1]")
    en_index = NearDuplicateIndex([pair.en for pair in held_out], max_postings=max_postings)
    fr_index = NearDuplicateIndex([pair.fr for pair in held_out], max_postings=max_postings)
    kept: list[Pair] = []
    removed_by_source: Counter = Counter()
    removed_by_side: Counter = Counter()
    examples: list[dict] = []
    for pair in pairs:
        en_match = en_index.match(pair.en, threshold=threshold)
        fr_match = fr_index.match(pair.fr, threshold=threshold)
        if en_match is None and fr_match is None:
            kept.append(pair)
            continue
        source = pair.source or "unknown"
        removed_by_source[source] += 1
        side = "en" if en_match is not None else "fr"
        removed_by_side[side] += 1
        if len(examples) < 20:
            examples.append(
                {
                    "pair_id": pair.pair_id,
                    "source": source,
                    "side": side,
                    "similarity": round((en_match or fr_match)[1], 4),  # type: ignore[index]
                }
            )
    return kept, {
        "threshold": threshold,
        "checked_pairs": len(pairs),
        "held_out_pairs": len(held_out),
        "removed": len(pairs) - len(kept),
        "removed_by_source": dict(sorted(removed_by_source.items())),
        "removed_by_side": dict(sorted(removed_by_side.items())),
        "examples": examples,
        "method": "word 3-gram Jaccard; exact normalized matches also score 1.0",
    }


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
