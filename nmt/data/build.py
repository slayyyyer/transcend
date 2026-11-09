"""Corpus preparation entry points with machine-readable manifests."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from itertools import zip_longest
from pathlib import Path

from ..config import write_json
from ..tokenizer import ByteBPETokenizer, normalize_text
from .acquire import SOURCES, acquire, extract, sha256_file
from .prepare import (
    Pair,
    assert_disjoint,
    clean_pairs,
    length_profile,
    read_parallel,
    read_sgm,
    read_tsv,
    remove_cross_split_leakage,
    split_pairs,
    write_benchmark,
    write_splits,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    return dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def prepare_fixtures(out_dir: Path, *, fixture_path: Path | None = None) -> dict:
    fixture_path = fixture_path or REPO_ROOT / "data" / "fixtures" / "en-fr-fixtures.tsv"
    pairs = read_tsv(fixture_path)
    if not pairs:
        raise RuntimeError(f"no fixture pairs in {fixture_path}")
    cleaned, stats = clean_pairs(pairs, min_chars=1, max_char_ratio=4.0)
    splits = split_pairs(cleaned, salt="fixtures-v1", dev_permyriad=1000, test_permyriad=1000)
    splits, leak = remove_cross_split_leakage(splits)
    assertions = assert_disjoint(splits)
    hashes = write_splits(out_dir, splits)
    manifest = {
        "corpus": "transcend-en-fr-fixtures",
        "owner": "this project",
        "release": "1",
        "created_at_utc": _utc_now(),
        "source_path": str(fixture_path),
        "source_sha256": sha256_file(fixture_path),
        "terms_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "terms_summary": "Hand-authored for this repository and released CC0-1.0.",
        "intended_use": "smoke tests, memorization checks, export parity fixtures",
        "counts": {"before_cleaning": len(pairs), "after_cleaning": len(cleaned)},
        "cleaning": stats.as_dict(),
        "splits": {name: len(items) for name, items in splits.items()},
        "split_hashes": hashes,
        "blocking": "hash of pair id (fixtures are not documents)",
        "leakage": leak,
        "assertions": assertions,
        "length_profile": {name: length_profile(items, "en") for name, items in splits.items()},
    }
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "contamination_report.json", {**leak, **assertions})
    return manifest


def _read_parallel_sample(
    en_path: Path,
    fr_path: Path,
    *,
    stride: int,
    block_size: int,
    prefix: str,
    block_offset: int = 0,
) -> list[Pair]:
    """Sample every ``stride``-th aligned pair, reading both files in lockstep.

    Sampling by raw index (rather than striding each file separately) keeps the
    two languages aligned even when one file lacks a trailing newline.
    """
    pairs: list[Pair] = []
    # newline="\n" disables universal-newline translation: these releases contain
    # lone carriage returns inside text, which otherwise split lines differently
    # in the two languages and destroy alignment.
    with en_path.open("r", encoding="utf-8", errors="strict", newline="\n") as en_handle, fr_path.open(
        "r", encoding="utf-8", errors="strict", newline="\n"
    ) as fr_handle:
        for index, (en, fr) in enumerate(zip_longest(en_handle, fr_handle, fillvalue=None)):
            if en is None or fr is None:
                raise RuntimeError(f"{en_path.name} and {fr_path.name} have different line counts")
            if index % stride:
                continue
            pairs.append(
                Pair(
                    pair_id=f"{prefix}:{index:07d}",
                    block=block_offset + index // block_size,
                    en=en.rstrip("\n"),
                    fr=fr.rstrip("\n"),
                )
            )
    return pairs


def prepare_training_corpus(
    out_dir: Path,
    *,
    max_pairs: int = 300_000,
    block_size: int = 200,
    max_char_ratio: float = 3.0,
    max_download_gib: float = 5.0,
    europarl_stride: int | None = None,
    news_commentary_stride: int | None = None,
) -> dict:
    raw_dir = REPO_ROOT / "data" / "raw"
    acquisition = acquire(
        ["europarl_v7_fr_en", "wmt14_nc_v9"], raw_dir, max_download_gib=max_download_gib
    )
    archives = {record["key"]: Path(record["path"]) for record in acquisition["sources"]}

    europarl_dir = raw_dir / "europarl-v7"
    nc_dir = raw_dir / "wmt14-nc-v9"
    extract(archives["europarl_v7_fr_en"], europarl_dir)
    extract(archives["wmt14_nc_v9"], nc_dir)

    europarl_en = europarl_dir / "europarl-v7.fr-en.en"
    europarl_fr = europarl_dir / "europarl-v7.fr-en.fr"
    nc_en = nc_dir / "training" / "news-commentary-v9.fr-en.en"
    nc_fr = nc_dir / "training" / "news-commentary-v9.fr-en.fr"

    europarl_lines = sum(1 for _ in europarl_en.open("r", encoding="utf-8"))
    nc_lines = sum(1 for _ in nc_en.open("r", encoding="utf-8"))
    # Deterministic strides spread sampling over the whole release while keeping
    # memory bounded on the target machine. Explicit stride overrides let a
    # controlled data-expansion run keep every aligned pair; the defaults preserve
    # the original max-pairs sampler.
    if europarl_stride is not None and europarl_stride < 1:
        raise ValueError("europarl_stride must be at least 1")
    if news_commentary_stride is not None and news_commentary_stride < 1:
        raise ValueError("news_commentary_stride must be at least 1")
    share = 0.62
    if europarl_stride is None:
        europarl_stride = max(1, math.ceil((europarl_lines * share) / max_pairs))
    nc_stride = news_commentary_stride
    if nc_stride is None:
        nc_stride = max(1, math.ceil((nc_lines * (1 - share)) / max_pairs))

    pairs: list[Pair] = []
    pairs.extend(
        _read_parallel_sample(
            europarl_en,
            europarl_fr,
            stride=europarl_stride,
            block_size=block_size,
            prefix="europarl",
        )
    )
    pairs.extend(
        _read_parallel_sample(
            nc_en,
            nc_fr,
            stride=nc_stride,
            block_size=block_size,
            prefix="nc",
            block_offset=10_000_000,
        )
    )
    pair_count_before_clean = len(pairs)

    cleaned, stats = clean_pairs(pairs, min_chars=1, max_char_ratio=max_char_ratio)
    del pairs
    cleaned_count = len(cleaned)
    splits = split_pairs(cleaned, salt="europarl-nc-v1")
    del cleaned
    splits, leak = remove_cross_split_leakage(splits)
    assertions = assert_disjoint(splits)
    hashes = write_splits(out_dir, splits)

    file_hashes = {
        "europarl_en": sha256_file(europarl_en),
        "europarl_fr": sha256_file(europarl_fr),
        "nc_en": sha256_file(nc_en),
        "nc_fr": sha256_file(nc_fr),
    }
    manifest = {
        "corpus": "europarl_v7 + wmt14_news_commentary_v9",
        "created_at_utc": _utc_now(),
        "acquisition": acquisition,
        "sources": [
            {
                "key": key,
                "owner": SOURCES[key].owner,
                "corpus": SOURCES[key].corpus,
                "release": SOURCES[key].release,
                "url": SOURCES[key].url,
                "terms_url": SOURCES[key].terms_url,
                "terms_summary": SOURCES[key].terms_summary,
                "intended_use": SOURCES[key].intended_use,
                "redistribution": SOURCES[key].redistribution,
            }
            for key in ("europarl_v7_fr_en", "wmt14_nc_v9")
        ],
        "file_sha256": file_hashes,
        "document_identity": (
            "Europarl v7 and News Commentary v9 are distributed as flat line-aligned text "
            "without document ids. Pairs are grouped into opaque contiguous blocks of "
            f"{block_size} for split locality. These blocks are NOT claimed to be documents."
        ),
        "sampling": {
            "europarl_stride": europarl_stride,
            "news_commentary_stride": nc_stride,
            "stride_overridden": europarl_stride != max(1, math.ceil((europarl_lines * share) / max_pairs))
            or nc_stride != max(1, math.ceil((nc_lines * (1 - share)) / max_pairs)),
            "europarl_source_lines": europarl_lines,
            "news_commentary_source_lines": nc_lines,
            "pairs_before_cleaning": pair_count_before_clean,
        },
        "counts": {
            "before_cleaning": pair_count_before_clean,
            "after_cleaning": cleaned_count,
            "train": len(splits["train"]),
            "dev": len(splits["dev"]),
            "test": len(splits["test"]),
        },
        "cleaning": stats.as_dict(),
        "split_hashes": hashes,
        "leakage": leak,
        "assertions": assertions,
        "length_profile": {
            name: {"en": length_profile(items, "en"), "fr": length_profile(items, "fr")}
            for name, items in splits.items()
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "contamination_report.json", {**leak, **assertions})
    return manifest


def prepare_newstest(out_dir: Path, *, max_download_gib: float = 5.0) -> dict:
    raw_dir = REPO_ROOT / "data" / "raw"
    acquisition = acquire(
        ["wmt14_newstest_fr_en", "wmt13_newstest"], raw_dir, max_download_gib=max_download_gib
    )
    archives = {record["key"]: Path(record["path"]) for record in acquisition["sources"]}
    test_dir = raw_dir / "wmt14-test-full"
    extract(archives["wmt14_newstest_fr_en"], test_dir)

    en_src = test_dir / "test-full" / "newstest2014-fren-src.en.sgm"
    fr_ref = test_dir / "test-full" / "newstest2014-fren-ref.fr.sgm"
    fr_src = test_dir / "test-full" / "newstest2014-fren-src.fr.sgm"
    en_ref = test_dir / "test-full" / "newstest2014-fren-ref.en.sgm"

    doc_en, seg_en, en_source = read_sgm(en_src)
    doc_fr, seg_fr, fr_reference = read_sgm(fr_ref)
    _, _, fr_source = read_sgm(fr_src)
    _, _, en_reference = read_sgm(en_ref)
    if not (len(en_source) == len(fr_reference) == len(fr_source) == len(en_reference)):
        raise RuntimeError("newstest2014 FR-EN files have inconsistent segment counts")
    if seg_en != seg_fr:
        raise RuntimeError("newstest2014 segment ids differ between source and reference")

    en_fr_rows = [
        {"id": f"newstest2014:{i:04d}", "docid": doc_en[i], "src": en_source[i], "ref": fr_reference[i]}
        for i in range(len(en_source))
    ]
    fr_en_rows = [
        {"id": f"newstest2014:{i:04d}", "docid": doc_en[i], "src": fr_source[i], "ref": en_reference[i]}
        for i in range(len(en_source))
    ]
    en_fr = write_benchmark(out_dir, en_fr_rows, name="newstest2014.en-fr")
    fr_en = write_benchmark(out_dir, fr_en_rows, name="newstest2014.fr-en")
    manifest = {
        "corpus": "newstest2014",
        "variant": "wmt14/full (test-full.tgz)",
        "directions": {
            "en-fr": {
                "source_file": str(en_src),
                "reference_file": str(fr_ref),
                **en_fr,
            },
            "fr-en": {
                "source_file": str(fr_src),
                "reference_file": str(en_ref),
                **fr_en,
                "note": "official WMT14 FR->EN source and reference files in the same archive",
            },
        },
        "segments": len(en_source),
        "documents": len(set(doc_en)),
        "file_sha256": {
            "en_src": sha256_file(en_src),
            "fr_ref": sha256_file(fr_ref),
            "fr_src": sha256_file(fr_src),
            "en_ref": sha256_file(en_ref),
        },
        "acquisition": acquisition,
        "terms_url": SOURCES["wmt14_newstest_fr_en"].terms_url,
        "terms_summary": SOURCES["wmt14_newstest_fr_en"].terms_summary,
        "newstest2013_status": (
            "The WMT13 test archive distributed at statmt.org contains source files only; "
            "it provides no parallel English/French reference pair. newstest2013 is therefore "
            "not used as a bilingual dev set here, and the in-domain Europarl dev split is "
            "used for development decisions instead."
        ),
        "created_at_utc": _utc_now(),
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def train_tokenizer(
    out_dir: Path,
    *,
    train_dir: Path,
    vocab_size: int,
    min_pair_frequency: int = 2,
    seed: int = 1337,
    limits: tuple[int, int] = (96, 104),
) -> ByteBPETokenizer:
    train_pairs = read_tsv(train_dir / "train.tsv")
    en_texts = [pair.en for pair in train_pairs]
    fr_texts = [pair.fr for pair in train_pairs]
    digest = hashlib.sha256()
    for text in en_texts:
        digest.update(normalize_text(text).encode("utf-8"))
    for text in fr_texts:
        digest.update(normalize_text(text).encode("utf-8"))
    meta = {
        "seed": seed,
        "en_lines": len(en_texts),
        "fr_lines": len(fr_texts),
        "text_sha256": digest.hexdigest(),
        "trained_split": "train",
        "sampling": "all train pairs, both languages, balanced by line",
    }
    tokenizer = ByteBPETokenizer.train(
        en_texts + fr_texts,
        vocab_size=vocab_size,
        min_pair_frequency=min_pair_frequency,
        meta=meta,
    )
    tokenizer.save(out_dir)

    coverage = {}
    for split in ("train", "dev", "test"):
        path = train_dir / f"{split}.tsv"
        if not path.exists():
            continue
        pairs = read_tsv(path)
        stats = {
            "pairs": len(pairs),
            "en": {"p50": 0, "p90": 0, "p99": 0, "max": 0, "within_limit": 0},
            "fr": {"p50": 0, "p90": 0, "p99": 0, "max": 0, "within_limit": 0},
        }
        for language, field in (("en", "en"), ("fr", "fr")):
            counts = sorted(
                tokenizer.encode(getattr(pair, field), lang="fr" if language == "en" else "en", add_eos=True).__len__()
                for pair in pairs
            )
            if counts:
                def pick(q: float) -> int:
                    return counts[min(len(counts) - 1, int(round(q * (len(counts) - 1))))]

                limit = limits[0]
                stats[language] = {
                    "p50": pick(0.5),
                    "p90": pick(0.9),
                    "p99": pick(0.99),
                    "max": counts[-1],
                    "within_limit": round(sum(1 for c in counts if c <= limit) / len(counts), 4),
                    "limit": limit,
                }
        coverage[split] = stats
    # Which sample texts were used is recorded by hash; sample bytes are small.
    report = {
        "vocab_size": tokenizer.vocab_size,
        "merges": len(tokenizer.merges),
        "min_pair_frequency": min_pair_frequency,
        "target_vocab_size": vocab_size,
        "trained_on": meta,
        "coverage": coverage,
        "normalization": "Unicode NFC; ASCII whitespace collapsed; accents and case preserved",
    }
    write_json(out_dir / "tokenizer_report.json", report)
    return tokenizer
