"""Controlled mixtures on top of an already prepared baseline.

The baseline splits are copied verbatim. New sources are cleaned, capped,
split by their own group metadata, deduplicated against the baseline, and then
checked for exact and approximate held-out contamination. This keeps the
"existing data only" arm reproducible instead of silently rebuilding it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections import Counter
from pathlib import Path

from ..config import write_json
from .acquire import SOURCES, acquire, sha256_file
from .external import read_paracrawl, read_tatoeba
from .prepare import (
    Pair,
    assert_disjoint,
    cap_pairs,
    clean_pairs,
    dedup_text,
    deduplicate_pairs,
    filter_near_duplicates,
    length_profile,
    read_tsv,
    split_pairs,
    write_splits,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _pair_key(pair: Pair) -> tuple[str, str]:
    return tuple(sorted((dedup_text(pair.en), dedup_text(pair.fr))))


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _review_pairs(review_dir: Path | None) -> tuple[list[Pair], dict]:
    if review_dir is None or not (review_dir / "review_pairs.tsv").exists():
        return [], {}
    pairs = read_tsv(review_dir / "review_pairs.tsv")
    return pairs, _read_json(review_dir / "manifest.json")


def _source_entry(
    key: str,
    read_stats: dict,
    cleaning: dict,
    sampling: dict,
    split_counts: dict,
) -> dict:
    spec = SOURCES[key]
    return {
        "key": key,
        "corpus": spec.corpus,
        "release": spec.release,
        "url": spec.url,
        "terms_url": spec.terms_url,
        "license": spec.license,
        "attribution": spec.attribution,
        "read": read_stats,
        "cleaning": cleaning,
        "sampling": sampling,
        "split_counts": split_counts,
    }


def prepare_expanded_corpus(
    out_dir: Path,
    *,
    base_dir: Path = REPO_ROOT / "data" / "prepared" / "europarl_full",
    include_tatoeba: bool = False,
    include_paracrawl: bool = False,
    tatoeba_max_pairs: int = 100_000,
    paracrawl_max_pairs: int = 100_000,
    paracrawl_max_bytes: int = 64 * 1024 * 1024,
    max_download_gib: float = 5.0,
    block_size: int = 200,
    seed: int = 1337,
    max_char_ratio: float = 3.0,
    max_word_ratio: float = 4.0,
    near_duplicate_threshold: float = 0.85,
    review_dir: Path | None = None,
    audit_base_near_duplicates: bool = True,
) -> dict:
    if not include_tatoeba and not include_paracrawl:
        raise ValueError("select at least one of include_tatoeba or include_paracrawl")
    if not (base_dir / "train.tsv").exists():
        raise FileNotFoundError(f"baseline prepared directory is missing train.tsv: {base_dir}")
    if tatoeba_max_pairs < 0 or paracrawl_max_pairs < 0 or paracrawl_max_bytes < 1:
        raise ValueError("sample caps must be non-negative and the ParaCrawl byte cap positive")

    base = {name: read_tsv(base_dir / f"{name}.tsv") for name in ("train", "dev", "test")}
    base_manifest = _read_json(base_dir / "manifest.json")
    if not base["train"]:
        raise RuntimeError(f"baseline has no training pairs: {base_dir}")

    keys: list[str] = []
    if include_tatoeba:
        keys.extend(["tatoeba_eng_sentences", "tatoeba_fra_sentences", "tatoeba_eng_fra_links"])
    if include_paracrawl:
        keys.append("paracrawl_v9_en_fr")
    acquisition = acquire(
        keys,
        REPO_ROOT / "data" / "raw",
        max_download_gib=max_download_gib,
        limits={"paracrawl_v9_en_fr": paracrawl_max_bytes},
    )
    paths = {record["key"]: Path(record["path"]) for record in acquisition["sources"]}

    new_pairs: list[Pair] = []
    source_entries: list[dict] = []
    source_split_counts: dict[str, dict] = {}

    if include_tatoeba:
        raw, read_stats = read_tatoeba(
            paths["tatoeba_eng_sentences"],
            paths["tatoeba_fra_sentences"],
            paths["tatoeba_eng_fra_links"],
            max_pairs=0,
            seed=seed,
        )
        cleaned, cleaning = clean_pairs(
            raw,
            max_char_ratio=max_char_ratio,
            max_word_ratio=max_word_ratio,
            quality_filter=True,
            language_filter=False,
        )
        capped, sampling = cap_pairs(cleaned, tatoeba_max_pairs, seed=seed, by_block=True)
        new_pairs.extend(capped)
        source_entries.append(
            _source_entry(
                "tatoeba_eng_sentences",
                read_stats.as_dict(),
                cleaning.as_dict(),
                sampling,
                {},
            )
        )
        source_split_counts["tatoeba"] = {"pairs_after_clean": len(cleaned), "selected": len(capped)}

    if include_paracrawl:
        raw, read_stats = read_paracrawl(
            paths["paracrawl_v9_en_fr"],
            max_bytes=paracrawl_max_bytes,
            block_size=block_size,
        )
        cleaned, cleaning = clean_pairs(
            raw,
            max_char_ratio=max_char_ratio,
            max_word_ratio=max_word_ratio,
            quality_filter=True,
            language_filter=True,
        )
        capped, sampling = cap_pairs(cleaned, paracrawl_max_pairs, seed=seed, by_block=False)
        new_pairs.extend(capped)
        source_entries.append(
            _source_entry(
                "paracrawl_v9_en_fr",
                read_stats.as_dict(),
                cleaning.as_dict(),
                sampling,
                {},
            )
        )
        source_split_counts["paracrawl"] = {"pairs_after_clean": len(cleaned), "selected": len(capped)}

    # Deduplicate within and across the newly added sources before assigning
    # splits. The baseline is kept verbatim and filtered against below.
    new_pairs, new_source_dedup = deduplicate_pairs(new_pairs, aggressive=True)
    new_splits = split_pairs(new_pairs, salt=f"expanded-{seed}-v1")

    base_keys = {name: {_pair_key(pair) for pair in pairs} for name, pairs in base.items()}
    removed = Counter()
    new_held_keys = {
        _pair_key(pair) for pair in new_splits["dev"] + new_splits["test"]
    }

    new_train: list[Pair] = []
    for pair in new_splits["train"]:
        key = _pair_key(pair)
        if key in base_keys["train"]:
            removed["new_train_duplicate_base_train"] += 1
            continue
        if key in base_keys["dev"] or key in base_keys["test"]:
            removed["new_train_overlap_base_heldout"] += 1
            continue
        if key in new_held_keys:
            removed["new_train_overlap_new_heldout"] += 1
            continue
        new_train.append(pair)
    new_train, train_dedup = deduplicate_pairs(new_train, aggressive=True)

    new_dev: list[Pair] = []
    new_test: list[Pair] = []
    test_keys = {_pair_key(pair) for pair in new_splits["test"] if _pair_key(pair) not in base_keys["train"] and _pair_key(pair) not in base_keys["dev"] and _pair_key(pair) not in base_keys["test"]}
    for pair in new_splits["dev"]:
        key = _pair_key(pair)
        if key in base_keys["train"]:
            removed["new_dev_overlap_base_train"] += 1
            continue
        if key in base_keys["dev"] or key in base_keys["test"] or key in test_keys:
            removed["new_dev_duplicate_heldout"] += 1
            continue
        new_dev.append(pair)
    for pair in new_splits["test"]:
        key = _pair_key(pair)
        if key in base_keys["train"]:
            removed["new_test_overlap_base_train"] += 1
            continue
        if key in base_keys["dev"] or key in base_keys["test"]:
            removed["new_test_duplicate_base_heldout"] += 1
            continue
        new_test.append(pair)
    new_dev, dev_dedup = deduplicate_pairs(new_dev, aggressive=True)
    new_test, test_dedup = deduplicate_pairs(new_test, aggressive=True)

    review_pairs, review_manifest = _review_pairs(review_dir)
    review_en = {pair.en for pair in review_pairs}
    review_fr = {pair.fr for pair in review_pairs}
    exact_review_hits = sum(
        1
        for pair in new_train + new_dev + new_test
        if pair.en in review_en or pair.fr in review_fr
    )
    if exact_review_hits:
        raise RuntimeError(
            f"new data contains {exact_review_hits} exact review-set rows; refusing to train on them"
        )

    near: dict[str, dict] = {}
    if near_duplicate_threshold > 0:
        held_out = base["dev"] + base["test"] + new_dev + new_test + review_pairs
        new_train, near["new_train_vs_held_out"] = filter_near_duplicates(
            new_train, held_out, threshold=near_duplicate_threshold
        )
        new_dev, near["new_dev_vs_base_train"] = filter_near_duplicates(
            new_dev, base["train"], threshold=near_duplicate_threshold
        )
        new_test, near["new_test_vs_base_train"] = filter_near_duplicates(
            new_test, base["train"], threshold=near_duplicate_threshold
        )
        if audit_base_near_duplicates:
            _, near["base_train_audit"] = filter_near_duplicates(
                base["train"],
                base["dev"] + base["test"] + review_pairs,
                threshold=near_duplicate_threshold,
            )
    else:
        near = {"enabled": False}

    final = {
        "train": base["train"] + new_train,
        "dev": base["dev"] + new_dev,
        "test": base["test"] + new_test,
    }
    assertions = assert_disjoint(final)
    final_review_hits = sum(
        1
        for pair in final["train"] + final["dev"] + final["test"]
        if pair.en in review_en or pair.fr in review_fr
    )
    if final_review_hits:
        raise RuntimeError(f"final corpus contains {final_review_hits} exact review-set rows")

    hashes = write_splits(out_dir, final)
    split_counts = {
        "train": len(final["train"]),
        "dev": len(final["dev"]),
        "test": len(final["test"]),
    }
    added_counts = {
        "tatoeba": {"train": len([p for p in new_train if p.source == "tatoeba"]), "dev": len([p for p in new_dev if p.source == "tatoeba"]), "test": len([p for p in new_test if p.source == "tatoeba"])}
        if include_tatoeba
        else {},
        "paracrawl": {"train": len([p for p in new_train if p.source == "paracrawl"]), "dev": len([p for p in new_dev if p.source == "paracrawl"]), "test": len([p for p in new_test if p.source == "paracrawl"])}
        if include_paracrawl
        else {},
    }

    for entry in source_entries:
        key = entry["key"]
        source = "tatoeba" if key.startswith("tatoeba") else "paracrawl"
        entry["split_counts"] = added_counts[source]

    manifest = {
        "corpus": "europarl_v7 + wmt14_news_commentary_v9 + "
        + " + ".join(
            name for name, include in (("tatoeba", include_tatoeba), ("paracrawl_v9", include_paracrawl)) if include
        ),
        "created_at_utc": _utc_now(),
        "base": {
            "path": str(base_dir),
            "manifest_sha256": sha256_file(base_dir / "manifest.json") if (base_dir / "manifest.json").exists() else None,
            "counts": {name: len(pairs) for name, pairs in base.items()},
            "split_hashes": {name: hashlib.sha256((base_dir / f"{name}.tsv").read_bytes()).hexdigest() for name in base},
            "manifest": base_manifest,
        },
        "sources": source_entries,
        "acquisition": acquisition,
        "deduplication": {
            "new_sources": new_source_dedup,
            "new_train_after_base_filter": train_dedup,
            "new_dev": dev_dedup,
            "new_test": test_dedup,
            "removed": dict(sorted(removed.items())),
            "baseline_is_copied_verbatim": True,
        },
        "near_duplicate_contamination": near,
        "review_set": {
            "path": str(review_dir) if review_dir else None,
            "pairs": len(review_pairs),
            "human_reviewed": bool(review_manifest.get("human_reviewed", False)),
            "reference_status": review_manifest.get("reference_status"),
            "exact_overlap_in_final": final_review_hits,
            "excluded_from_training": True,
        },
        "counts": {
            "before_new_cleaning": sum(entry["read"].get("pairs", 0) for entry in source_entries),
            "after_new_cleaning": sum(entry["cleaning"].get("kept", 0) for entry in source_entries),
            "new_selected": len(new_pairs),
            "train": split_counts["train"],
            "dev": split_counts["dev"],
            "test": split_counts["test"],
        },
        "added_counts": added_counts,
        "split_hashes": hashes,
        "assertions": assertions,
        "length_profile": {
            name: {"en": length_profile(items, "en"), "fr": length_profile(items, "fr")}
            for name, items in final.items()
        },
        "document_identity": (
            "Baseline rows keep their existing opaque blocks. Tatoeba rows sharing "
            "an English sentence id stay in one split block. ParaCrawl has no document "
            "ids in the bounded TXT prefix, so contiguous 200-pair blocks are used. "
            "None of these blocks is claimed to be a document."
        ),
        "tokenizer_policy": (
            "Controlled comparisons must use the frozen 8,000-entry tokenizer. Any new "
            "tokenizer must be trained only on the mixture train.tsv; do not retokenize "
            "when continuing an existing checkpoint."
        ),
        "sampling": {
            "seed": seed,
            "tatoeba_max_pairs": tatoeba_max_pairs,
            "paracrawl_max_pairs": paracrawl_max_pairs,
            "paracrawl_max_bytes": paracrawl_max_bytes,
            "near_duplicate_threshold": near_duplicate_threshold,
            "audit_base_near_duplicates": audit_base_near_duplicates,
        },
        "limitations": [
            "Tatoeba is weekly-updated; checksums in the acquisition manifest pin this run, not future downloads.",
            "Tatoeba weekly files do not carry a per-row license or review flag; CC BY attribution is required.",
            "ParaCrawl v9 TXT has no Bicleaner score; filtering is heuristic and the sample is a bounded prefix, not a random sample.",
            "Near-duplicate detection uses word 3-gram Jaccard at a documented threshold and may miss paraphrases.",
            "New source dev/test additions are small and are not substitutes for the existing news evaluation.",
        ],
    }
    write_json(out_dir / "manifest.json", manifest)
    write_json(
        out_dir / "contamination_report.json",
        {
            "assertions": assertions,
            "near_duplicate": near,
            "deduplication": manifest["deduplication"],
            "review_set": manifest["review_set"],
        },
    )
    return manifest