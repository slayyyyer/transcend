"""Preparation and auditing helpers for the small everyday-language review set."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ..config import write_json
from .prepare import Pair, normalize_text, write_benchmark

REVIEW_CATEGORIES = ("question", "negation", "names", "numbers", "idiom", "tu_vous")


@dataclass(frozen=True)
class ReviewRow:
    review_id: str
    category: str
    en: str
    fr: str
    reference_review: str


def read_review_source(path: Path) -> list[ReviewRow]:
    rows: list[ReviewRow] = []
    lines = path.read_text(encoding="utf-8").split("\n")
    if not lines or not lines[0].strip():
        raise ValueError(f"review source is empty: {path}")
    header = lines[0].split("\t")
    if header != ["id", "category", "en", "fr", "reference_review"]:
        raise ValueError(f"unexpected review header in {path}: {header}")
    seen: set[str] = set()
    for line_number, line in enumerate(lines[1:], start=2):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 columns, got {len(parts)}")
        review_id, category, en, fr, reference_review = parts
        if review_id in seen:
            raise ValueError(f"duplicate review id {review_id}")
        if category not in REVIEW_CATEGORIES:
            raise ValueError(f"unknown review category {category!r}")
        if not normalize_text(en) or not normalize_text(fr):
            raise ValueError(f"empty review text at {path}:{line_number}")
        seen.add(review_id)
        rows.append(
            ReviewRow(
                review_id=review_id,
                category=category,
                en=normalize_text(en),
                fr=normalize_text(fr),
                reference_review=reference_review,
            )
        )
    missing = [category for category in REVIEW_CATEGORIES if not any(row.category == category for row in rows)]
    if missing:
        raise ValueError(f"review set is missing categories: {missing}")
    return rows


def prepare_review_set(out_dir: Path, *, source_path: Path) -> dict:
    rows = read_review_source(source_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "review_pairs.tsv"
    pairs_path.write_text(
        "".join(f"review:{row.review_id}\t{row.en}\t{row.fr}\n" for row in rows),
        encoding="utf-8",
    )
    (out_dir / "review.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "id": row.review_id,
                    "category": row.category,
                    "en": row.en,
                    "fr": row.fr,
                    "reference_review": row.reference_review,
                    "excluded_from_training": True,
                },
                ensure_ascii=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    en_fr = [
        {
            "id": f"review:{row.review_id}",
            "category": row.category,
            "src": row.en,
            "ref": row.fr,
            "reference_review": row.reference_review,
        }
        for row in rows
    ]
    fr_en = [
        {
            "id": f"review:{row.review_id}",
            "category": row.category,
            "src": row.fr,
            "ref": row.en,
            "reference_review": row.reference_review,
        }
        for row in rows
    ]
    directions = {
        "en-fr": write_benchmark(out_dir, en_fr, name="everyday_review.en-fr"),
        "fr-en": write_benchmark(out_dir, fr_en, name="everyday_review.fr-en"),
    }
    counts = Counter(row.category for row in rows)
    review_flags = Counter(row.reference_review for row in rows)
    human_reviewed = all(flag.startswith("bilingual_human_reviewed") for flag in review_flags)
    manifest = {
        "corpus": "transcend-everyday-review",
        "created_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "source_path": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "rows": len(rows),
        "categories": {category: counts.get(category, 0) for category in REVIEW_CATEGORIES},
        "reference_review": dict(sorted(review_flags.items())),
        "human_reviewed": human_reviewed,
        "reference_status": (
            "bilingual_human_reviewed" if human_reviewed else "not_bilingual_human_reviewed"
        ),
        "reviewer": None,
        "license": "CC0-1.0 for this project-authored review source; no third-party text",
        "excluded_from_training": True,
        "intended_use": "small diagnostic review of everyday constructions; not a benchmark estimate",
        "directions": directions,
        "review_pairs": {
            "path": str(pairs_path),
            "sha256": hashlib.sha256(pairs_path.read_bytes()).hexdigest(),
        },
        "limitations": [
            "The French references are project-authored drafts and have NOT received qualified bilingual human review.",
            "This is a diagnostic review set, not a statistically representative benchmark.",
            "The English prompts for tu/vous are intentionally repeated so the two French address forms can be compared.",
        ],
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def review_pairs(directory: Path) -> list[Pair]:
    from .prepare import read_tsv

    path = directory / "review_pairs.tsv"
    if not path.exists():
        raise FileNotFoundError(path)
    return read_tsv(path)