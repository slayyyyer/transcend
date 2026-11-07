"""Blind bilingual review packet generation.

The packet is a TSV with empty reviewer columns; it never invents judgements.
Sampling is deterministic and documented so the packet can be regenerated.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

DIAGNOSTIC_MARKERS = {
    "negation": ("not ", "n't", "never", "no ", "ne ", "n'", "jamais", "rien", "personne"),
    "quantity_or_unit": ("%", "percent", "pour cent", "km", "kilometre", "kilomètre", "euro", "million", "thousand", "mille"),
    "name_or_acronym": ("mr ", "mrs ", "ms ", "president", "président", "minister", "ministre", "eu ", "ue "),
    "pronoun": ("he ", "she ", "il ", "elle ", "they ", "ils ", "elles "),
    "formal_address": ("vous", "your", "votre", "s'il vous plaît", "please"),
}


def _flags(text: str) -> list[str]:
    lowered = f" {text.lower()} "
    return [name for name, markers in DIAGNOSTIC_MARKERS.items() if any(m in lowered for m in markers)]


def build_review_packet(
    report_path: Path,
    out_dir: Path,
    *,
    per_direction: int = 100,
    seed: int = 20261107,
) -> dict:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = report["sources"]
    references = report["references"]
    models = report["models"]
    labels = list(models)
    preferred = labels[-1]
    if "candidate" in models:
        preferred = "candidate"
    hypotheses = models[preferred]["hypotheses"]
    doc_ids = report.get("doc_ids") or [""] * len(sources)

    ranked = sorted(
        range(len(sources)),
        key=lambda index: hashlib.sha256(f"{seed}:{index}".encode()).hexdigest(),
    )
    # Stratify: take up to a quarter of the sample from segments that carry at
    # least one diagnostic marker, then fill deterministically.
    flagged = [index for index in ranked if _flags(sources[index])]
    plain = [index for index in ranked if index not in set(flagged)]
    diagnostic_quota = min(len(flagged), max(1, per_direction // 4))
    selected = flagged[:diagnostic_quota] + plain[: per_direction - diagnostic_quota]
    selected = sorted(selected)[:per_direction]

    path = out_dir / f"review_{report['direction']}.tsv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "id",
                "direction",
                "document",
                "source",
                "reference",
                "system_output",
                "diagnostic_flags",
                "reviewer_acceptability(OK/MINOR/MAJOR/CRITICAL)",
                "reviewer_error_categories",
                "reviewer_notes",
                "alternative_acceptable_translation",
            ]
        )
        for index in selected:
            writer.writerow(
                [
                    index,
                    report["direction"],
                    doc_ids[index],
                    sources[index],
                    references[index],
                    hypotheses[index],
                    ",".join(_flags(sources[index])),
                    "",
                    "",
                    "",
                    "",
                ]
            )

    guide = out_dir / "README.md"
    guide.write_text(
        f"""# Blind bilingual review packet — {report['direction']}

* Source: `{report['benchmark']}` ({report['n']} segments, {report.get('documents')} documents).
* System: `{preferred}` checkpoint step {models[preferred].get('step')}.
* Sample size: {len(selected)} of {len(sources)} segments.
* Sampling policy: deterministic SHA-256 ranking with seed {seed}; up to
  {diagnostic_quota} segments carrying a diagnostic marker (negation, quantity/unit,
  name/acronym, pronoun, formal address) and the remainder from the general pool.
* Blinding: the system identity is not shown to the reviewer; outputs are not
  pre-marked as correct.

## Instructions

1. Mark each row OK / MINOR / MAJOR / CRITICAL for meaning preservation.
2. Record error categories (omission, addition, mistranslation, grammar,
   agreement, terminology, punctuation) in the free-text column.
3. If a different wording would also be acceptable, write it in the last column.
   do not require one exact wording.
4. A round-trip translation is **not** evidence of correctness.

Human evaluation status: **pending** until a qualified bilingual reviewer
completes the packet. No reviewer judgements are fabricated in this repository.
""",
        encoding="utf-8",
    )
    return {"path": str(path), "count": len(selected), "seed": seed, "model": preferred}
