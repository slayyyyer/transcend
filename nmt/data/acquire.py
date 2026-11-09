"""Bounded, auditable corpus acquisition.

Every download is pinned by URL, recorded with its SHA-256, and checked against
an explicit byte budget. Archives are kept under ``data/raw`` (git-ignored) and
only metadata is committed.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

GIB = 1024**3


@dataclass(frozen=True)
class SourceSpec:
    key: str
    owner: str
    corpus: str
    release: str
    url: str
    archive: str
    terms_url: str
    terms_summary: str
    intended_use: str
    redistribution: str
    license: str = ""
    attribution: str = ""
    format: str = "tar.gz"
    subdir: str = "archives"
    expected_bytes: int | None = None
    download_limit_bytes: int | None = None


SOURCES: dict[str, SourceSpec] = {
    "europarl_v7_fr_en": SourceSpec(
        key="europarl_v7_fr_en",
        owner="European Parliament",
        corpus="Europarl",
        release="v7",
        url="https://www.statmt.org/europarl/v7/fr-en.tgz",
        archive="europarl-v7.fr-en.tgz",
        terms_url="https://www.statmt.org/europarl/",
        terms_summary=(
            "Europarl v7 is distributed by statmt.org for research use. The release does not "
            "carry a standard permissive licence; OPUS lists the licence as unknown. "
            "Reuse/redistribution rights are unresolved here."
        ),
        intended_use="training the research model in this repository",
        redistribution="unresolved: raw text and derived weights are treated as research-only",
    ),
    "wmt14_nc_v9": SourceSpec(
        key="wmt14_nc_v9",
        owner="WMT / News Commentary (source publishers)",
        corpus="News Commentary",
        release="v9 (WMT14)",
        url="https://www.statmt.org/wmt14/training-parallel-nc-v9.tgz",
        archive="wmt14-nc-v9.tgz",
        terms_url="https://www.statmt.org/wmt14/translation-task.html",
        terms_summary=(
            "Distributed for the WMT14 translation task. WMT distribution does not grant "
            "blanket commercial or redistribution rights; the underlying articles remain "
            "the publishers' copyright."
        ),
        intended_use="training the research model in this repository",
        redistribution="unresolved: research use recorded, redistribution not asserted",
    ),
    "wmt14_newstest_fr_en": SourceSpec(
        key="wmt14_newstest_fr_en",
        owner="WMT",
        corpus="newstest2014",
        release="WMT14 test-full",
        url="https://www.statmt.org/wmt14/test-full.tgz",
        archive="wmt14-test-full.tgz",
        terms_url="https://www.statmt.org/wmt14/translation-task.html",
        terms_summary="Evaluation set distributed with the WMT14 shared task.",
        intended_use="held-out final evaluation only; never training or tuning",
        redistribution="evaluation references are not redistributed by this repository",
    ),
    "wmt13_newstest": SourceSpec(
        key="wmt13_newstest",
        owner="WMT",
        corpus="newstest2013",
        release="WMT13 test",
        url="https://www.statmt.org/wmt13/test.tgz",
        archive="wmt13-test.tgz",
        terms_url="https://www.statmt.org/wmt13/translation-task.html",
        terms_summary=(
            "This archive contains source files only. It does not contain a parallel "
            "English/French reference pair, so it is not usable as a bilingual dev set."
        ),
        intended_use="recorded for provenance; not used as a bilingual benchmark",
        redistribution="evaluation data is not redistributed by this repository",
    ),
    "tatoeba_eng_sentences": SourceSpec(
        key="tatoeba_eng_sentences",
        owner="Tatoeba contributors",
        corpus="Tatoeba",
        release="weekly export 2026-11-07",
        url="https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2",
        archive="eng_sentences.tsv.bz2",
        terms_url="https://tatoeba.org/en/terms_of_use",
        terms_summary=(
            "Tatoeba sentence text is released under CC BY 2.0 FR by default; "
            "some rows are also available under CC0 1.0. Attribution to Tatoeba "
            "and the sentence contributors is required for CC BY rows. The weekly "
            "per-language sentence file does not carry a per-row license column."
        ),
        intended_use="training the research model in this repository",
        redistribution=(
            "CC BY 2.0 FR requires attribution; some rows are CC0 1.0; raw weekly "
            "exports are not redistributed by this repository"
        ),
        license="CC BY 2.0 FR (some rows CC0 1.0)",
        attribution=(
            "Credit Tatoeba (https://tatoeba.org) and its contributors; keep the "
            "Tatoeba sentence IDs in derived data so attribution can be resolved"
        ),
        format="tsv.bz2",
        subdir="tatoeba",
        expected_bytes=24_933_870,
    ),
    "tatoeba_fra_sentences": SourceSpec(
        key="tatoeba_fra_sentences",
        owner="Tatoeba contributors",
        corpus="Tatoeba",
        release="weekly export 2026-11-07",
        url="https://downloads.tatoeba.org/exports/per_language/fra/fra_sentences.tsv.bz2",
        archive="fra_sentences.tsv.bz2",
        terms_url="https://tatoeba.org/en/terms_of_use",
        terms_summary=(
            "Tatoeba sentence text is released under CC BY 2.0 FR by default; "
            "some rows are also available under CC0 1.0. Attribution to Tatoeba "
            "and the sentence contributors is required for CC BY rows. The weekly "
            "per-language sentence file does not carry a per-row license column."
        ),
        intended_use="training the research model in this repository",
        redistribution=(
            "CC BY 2.0 FR requires attribution; some rows are CC0 1.0; raw weekly "
            "exports are not redistributed by this repository"
        ),
        license="CC BY 2.0 FR (some rows CC0 1.0)",
        attribution=(
            "Credit Tatoeba (https://tatoeba.org) and its contributors; keep the "
            "Tatoeba sentence IDs in derived data so attribution can be resolved"
        ),
        format="tsv.bz2",
        subdir="tatoeba",
        expected_bytes=9_798_519,
    ),
    "tatoeba_eng_fra_links": SourceSpec(
        key="tatoeba_eng_fra_links",
        owner="Tatoeba contributors",
        corpus="Tatoeba",
        release="weekly export 2026-11-07",
        url="https://downloads.tatoeba.org/exports/per_language/eng/eng-fra_links.tsv.bz2",
        archive="eng-fra_links.tsv.bz2",
        terms_url="https://tatoeba.org/en/terms_of_use",
        terms_summary=(
            "Tatoeba sentence text is released under CC BY 2.0 FR by default; "
            "some rows are also available under CC0 1.0. Attribution to Tatoeba "
            "and the sentence contributors is required for CC BY rows. This file "
            "contains sentence IDs only and is covered by the same terms."
        ),
        intended_use="training the research model in this repository",
        redistribution=(
            "CC BY 2.0 FR requires attribution; some rows are CC0 1.0; raw weekly "
            "exports are not redistributed by this repository"
        ),
        license="CC BY 2.0 FR (some rows CC0 1.0)",
        attribution=(
            "Credit Tatoeba (https://tatoeba.org) and its contributors; keep the "
            "Tatoeba sentence IDs in derived data so attribution can be resolved"
        ),
        format="tsv.bz2",
        subdir="tatoeba",
        expected_bytes=2_675_177,
    ),
    "paracrawl_v9_en_fr": SourceSpec(
        key="paracrawl_v9_en_fr",
        owner="ParaCrawl project; original website publishers retain their text rights",
        corpus="ParaCrawl",
        release="v9 (2021-09-16)",
        url="https://web-language-models.s3.amazonaws.com/paracrawl/release9/en-fr/en-fr.txt.gz",
        archive="en-fr.v9.sample.txt.gz",
        terms_url="https://paracrawl.eu/",
        terms_summary=(
            "ParaCrawl releases the packaging of its parallel data under CC0 1.0, "
            "but states that it does not own the underlying website text. The "
            "project asks users to acknowledge ParaCrawl and provides a notice and "
            "take-down process. The v9 EN-FR TXT file has two columns (source and "
            "target) and no per-row quality score."
        ),
        intended_use="bounded, filtered training sample for the research model in this repository",
        redistribution=(
            "CC0 1.0 packaging; underlying text rights are not asserted; raw data "
            "is not redistributed by this repository"
        ),
        license="CC0 1.0 (packaging); underlying text not owned by ParaCrawl",
        attribution=(
            "Acknowledge the ParaCrawl project (https://paracrawl.eu) and cite "
            "ParaCrawl: Web-Scale Acquisition of Parallel Corpora (ACL 2020)"
        ),
        format="tsv.gz",
        subdir="paracrawl",
        expected_bytes=21_863_614_702,
        download_limit_bytes=64 * 1024 * 1024,
    ),
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def download(
    spec: SourceSpec,
    raw_dir: Path,
    *,
    budget_bytes: int,
    already_used: int,
    limit_bytes: int | None = None,
) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / spec.archive
    limit = spec.download_limit_bytes if limit_bytes is None else limit_bytes
    if limit is not None and limit < 1:
        raise ValueError("download limit must be at least one byte")
    if target.exists():
        size = target.stat().st_size
        partial = bool(limit is not None and size >= limit)
        return {
            "key": spec.key,
            "path": str(target),
            "bytes": size,
            "sha256": sha256_file(target),
            "downloaded": False,
            "partial": partial,
            "source_bytes": spec.expected_bytes,
            "download_limit_bytes": limit,
        }
    headers = {"User-Agent": "transcend-nmt/0.1"}
    if limit is not None:
        headers["Range"] = f"bytes=0-{limit - 1}"
    request = urllib.request.Request(spec.url, headers=headers)
    temp = target.with_suffix(target.suffix + ".part")
    written = 0
    response_headers: dict[str, str] = {}
    source_bytes = spec.expected_bytes
    with urllib.request.urlopen(request, timeout=120) as response, temp.open("wb") as handle:
        response_headers = {key.lower(): value for key, value in response.headers.items()}
        content_range = response_headers.get("content-range", "")
        if "/" in content_range:
            total = content_range.rsplit("/", 1)[-1]
            if total.isdigit():
                source_bytes = int(total)
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            if limit is not None:
                remaining = limit - written
                if remaining <= 0:
                    break
                chunk = chunk[:remaining]
            written += len(chunk)
            if already_used + written > budget_bytes:
                handle.close()
                temp.unlink(missing_ok=True)
                raise RuntimeError(
                    f"download budget exceeded while fetching {spec.key}: "
                    f"{already_used + written} > {budget_bytes} bytes"
                )
            handle.write(chunk)
            if limit is not None and written >= limit:
                break
    temp.replace(target)
    size = target.stat().st_size
    partial = bool(limit is not None and size >= limit and (source_bytes is None or size < source_bytes))
    return {
        "key": spec.key,
        "path": str(target),
        "bytes": size,
        "sha256": sha256_file(target),
        "downloaded": True,
        "partial": partial,
        "source_bytes": source_bytes,
        "download_limit_bytes": limit,
        "etag": response_headers.get("etag"),
        "last_modified": response_headers.get("last-modified"),
        "content_range": response_headers.get("content-range"),
    }


def acquire(
    keys: list[str],
    raw_dir: Path,
    *,
    max_download_gib: float = 5.0,
    limits: dict[str, int] | None = None,
) -> dict:
    budget = int(max_download_gib * GIB)
    used = 0
    records = []
    for key in keys:
        spec = SOURCES[key]
        record = download(
            spec,
            raw_dir / spec.subdir,
            budget_bytes=budget,
            already_used=used,
            limit_bytes=(limits or {}).get(key),
        )
        used += record["bytes"]
        record.update(
            {
                "owner": spec.owner,
                "corpus": spec.corpus,
                "release": spec.release,
                "url": spec.url,
                "terms_url": spec.terms_url,
                "terms_summary": spec.terms_summary,
                "intended_use": spec.intended_use,
                "redistribution": spec.redistribution,
                "license": spec.license,
                "attribution": spec.attribution,
                "format": spec.format,
                "subdir": spec.subdir,
                "retrieved_at_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }
        )
        records.append(record)
    manifest = {
        "total_bytes": used,
        "total_gib": round(used / GIB, 3),
        "budget_gib": max_download_gib,
        "sources": records,
    }
    return manifest


def extract(archive: Path, destination: Path, members: list[str] | None = None) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            if members is not None and member.name not in members:
                continue
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                continue
            with source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            extracted.append(member.name)
    return extracted


def disk_free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
