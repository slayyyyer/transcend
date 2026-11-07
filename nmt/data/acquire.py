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
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def download(spec: SourceSpec, raw_dir: Path, *, budget_bytes: int, already_used: int) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / spec.archive
    if target.exists():
        size = target.stat().st_size
        return {
            "key": spec.key,
            "path": str(target),
            "bytes": size,
            "sha256": sha256_file(target),
            "downloaded": False,
        }
    request = urllib.request.Request(spec.url, headers={"User-Agent": "transcend-nmt/0.1"})
    temp = target.with_suffix(target.suffix + ".part")
    written = 0
    with urllib.request.urlopen(request, timeout=120) as response, temp.open("wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            written += len(chunk)
            if already_used + written > budget_bytes:
                handle.close()
                temp.unlink(missing_ok=True)
                raise RuntimeError(
                    f"download budget exceeded while fetching {spec.key}: "
                    f"{already_used + written} > {budget_bytes} bytes"
                )
            handle.write(chunk)
    temp.replace(target)
    return {
        "key": spec.key,
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        "downloaded": True,
    }


def acquire(keys: list[str], raw_dir: Path, *, max_download_gib: float = 5.0) -> dict:
    budget = int(max_download_gib * GIB)
    used = 0
    records = []
    for key in keys:
        spec = SOURCES[key]
        record = download(spec, raw_dir / "archives", budget_bytes=budget, already_used=used)
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
