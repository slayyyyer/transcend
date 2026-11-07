"""API-assisted comparison of greedy and tuned-beam translations.

This is deliberately framed as machine-assisted review. The API output is a
third machine translation, not a reference and not a human judgment. The default
provider is the no-key Google ``gtx`` endpoint; MyMemory is also available but
is translation-memory-backed and can return reference-like matches, so it is not
an independent MT baseline.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Callable, Sequence

from sacrebleu.metrics import CHRF

from .evaluate import corpus_scores
from .review_packet import _flags, select_indices

MYMEMORY_ENDPOINT = "https://api.mymemory.translated.net/get"
GOOGLE_GTX_ENDPOINT = "https://translate.googleapis.com/translate_a/single"
DEFAULT_PROVIDER = "google-gtx"
DEFAULT_ENDPOINTS = {
    "google-gtx": GOOGLE_GTX_ENDPOINT,
    "mymemory": MYMEMORY_ENDPOINT,
}
PROVIDER_LABELS = {
    "google-gtx": "Google Translate (gtx, unofficial)",
    "mymemory": "MyMemory",
}
PROVIDER_NOTES = {
    "google-gtx": (
        "The Google gtx endpoint is free and needs no key, but it is an "
        "undocumented public endpoint with no SLA and may rate-limit or change."
    ),
    "mymemory": (
        "MyMemory is translation-memory-backed and may return reference-like "
        "matches; treat it as retrieval, not an independent MT baseline."
    ),
}
DEFAULT_MAX_QUERY_BYTES = 450


class ApiError(RuntimeError):
    """A translation request failed or returned no usable output."""


def byte_chunks(text: str, max_bytes: int = DEFAULT_MAX_QUERY_BYTES) -> list[str]:
    """Split text into chunks whose UTF-8 encoding fits the API limit.

    MyMemory's GET endpoint accepts at most 500 bytes of query text. Splitting
    at whitespace keeps ordinary words intact; a single overlong token is split
    as a last resort. The harness records how many rows were chunked.
    """
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    if not text:
        return []
    if len(text.encode("utf-8")) <= max_bytes:
        return [text]

    chunks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            chunks.append(" ".join(current))
            current.clear()

    for word in text.split():
        if len(word.encode("utf-8")) > max_bytes:
            flush()
            piece = ""
            for character in word:
                if piece and len((piece + character).encode("utf-8")) > max_bytes:
                    chunks.append(piece)
                    piece = character
                else:
                    piece += character
            if piece:
                current.append(piece)
            continue
        candidate = " ".join(current + [word])
        if current and len(candidate.encode("utf-8")) > max_bytes:
            flush()
        current.append(word)
    flush()
    return chunks


def _fetch_json(url: str, timeout: float) -> dict:
    request = urllib.request.Request(
        url, headers={"User-Agent": "transcend-review-harness/0.1"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class MyMemoryTranslator:
    """Small no-key MyMemory client with retries and source chunking."""

    def __init__(
        self,
        *,
        endpoint: str = MYMEMORY_ENDPOINT,
        email: str | None = None,
        timeout_s: float = 30.0,
        retries: int = 2,
        backoff_s: float = 2.0,
        sleep_s: float = 0.5,
        max_query_bytes: int = DEFAULT_MAX_QUERY_BYTES,
        fetch: Callable[[str, float], dict] | None = None,
    ) -> None:
        if max_query_bytes > 500:
            raise ValueError("MyMemory caps each query at 500 UTF-8 bytes")
        self.endpoint = endpoint
        self.email = email
        self.timeout_s = timeout_s
        self.retries = max(0, retries)
        self.backoff_s = max(0.0, backoff_s)
        self.sleep_s = max(0.0, sleep_s)
        self.max_query_bytes = max_query_bytes
        self.fetch = fetch or _fetch_json

    def translate(self, text: str, source_lang: str, target_lang: str) -> tuple[str, int]:
        chunks = byte_chunks(text, self.max_query_bytes)
        if not chunks:
            return "", 0
        translated: list[str] = []
        for chunk in chunks:
            translated.append(self._request(chunk, source_lang, target_lang))
            if self.sleep_s:
                time.sleep(self.sleep_s)
        return " ".join(part.strip() for part in translated if part.strip()), len(chunks)

    def _request(self, text: str, source_lang: str, target_lang: str) -> str:
        params = {
            "q": text,
            "langpair": f"{source_lang}|{target_lang}",
            "mt": "1",
        }
        if self.email:
            params["de"] = self.email
        url = f"{self.endpoint}?{urllib.parse.urlencode(params)}"
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                payload = self.fetch(url, self.timeout_s)
                status = payload.get("responseStatus")
                if status not in (None, 200, "200"):
                    raise ApiError(f"MyMemory response status {status}")
                if payload.get("quotaFinished"):
                    raise ApiError("MyMemory quota exhausted")
                result = str((payload.get("responseData") or {}).get("translatedText", "")).strip()
                if not result:
                    raise ApiError("MyMemory returned an empty translation")
                if result.upper().startswith("MYMEMORY WARNING"):
                    raise ApiError("MyMemory quota or usage warning")
                return html.unescape(result)
            except Exception as exc:  # HTTP errors, JSON errors, warnings
                last_error = exc
                if attempt < self.retries and self.backoff_s:
                    time.sleep(self.backoff_s * (2**attempt))
        # Do not include the request URL here: it can contain the optional email.
        raise ApiError(f"MyMemory request failed: {type(last_error).__name__}") from last_error


class GoogleGtxTranslator:
    """No-key Google Translate ``gtx`` client with retries and chunking.

    This is an unofficial endpoint, not the paid Google Cloud Translation API.
    It is suitable for spot checks only; callers should keep the request rate
    low and record that the endpoint has no availability guarantee.
    """

    def __init__(
        self,
        *,
        endpoint: str = GOOGLE_GTX_ENDPOINT,
        timeout_s: float = 30.0,
        retries: int = 2,
        backoff_s: float = 2.0,
        sleep_s: float = 0.5,
        max_query_bytes: int = DEFAULT_MAX_QUERY_BYTES,
        fetch: Callable[[str, float], object] | None = None,
    ) -> None:
        if max_query_bytes < 1:
            raise ValueError("max_query_bytes must be positive")
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.retries = max(0, retries)
        self.backoff_s = max(0.0, backoff_s)
        self.sleep_s = max(0.0, sleep_s)
        self.max_query_bytes = max_query_bytes
        self.fetch = fetch or _fetch_json

    def translate(self, text: str, source_lang: str, target_lang: str) -> tuple[str, int]:
        chunks = byte_chunks(text, self.max_query_bytes)
        if not chunks:
            return "", 0
        translated: list[str] = []
        for chunk in chunks:
            translated.append(self._request(chunk, source_lang, target_lang))
            if self.sleep_s:
                time.sleep(self.sleep_s)
        return " ".join(part.strip() for part in translated if part.strip()), len(chunks)

    def _request(self, text: str, source_lang: str, target_lang: str) -> str:
        params = {
            "client": "gtx",
            "sl": source_lang,
            "tl": target_lang,
            "dt": "t",
            "q": text,
        }
        url = f"{self.endpoint}?{urllib.parse.urlencode(params)}"
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                payload = self.fetch(url, self.timeout_s)
                if not isinstance(payload, list) or not payload or not isinstance(payload[0], list):
                    raise ApiError("Google gtx returned an unexpected response")
                result = "".join(
                    str(segment[0])
                    for segment in payload[0]
                    if isinstance(segment, list) and segment and segment[0]
                ).strip()
                if not result:
                    raise ApiError("Google gtx returned an empty translation")
                return html.unescape(result)
            except Exception as exc:  # HTTP errors, JSON errors, malformed payloads
                last_error = exc
                if attempt < self.retries and self.backoff_s:
                    time.sleep(self.backoff_s * (2**attempt))
        raise ApiError(f"Google gtx request failed: {type(last_error).__name__}") from last_error


def _load_report(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    models = data.get("models", {})
    if not models:
        raise ValueError(f"no models in report: {path}")
    preferred = "candidate" if "candidate" in models else next(iter(models))
    hypotheses = models[preferred].get("hypotheses")
    if not isinstance(hypotheses, list):
        raise ValueError(f"no hypotheses for {preferred} in {path}")
    if len(hypotheses) != len(data.get("sources", [])) or len(hypotheses) != len(data.get("references", [])):
        raise ValueError(f"length mismatch in report: {path}")
    return {
        "direction": data["direction"],
        "sources": data["sources"],
        "references": data["references"],
        "doc_ids": data.get("doc_ids") or [""] * len(hypotheses),
        "hypotheses": hypotheses,
        "model": preferred,
        "step": models[preferred].get("step"),
        "checkpoint": models[preferred].get("checkpoint"),
    }


def _cache_key(provider: str, direction: str, text: str) -> str:
    return hashlib.sha256(f"{provider}\0{direction}\0{text}".encode("utf-8")).hexdigest()


def _load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("entries", data) if isinstance(data, dict) else {}
    return entries if isinstance(entries, dict) else {}


def _save_cache(path: Path, entries: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"entries": entries}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _sentence_chrf(metric: CHRF, hypothesis: str, reference: str) -> float:
    return round(float(metric.sentence_score(hypothesis, [reference]).score), 3)


def _wlt(rows: Sequence[dict], candidate: str, baseline: str) -> dict:
    wins = losses = ties = 0
    for row in rows:
        candidate_score = row.get(candidate)
        baseline_score = row.get(baseline)
        if candidate_score is None or baseline_score is None:
            continue
        if candidate_score > baseline_score:
            wins += 1
        elif candidate_score < baseline_score:
            losses += 1
        else:
            ties += 1
    return {"candidate_wins": wins, "baseline_wins": losses, "ties": ties}


def _mean_words(rows: Sequence[dict], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return round(mean(values), 3) if values else None


def build_api_review(
    *,
    greedy_report: Path,
    beam_report: Path,
    out_dir: Path,
    per_direction: int = 25,
    seed: int = 20261107,
    provider: str = DEFAULT_PROVIDER,
    endpoint: str | None = None,
    email: str | None = None,
    cache_path: Path | None = None,
    sleep_s: float = 0.5,
    retries: int = 2,
    timeout_s: float = 30.0,
    max_query_bytes: int = DEFAULT_MAX_QUERY_BYTES,
    dry_run: bool = False,
) -> dict:
    if provider not in DEFAULT_ENDPOINTS:
        raise ValueError(f"unsupported provider: {provider}")
    if provider == "mymemory" and max_query_bytes > 500:
        raise ValueError("MyMemory caps each query at 500 UTF-8 bytes")
    endpoint = endpoint or DEFAULT_ENDPOINTS[provider]
    if per_direction < 0:
        raise ValueError("per_direction cannot be negative")

    greedy = _load_report(Path(greedy_report))
    beam = _load_report(Path(beam_report))
    if greedy["direction"] != beam["direction"]:
        raise ValueError("greedy and beam reports have different directions")
    if greedy["sources"] != beam["sources"] or greedy["references"] != beam["references"]:
        raise ValueError("greedy and beam reports do not cover identical rows")
    direction = greedy["direction"]
    source_lang, target_lang = direction.split("-")
    selected, diagnostic_quota = select_indices(
        greedy["sources"], per_direction=per_direction, seed=seed
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(cache_path) if cache_path else out_dir / "api_cache.json"
    cache = _load_cache(cache_path)
    translator: MyMemoryTranslator | GoogleGtxTranslator
    if provider == "mymemory":
        translator = MyMemoryTranslator(
            endpoint=endpoint,
            email=email,
            timeout_s=timeout_s,
            retries=retries,
            sleep_s=sleep_s,
            max_query_bytes=max_query_bytes,
        )
    else:
        translator = GoogleGtxTranslator(
            endpoint=endpoint,
            timeout_s=timeout_s,
            retries=retries,
            sleep_s=sleep_s,
            max_query_bytes=max_query_bytes,
        )
    chrf = CHRF(word_order=2, lowercase=False)
    rows: list[dict] = []
    status_counts: Counter[str] = Counter()
    api_calls = 0
    cache_hits = 0
    chunked_rows = 0

    for index in selected:
        source = greedy["sources"][index]
        reference = greedy["references"][index]
        greedy_hypothesis = greedy["hypotheses"][index]
        beam_hypothesis = beam["hypotheses"][index]
        api_hypothesis = ""
        api_status = "dry_run"
        api_chunks = 0
        if not dry_run:
            if not source.strip():
                api_status = "empty_source"
            else:
                key = _cache_key(provider, direction, source)
                cached = cache.get(key)
                if isinstance(cached, dict) and cached.get("translation"):
                    api_hypothesis = str(cached["translation"])
                    api_chunks = int(cached.get("chunks", 0))
                    api_status = "cache"
                    cache_hits += 1
                else:
                    try:
                        api_hypothesis, api_chunks = translator.translate(
                            source, source_lang, target_lang
                        )
                        api_status = "ok"
                        api_calls += 1
                        cache[key] = {
                            "translation": api_hypothesis,
                            "chunks": api_chunks,
                        }
                        _save_cache(cache_path, cache)
                        if api_chunks > 1:
                            chunked_rows += 1
                    except Exception as exc:
                        api_status = f"error:{type(exc).__name__}"
        status_counts[api_status] += 1

        greedy_score = _sentence_chrf(chrf, greedy_hypothesis, reference)
        beam_score = _sentence_chrf(chrf, beam_hypothesis, reference)
        api_score = _sentence_chrf(chrf, api_hypothesis, reference) if api_hypothesis else None
        rows.append(
            {
                "id": index,
                "direction": direction,
                "document": greedy["doc_ids"][index],
                "source": source,
                "reference": reference,
                "greedy": greedy_hypothesis,
                "tuned_beam": beam_hypothesis,
                "api": api_hypothesis,
                "greedy_chrfpp": greedy_score,
                "beam_chrfpp": beam_score,
                "api_chrfpp": api_score,
                "beam_minus_greedy": round(beam_score - greedy_score, 3),
                "api_minus_greedy": round(api_score - greedy_score, 3) if api_score is not None else None,
                "api_minus_beam": round(api_score - beam_score, 3) if api_score is not None else None,
                "greedy_words": len(greedy_hypothesis.split()),
                "beam_words": len(beam_hypothesis.split()),
                "api_words": len(api_hypothesis.split()) if api_hypothesis else None,
                "api_chunks": api_chunks,
                "api_status": api_status,
                "diagnostic_flags": ",".join(_flags(source)),
            }
        )

    tsv_path = out_dir / f"review_api_{direction}.tsv"
    columns = list(rows[0]) if rows else [
        "id", "direction", "document", "source", "reference", "greedy",
        "tuned_beam", "api", "greedy_chrfpp", "beam_chrfpp", "api_chrfpp",
        "beam_minus_greedy", "api_minus_greedy", "api_minus_beam",
        "greedy_words", "beam_words", "api_words", "api_chunks", "api_status",
        "diagnostic_flags",
    ]
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [row for row in rows if row["api_chrfpp"] is not None]
    scores = {}
    if valid_rows:
        valid_refs = [row["reference"] for row in valid_rows]
        scores = {
            "greedy": corpus_scores([row["greedy"] for row in valid_rows], valid_refs),
            "tuned_beam": corpus_scores([row["tuned_beam"] for row in valid_rows], valid_refs),
            "api": corpus_scores([row["api"] for row in valid_rows], valid_refs),
        }
    provider_label = PROVIDER_LABELS[provider]
    provider_note = PROVIDER_NOTES[provider]
    summary = {
        "provider": provider,
        "provider_label": provider_label,
        "endpoint": endpoint,
        "machine_translation": True,
        "human_evaluation": False,
        "direction": direction,
        "per_direction": per_direction,
        "sample_count": len(rows),
        "diagnostic_quota": diagnostic_quota,
        "sample_policy": (
            "Deterministic SHA-256 ranking with the same diagnostic-marker "
            "stratification as the blind review packet."
        ),
        "models": {
            "greedy": {
                "label": greedy["model"],
                "step": greedy["step"],
                "checkpoint": greedy["checkpoint"],
            },
            "tuned_beam": {
                "label": beam["model"],
                "step": beam["step"],
                "checkpoint": beam["checkpoint"],
            },
        },
        "decoder_settings": {
            "greedy": "beam 1",
            "tuned_beam": "beam 4, length penalty 1.2, minimum 16 new tokens",
        },
        "api": {
            "provider": provider,
            "independently_trained_from_reference": provider == "google-gtx",
            "dry_run": dry_run,
            "email_supplied": bool(email),
            "cache_path": str(cache_path),
            "max_query_bytes": max_query_bytes,
            "calls": api_calls,
            "cache_hits": cache_hits,
            "chunked_rows": chunked_rows,
            "status_counts": dict(status_counts),
        },
        "scored_rows": len(valid_rows),
        "scores": scores,
        "mean_words": {
            "greedy": _mean_words(valid_rows, "greedy_words"),
            "tuned_beam": _mean_words(valid_rows, "beam_words"),
            "api": _mean_words(valid_rows, "api_words"),
        },
        "sentence_chrfpp": {
            "beam_vs_greedy": _wlt(rows, "beam_chrfpp", "greedy_chrfpp"),
            "api_vs_greedy": _wlt(valid_rows, "api_chrfpp", "greedy_chrfpp"),
            "api_vs_beam": _wlt(valid_rows, "api_chrfpp", "beam_chrfpp"),
        },
        "caveats": [
            "API output is machine translation, not a gold reference or human judgment.",
            "The sample is diagnostic-stratified and is not the full newstest2014 set.",
            "Long sources are chunked; chunking can degrade translation quality.",
            provider_note,
            "This report must not be described as a human evaluation.",
        ],
    }
    summary_path = out_dir / f"review_api_{direction}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    lines = [
        f"# API-assisted review — {direction}",
        "",
        "This is **machine-assisted review**, not a human evaluation. The API "
        "translations are a third machine opinion; they are not gold references.",
        "",
        f"Sample: {len(rows)} segments selected deterministically (seed {seed}) "
        "with the same diagnostic-marker stratification as the blind review packet.",
        f"Checkpoint: `{greedy['model']}` step {greedy['step']} (greedy); "
        f"`{beam['model']}` step {beam['step']} (tuned beam).",
        f"API: {provider_label}, {api_calls} network calls, {cache_hits} cache hits, "
        f"{len(valid_rows)} rows usable. Long sources were split at "
        f"{max_query_bytes} UTF-8 bytes.",
        "",
        "| System | BLEU | chrF++ | mean words |",
        "| --- | ---: | ---: | ---: |",
    ]
    if scores:
        for key, label in (("greedy", "Greedy"), ("tuned_beam", "Tuned beam"), ("api", provider_label)):
            entry = scores[key]
            lines.append(
                f"| {label} | {entry['bleu']:.3f} | {entry['chrfpp']:.3f} | "
                f"{fmt(summary['mean_words'][key])} |"
            )
    else:
        lines.append("| No usable API rows | n/a | n/a | n/a |")
    lines += [
        "",
        "| Comparison | candidate wins | baseline wins | ties |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in (
        ("beam_vs_greedy", "Beam vs greedy"),
        ("api_vs_greedy", f"{provider_label} vs greedy"),
        ("api_vs_beam", f"{provider_label} vs beam"),
    ):
        comparison = summary["sentence_chrfpp"][key]
        lines.append(
            f"| {label} | {comparison['candidate_wins']} | "
            f"{comparison['baseline_wins']} | {comparison['ties']} |"
        )
    lines += [
        "",
        "API status counts: " + ", ".join(
            f"`{status}`={count}" for status, count in summary["api"]["status_counts"].items()
        ) + ".",
        "",
        provider_note + " The harness caches responses and sleeps between requests. "
        + ("Use `--email` only if you are comfortable sending that address to "
           "MyMemory; it is never written to the report. " if provider == "mymemory" else "")
        + f"The cache path is `{cache_path}`.",
        "",
        "Caveats: the sample is not the full benchmark; chunking can degrade "
        "translations; API output is not a human judgment; and no human evaluation "
        "is claimed by this report.",
        "",
    ]
    markdown_path = out_dir / f"review_api_{direction}.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    summary["paths"] = {
        "tsv": str(tsv_path),
        "summary_json": str(summary_path),
        "markdown": str(markdown_path),
    }
    return summary
