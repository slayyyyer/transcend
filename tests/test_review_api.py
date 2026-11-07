import json

from nmt.review_api import (
    GoogleGtxTranslator,
    MyMemoryTranslator,
    _cache_key,
    build_api_review,
    byte_chunks,
)


def _report(direction: str, sources: list[str], references: list[str], hypotheses: list[str]) -> dict:
    return {
        "direction": direction,
        "sources": sources,
        "references": references,
        "doc_ids": [f"doc-{index}" for index in range(len(sources))],
        "models": {"candidate": {"hypotheses": hypotheses}},
    }


def test_byte_chunks_respect_utf8_byte_limit_and_words():
    text = ("Un été très long avec des mots et des accents. " * 8).strip()
    chunks = byte_chunks(text, max_bytes=40)
    assert chunks
    assert all(len(chunk.encode("utf-8")) <= 40 for chunk in chunks)
    assert " ".join(chunks).split() == text.split()


def test_mymemory_translator_parses_and_unescapes():
    seen = []

    def fake_fetch(url, timeout):
        seen.append((url, timeout))
        return {
            "responseStatus": 200,
            "responseData": {"translatedText": "Bonjour &amp; bienvenue"},
        }

    translator = MyMemoryTranslator(fetch=fake_fetch, retries=0, sleep_s=0)
    translated, chunks = translator.translate("Hello and welcome", "en", "fr")
    assert translated == "Bonjour & bienvenue"
    assert chunks == 1
    assert len(seen) == 1
    assert "langpair=en%7Cfr" in seen[0][0]


def test_google_gtx_translator_joins_segments():
    seen = []

    def fake_fetch(url, timeout):
        seen.append((url, timeout))
        return [
            [
                ["Bonjour", "Hello", None, None, 10],
                [" tout le monde", " everyone", None, None, 3],
            ],
            None,
            "en",
        ]

    translator = GoogleGtxTranslator(fetch=fake_fetch, retries=0, sleep_s=0)
    translated, chunks = translator.translate("Hello everyone", "en", "fr")
    assert translated == "Bonjour tout le monde"
    assert chunks == 1
    assert len(seen) == 1
    assert "client=gtx" in seen[0][0]
    assert "tl=fr" in seen[0][0]


def test_cache_keys_are_provider_scoped():
    assert _cache_key("google-gtx", "en-fr", "Hello") != _cache_key(
        "mymemory", "en-fr", "Hello"
    )


def test_dry_run_writes_side_by_side_review_without_network(tmp_path):
    sources = ["Hello.", "A longer public news sentence.", "Third source."]
    references = ["Bonjour.", "Une phrase de presse plus longue.", "Troisième source."]
    greedy = _report("en-fr", sources, references, ["Bonjour.", "Une phrase.", "Troisième."])
    beam = _report("en-fr", sources, references, ["Bonjour.", "Une phrase plus longue.", "Troisième source."])
    greedy_path = tmp_path / "greedy.json"
    beam_path = tmp_path / "beam.json"
    greedy_path.write_text(json.dumps(greedy), encoding="utf-8")
    beam_path.write_text(json.dumps(beam), encoding="utf-8")

    summary = build_api_review(
        greedy_report=greedy_path,
        beam_report=beam_path,
        out_dir=tmp_path / "out",
        per_direction=2,
        dry_run=True,
        cache_path=tmp_path / "out" / "cache.json",
    )

    assert summary["sample_count"] == 2
    assert summary["provider"] == "google-gtx"
    assert summary["provider_label"] == "Google Translate (gtx, unofficial)"
    assert summary["machine_translation"] is True
    assert summary["human_evaluation"] is False
    assert summary["api"]["status_counts"] == {"dry_run": 2}
    assert not (tmp_path / "out" / "cache.json").exists()
    tsv = (tmp_path / "out" / "review_api_en-fr.tsv").read_text(encoding="utf-8")
    assert "greedy" in tsv and "tuned_beam" in tsv and "api" in tsv
    markdown = (tmp_path / "out" / "review_api_en-fr.md").read_text(encoding="utf-8")
    assert "not a human evaluation" in markdown


def test_final_only_report_is_labelled_in_outputs(tmp_path):
    sources = ["Hello."]
    references = ["Bonjour."]
    report = _report("en-fr", sources, references, ["Bonjour."])
    report["models"] = {
        "final": {
            "hypotheses": ["Bonjour."],
            "step": 15897,
            "checkpoint": "models/accel_pilot/checkpoint.pt",
        }
    }
    path = tmp_path / "final.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    summary = build_api_review(
        greedy_report=path,
        beam_report=path,
        out_dir=tmp_path / "out",
        per_direction=1,
        dry_run=True,
    )

    assert summary["models"]["greedy"]["label"] == "final"
    assert summary["models"]["greedy"]["step"] == 15897
    markdown = (tmp_path / "out" / "review_api_en-fr.md").read_text(encoding="utf-8")
    assert "`final` step 15897" in markdown


def test_reports_must_cover_identical_rows(tmp_path):
    greedy_path = tmp_path / "greedy.json"
    beam_path = tmp_path / "beam.json"
    greedy_path.write_text(
        json.dumps(_report("en-fr", ["A."], ["Un."], ["Un."])), encoding="utf-8"
    )
    beam_path.write_text(
        json.dumps(_report("en-fr", ["B."], ["Deux."], ["Deux."])), encoding="utf-8"
    )
    try:
        build_api_review(
            greedy_report=greedy_path,
            beam_report=beam_path,
            out_dir=tmp_path / "out",
            dry_run=True,
        )
    except ValueError as exc:
        assert "identical rows" in str(exc)
    else:
        raise AssertionError("mismatched reports should be rejected")
