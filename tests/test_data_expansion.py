import bz2
import gzip
import json
from pathlib import Path

import pytest

from nmt.data import mixture
from nmt.data.acquire import SOURCES
from nmt.data.external import read_paracrawl, read_tatoeba
from nmt.data.prepare import (
    Pair,
    cap_pairs,
    clean_pairs,
    deduplicate_pairs,
    filter_near_duplicates,
)
from nmt.data.review import prepare_review_set


def _write_tatoeba(directory: Path) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "eng": directory / "eng_sentences.tsv.bz2",
        "fra": directory / "fra_sentences.tsv.bz2",
        "links": directory / "eng-fra_links.tsv.bz2",
    }
    rows = {
        "eng": "1\teng\tHello, how are you?\n2\teng\tWhere is the station?\n3\teng\tI have two brothers.\n",
        "fra": "10\tfra\tBonjour, comment allez-vous ?\n11\tfra\tSalut, ça va ?\n12\tfra\tOù est la gare ?\n13\tfra\tJ'ai deux frères.\n",
        "links": "1\t10\n1\t11\n2\t12\n3\t13\n",
    }
    for name, text in rows.items():
        with bz2.open(paths[name], "wt", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    return paths


def _write_paracrawl(path: Path) -> None:
    rows = [
        "Where can I buy a ticket?\tOù puis-je acheter un billet ?",
        "The weather is nice today.\tIl fait beau aujourd'hui.",
        "This is English only.\tThis is also English only.",
        "aaaaaaaaaaaaaaaaaaaa\tBonjour.",
        "Bad row without a tab",
    ]
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(rows) + "\n")


def test_source_metadata_records_versions_terms_and_limits():
    assert "tatoeba" in SOURCES["tatoeba_eng_sentences"].corpus.lower()
    assert "CC BY 2.0 FR" in SOURCES["tatoeba_eng_sentences"].license
    assert "Tatoeba" in SOURCES["tatoeba_eng_sentences"].attribution
    paracrawl = SOURCES["paracrawl_v9_en_fr"]
    assert paracrawl.release.startswith("v9")
    assert "CC0" in paracrawl.license
    assert paracrawl.download_limit_bytes == 64 * 1024 * 1024
    assert paracrawl.expected_bytes > 20_000_000_000


def test_tatoeba_groups_translations_and_caps_without_splitting_a_group(tmp_path: Path):
    paths = _write_tatoeba(tmp_path / "tatoeba")
    pairs, stats = read_tatoeba(paths["eng"], paths["fra"], paths["links"])
    assert len(pairs) == 4
    assert {pair.pair_id for pair in pairs} == {
        "tatoeba:1:10",
        "tatoeba:1:11",
        "tatoeba:2:12",
        "tatoeba:3:13",
    }
    assert len({pair.block for pair in pairs if pair.en.startswith("Hello")}) == 1
    assert stats.pairs == 4
    assert stats.groups == 3

    capped, cap = cap_pairs(pairs, 2, seed=7, by_block=True)
    assert cap["by_block"] is True
    assert len(capped) <= 3  # the first selected group may contain two translations
    assert len({pair.block for pair in capped}) <= 2


def test_paracrawl_parser_and_bounded_gzip(tmp_path: Path):
    path = tmp_path / "en-fr.txt.gz"
    _write_paracrawl(path)
    pairs, stats = read_paracrawl(path, max_bytes=10_000, block_size=2)
    assert [pair.en for pair in pairs] == [
        "Where can I buy a ticket?",
        "The weather is nice today.",
        "This is English only.",
        "aaaaaaaaaaaaaaaaaaaa",
    ]
    assert stats.malformed_lines == 1
    assert stats.partial is False
    assert stats.score_column is False
    assert pairs[0].source == "paracrawl"
    assert pairs[0].block != pairs[2].block


def test_clean_pairs_filters_wrong_language_repetition_and_malformed():
    pairs = [
        Pair("x:1", 1, "Where is the pharmacy?", "Où est la pharmacie ?", "x"),
        Pair("x:2", 2, "This is English only.", "This is also English only.", "x"),
        Pair("x:3", 3, "Hello hello hello hello hello.", "Bonjour bonjour bonjour bonjour.", "x"),
        Pair("x:4", 4, "Bad \ufffd text", "Mauvais texte", "x"),
    ]
    kept, stats = clean_pairs(pairs, quality_filter=True, language_filter=True)
    assert [pair.pair_id for pair in kept] == ["x:1"]
    assert stats.dropped_language == 1
    assert stats.dropped_repetition == 1
    assert stats.dropped_malformed == 1


def test_deduplication_is_cross_source_and_direction_agnostic():
    pairs = [
        Pair("europarl:1", 1, "The train is late.", "Le train est en retard.", "europarl"),
        Pair("tatoeba:1:2", 2, "the train is late", "le train est en retard", "tatoeba"),
        Pair("paracrawl:3", 3, "Le train est en retard", "The train is late.", "paracrawl"),
    ]
    kept, report = deduplicate_pairs(pairs, aggressive=True)
    assert len(kept) == 1
    assert report["removed_duplicate"] == 2
    assert report["removed_by_source"] == {"paracrawl": 1, "tatoeba": 1}


def test_near_duplicate_filter_catches_small_edit():
    held = [Pair("held:1", 1, "Where is the nearest pharmacy?", "Où est la pharmacie la plus proche ?", "held")]
    train = [
        Pair("train:1", 2, "Where is the nearest pharmacy ?", "Où est la pharmacie la plus proche ?", "train"),
        Pair("train:2", 3, "I have two brothers and one sister.", "J'ai deux frères et une sœur.", "train"),
    ]
    kept, report = filter_near_duplicates(train, held, threshold=0.8)
    assert [pair.pair_id for pair in kept] == ["train:2"]
    assert report["removed"] == 1
    assert report["removed_by_side"]["en"] == 1


def test_review_set_manifest_marks_human_review_status(tmp_path: Path):
    source = Path("data/review/everyday_review.tsv")
    manifest = prepare_review_set(tmp_path / "review", source_path=source)
    assert manifest["human_reviewed"] is False
    assert manifest["reference_status"] == "not_bilingual_human_reviewed"
    assert set(manifest["categories"]) == {"question", "negation", "names", "numbers", "idiom", "tu_vous"}
    assert manifest["rows"] == 24
    assert (tmp_path / "review" / "review_pairs.tsv").exists()
    assert (tmp_path / "review" / "everyday_review.en-fr.jsonl").exists()
    rows = (tmp_path / "review" / "review.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 24
    assert json.loads(rows[0])["excluded_from_training"] is True


def test_expanded_corpus_preserves_baseline_and_adds_both_sources(tmp_path: Path, monkeypatch):
    # Build a tiny real baseline from the project fixtures.
    from nmt.data.build import prepare_fixtures

    base = tmp_path / "base"
    prepare_fixtures(base)
    tatoeba = _write_tatoeba(tmp_path / "tatoeba")
    paracrawl = tmp_path / "paracrawl" / "en-fr.v9.sample.txt.gz"
    paracrawl.parent.mkdir()
    _write_paracrawl(paracrawl)
    review = tmp_path / "review"
    prepare_review_set(review, source_path=Path("data/review/everyday_review.tsv"))

    paths = {
        "tatoeba_eng_sentences": tatoeba["eng"],
        "tatoeba_fra_sentences": tatoeba["fra"],
        "tatoeba_eng_fra_links": tatoeba["links"],
        "paracrawl_v9_en_fr": paracrawl,
    }

    def fake_acquire(keys, raw_dir, **kwargs):
        return {
            "total_bytes": sum(paths[key].stat().st_size for key in keys),
            "sources": [
                {
                    "key": key,
                    "path": str(paths[key]),
                    "bytes": paths[key].stat().st_size,
                    "sha256": "test",
                    "downloaded": False,
                    "partial": key == "paracrawl_v9_en_fr",
                }
                for key in keys
            ],
        }

    monkeypatch.setattr(mixture, "acquire", fake_acquire)
    out = tmp_path / "expanded"
    manifest = mixture.prepare_expanded_corpus(
        out,
        base_dir=base,
        include_tatoeba=True,
        include_paracrawl=True,
        tatoeba_max_pairs=100,
        paracrawl_max_pairs=100,
        paracrawl_max_bytes=10_000,
        near_duplicate_threshold=0.85,
        review_dir=review,
    )
    assert manifest["base"]["counts"]["train"] == 124
    assert manifest["added_counts"]["tatoeba"]["train"] > 0
    assert manifest["added_counts"]["paracrawl"]["train"] > 0
    assert manifest["review_set"]["excluded_from_training"] is True
    assert manifest["review_set"]["human_reviewed"] is False
    base_train = {line.split("\t")[0] for line in (base / "train.tsv").read_text(encoding="utf-8").splitlines()}
    expanded_train = {line.split("\t")[0] for line in (out / "train.tsv").read_text(encoding="utf-8").splitlines()}
    assert base_train <= expanded_train
    for name in ("train", "dev", "test"):
        assert (out / f"{name}.tsv").exists()
    assert (out / "contamination_report.json").exists()
