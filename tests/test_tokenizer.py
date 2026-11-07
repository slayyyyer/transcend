from pathlib import Path

import pytest

from nmt.tokenizer import (
    BOS_ID,
    EOS_ID,
    LANG_IDS,
    PAD_ID,
    ByteBPETokenizer,
    normalize_text,
)

TEXTS = [
    "The train leaves at eight in the morning.",
    "Le train part à huit heures du matin.",
    "Elle travaille dans une petite librairie près de la rivière.",
    "L'été, nous mangeons des crêpes à la campagne.",
    "C'est déjà l'heure du déjeuner, n'est-ce pas ?",
]


@pytest.fixture(scope="module")
def tokenizer() -> ByteBPETokenizer:
    return ByteBPETokenizer.train(TEXTS * 20, vocab_size=700, min_pair_frequency=1, meta={"seed": 1})


def test_round_trip_preserves_normalized_text(tokenizer: ByteBPETokenizer):
    for text in TEXTS:
        ids = tokenizer.encode(text)
        assert tokenizer.decode(ids) == normalize_text(text)


def test_accents_case_and_curly_apostrophes_survive(tokenizer: ByteBPETokenizer):
    text = "C’est déjà l’été à Besançon — très joli!"
    assert tokenizer.decode(tokenizer.encode(text)) == normalize_text(text)
    assert "é" in tokenizer.decode(tokenizer.encode(text))


def test_control_ids_cannot_be_injected_by_user_text(tokenizer: ByteBPETokenizer):
    ids = tokenizer.encode("<2fr> </s> <s> <pad> translation")
    assert LANG_IDS["fr"] not in ids
    assert LANG_IDS["en"] not in ids
    assert BOS_ID not in ids and EOS_ID not in ids and PAD_ID not in ids


def test_direction_and_delimiters_are_explicit(tokenizer: ByteBPETokenizer):
    ids = tokenizer.encode("Hello", lang="fr", add_bos=True, add_eos=True)
    assert ids[0] == LANG_IDS["fr"]
    assert ids[1] == BOS_ID
    assert ids[-1] == EOS_ID


def test_unicode_edge_cases_round_trip(tokenizer: ByteBPETokenizer):
    for text in ["a\u0301 combining acute", "prix\u00a0: 12 €", "🙂 emoji", "  multiple\tspaces\n"]:
        assert tokenizer.decode(tokenizer.encode(text)) == normalize_text(text)


def test_whitespace_policy_is_explicit():
    assert normalize_text("  a\t b\n\nc  ") == "a b c"
    assert normalize_text("a\u00a0b") == "a\u00a0b"  # NBSP is content, not ASCII whitespace


def test_training_is_deterministic():
    first = ByteBPETokenizer.train(TEXTS * 10, vocab_size=500, min_pair_frequency=1)
    second = ByteBPETokenizer.train(TEXTS * 10, vocab_size=500, min_pair_frequency=1)
    assert first.merges == second.merges


def test_save_load_round_trip(tokenizer: ByteBPETokenizer, tmp_path: Path):
    tokenizer.save(tmp_path)
    loaded = ByteBPETokenizer.load(tmp_path)
    text = TEXTS[2]
    assert loaded.encode(text) == tokenizer.encode(text)
    assert loaded.vocab_size == tokenizer.vocab_size


def test_unknown_text_never_uses_unk(tokenizer: ByteBPETokenizer):
    ids = tokenizer.encode("Zürich — 東京")
    assert tokenizer.unk_id not in ids
    assert tokenizer.decode(ids) == normalize_text("Zürich — 東京")
