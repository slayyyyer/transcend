"""Generate the Java tokenizer fixture and golden vectors from the Python code.

The fixture tokenizer is trained on the repository's CC0 fixture corpus only, so
the Java tests never need a trained model or a corpus download.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nmt.tokenizer import ByteBPETokenizer  # noqa: E402

RESOURCES = ROOT / "java" / "src" / "test" / "resources"
CASES = [
    ("The train leaves at eight in the morning.", "fr"),
    ("Elle travaille dans une petite librairie près de la rivière.", "en"),
    ("L'été, nous mangeons des crêpes à la campagne.", "en"),
    ("C’est déjà l’heure du déjeuner, n’est-ce pas ?", "en"),
    ("prix\u00a0: 12 € — très joli!", "en"),
]


def main() -> None:
    fixture = (ROOT / "data" / "fixtures" / "en-fr-fixtures.tsv").read_text(encoding="utf-8")
    texts = []
    for line in fixture.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        _, en, fr = line.split("\t")
        texts.extend([en, fr])
    tokenizer = ByteBPETokenizer.train(texts, vocab_size=900, min_pair_frequency=1, meta={"seed": 7})
    target = RESOURCES / "fixture_tokenizer"
    tokenizer.save(target)

    cases = []
    for text, direction in CASES:
        ids = tokenizer.encode(text, lang=direction, add_eos=True)
        cases.append(
            {
                "text": text,
                "target": direction,
                "ids": ids,
                "decoded": tokenizer.decode(ids),
                "vocab_size": tokenizer.vocab_size,
            }
        )
    (RESOURCES / "tokenizer_golden.json").write_text(
        json.dumps({"cases": cases}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote fixture tokenizer (vocab={tokenizer.vocab_size}) and {len(cases)} golden cases")


if __name__ == "__main__":
    main()
