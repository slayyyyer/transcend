"""Configuration objects and YAML loading.

All architecture/training decisions used by a run are recorded in a single
resolved config file that is written next to every checkpoint, so a run can be
reproduced from its artifact directory alone.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a config file is missing or internally inconsistent."""


@dataclass(frozen=True)
class ModelConfig:
    """Encoder-decoder Transformer hyper-parameters.

    ``vocab_size`` is normally resolved from the frozen tokenizer artifact
    ("auto" in the YAML); it is never inherited from a pretrained model.
    """

    vocab_size: int
    d_model: int = 128
    encoder_layers: int = 2
    decoder_layers: int = 2
    attention_heads: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1
    max_positions: int = 512
    layernorm_eps: float = 1e-5
    activation: str = "gelu"
    tie_embeddings: bool = True
    position_scheme: str = "sinusoidal"

    def __post_init__(self) -> None:
        if self.d_model % self.attention_heads:
            raise ConfigError(
                f"d_model={self.d_model} not divisible by attention_heads={self.attention_heads}"
            )
        if self.vocab_size <= 0:
            raise ConfigError("vocab_size must be positive once resolved")


@dataclass(frozen=True)
class TokenizerConfig:
    target_vocab_size: int = 8000
    min_pair_frequency: int = 2


@dataclass(frozen=True)
class DataConfig:
    source: str = "fixtures"
    train_pairs: int = 0
    max_source_tokens: int = 96
    max_target_tokens: int = 104
    max_char_ratio: float = 3.0
    min_chars: int = 1
    block_size: int = 200
    max_download_gib: float = 5.0
    max_working_gib: float = 20.0


@dataclass(frozen=True)
class TrainConfig:
    device: str = "auto"
    precision: str = "auto"
    num_threads: int = 8
    peak_lr: float = 5e-4
    warmup_updates: int = 200
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-9
    weight_decay: float = 0.01
    clip_norm: float = 1.0
    label_smoothing: float = 0.1
    token_budget: int = 2048
    max_micro_batch: int = 32
    max_steps: int = 4000
    max_train_tokens: int = 8_000_000
    max_walltime_s: int = 1800
    checkpoint_every: int = 250
    validate_every: int = 250
    patience: int = 6
    grad_accum_safety: int = 64


@dataclass(frozen=True)
class EvalConfig:
    max_new_tokens: int = 104
    beam_size: int = 1
    dev_limit: int = 200
    length_penalty: float = 1.0


@dataclass(frozen=True)
class Experiment:
    name: str
    seed: int
    model: ModelConfig
    tokenizer: TokenizerConfig
    data: DataConfig
    train: TrainConfig
    eval: EvalConfig
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_json(self) -> str:
        payload = {
            "name": self.name,
            "seed": self.seed,
            "model": asdict(self.model),
            "tokenizer": asdict(self.tokenizer),
            "data": asdict(self.data),
            "train": asdict(self.train),
            "eval": asdict(self.eval),
        }
        return json.dumps(payload, indent=2, sort_keys=True)


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key) or {}
    if not isinstance(value, dict):
        raise ConfigError(f"config section '{key}' must be a mapping")
    return value


def load_experiment(path: str | Path, *, vocab_size: int | None = None) -> Experiment:
    """Load a YAML preset. ``vocab_size`` overrides the artifact-derived size."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return experiment_from_dict(raw, vocab_size=vocab_size, name=path.stem)


def experiment_from_dict(
    raw: dict[str, Any], *, vocab_size: int | None = None, name: str | None = None
) -> Experiment:
    """Build an Experiment from an already-parsed mapping (e.g. a checkpoint)."""
    if not isinstance(raw, dict):
        raise ConfigError("experiment config must be a mapping")
    model_raw = _section(raw, "model")
    if vocab_size is not None:
        model_raw["vocab_size"] = int(vocab_size)
    elif str(model_raw.get("vocab_size", "auto")) == "auto":
        raise ConfigError(
            "model.vocab_size is 'auto': resolve it from a tokenizer artifact via "
            "load_experiment(..., vocab_size=tokenizer.vocab_size)"
        )
    model_raw["vocab_size"] = int(model_raw["vocab_size"])
    model = ModelConfig(**model_raw)

    train_raw = _section(raw, "train")
    if "betas" in train_raw:
        train_raw["betas"] = tuple(train_raw["betas"])
    train = TrainConfig(**train_raw)

    if train.token_budget <= 0:
        raise ConfigError("train.token_budget must be positive")
    if train.max_walltime_s <= 0:
        raise ConfigError("train.max_walltime_s must be positive")

    return Experiment(
        name=str(raw.get("name", name or "experiment")),
        seed=int(raw.get("seed", 1337)),
        model=model,
        tokenizer=TokenizerConfig(**_section(raw, "tokenizer")),
        data=DataConfig(**_section(raw, "data")),
        train=train,
        eval=EvalConfig(**_section(raw, "eval")),
        raw=raw,
    )


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return path
