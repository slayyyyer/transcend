"""Python/ONNX Runtime parity checks for tokenizer, masks, cache and logits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

from .config import ModelConfig, load_json
from .decode import greedy_decode_batch
from .model import build_causal_mask, build_model, build_padding_mask
from .tokenizer import BOS_ID, ByteBPETokenizer

ATOL = 1e-4
RTOL = 1e-3


def _session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 2
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def _max_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def run_parity(model_dir: str | Path, *, samples: int = 3) -> dict[str, Any]:
    model_dir = Path(model_dir)
    tokenizer = ByteBPETokenizer.load(model_dir / "tokenizer")
    config_payload = load_json(model_dir / "model_config.json")
    model_cfg = ModelConfig(**config_payload["config"])
    model = build_model(model_cfg, seed=0)
    state = torch.load(model_dir / "torch_state_dict.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    encoder = _session(model_dir / "onnx" / "encoder.onnx")
    decoder_init = _session(model_dir / "onnx" / "decoder_init.onnx")
    decoder_step = _session(model_dir / "onnx" / "decoder_step.onnx")

    texts = [
        ("The train leaves at eight in the morning.", "fr"),
        ("Elle travaille dans une petite librairie près de la rivière.", "en"),
        ("L'été, nous mangeons des crêpes à la campagne.", "en"),
    ][:samples]

    cache_names = config_payload["onnx"]["cache_order"]
    report: dict[str, Any] = {
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "onnxruntime": ort.__version__,
        "torch": torch.__version__,
        "cases": [],
    }
    worst = 0.0
    for text, target_lang in texts:
        ids = tokenizer.encode(text, lang=target_lang, add_eos=True)
        src_ids = torch.tensor([ids], dtype=torch.long)
        src_mask = torch.zeros_like(src_ids, dtype=torch.bool)
        src_pos = torch.arange(src_ids.size(1), dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            memory_torch = model.encode(src_ids, src_mask, src_pos, training=False)

        memory_onnx = encoder.run(
            None,
            {
                "src_ids": src_ids.numpy(),
                "src_pad_mask": src_mask.numpy(),
                "position_ids": src_pos.numpy(),
            },
        )[0]
        encoder_diff = _max_diff(memory_torch.numpy(), memory_onnx)

        prefix = torch.tensor([[BOS_ID, 42]], dtype=torch.long)
        tgt_pos = torch.arange(prefix.size(1), dtype=torch.long).unsqueeze(0)
        self_mask = build_causal_mask(prefix.size(1), prefix.size(1), 0, torch.device("cpu"), memory_torch.dtype)
        cross_mask = build_padding_mask(src_mask, memory_torch.dtype)
        with torch.no_grad():
            logits_torch, caches = model.decode(
                prefix, memory_torch, self_mask=self_mask, cross_mask=cross_mask, position_ids=tgt_pos
            )
        outputs = decoder_init.run(
            None,
            {
                "tgt_ids": prefix.numpy(),
                "memory": memory_onnx,
                "self_mask": self_mask.numpy(),
                "cross_mask": cross_mask.numpy(),
                "position_ids": tgt_pos.numpy(),
            },
        )
        logits_onnx = outputs[0]
        logits_diff = _max_diff(logits_torch.numpy(), logits_onnx)
        flat = list(outputs[1:])
        cache_diffs = {}
        for index, layer in enumerate(caches):
            for offset, name in enumerate(("k", "v", "cross_k", "cross_v")):
                value = outputs[1 + 4 * index + offset]
                cache_diffs[f"layer{index}_{name}"] = _max_diff(layer[name].numpy(), value)

        next_token = torch.tensor([[77]], dtype=torch.long)
        next_pos = torch.tensor([[prefix.size(1)]], dtype=torch.long)
        with torch.no_grad():
            step_logits, _ = model.decode(
                next_token, None, position_ids=next_pos, past=caches
            )
        step_outputs = decoder_step.run(
            None,
            {
                "tgt_ids": next_token.numpy(),
                "position_ids": next_pos.numpy(),
                **{f"past_{name}": value for name, value in zip(cache_names, flat)},
            },
        )
        step_diff = _max_diff(step_logits.numpy(), step_outputs[0])

        case_worst = max([encoder_diff, logits_diff, step_diff, *cache_diffs.values()])
        worst = max(worst, case_worst)
        report["cases"].append(
            {
                "text": text,
                "source_token_ids": ids,
                "encoder_max_abs_diff": encoder_diff,
                "init_logits_max_abs_diff": logits_diff,
                "step_logits_max_abs_diff": step_diff,
                "cache_max_abs_diff": max(cache_diffs.values()),
                "argmax_token_agreement": bool(
                    np.array_equal(np.argmax(logits_torch.numpy(), -1), np.argmax(logits_onnx, -1))
                ),
                "passed": case_worst <= ATOL,
            }
        )

    ids = tokenizer.encode(texts[0][0], lang=texts[0][1], add_eos=True)
    torch_ids = greedy_decode_batch(
        model,
        [ids],
        max_new_tokens=12,
        eos_id=tokenizer.eos_id,
        pad_id=tokenizer.pad_id,
        device=torch.device("cpu"),
    )[0].ids
    onnx_ids = _onnx_greedy(encoder, decoder_init, decoder_step, ids, tokenizer, max_new_tokens=12)
    report["generation_parity"] = {
        "python_ids": torch_ids,
        "onnx_ids": onnx_ids,
        "identical": torch_ids == onnx_ids,
    }
    report["worst_max_abs_diff"] = worst
    report["passed"] = bool(worst <= ATOL and report["generation_parity"]["identical"])
    return report


def _onnx_greedy(encoder, decoder_init, decoder_step, ids, tokenizer: ByteBPETokenizer, *, max_new_tokens: int) -> list[int]:
    src_ids = np.asarray([ids], dtype=np.int64)
    src_mask = np.zeros_like(src_ids, dtype=bool)
    src_pos = np.arange(src_ids.shape[1], dtype=np.int64)[None, :]
    memory = encoder.run(
        None, {"src_ids": src_ids, "src_pad_mask": src_mask, "position_ids": src_pos}
    )[0]

    prefix = np.asarray([[BOS_ID]], dtype=np.int64)
    prefix_pos = np.zeros((1, 1), dtype=np.int64)
    self_mask = np.zeros((1, 1, 1, 1), dtype=np.float32)
    cross_mask = np.zeros((1, 1, 1, src_ids.shape[1]), dtype=np.float32)
    outputs = decoder_init.run(
        None,
        {
            "tgt_ids": prefix,
            "memory": memory,
            "self_mask": self_mask,
            "cross_mask": cross_mask,
            "position_ids": prefix_pos,
        },
    )
    logits, caches = outputs[0], list(outputs[1:])
    past_names = [item.name for item in decoder_step.get_inputs()[2:]]
    suppressed = {
        tokenizer.pad_id,
        tokenizer.bos_id,
        tokenizer.unk_id,
        tokenizer.lang_id("en"),
        tokenizer.lang_id("fr"),
    }
    generated: list[int] = []
    for step in range(max_new_tokens):
        row = logits[0, -1].copy()
        for token in suppressed:
            row[token] = -np.inf
        next_id = int(np.argmax(row))
        if next_id == tokenizer.eos_id:
            break
        generated.append(next_id)
        step_inputs = {
            "tgt_ids": np.asarray([[next_id]], dtype=np.int64),
            "position_ids": np.asarray([[step + 1]], dtype=np.int64),
        }
        step_inputs.update({name: value for name, value in zip(past_names, caches)})
        outputs = decoder_step.run(None, step_inputs)
        logits, caches = outputs[0], list(outputs[1:])
    return generated


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Check Python / ONNX Runtime parity")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    report = run_parity(args.model_dir)
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
