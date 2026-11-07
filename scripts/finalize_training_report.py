"""Rebuild training_report.json for a run that was stopped by hand.

The accelerator pilot was terminated with taskkill after it overran its walltime
limit inside a validation pass, so ``run_training`` never wrote its report. This
script reconstructs the report from files the run did write (history.json,
run_manifest.json, resolved_config.json, checkpoint.pt) and records that origin
explicitly so the document is not mistaken for a normal end-of-run report.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "models" / "accel_pilot"


def main() -> None:
    history = json.loads((RUN / "history.json").read_text(encoding="utf-8"))
    manifest = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
    config = json.loads((RUN / "resolved_config.json").read_text(encoding="utf-8"))
    payload = torch.load(RUN / "best.pt", map_location="cpu", weights_only=False)

    last = history[-1]
    report = {
        "report_origin": (
            "reconstructed from history.json/run_manifest.json/resolved_config.json/checkpoint.pt "
            "after the process was stopped manually; run_training did not write its own report"
        ),
        "run_name": manifest.get("run_name", "accel_pilot"),
        "device": payload.get("device"),
        "precision": payload.get("precision"),
        "global_step": int(payload.get("global_step", 0)),
        "tokens_seen": int(payload.get("tokens_seen", 0)),
        "best_metric": float(payload.get("best_metric", float("nan"))),
        "stop_reason": "manual_stop_after_walltime_overrun",
        "parameter_counts": config.get("model", {}).get("parameter_counts"),
        "model": config.get("model"),
        "data": config.get("data"),
        "train": config.get("train"),
        "eval": config.get("eval"),
        "validation_history": [
            {
                "step": entry["step"],
                "train_loss_per_token": entry.get("train_loss_per_token"),
                "mean_chrfpp": entry["validation"]["mean_chrfpp"],
                "en-fr": entry["validation"]["directions"]["en-fr"]["chrfpp"],
                "fr-en": entry["validation"]["directions"]["fr-en"]["chrfpp"],
            }
            for entry in history
        ],
        "last_validation": last["validation"],
    }
    out = RUN / "training_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(report["validation_history"], indent=2))


if __name__ == "__main__":
    main()
