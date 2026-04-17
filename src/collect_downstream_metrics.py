#!/usr/bin/env python3
"""
Collect downstream-task metrics across the 5-run error-analysis pipeline.

Given a list of `run_k` output directories produced by the downstream
training scripts (NER encoder, ICD-10 multi-label, PLM-ICD), this script
discovers the relevant metrics file inside each directory, extracts the
same headline metric per task, and aggregates mean ± std across runs.

The downstream scripts write metrics in slightly different shapes, so the
collector knows how to read each of them:

* NER encoder (`train_ner_encoder.py`):
    `<dir>/test_metrics.json`    → {"f1": ..., "precision": ..., "recall": ...}

* Multi-label classifier (`train_multilabel_classification.py`):
    `<dir>/final_eval_metrics.json`
    `<dir>/eval_results.json`    → whichever of `f1_sample`, `f1_micro`,
                                   `f1_macro` is present.

* PLM-ICD (`run_plm_icd_mimic.sh` wrapping `main.py`):
    `<dir>/metrics.json` or `<dir>/test_metrics.json`
    Uses `f1_micro` / `f1_macro` when present.

If a directory does not yet contain a recognised metrics file, the run is
reported with `"status": "missing"` and skipped from the aggregate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Metric discovery per task family
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _first_existing(dir_: Path, names: List[str]) -> Optional[Path]:
    for n in names:
        p = dir_ / n
        if p.is_file():
            return p
    return None


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            flat.update(_flatten(v, key))
        elif isinstance(v, (int, float)):
            flat[key] = float(v)
    return flat


def extract_ner_metrics(dir_: Path) -> Optional[Dict[str, float]]:
    """Look for the NER encoder test metrics.

    Also computes `f1_macro_no_uncertain`: the macro F1 across all per-class
    `test_*_f1` keys excluding any class whose name contains "uncertain".
    The RadGraph `Anatomy::uncertain` / `Observation::uncertain` buckets have
    tiny support and drag the overall F1 down, so we report both with and
    without them.
    """
    for name in ("test_metrics.json", "eval_metrics.json", "metrics.json"):
        data = _load_json(dir_ / name)
        if data is None:
            continue
        flat = _flatten(data)
        out = {}
        for key in ("f1", "f1_span", "overall_f1", "eval_f1", "test.f1", "test_f1"):
            if key in flat:
                out["f1"] = flat[key]
                break
        for key in ("precision", "eval_precision", "overall_precision"):
            if key in flat:
                out["precision"] = flat[key]
                break
        for key in ("recall", "eval_recall", "overall_recall"):
            if key in flat:
                out["recall"] = flat[key]
                break

        # Macro F1 over per-class keys, excluding `*uncertain*`. Classes with
        # zero support are also dropped so an absent class doesn't pull the
        # macro towards 0.
        per_class_f1: List[Tuple[str, float, float]] = []
        for k, v in flat.items():
            if not (k.startswith("test_") and k.endswith("_f1")):
                continue
            cls = k[len("test_"):-len("_f1")]
            if cls in ("",):
                continue
            # Skip the overall "f1" key (test_f1 → cls="")
            sup_key = f"test_{cls}_support"
            support = float(flat.get(sup_key, 0.0))
            per_class_f1.append((cls, float(v), support))
        filtered = [
            (cls, f, s) for (cls, f, s) in per_class_f1
            if "uncertain" not in cls.lower() and s > 0
        ]
        if filtered:
            macro = sum(f for _, f, _ in filtered) / len(filtered)
            out["f1_macro_no_uncertain"] = round(macro, 6)
            # Support-weighted average as an alternative headline.
            total_sup = sum(s for _, _, s in filtered)
            if total_sup > 0:
                wavg = sum(f * s for _, f, s in filtered) / total_sup
                out["f1_weighted_no_uncertain"] = round(wavg, 6)

        if out:
            return out
    return None


def extract_icd10_metrics(dir_: Path) -> Optional[Dict[str, float]]:
    for name in (
        "final_metrics.json",
        "final_eval_metrics.json",
        "eval_results.json",
        "metrics.json",
    ):
        data = _load_json(dir_ / name)
        if data is None:
            continue
        flat = _flatten(data)
        out = {}
        # train_multilabel_classification.py writes keys like
        # `eval_f1_sample`, `eval_f1_micro_flat`, `eval_f1_macro_flat`.
        for out_key, candidates in (
            ("f1_sample",
                ("f1_sample", "eval_f1_sample", "sample_f1")),
            ("f1_micro",
                ("f1_micro_flat", "eval_f1_micro_flat",
                 "f1_micro", "eval_f1_micro", "micro_f1")),
            ("f1_macro",
                ("f1_macro_flat", "eval_f1_macro_flat",
                 "f1_macro", "eval_f1_macro", "macro_f1")),
        ):
            for c in candidates:
                if c in flat:
                    out[out_key] = flat[c]
                    break
        if out:
            return out
    return None


def extract_plmicd_metrics(dir_: Path) -> Optional[Dict[str, float]]:
    """Read PLM-ICD test metrics for the latest epoch.

    PLM-ICD's `main.py` writes one file per epoch and split:
        metrics_train_val_epoch_<N>.json
        metrics_val_epoch_<N>.json
        metrics_test_epoch_<N>.json
    Each file has the shape:
        {"<split>": {"all": {...}, "icd10_diag": {...}, "icd10_proc": {...}}}
    We want the headline numbers from the *test* split of the latest epoch.
    """
    test_files = list(dir_.glob("metrics_test_epoch_*.json"))
    if not test_files:
        return None

    def _epoch(p: Path) -> int:
        try:
            return int(p.stem.rsplit("_", 1)[1])
        except Exception:
            return -1

    test_files.sort(key=_epoch)
    data = _load_json(test_files[-1])
    if not data:
        return None

    # Schema: {"test": {"all": {...}, "icd10_diag": {...}, "icd10_proc": {...}}}
    test_block = data.get("test") or {}
    all_block = test_block.get("all") or {}
    if not all_block:
        return None

    out: Dict[str, float] = {}
    for key in (
        "f1_micro",
        "f1_macro",
        "precision_micro",
        "recall_micro",
        "exact_match_ratio",
        "map",
        "precision@1",
        "recall@1",
        "precision@5",
        "recall@5",
        "precision@8",
        "recall@8",
        "precision@15",
        "recall@15",
    ):
        if key in all_block:
            try:
                out[key] = float(all_block[key])
            except (TypeError, ValueError):
                pass
    return out or None


_EXTRACTORS = {
    "ner": extract_ner_metrics,
    "icd10": extract_icd10_metrics,
    "plmicd": extract_plmicd_metrics,
}


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def _mean_std(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    m = mean(values)
    s = pstdev(values) if len(values) > 1 else 0.0
    return {"mean": round(m, 6), "std": round(s, 6), "n": len(values)}


def aggregate(per_run: Dict[str, Optional[Dict[str, float]]]) -> Dict[str, Any]:
    keys: List[str] = []
    for v in per_run.values():
        if isinstance(v, dict):
            for k in v.keys():
                if k not in keys:
                    keys.append(k)
    agg: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = [v[k] for v in per_run.values() if isinstance(v, dict) and k in v]
        agg[k] = _mean_std(vals)
    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate downstream-task metrics across the 5 error-analysis runs",
    )
    parser.add_argument("--task", required=True, choices=["ner", "icd10", "plmicd"],
                        help="Which extractor to use.")
    parser.add_argument("--run-dir", action="append", required=True,
                        help="Repeat: one per run, e.g. --run-dir .../run1 --run-dir .../run2 ...")
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--fraction", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None,
                        help="Optional variant tag (e.g. mimic small/large)")
    args = parser.parse_args()

    extractor = _EXTRACTORS[args.task]
    per_run: Dict[str, Optional[Dict[str, float]]] = {}
    for i, dir_ in enumerate(args.run_dir, start=1):
        p = Path(dir_)
        if not p.is_dir():
            per_run[f"run{i}"] = None
            print(f"[collect] run{i}: {p} (missing)")
            continue
        metrics = extractor(p)
        per_run[f"run{i}"] = metrics
        if metrics is None:
            print(f"[collect] run{i}: {p} (no metrics file found)")
        else:
            pretty = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"[collect] run{i}: {pretty}")

    report = {
        "task": args.task,
        "dataset": args.dataset,
        "fraction": args.fraction,
        "variant": args.variant,
        "per_run": per_run,
        "aggregate": aggregate(per_run),
    }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[collect] Report → {args.output_file}")
    print("\nAggregate (mean ± std):")
    for k, v in report["aggregate"].items():
        if v.get("mean") is None:
            continue
        print(f"  {k:20s} {v['mean']:.4f} ± {v['std']:.4f}  (n={v['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
