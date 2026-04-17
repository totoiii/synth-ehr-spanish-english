#!/usr/bin/env python3
"""
Run fidelity + privacy metrics across the N outputs inside a synth_*.json.

`generate_all_vllm.py` writes a single JSON array where each record has
`original_text` and `output1 .. outputN` (one text per sampled completion).
This script treats each `output_k` field as an independent "run", computes
fidelity and privacy metrics against the shared `original_text` column, and
aggregates the per-run numbers as mean and population-std.

The output is a single JSON report that can be consumed by the driver
(`scripts/run_full_pipeline.sh`) or inspected by hand:

    {
      "input_file": ".../synth_cares_5pct_...json",
      "dataset": "cares",
      "n_runs": 5,
      "num_records": 2253,
      "per_run": { "run1": {...}, "run2": {...}, ... },
      "aggregate": {
          "fidelity": { "avg_tokens_per_document": {"mean":..., "std":...}, ... },
          "privacy":  { "8gram_overlap_percent": {"mean":..., "std":...}, ... }
      }
    }
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Tuple

# Make sibling modules importable when run as a standalone script.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Lazy imports of NLTK are deferred to avoid paying the import cost when
# fidelity is skipped (unlikely, but keeps the smoke test snappy).
import nltk  # type: ignore


def _ensure_nltk_tokenizers() -> None:
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except Exception:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                # Non-fatal: analyze_texts will surface the underlying error.
                pass


# ---------------------------------------------------------------------------
# Fidelity (word-level statistics) — self-contained, no nltk.download side-effects
# ---------------------------------------------------------------------------
def analyze_texts(texts: List[str]) -> Dict[str, float]:
    """Compute length / vocabulary statistics for a list of strings."""
    from nltk.tokenize import sent_tokenize, word_tokenize  # local import

    total_sentences = 0
    total_tokens = 0
    unique_words: set = set()
    doc_count = 0

    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        doc_count += 1
        sentences = sent_tokenize(text)
        total_sentences += len(sentences)
        for sent in sentences:
            toks = word_tokenize(sent)
            total_tokens += len(toks)
            unique_words.update(toks)

    def _safe_div(a: float, b: float) -> float:
        return a / b if b else 0.0

    return {
        "num_documents": doc_count,
        "num_sentences": total_sentences,
        "num_tokens": total_tokens,
        "avg_sentences_per_document": round(_safe_div(total_sentences, doc_count), 4),
        "avg_tokens_per_document": round(_safe_div(total_tokens, doc_count), 4),
        "avg_tokens_per_sentence": round(_safe_div(total_tokens, total_sentences), 4),
        "unique_tokens": len(unique_words),
        "unique_token_ratio": round(_safe_div(len(unique_words), total_tokens), 6),
    }


# ---------------------------------------------------------------------------
# Privacy (n-gram overlap + optional ROUGE-5 recall)
# ---------------------------------------------------------------------------
def _ngrams(text: str, n: int) -> set:
    words = text.strip().lower().split()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def ngram_overlap(
    original_texts: List[str], generated_texts: List[str], n: int
) -> Dict[str, float]:
    orig = set()
    for t in original_texts:
        if isinstance(t, str):
            orig |= _ngrams(t, n)
    gen = set()
    for t in generated_texts:
        if isinstance(t, str):
            gen |= _ngrams(t, n)
    common = orig & gen
    union = orig | gen
    ratio = len(common) / len(union) if union else 0.0
    return {
        f"original_unique_{n}grams": len(orig),
        f"generated_unique_{n}grams": len(gen),
        f"common_{n}grams": len(common),
        f"{n}gram_overlap_ratio": round(ratio, 6),
        f"{n}gram_overlap_percent": round(ratio * 100, 4),
    }


def rouge5_recall(
    original_texts: List[str],
    generated_texts: List[str],
    max_samples: int,
    n_jobs: int,
) -> Dict[str, float]:
    """Mean ROUGE-5 recall of best-matching training doc for each generated doc.

    Kept light: optional dependency, optional sampling, optional parallelism.
    """
    try:
        from rouge_score import rouge_scorer  # type: ignore
    except ImportError:
        return {"error": "rouge-score package not installed"}

    scorer = rouge_scorer.RougeScorer(["rouge5"], use_stemmer=False)
    sampled = generated_texts[:max_samples] if max_samples else generated_texts

    def best_recall(doc: str) -> float:
        best = 0.0
        for real in original_texts:
            if not isinstance(real, str):
                continue
            r = scorer.score(doc, real)["rouge5"].recall
            if r > best:
                best = r
        return best

    recalls: List[float] = []
    if n_jobs > 1:
        try:
            from joblib import Parallel, delayed  # type: ignore
            recalls = list(
                Parallel(n_jobs=n_jobs)(
                    delayed(best_recall)(d) for d in sampled if isinstance(d, str) and d.strip()
                )
            )
        except ImportError:
            n_jobs = 1
    if n_jobs == 1:
        recalls = [best_recall(d) for d in sampled if isinstance(d, str) and d.strip()]

    if not recalls:
        return {"error": "no non-empty generations"}
    return {
        "evaluated": len(recalls),
        "mean_recall": round(sum(recalls) / len(recalls), 6),
        "max_recall": round(max(recalls), 6),
        "min_recall": round(min(recalls), 6),
        "median_recall": round(sorted(recalls)[len(recalls) // 2], 6),
    }


# ---------------------------------------------------------------------------
# Per-run driver
# ---------------------------------------------------------------------------
def load_synth_json(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array at {path}, got {type(data).__name__}")
    return data


def detect_n_runs(records: List[Dict[str, Any]]) -> int:
    if not records:
        return 0
    sample = records[0]
    keys = [k for k in sample.keys() if k.startswith("output") and k[6:].isdigit()]
    return max((int(k[6:]) for k in keys), default=0)


def evaluate_one_run(
    original_texts: List[str],
    generated_texts: List[str],
    *,
    run_rouge: bool,
    rouge_max_samples: int,
    rouge_n_jobs: int,
) -> Dict[str, Any]:
    """Run fidelity + privacy for a single (orig, generated) pair."""
    fidelity_orig = analyze_texts(original_texts)
    fidelity_gen = analyze_texts(generated_texts)

    privacy: Dict[str, Any] = {}
    privacy.update(ngram_overlap(original_texts, generated_texts, n=8))
    privacy.update(ngram_overlap(original_texts, generated_texts, n=5))
    if run_rouge:
        privacy["rouge5_recall"] = rouge5_recall(
            original_texts, generated_texts, rouge_max_samples, rouge_n_jobs
        )

    return {
        "fidelity": {"original": fidelity_orig, "generated": fidelity_gen},
        "privacy": privacy,
    }


# ---------------------------------------------------------------------------
# Aggregation (mean ± std) across the N runs.
# ---------------------------------------------------------------------------
# Metrics we care about for plotting / tables. Listed explicitly so the
# aggregation is stable even when the per-run dict gains new fields.
_FIDELITY_KEYS = (
    "avg_sentences_per_document",
    "avg_tokens_per_document",
    "avg_tokens_per_sentence",
    "unique_tokens",
    "unique_token_ratio",
)
_PRIVACY_KEYS = (
    "8gram_overlap_percent",
    "5gram_overlap_percent",
)


def _mean_std(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    m = mean(values)
    s = pstdev(values) if len(values) > 1 else 0.0
    return {"mean": round(m, 6), "std": round(s, 6), "n": len(values)}


def aggregate_runs(per_run: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    runs = list(per_run.values())
    agg_fid: Dict[str, Dict[str, float]] = {}
    for k in _FIDELITY_KEYS:
        agg_fid[k] = _mean_std([r["fidelity"]["generated"].get(k, 0.0) for r in runs])

    # Real-text fidelity is a single value (same original_text across runs)
    # but we still wrap it in the mean/std shape for schema consistency.
    agg_fid_real: Dict[str, Dict[str, float]] = {}
    if runs:
        real = runs[0]["fidelity"]["original"]
        for k in _FIDELITY_KEYS:
            v = real.get(k, 0.0)
            agg_fid_real[k] = {"mean": round(float(v), 6), "std": 0.0, "n": 1}

    agg_priv: Dict[str, Dict[str, float]] = {}
    for k in _PRIVACY_KEYS:
        agg_priv[k] = _mean_std([r["privacy"].get(k, 0.0) for r in runs])

    # ROUGE-5 (optional)
    rouge_means = [
        r["privacy"].get("rouge5_recall", {}).get("mean_recall")
        for r in runs
        if isinstance(r["privacy"].get("rouge5_recall"), dict)
        and "mean_recall" in r["privacy"]["rouge5_recall"]
    ]
    if rouge_means:
        agg_priv["rouge5_mean_recall"] = _mean_std(rouge_means)

    return {
        "fidelity_generated": agg_fid,
        "fidelity_real": agg_fid_real,
        "privacy": agg_priv,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-run fidelity+privacy evaluator")
    parser.add_argument("--input-file", required=True, type=Path,
                        help="Path to a synth_*.json (list of records with output1..outputN)")
    parser.add_argument("--output-file", type=Path, default=None,
                        help="Where to write the aggregated JSON report")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset tag, e.g. cares/cwlc/radgraph/mimic")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit the number of records evaluated (for smoke tests)")
    parser.add_argument("--run-rouge", action="store_true",
                        help="Also compute ROUGE-5 recall (slow; off by default)")
    parser.add_argument("--rouge-max-samples", type=int, default=200)
    parser.add_argument("--rouge-n-jobs", type=int, default=1)
    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"ERROR: input file not found: {args.input_file}", file=sys.stderr)
        return 2

    print(f"[multirun_eval] Loading {args.input_file}")
    records = load_synth_json(args.input_file)
    if args.max_samples:
        records = records[: args.max_samples]
    n_runs = detect_n_runs(records)
    print(f"[multirun_eval] records={len(records)}  detected n_runs={n_runs}")

    if n_runs == 0:
        print("ERROR: no output1..outputN fields found", file=sys.stderr)
        return 3

    _ensure_nltk_tokenizers()

    originals = [
        (r.get("original_text") or "") for r in records
    ]
    # Keep only records where original is non-empty to ensure a fair comparison.
    keep_mask = [bool(o.strip()) for o in originals]
    originals = [o for o, k in zip(originals, keep_mask) if k]
    if not originals:
        print("ERROR: no usable original_text in the input file", file=sys.stderr)
        return 4

    per_run: Dict[str, Dict[str, Any]] = {}
    per_run_times: Dict[str, float] = {}
    for k in range(1, n_runs + 1):
        col = f"output{k}"
        generated = [r.get(col, "") for r, keep in zip(records, keep_mask) if keep]
        non_empty = sum(1 for t in generated if isinstance(t, str) and t.strip())
        print(f"[multirun_eval] run{k}: {non_empty}/{len(generated)} non-empty generations")
        t0 = time.time()
        per_run[f"run{k}"] = evaluate_one_run(
            originals, generated,
            run_rouge=args.run_rouge,
            rouge_max_samples=args.rouge_max_samples,
            rouge_n_jobs=args.rouge_n_jobs,
        )
        per_run_times[f"run{k}"] = round(time.time() - t0, 2)

    aggregate = aggregate_runs(per_run)

    report = {
        "input_file": str(args.input_file),
        "dataset": args.dataset,
        "n_runs": n_runs,
        "num_records_used": len(originals),
        "per_run_seconds": per_run_times,
        "per_run": per_run,
        "aggregate": aggregate,
    }

    out = args.output_file
    if out is None:
        out = args.input_file.with_name(args.input_file.stem + "_multirun_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[multirun_eval] Saved report → {out}")
    # Pretty-print aggregate headlines.
    print("\nAggregated metrics (mean ± std across runs):")
    for group in ("fidelity_real", "fidelity_generated", "privacy"):
        print(f"  [{group}]")
        for k, v in aggregate[group].items():
            if v.get("mean") is None:
                continue
            print(f"    {k:35s} {v['mean']:.4f} ± {v['std']:.4f}  (n={v['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
