#!/usr/bin/env python3
"""
Materialise a per-run view of a synth_*.json file.

For error analysis we generate each prompt 5 times (output1..output5). The
existing downstream scripts were written when there was a single generation
per prompt and therefore read `output1` / `generated_text`. Rather than
patch every downstream script to accept an arbitrary `output_k`, we
materialise a copy of the synth file in which `output1` and `generated_text`
are both set to the chosen run's text. This keeps the downstream scripts
untouched and makes it trivial to run them 5× per fraction.

Usage:
    python src/materialize_runk.py \
        --input-file output/CARES/synth_cares_100pct_....json \
        --run-k 3 \
        --output-file output/CARES/runk/synth_cares_100pct_run3.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-file", type=Path, required=True)
    ap.add_argument("--run-k", type=int, required=True)
    ap.add_argument("--output-file", type=Path, required=True)
    args = ap.parse_args()

    if not args.input_file.is_file():
        print(f"ERROR: {args.input_file} not found", file=sys.stderr)
        return 2

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("ERROR: synth JSON must be a list of records", file=sys.stderr)
        return 3

    src = f"output{args.run_k}"
    if data and src not in data[0]:
        available = [k for k in data[0].keys() if k.startswith("output")]
        print(
            f"ERROR: field {src} not present in records "
            f"(available: {available})",
            file=sys.stderr,
        )
        return 4

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    materialised = []
    for rec in data:
        new = dict(rec)
        txt = new.get(src, "") or ""
        new["output1"] = txt
        new["generated_text"] = txt
        materialised.append(new)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(materialised, f, ensure_ascii=False)
    print(f"[materialize] {args.input_file.name} run{args.run_k} → {args.output_file}")
    print(f"[materialize] records={len(materialised)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
