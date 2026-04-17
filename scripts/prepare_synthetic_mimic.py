#!/usr/bin/env python3
"""
Convert synthetic MIMIC JSON arrays into PLM-ICD feather format.

Usage:
    python scripts/prepare_synthetic_mimic.py \
        --synth-json output/MIMIC/synth_mimic_small_5pct_20260305_041044.json \
        --eval-json data/mimic/mimic_m.json \
        --icd-csv data/mimic/ICD10_descriptions_mimic.csv \
        --text-source generated \
        --output-dir /path/to/plm_icd/files/data/synth_mimic \
        --output-tag 5pct_generated

Produces:
    synth_mimic_{tag}.feather       – data file with _id, text, target, icd10_diag, icd10_proc, num_words, num_targets
    synth_mimic_{tag}_split.feather – split assignments (_id, split)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
MIN_TARGET_COUNT = 10  # Same as PLM-ICD prepare_mimiciv.py


# --------------------------------------------------------------------------- #
# ICD-10 code utilities
# --------------------------------------------------------------------------- #
def is_icd10_diag(code: str) -> bool:
    """ICD-10-CM diagnosis codes: letter + 2+ digits (optionally with decimal)."""
    return bool(re.match(r"^[A-TV-Z]\d{2}", code))


def reformat_icd10_diag(code: str) -> str:
    """Add decimal point after first 3 characters for diagnosis codes."""
    code = code.replace(".", "")
    if len(code) > 3:
        return code[:3] + "." + code[3:]
    return code


def classify_and_format_code(code: str) -> tuple[str, str]:
    """Classify a code as 'diag' or 'proc' and reformat it.
    
    Returns (formatted_code, code_type) where code_type is 'diag' or 'proc'.
    """
    code = code.strip()
    if is_icd10_diag(code):
        return reformat_icd10_diag(code), "diag"
    else:
        return code, "proc"


# --------------------------------------------------------------------------- #
# JSON streaming (handles large files)
# --------------------------------------------------------------------------- #
def stream_json_array(file_path: str):
    """Load JSON array using json module (faster than custom streaming)."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for obj in data:
            yield obj


# --------------------------------------------------------------------------- #
# Description → code mapping
# --------------------------------------------------------------------------- #
def load_desc_to_code(csv_path: str) -> dict[str, str]:
    """Load ICD-10 descriptions CSV and return description→code mapping."""
    desc_to_code: dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["icd_version"] == "10":
                desc_to_code[row["long_title"]] = row["icd_code"]
    return desc_to_code


def map_descriptions_to_codes(
    input_text: str, desc_to_code: dict[str, str], sorted_descs: list[str]
) -> list[str]:
    """Map ICD-10 descriptions in input text to codes (fast)."""
    # The input is usually a comma-separated list. We could try to split it.
    # However, since some descriptions have commas, a simple split and check is best.
    # We can use a trie, or just do the naive loop but only for descriptions that exist in the text using .find()
    # Actually, the string is small, so we can split by ", " and try combining chunks if they aren't found.
    # A robust fast alternative is to check if any token in the desc is in the string before checking the whole desc.
    codes = []
    # Fast path: split by ", " and check. If found, continue. If not, it might be a description with a comma, 
    # so we fallback to a simpler approach for the remaining text.
    parts = input_text.split(", ")
    
    # Try exact matches first
    i = 0
    while i < len(parts):
        # Try joining up to 4 parts (max commas in a desc is usually 3-4)
        found = False
        for j in range(min(len(parts), i+4), i, -1):
            candidate = ", ".join(parts[i:j])
            if candidate in desc_to_code:
                codes.append(desc_to_code[candidate])
                i = j
                found = True
                break
        if not found:
            i += 1
            
    return codes


# --------------------------------------------------------------------------- #
# Text preprocessing (adapted from PLM-ICD preprocessing.py)
# --------------------------------------------------------------------------- #
def preprocess_text(text: str) -> str:
    """Apply the same preprocessing as PLM-ICD's preprocessing pipeline."""
    # Remove from discharge condition/instructions onwards
    for keyword in ["discharge condition", "discharge instruction", "discharge instructions"]:
        pos = text.lower().find(keyword)
        if pos != -1:
            text = text[:pos].strip()
            break

    # Remove beginning (admission dates etc) — keep from chief complaint
    for keyword in ["chief complaint", "major surgical or invasive procedure", "history of present illness"]:
        pos = text.lower().find(keyword)
        if pos != -1:
            text = text[pos:]
            break

    # Remove lines with medication headers
    lines = text.splitlines()
    cleaned = []
    skip = False
    for line in lines:
        if not skip:
            if re.search(r"\bmedications?\s*(?:[^:]*:)?\s*$", line, re.IGNORECASE):
                skip = True
            else:
                cleaned.append(line)
        else:
            if line.strip() == "":
                skip = False
    text = "\n".join(cleaned)

    # Remove vitals (lines starting with ___ HH:MM)
    lines = text.splitlines()
    cleaned = []
    skip = False
    for line in lines:
        if not skip:
            if re.match(r"^___\s*\d{2}:\d{2}.*$", line.strip()):
                skip = True
            else:
                cleaned.append(line)
        else:
            if line.strip() == "":
                skip = False
    text = "\n".join(cleaned)

    # Remove lab results (faster line-by-line check)
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        lower_line = line.lower()
        if not ("pertinent result" in lower_line or "admission lab" in lower_line or "discharge lab" in lower_line):
            cleaned.append(line)
    text = "\n".join(cleaned)

    # Remove facility lines and blank ___ lines
    lines = text.split("\n")
    lines = [l for l in lines if not (l.strip().lower() == "facility:" or l.strip() == "___")]
    text = "\n".join(lines)

    # Collapse multiple spaces
    text = re.sub(r" +", " ", text).strip()
    return text


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Prepare synthetic MIMIC data for PLM-ICD")
    parser.add_argument("--synth-json", required=True, help="Path to synthetic JSON file")
    parser.add_argument("--eval-json", required=True, help="Path to eval/test JSON (mimic_m.json)")
    parser.add_argument("--icd-csv", required=True, help="Path to ICD10_descriptions_mimic.csv")
    parser.add_argument("--text-source", default="generated", choices=["original", "generated"],
                        help="Which text field to use for training: 'original' (output) or 'generated' (output_k)")
    parser.add_argument("--text-field", default="output1",
                        help="When --text-source=generated, which synthetic field to read "
                             "(output1, output2, ..., output5). Used for multi-run error analysis.")
    parser.add_argument("--output-dir", required=True, help="Output directory for feather files")
    parser.add_argument("--output-tag", default="default", help="Tag for output filenames")
    parser.add_argument("--min-target-count", type=int, default=MIN_TARGET_COUNT,
                        help="Min occurrences for a code to be kept")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Limit training samples (for smoke tests)")
    parser.add_argument("--max-eval-samples", type=int, default=None,
                        help="Limit eval samples (for smoke tests). "
                             "The eval set (mimic_m.json) has ~20k rows; capping it is the "
                             "single biggest smoke-mode speedup because val+test passes "
                             "dominate wall time.")
    parser.add_argument("--val-fraction", type=float, default=0.5,
                        help="Fraction of eval data used for validation (rest is test)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load description→code mapping ----
    print("Loading ICD-10 descriptions...")
    desc_to_code = load_desc_to_code(args.icd_csv)
    sorted_descs = sorted(desc_to_code.keys(), key=len, reverse=True)
    print(f"  Loaded {len(desc_to_code)} ICD-10 descriptions")

    # ---- 2. Process training data (from synthetic JSON) ----
    print(f"\nProcessing training data from: {args.synth_json}")
    print(f"  Text source: {args.text_source}")
    train_records = []
    n_skipped_no_text = 0
    n_skipped_no_codes = 0
    for i, entry in enumerate(stream_json_array(args.synth_json)):
        # Get text
        if args.text_source == "original":
            text = entry.get("output", "")
        else:  # generated
            text = entry.get(args.text_field, entry.get("output1", ""))
        
        if not text or not isinstance(text, str) or not text.strip():
            n_skipped_no_text += 1
            continue

        # Map descriptions to codes
        input_text = entry.get("input", "")
        codes = map_descriptions_to_codes(input_text, desc_to_code, sorted_descs)
        if not codes:
            n_skipped_no_codes += 1
            continue

        # Classify codes
        diag_codes = []
        proc_codes = []
        all_codes = []
        for code in codes:
            formatted, code_type = classify_and_format_code(code)
            if code_type == "diag":
                diag_codes.append(formatted)
            else:
                proc_codes.append(formatted)
            all_codes.append(formatted)

        # Preprocess text
        clean_text = preprocess_text(text)
        if not clean_text.strip():
            clean_text = text  # fallback to raw text

        train_records.append({
            "_id": entry.get("id", i),
            "text": clean_text,
            "target": all_codes,
            "icd10_diag": diag_codes,
            "icd10_proc": proc_codes,
            "split": "train",
        })

        if args.max_train_samples and len(train_records) >= args.max_train_samples:
            break

    print(f"  Train records: {len(train_records)}")
    print(f"  Skipped (no text): {n_skipped_no_text}")
    print(f"  Skipped (no codes): {n_skipped_no_codes}")

    # ---- 3. Process eval data (from mimic_m.json) ----
    print(f"\nProcessing eval data from: {args.eval_json}")
    eval_records = []
    for i, entry in enumerate(stream_json_array(args.eval_json)):
        text = entry.get("output", "")
        if not text or not isinstance(text, str) or not text.strip():
            continue

        input_text = entry.get("input", "")
        codes = map_descriptions_to_codes(input_text, desc_to_code, sorted_descs)
        if not codes:
            continue

        diag_codes = []
        proc_codes = []
        all_codes = []
        for code in codes:
            formatted, code_type = classify_and_format_code(code)
            if code_type == "diag":
                diag_codes.append(formatted)
            else:
                proc_codes.append(formatted)
            all_codes.append(formatted)

        clean_text = preprocess_text(text)
        if not clean_text.strip():
            clean_text = text

        eval_records.append({
            "_id": entry.get("id", 100_000 + i),
            "text": clean_text,
            "target": all_codes,
            "icd10_diag": diag_codes,
            "icd10_proc": proc_codes,
        })

        if args.max_eval_samples and len(eval_records) >= args.max_eval_samples:
            break

    # Split eval into val/test
    np.random.seed(args.seed)
    indices = np.random.permutation(len(eval_records))
    n_val = int(len(eval_records) * args.val_fraction)
    for idx in indices[:n_val]:
        eval_records[idx]["split"] = "val"
    for idx in indices[n_val:]:
        eval_records[idx]["split"] = "test"

    print(f"  Eval records: {len(eval_records)} (val: {n_val}, test: {len(eval_records) - n_val})")

    # ---- 4. Combine and filter codes ----
    all_records = train_records + eval_records
    df = pd.DataFrame(all_records)

    # Filter rare codes
    print(f"\nFiltering codes with min_target_count={args.min_target_count}...")
    for col in ["icd10_diag", "icd10_proc"]:
        code_counts = Counter(code for codes in df[col] for code in codes)
        n_before = len(code_counts)
        codes_to_keep = {c for c, cnt in code_counts.items() if cnt >= args.min_target_count}
        df[col] = df[col].apply(lambda x: [c for c in x if c in codes_to_keep])
        print(f"  {col}: {n_before} unique → {len(codes_to_keep)} after filtering")

    # Rebuild target from filtered diag + proc
    df["target"] = df["icd10_diag"] + df["icd10_proc"]

    # Remove entries with empty target
    n_before = len(df)
    df = df[df["target"].apply(len) > 0]
    print(f"  Removed {n_before - len(df)} entries with empty target")
    print(f"  Final dataset size: {len(df)}")

    # Add num_words and num_targets
    df["num_words"] = df["text"].str.split().str.len()
    df["num_targets"] = df["target"].apply(len)

    # Print stats
    train_mask = df["split"] == "train"
    val_mask = df["split"] == "val"
    test_mask = df["split"] == "test"
    print(f"\n  Split sizes: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")

    all_target_codes = set(code for codes in df["target"] for code in codes)
    diag_codes_set = set(code for codes in df["icd10_diag"] for code in codes)
    proc_codes_set = set(code for codes in df["icd10_proc"] for code in codes)
    print(f"  Total unique codes: {len(all_target_codes)} (diag: {len(diag_codes_set)}, proc: {len(proc_codes_set)})")
    print(f"  Avg codes/example: {df['num_targets'].mean():.1f}")
    print(f"  Avg words/example: {df['num_words'].mean():.0f}")

    # ---- 5. Save feather files ----
    tag = args.output_tag
    data_path = out_dir / f"synth_mimic_{tag}.feather"
    split_path = out_dir / f"synth_mimic_{tag}_split.feather"

    # Data feather: needs target, icd10_diag, icd10_proc as list columns
    data_df = df[["_id", "text", "target", "icd10_diag", "icd10_proc", "num_words", "num_targets"]].copy()
    data_df = data_df.reset_index(drop=True)
    data_df.to_feather(data_path)
    print(f"\n  Saved data to: {data_path}")

    # Split feather
    split_df = df[["_id", "split"]].copy().reset_index(drop=True)
    split_df.to_feather(split_path)
    print(f"  Saved splits to: {split_path}")

    # ---- 6. Generate data config YAML ----
    config_content = f"""defaults:
  - defaults
dir: files/data/synth_mimic
data_filename: synth_mimic_{tag}.feather
split_filename: synth_mimic_{tag}_split.feather
code_column_names:
  - icd10_diag
  - icd10_proc
max_length: 4000
"""
    config_path = out_dir / f"data_config_{tag}.yaml"
    with open(config_path, "w") as f:
        f.write(config_content)
    print(f"  Generated data config: {config_path}")

    print("\n✅ Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
