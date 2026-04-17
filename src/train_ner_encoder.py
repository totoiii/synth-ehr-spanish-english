#!/usr/bin/env python3
"""Train a RoBERTa encoder for NER from CWLC-style JSONL generations.

Input JSONL format (per line):
{
  "idx": int,
  "doc_id": str,
  "mode": "cwlc",
  "tags": [["Label", "Value"], ...],
  "original_text": str,
  "generated_text": str,
  ...
}

We build token-level BIO labels by matching each tag "Value" as a substring span in the
selected text field. This is weak-supervision alignment (string matching), suitable for
utility experiments.

Two modes:
- --text-field original_text: uses tags as-is
- --text-field generated_text: first filters tags to keep only those whose surface form
  occurs in generated_text (regex), then aligns.

Splits:
- deterministic train/val/test split using a fixed seed.

Model:
- PlanTL-GOB-ES/roberta-base-bne (or user-provided)

Outputs:
- Saves model/tokenizer to output_dir
- Saves split metadata + test metrics JSON

"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from datasets import Dataset, DatasetDict

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    import evaluate

    _HAS_EVALUATE = True
except Exception:
    _HAS_EVALUATE = False


DEFAULT_SEED = 3407


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    # Try reading as a standard JSON list first
    print(f"Reading {path}...")
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                print(f"Loaded as JSON list with {len(data)} records.")
                return data
        except json.JSONDecodeError:
            pass  # Fallback to JSONL

    # Fallback: Read as JSONL
    print("Fallback to JSONL parsing...")
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping invalid line: {e}")
                continue
    return rows


def _normalize_whitespace(s: str) -> str:
    s = s.replace("\t", " ")
    s = re.sub(r"[ ]{2,}", " ", s)
    return s


def _compile_surface_regex(surface: str) -> Optional[re.Pattern]:
    surface = str(surface).strip()
    if not surface or surface == "-":
        return None

    tokens = [re.escape(t) for t in re.split(r"\s+", surface) if t]
    if not tokens:
        return None

    # word boundaries + flexible whitespace
    pattern = r"\b" + r"\s+".join(tokens) + r"\b"
    return re.compile(pattern, flags=re.IGNORECASE | re.MULTILINE)


def filter_tags_present_in_text(tags: List[List[str]], text: str) -> List[List[str]]:
    kept: List[List[str]] = []
    for lbl, val in tags:
        rx = _compile_surface_regex(val)
        if rx is None:
            continue
        if rx.search(text) is not None:
            kept.append([str(lbl), str(val)])
    return kept


def find_all_spans(text: str, surface: str) -> List[Tuple[int, int]]:
    rx = _compile_surface_regex(surface)
    if rx is None:
        return []
    return [(m.start(), m.end()) for m in rx.finditer(text)]


def build_label_list(rows: List[Dict[str, Any]]) -> List[str]:
    labels = set()
    for r in rows:
        for lbl, _ in (r.get("tags") or []):
            if isinstance(lbl, str) and lbl:
                labels.add(lbl)
    labels = sorted(labels)

    out = ["O"]
    for l in labels:
        out.append(f"B-{l}")
        out.append(f"I-{l}")
    return out


def split_rows(
    rows: List[Dict[str, Any]],
    seed: int,
    test_size: float,
    val_size: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_rows, test_rows = train_test_split(rows, test_size=test_size, random_state=seed, shuffle=True)

    if val_size <= 0:
        return train_rows, [], test_rows

    val_frac_of_train = val_size / (1.0 - test_size)
    train_rows, val_rows = train_test_split(train_rows, test_size=val_frac_of_train, random_state=seed, shuffle=True)
    return train_rows, val_rows, test_rows


@dataclass
class NerExample:
    id: str
    text: str
    tags: List[List[str]]


def rows_to_examples(rows: List[Dict[str, Any]], text_field: str, filter_tags_in_text: bool) -> List[NerExample]:
    examples: List[NerExample] = []
    for i, r in enumerate(rows):
        text = _normalize_whitespace(str(r.get(text_field) or ""))

        tags = r.get("tags") or []
        cleaned_tags: List[List[str]] = []
        for t in tags:
            if not isinstance(t, (list, tuple)) or len(t) != 2:
                continue
            lbl, val = t
            cleaned_tags.append([str(lbl), str(val)])

        if filter_tags_in_text:
            cleaned_tags = filter_tags_present_in_text(cleaned_tags, text)

        ex_id = str(r.get("doc_id") or r.get("idx") or i)
        examples.append(NerExample(id=ex_id, text=text, tags=cleaned_tags))
    return examples


def align_tags_to_tokens(
    tokenizer,
    example: NerExample,
    label2id: Dict[str, int],
    max_length: int,
) -> Dict[str, Any]:
    text = example.text

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
    )

    offsets: List[Tuple[int, int]] = encoding["offset_mapping"]
    labels = [label2id["O"]] * len(offsets)

    spans: List[Tuple[int, int, str]] = []
    for lbl, surface in example.tags:
        for s, e in find_all_spans(text, surface):
            spans.append((s, e, lbl))

    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    occupied = np.zeros(len(text), dtype=np.uint8) if len(text) else np.zeros(1, dtype=np.uint8)

    for start, end, lbl in spans:
        if start < 0 or end <= start or start >= len(text):
            continue
        end = min(end, len(text))
        if occupied[start:end].any():
            continue
        occupied[start:end] = 1

        first = True
        for i, (os_, oe_) in enumerate(offsets):
            if os_ == oe_:
                continue
            if oe_ <= start:
                continue
            if os_ >= end:
                break

            tag = f"B-{lbl}" if first else f"I-{lbl}"
            if tag in label2id:
                labels[i] = label2id[tag]
            first = False

    encoding.pop("offset_mapping")
    encoding["labels"] = labels
    encoding["id"] = example.id
    return encoding


def compute_metrics_fn(label_list: List[str]):
    if not _HAS_EVALUATE:
        return None

    seqeval = evaluate.load("seqeval")
    id2label = {i: l for i, l in enumerate(label_list)}

    def compute_metrics(p):
        preds, labs = p
        preds = np.argmax(preds, axis=-1)

        true_predictions = []
        true_labels = []
        for pred, lab in zip(preds, labs):
            cur_preds = []
            cur_labs = []
            for p_i, l_i in zip(pred, lab):
                if l_i == -100:
                    continue
                cur_preds.append(id2label[int(p_i)])
                cur_labs.append(id2label[int(l_i)])
            true_predictions.append(cur_preds)
            true_labels.append(cur_labs)

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        print(results)
        # Build output with overall metrics
        output = {
            "precision": results.get("overall_precision", 0.0),
            "recall": results.get("overall_recall", 0.0),
            "f1": results.get("overall_f1", 0.0),
            "accuracy": results.get("overall_accuracy", 0.0),
        }
        
        # Add per-class metrics (e.g., Abbreviation, Disease, Medication, etc.)
        for key, value in results.items():
            if isinstance(value, dict) and key not in ("overall_precision", "overall_recall", "overall_f1", "overall_accuracy"):
                # key is the entity type (e.g., "Abbreviation", "Disease")
                output[f"{key}_precision"] = value.get("precision", 0.0)
                output[f"{key}_recall"] = value.get("recall", 0.0)
                output[f"{key}_f1"] = value.get("f1", 0.0)
                output[f"{key}_support"] = value.get("number", 0)
        
        return output

    return compute_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train RoBERTa encoder NER from generated CWLC JSONL")

    parser.add_argument("--jsonl", type=str, required=True, help="Path to generated JSONL")
    parser.add_argument(
        "--model-name",
        type=str,
        default="dccuchile/bert-base-spanish-wwm-cased",
        help="HF encoder model name",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="original_text",
        choices=["original_text", "generated_text",
                 "output1", "output2", "output3", "output4", "output5"],
        help="Which text field to use",
    )
    parser.add_argument("--output-dir", type=str, default="./output/ner_roberta_base_bne")

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)

    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=3,
                        help="Stop training if val F1 does not improve for this many epochs")

    parser.add_argument("--eval-only", action="store_true", help="Load from --output-dir and eval on test")

    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(args.jsonl)
    # Normalize rows: if "ner_entities" is present but "tags" is not, convert it.
    for r in rows:
        if isinstance(r, dict):
            if r.get("tags") is None and r.get("ner_entities") is not None:
                # Convert ner_entities to tags
                # ner_entities: [{"label": "...", "text": "...", ...}, ...]
                # tags: [[Label, Value], ...]
                new_tags = []
                for ent in r["ner_entities"]:
                    if isinstance(ent, dict) and "label" in ent and "text" in ent:
                        new_tags.append([ent["label"], ent["text"]])
                r["tags"] = new_tags

    rows = [
        r
        for r in rows
        if isinstance(r, dict)
        and r.get("tags") is not None
        and (r.get("original_text") is not None or r.get("generated_text") is not None or r.get("output1") is not None)
    ]
    if not rows:
        raise ValueError("No valid rows found in JSONL (need tags and original_text/generated_text/output1)")

    train_rows, val_rows, test_rows = split_rows(rows, seed=args.seed, test_size=args.test_size, val_size=args.val_size)

    filter_tags_in_text = args.text_field != "original_text"

    # Train/val use the selected text_field (original or generated)
    train_ex = rows_to_examples(train_rows, text_field=args.text_field, filter_tags_in_text=filter_tags_in_text)
    val_ex = rows_to_examples(val_rows, text_field=args.text_field, filter_tags_in_text=filter_tags_in_text)
    
    # Test ALWAYS uses original_text (no filtering) for fair evaluation
    test_ex = rows_to_examples(test_rows, text_field="original_text", filter_tags_in_text=False)

    label_list = build_label_list(rows)
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def examples_to_dataset(examples: List[NerExample]) -> Dataset:
        return Dataset.from_list([{"id": e.id, "text": e.text, "tags": e.tags} for e in examples])

    ds = DatasetDict(
        {
            "train": examples_to_dataset(train_ex),
            "validation": examples_to_dataset(val_ex),
            "test": examples_to_dataset(test_ex),
        }
    )

    tokenized = ds.map(
        lambda r: align_tags_to_tokens(
            tokenizer,
            NerExample(id=r["id"], text=r["text"], tags=r["tags"]),
            label2id,
            args.max_length,
        ),
        remove_columns=ds["train"].column_names,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    if args.eval_only:
        model = AutoModelForTokenClassification.from_pretrained(
            out_dir,
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id,
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_name,
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id,
        )

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        overwrite_output_dir=True,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=_HAS_EVALUATE,
        metric_for_best_model="f1" if _HAS_EVALUATE else None,
        greater_is_better=True,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
    )

    compute_metrics = compute_metrics_fn(label_list)

    callbacks = []
    if _HAS_EVALUATE:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    split_meta = {
        "seed": args.seed,
        "text_field": args.text_field,
        "filter_tags_in_text": filter_tags_in_text,
        "n_total": len(rows),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_test": len(test_rows),
        "test_size": args.test_size,
        "val_size": args.val_size,
        "label_count": len(label_list),
        "model_name": args.model_name,
        "max_length": args.max_length,
    }
    (out_dir / "split_meta.json").write_text(json.dumps(split_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.eval_only:
        trainer.train()
        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

    metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nTest metrics:")
    for k in sorted(metrics.keys()):
        print(f"  {k}: {metrics[k]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())