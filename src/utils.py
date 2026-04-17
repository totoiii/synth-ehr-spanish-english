"""
Utility functions for dataset preparation.
"""
from __future__ import annotations

import os
import json
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from datasets import Dataset, concatenate_datasets


def _parse_ann_file(ann_path: Optional[Path]) -> List[Tuple[str, str]]:
    """Parse a .ann file (BRAT-like) into a list of (label, text) pairs.
    Ignores lines starting with 'A' or 'R'.
    """
    tags: List[Tuple[str, str]] = []
    if ann_path is None or not ann_path.exists():
        return tags

    with open(ann_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("A") or line.startswith("R"):
                continue
            # Expected format (simplified): T1	Label start end	Text
            parts = line.split("\t")
            if len(parts) >= 3 and parts[0].startswith("T"):
                label_info = parts[1].split()
                if len(label_info) >= 1:
                    label = label_info[0]
                    text = parts[2]
                    tags.append((label, text))
    return tags


def build_cwlc_dataset_from_zip(zip_path: str, extract_dir: Optional[str] = None) -> Dataset:
    """Build a Hugging Face Dataset from a ZIP containing .txt and .ann files.

    - zip_path: path to the ZIP file.
    - extract_dir: optional extraction directory. If None, a temp folder is created next to the ZIP.

    Returns a datasets.Dataset with columns: doc_id, text, tags (list of [label, value]).
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP not found: {zip_path}")

    if extract_dir is None:
        extract_dir = zip_path.parent / (zip_path.stem + "_extracted")
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Extract ZIP
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # Find files
    txt_files = list(extract_dir.rglob("*.txt"))
    ann_files = list(extract_dir.rglob("*.ann"))

    # Map doc_id -> {txt, ann}
    doc_dict: Dict[str, Dict[str, Optional[Path]]] = {}

    for txt in txt_files:
        doc_id = txt.stem
        doc_dict.setdefault(doc_id, {"txt": None, "ann": None})
        doc_dict[doc_id]["txt"] = txt

    for ann in ann_files:
        doc_id = ann.stem
        doc_dict.setdefault(doc_id, {"txt": None, "ann": None})
        doc_dict[doc_id]["ann"] = ann

    texts: List[str] = []
    tags_list: List[List[List[str]]] = []  # list of [ [label, text], ... ]
    doc_ids: List[str] = []

    for doc_id, paths in doc_dict.items():
        txt_path = paths.get("txt")
        if txt_path is None:
            continue
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        tags = _parse_ann_file(paths.get("ann"))
        # Convert tuples to lists for JSON/HF compatibility
        tag_pairs: List[List[str]] = [[lbl, val] for (lbl, val) in tags]
        texts.append(text)
        tags_list.append(tag_pairs)
        doc_ids.append(doc_id)

    data_dict: Dict[str, Any] = {
        "doc_id": doc_ids,
        "text": texts,
        "tags": tags_list,
    }

    return Dataset.from_dict(data_dict)


def build_radgraph_dataset_from_jsonl(jsonl_paths: List[str]) -> Dataset:
    """Build a Hugging Face Dataset from one or more RadGraph-XL JSONL files.

    RadGraph-XL JSONL format (each line is a JSON object):
    {
        "dataset": "...",
        "doc_key": 0,
        "sentences": [["token1", "token2", ...]],
        "ner": [[[start, end, "Type::status"], ...]],
        "relations": [...]  # ignored
    }

    This function extracts NER entities as (entity_text, label) pairs.
    The full text is reconstructed from the tokenized sentences.

    Returns a datasets.Dataset with columns:
        - doc_id: str (dataset + "_" + doc_key)
        - text: str (full clinical note reconstructed from tokens)
        - ner_entities: List[Dict] with keys: text, label, start_idx, end_idx
    """
    all_records: List[Dict[str, Any]] = []

    for jsonl_path in jsonl_paths:
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("//"):
                    # Skip empty lines and comments
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON at {path}:{line_num}: {e}")
                    continue

                dataset_name = record.get("dataset", "unknown")
                doc_key = record.get("doc_key", line_num)
                doc_id = f"{dataset_name}_{doc_key}"

                # Reconstruct text from tokenized sentences
                sentences = record.get("sentences", [])
                # Flatten all tokens
                tokens: List[str] = []
                for sent in sentences:
                    if isinstance(sent, list):
                        tokens.extend(sent)

                # Reconstruct text (join with spaces, could be improved with smarter logic)
                full_text = " ".join(tokens)

                # Extract NER entities
                ner_data = record.get("ner", [])
                ner_entities: List[Dict[str, Any]] = []

                # ner is typically a list of lists (one per sentence)
                for sent_ner in ner_data:
                    if not isinstance(sent_ner, list):
                        continue
                    for entity in sent_ner:
                        if not isinstance(entity, list) or len(entity) < 3:
                            continue
                        start_idx, end_idx, label = entity[0], entity[1], entity[2]
                        # Extract entity text from tokens (inclusive indices)
                        if 0 <= start_idx < len(tokens) and 0 <= end_idx < len(tokens):
                            entity_tokens = tokens[start_idx : end_idx + 1]
                            entity_text = " ".join(entity_tokens)
                            ner_entities.append({
                                "text": entity_text,
                                "label": label,
                                "start_idx": start_idx,
                                "end_idx": end_idx,
                            })

                all_records.append({
                    "doc_id": doc_id,
                    "text": full_text,
                    "ner_entities": ner_entities,
                    "dataset_source": dataset_name,
                })

    if not all_records:
        raise ValueError("No valid records found in the provided JSONL files")

    # Convert to HF Dataset
    data_dict: Dict[str, List[Any]] = {
        "doc_id": [r["doc_id"] for r in all_records],
        "text": [r["text"] for r in all_records],
        "ner_entities": [r["ner_entities"] for r in all_records],
        "dataset_source": [r["dataset_source"] for r in all_records],
    }

    return Dataset.from_dict(data_dict)