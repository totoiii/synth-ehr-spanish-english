#!/usr/bin/env python3
"""Export Unsloth-style chat rows into an Axolotl-friendly JSONL (prompt/completion).

Axolotl supports multiple dataset formats; the simplest for our pipeline is a JSONL with:
- prompt: string
- completion: string

We convert from our existing SFT chat format:
- either a row is a list of messages
- or a row is a dict: {"messages": [...]}.

Each message is expected to be: {"role": "system"|"user"|"assistant", "content": "..."}

We export:
- prompt = concatenation of all non-assistant messages in order (system+user)
- completion = concatenation of all assistant messages in order

This keeps things model-agnostic and avoids relying on a chat template inside Axolotl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _get_messages(row: Any) -> List[Dict[str, str]]:
    if isinstance(row, list):
        return row
    if isinstance(row, dict) and isinstance(row.get("messages"), list):
        return row["messages"]
    raise ValueError("Row is not a messages list nor a {messages: [...]} dict")


def _to_prompt_completion(messages: List[Dict[str, str]]) -> Tuple[str, str]:
    prompt_parts: List[str] = []
    completion_parts: List[str] = []

    for m in messages:
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "").rstrip()
        if not content:
            continue

        if role == "assistant":
            completion_parts.append(content)
        else:
            # keep role markers minimal but explicit
            if role == "system":
                prompt_parts.append(f"[SYSTEM]\n{content}\n")
            elif role == "user":
                prompt_parts.append(f"[USER]\n{content}\n")
            else:
                prompt_parts.append(f"[{role.upper() or 'MESSAGE'}]\n{content}\n")

    prompt = "\n".join(p.strip() for p in prompt_parts if p.strip()).strip() + "\n\n[ASSISTANT]\n"
    completion = "\n".join(c.strip() for c in completion_parts if c.strip()).strip()

    return prompt, completion


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", required=True, help="Input JSONL in our chat format")
    ap.add_argument("--out-jsonl", required=True, help="Output JSONL with prompt/completion")
    ap.add_argument("--max-rows", type=int, default=0, help="If >0, limit exported rows")
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_out = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            row = json.loads(line)
            messages = _get_messages(row)
            prompt, completion = _to_prompt_completion(messages)
            if not completion:
                continue

            fout.write(json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False) + "\n")
            n_out += 1

            if args.max_rows and n_out >= args.max_rows:
                break

    print(json.dumps({"in_rows": n_in, "out_rows": n_out, "out": str(out_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
