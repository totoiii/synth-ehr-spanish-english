#!/usr/bin/env python3
"""
Generate clinical notes for ALL examples in CARES dataset using fine-tuned GPT-OSS 20B
"""

import os
import json

# CRITICAL: Disable all torch compile/inductor/Unsloth fast inference BEFORE importing torch
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["PYTORCH_JIT_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_FAST_INFERENCE"] = "1"  # ensure no fast patches

import torch
import torch._dynamo

# Completely disable dynamo/compile
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 0
try:
    torch.compiler.disable()
except Exception:
    pass

from datetime import datetime
from pathlib import Path
from datasets import load_dataset
from unsloth import FastLanguageModel
from tqdm import tqdm
import argparse
import pandas as pd

# Local import
import sys
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))
try:
    from utils import build_cwlc_dataset_from_zip, build_radgraph_dataset_from_jsonl
except Exception:
    build_cwlc_dataset_from_zip = None
    build_radgraph_dataset_from_jsonl = None


def _as_bool_flag(value: str) -> bool:
    """Parse common CLI boolean values."""
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--output-file", type=str, default=None, help="Output JSONL file")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--reasoning-effort", type=str, default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--dataset", type=str, default="chizhikchi/CARES", help="HF dataset: 'chizhikchi/CARES'")
    parser.add_argument("--split", type=str, default=None, help="Split to use (default: 'train' if available)")
    parser.add_argument("--cwlc-zip", type=str, default=None, help="Path to CWLC ZIP (overrides --dataset)")
    parser.add_argument("--radgraph-jsonl", type=str, nargs="+", default=None, help="Path(s) to RadGraph-XL JSONL files (overrides --dataset)")
    parser.add_argument(
        "--use-vllm",
        type=_as_bool_flag,
        default=True,
        help="Use vLLM for inference (True/False). Default: True",
    )
    parser.add_argument(
        "--vllm-tensor-parallel",
        type=int,
        default=None,
        help="vLLM tensor_parallel_size. Default: number of visible CUDA devices (or 1).",
    )
    parser.add_argument(
        "--vllm-gpu-memory-util",
        type=float,
        default=0.85,
        help="vLLM GPU memory utilization (0-1). Default: 0.85",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature. Default: 0.7 (used for both paths)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling. Default: 0.9 (used for both paths)",
    )
    args = parser.parse_args()

    # Set output file
    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_file = f"results/generated_gpt_oss_{timestamp}.jsonl"

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Generating Clinical Notes with Fine-tuned GPT-OSS 20B")
    print("=" * 70)

    # Load model
    llm = None
    tokenizer = None
    model = None
    if args.use_vllm:
        print(f"\n1. Loading model with vLLM from: {args.model_path}")
        try:
            from vllm import LLM, SamplingParams
        except Exception as e:
            raise RuntimeError(
                "vLLM is not available. Install it first (often requires Linux/WSL2 + CUDA). "
                "Re-run with --use-vllm False to use Unsloth instead."
            ) from e

        # Best-effort default for tensor parallel
        tp = args.vllm_tensor_parallel
        if tp is None:
            try:
                tp = torch.cuda.device_count() if torch.cuda.is_available() else 1
            except Exception:
                tp = 1
        tp = max(int(tp), 1)

        # vLLM expects a HF model repo or local path.
        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=tp,
            gpu_memory_utilization=float(args.vllm_gpu_memory_util),
        )
        sampling_params = SamplingParams(
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            max_tokens=int(args.max_new_tokens),
        )
        print(f"✓ vLLM loaded (tensor_parallel_size={tp})")
    else:
        print(f"\n1. Loading fine-tuned model with Unsloth from: {args.model_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_path,
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True,
        )
        # Avoid any Unsloth for_inference() patches which may trigger dynamo
        model.eval()
        if torch.cuda.is_available():
            model = model.to("cuda")
        print("✓ Model loaded (compile/dynamo disabled)")

    # Load dataset
    if args.radgraph_jsonl:
        if build_radgraph_dataset_from_jsonl is None:
            raise RuntimeError("utils.build_radgraph_dataset_from_jsonl is not available")
        print(f"\n2. Building RadGraph dataset from JSONL: {args.radgraph_jsonl}")
        ds_split = build_radgraph_dataset_from_jsonl(args.radgraph_jsonl)
        mode = "radgraph"
        print(f"✓ RadGraph docs: {len(ds_split)}")
    elif args.cwlc_zip:
        if build_cwlc_dataset_from_zip is None:
            raise RuntimeError("utils.build_cwlc_dataset_from_zip is not available")
        print(f"\n2. Building dataset from ZIP: {args.cwlc_zip}")
        ds_split = build_cwlc_dataset_from_zip(args.cwlc_zip)
        mode = "cwlc"
        print(f"✓ CWLC docs: {len(ds_split)}")
    else:
        print(f"\n2. Loading dataset: {args.dataset} ...")
        dataset = load_dataset(args.dataset)
        available_splits = list(dataset.keys())
        split_name = args.split or ("train" if "train" in dataset else available_splits[0])
        ds_split = dataset[split_name]
        print(f"✓ Available splits: {available_splits}")
        print(f"✓ Using split: {split_name} | rows: {len(ds_split)}")
        dataset_lower = args.dataset.lower()
        mode = "cares" if "cares" in dataset_lower else ("smc" if "smc" in dataset_lower else "cwlc")

    # Prepare iterable
    if mode == "cares":
        examples_iter = ds_split
        print("✓ Mode: CARES (ICD-10 -> clinical note)")
    elif mode == "radgraph":
        print("✓ Mode: RadGraph-XL (NER entities -> radiology report)")
        examples_iter = ds_split
    else:
        print("✓ Mode: CWLC (entities -> clinical note)")
        examples_iter = ds_split

    total_examples = len(examples_iter) if not isinstance(examples_iter, (list, tuple)) else len(examples_iter)
    print(f"\n3. Generating {total_examples} clinical notes...")
    print(f"   Output: {args.output_file}")
    print(f"   Use vLLM: {args.use_vllm}")
    print(f"   Reasoning effort: {args.reasoning_effort}")
    print(f"   Max tokens: {args.max_new_tokens}")
    print(f"   temperature={args.temperature} top_p={args.top_p}")

    def build_messages(example):
        if mode == "cares":
            icd10_codes = example.get("icd10", [])
            
            # INCLUDE DESCRIPTION IN SPANISH

            with open(Path(__file__).parent.parent / "data" / "cares_icd10_repetitions.txt", "r") as f:
                lines = f.readlines()
                codes_to_remove = set()
                for line in lines:
                    parts = line.split(":")
                    if len(parts) == 2:
                        code = parts[0].strip()
                        count = int(parts[1].strip())
                        if count <= 10:
                            codes_to_remove.add(code)

            icd10_codes = [code for code in icd10_codes if code not in codes_to_remove]
            # Incluye la descripcion de los codigos ICD-10 que estan en /home/lmiranda/ehr-synthetic-bmc-2026/data/cares_icd10_descriptions_es.json
            # el formato de este json es
            # {
            #   "E11.9": "Diabetes mellitus tipo 2 sin complicaciones",
            #   "I10": "Hipertensión esencial (primaria)",
            #   ...
            # }
            with open(Path(__file__).parent.parent / "data" / "cares_icd10_descriptions_es.json", "r") as f:
                icd10_descriptions = json.load(f)
            icd10_codes_with_desc = []
            for code in icd10_codes:
                description = icd10_descriptions.get(code, "Descripción no disponible")
                icd10_codes_with_desc.append(f"{code} ({description})")
            icd10_codes = icd10_codes_with_desc

            icd10_str = ", ".join(icd10_codes) if isinstance(icd10_codes, list) else str(icd10_codes)

            return [
                {
                    "role": "developer",
                    "content": "reasoning language: Spanish\n\nYou are a medical assistant that generates clinical notes from ICD-10 diagnosis codes.",
                },
                {"role": "user", "content": f"Generate a detailed clinical note for the following ICD-10 codes: {icd10_str}"},
            ]

        if mode == "radgraph":
            # RadGraph-XL: NER entities -> radiology report
            ner_entities = example.get("ner_entities", []) or []
            pairs = []
            for entity in ner_entities:
                if isinstance(entity, dict):
                    label = entity.get("label", "Unknown")
                    entity_text = entity.get("text", "")
                    if entity_text:
                        pairs.append(f"({label}, {entity_text})")
            pairs_str = ", ".join(pairs)
            return [
                {
                    "role": "developer",
                    "content": (
                        "reasoning language: English\n\n"
                        "You are a radiology expert. Generate a detailed radiology report from the given medical entities. "
                        "The entities include anatomical structures and clinical observations with their presence status."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Generate a complete radiology report based on the following medical entities: [{pairs_str}]",
                },
            ]

        # CWLC
        tags = example.get("tags", []) or []
        pairs = []
        for item in tags:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                pairs.append(f"({item[0]}, {item[1]})")
        pairs_str = ", ".join(pairs)
        return [
            {
                "role": "user",
                "content": f"A partir de las siguiente Entidades Médicas [{pairs_str}] debes redactar una nota clínica que las contenga",
            },
        ]

    def messages_to_prompt(messages):
        """Convert chat messages to a plain prompt for vLLM.

        vLLM supports chat templates via tokenizer in newer versions, but to keep this script
        self-contained we create a stable textual prompt.
        """
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"[{role.upper()}]\n{content}")
        parts.append("[ASSISTANT]\n")
        return "\n\n".join(parts)

    with open(args.output_file, "w", encoding="utf-8") as f:
        if args.use_vllm:
            # Generate one-by-one for streaming-like safety and easy per-row error handling.
            for idx, example in enumerate(tqdm(examples_iter, desc="Generating")):
                messages = build_messages(example)
                prompt = messages_to_prompt(messages)
                try:
                    outs = llm.generate([prompt], sampling_params)  # type: ignore[name-defined]
                    # single prompt => outs[0]
                    generated_note = "".join([o.text for o in outs[0].outputs]).strip() if outs else ""
                except Exception as e:
                    print(f"\n❌ vLLM error generating example {idx}: {e}")
                    generated_note = f"ERROR: {str(e)}"

                if mode == "cares":
                    result = {
                        "idx": idx,
                        "mode": mode,
                        "icd10": example.get("icd10", []),
                        "original_text": example.get("full_text", ""),
                        "generated_text": generated_note,
                        "reasoning_effort": args.reasoning_effort,
                    }
                elif mode == "radgraph":
                    result = {
                        "idx": idx,
                        "mode": mode,
                        "doc_id": example.get("doc_id", None),
                        "dataset_source": example.get("dataset_source", None),
                        "ner_entities": example.get("ner_entities", []),
                        "original_text": example.get("text", ""),
                        "generated_text": generated_note,
                        "reasoning_effort": args.reasoning_effort,
                    }
                else:
                    result = {
                        "idx": idx,
                        "mode": mode,
                        "doc_id": example.get("doc_id", None),
                        "tags": example.get("tags", []),
                        "original_text": example.get("text", ""),
                        "generated_text": generated_note,
                        "reasoning_effort": args.reasoning_effort,
                    }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
        else:
            for idx, example in enumerate(tqdm(examples_iter, desc="Generating")):
                messages = build_messages(example)

                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    reasoning_effort=args.reasoning_effort,
                )
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}

                with torch.no_grad():
                    try:
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            temperature=float(args.temperature),
                            top_p=float(args.top_p),
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            use_cache=True,
                        )
                    except Exception as e:
                        print(f"\n❌ Error generating example {idx}: {e}")
                        result = {
                            "idx": idx,
                            "mode": mode,
                            "generated_text": f"ERROR: {str(e)}",
                            "reasoning_effort": args.reasoning_effort,
                        }
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f.flush()
                        continue

                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if "<|start|>assistant<|message|>" in generated_text:
                    generated_note = generated_text.split("<|start|>assistant<|message|>")[-1]
                    if "<|end|>" in generated_note:
                        generated_note = generated_note.split("<|end|>")[0]
                    elif "<|return|>" in generated_note:
                        generated_note = generated_note.split("<|return|>")[0]
                else:
                    generated_note = generated_text

                if mode == "cares":
                    result = {
                        "idx": idx,
                        "mode": mode,
                        "icd10": example.get("icd10", []),
                        "original_text": example.get("full_text", ""),
                        "generated_text": generated_note.strip(),
                        "reasoning_effort": args.reasoning_effort,
                    }
                elif mode == "radgraph":
                    result = {
                        "idx": idx,
                        "mode": mode,
                        "doc_id": example.get("doc_id", None),
                        "dataset_source": example.get("dataset_source", None),
                        "ner_entities": example.get("ner_entities", []),
                        "original_text": example.get("text", ""),
                        "generated_text": generated_note.strip(),
                        "reasoning_effort": args.reasoning_effort,
                    }
                else:
                    result = {
                        "idx": idx,
                        "mode": mode,
                        "doc_id": example.get("doc_id", None),
                        "tags": example.get("tags", []),
                        "original_text": example.get("text", ""),
                        "generated_text": generated_note.strip(),
                        "reasoning_effort": args.reasoning_effort,
                    }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()

    print(f"\n✓ Generation completed!")
    print(f"✓ Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
