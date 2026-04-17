#!/usr/bin/env python3
"""
Generate synthetic clinical notes using vLLM for fast batch inference.
Supports 4 datasets: MIMIC, CARES, CWLC, RADGRAPH.

Usage examples:

  # MIMIC (Alpaca format JSON)
  python src/generate_all_vllm.py --dataset mimic \
      --model-path output/gpt-oss-20b-mimic-merged \
      --mimic-json data/mimic/mimic_s.json \
      --max-samples 10

  # CARES (HuggingFace dataset)
  python src/generate_all_vllm.py --dataset cares \
      --model-path output/gpt-oss-20b-cares-merged

  # CWLC (ZIP with .txt/.ann files)
  python src/generate_all_vllm.py --dataset cwlc \
      --model-path output/gpt-oss-20b-cwlc-merged \
      --cwlc-zip data/cwlc.zip

  # RADGRAPH (JSONL files)
  python src/generate_all_vllm.py --dataset radgraph \
      --model-path output/gpt-oss-20b-radgraph-merged \
      --radgraph-jsonl data/RADGRAPH/file1.jsonl data/RADGRAPH/file2.jsonl
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("ERROR: vLLM not installed.")
    print("Activate the vllm venv: source .venv-vllm/bin/activate")
    sys.exit(1)

# Local imports for CWLC and RADGRAPH loaders
sys.path.append(str(Path(__file__).parent))
try:
    from utils import build_cwlc_dataset_from_zip, build_radgraph_dataset_from_jsonl
except ImportError:
    build_cwlc_dataset_from_zip = None
    build_radgraph_dataset_from_jsonl = None

# =============================================================================
# Prompt formatting — uses the model's tokenizer chat template
# =============================================================================
# All loaders return (records, messages_list) where each element in
# messages_list is a list of message dicts [{"role": ..., "content": ...}].
# After the model is loaded, we use tokenizer.apply_chat_template() to
# convert them to the exact same format seen during training.


# =============================================================================
# Dataset loaders — each returns (records: list[dict], prompts: list[str])
# =============================================================================

def load_mimic(args):
    """Load MIMIC Alpaca-format JSON — returns messages matching training format."""
    if not args.mimic_json:
        print("ERROR: --mimic-json is required for MIMIC dataset")
        sys.exit(1)

    with open(args.mimic_json, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    print(f"   Total records in file: {len(all_data)}")
    sample = all_data[0]
    if not {"instruction", "input"}.issubset(sample.keys()):
        print(f"   ERROR: Expected keys 'instruction','input', got {list(sample.keys())}")
        sys.exit(1)
    print(f"   Keys: {list(sample.keys())}")

    all_messages = []
    for rec in all_data:
        instruction = rec["instruction"]
        input_text = rec.get("input", "")
        if input_text:
            user_content = f"{instruction}\n\nInput: {input_text}"
        else:
            user_content = instruction
        # In the MIMIC alpaca files the reference clinical note lives in the
        # `output` field. The rest of the pipeline (fidelity/privacy eval,
        # materialise_runk, multirun_eval) all read `original_text`, so
        # mirror `output` into `original_text` right here to keep every
        # downstream consumer uniform.
        if "original_text" not in rec:
            rec["original_text"] = rec.get("output", "") or ""
        messages = [
            {
                "role": "developer",
                "content": (
                    "reasoning language: English\n\n"
                    "You are a medical expert that generates clinical discharge summaries "
                    "from procedure and diagnosis code descriptions."
                ),
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        all_messages.append(messages)
    return all_data, all_messages


def load_cares(args):
    """Load CARES from HuggingFace."""
    from datasets import load_dataset

    hf_name = args.hf_dataset or "chizhikchi/CARES"
    print(f"   Loading HF dataset: {hf_name}")
    dataset = load_dataset(hf_name)
    split_name = args.split or ("train" if "train" in dataset else list(dataset.keys())[0])
    ds = dataset[split_name]
    print(f"   Split: {split_name} | rows: {len(ds)}")

    # Load ICD-10 descriptions and repetition filter
    data_dir = Path(__file__).parent.parent / "data"

    codes_to_remove = set()
    #rep_file = data_dir / "cares_icd10_repetitions.txt"
    #if rep_file.exists():
    #    with open(rep_file, "r") as f:
    #        for line in f:
    #            parts = line.split(":")
    #            if len(parts) == 2:
    #                code = parts[0].strip()
    #                count = int(parts[1].strip())
    #                if count <= 10:
    #                    codes_to_remove.add(code)

    icd10_descriptions = {}
    desc_file = data_dir / "cares_icd10_descriptions_es.json"
    if desc_file.exists():
        with open(desc_file, "r") as f:
            icd10_descriptions = json.load(f)

    records = []
    all_messages = []
    for i, example in enumerate(ds):
        icd10_codes = example.get("icd10", [])
        icd10_codes = [c for c in icd10_codes if c not in codes_to_remove]

        icd10_with_desc = []
        for code in icd10_codes:
            desc = icd10_descriptions.get(code, "Descripción no disponible")
            icd10_with_desc.append(f"{code} ({desc})")

        icd10_str = ", ".join(icd10_with_desc) if icd10_with_desc else ", ".join(icd10_codes)

        messages = [
            {
                "role": "developer",
                "content": "reasoning language: Spanish\n\nYou are a medical assistant that generates clinical notes from ICD-10 diagnosis codes.",
            },
            {
                "role": "user",
                "content": f"Generate a detailed clinical note for the following ICD-10 codes: {icd10_str}",
            },
        ]

        records.append({
            "id": i,
            "icd10": example.get("icd10", []),
            "original_text": example.get("full_text", ""),
        })
        all_messages.append(messages)

    return records, all_messages


def load_cwlc(args):
    """Load CWLC from ZIP."""
    if build_cwlc_dataset_from_zip is None:
        print("ERROR: utils.build_cwlc_dataset_from_zip not available")
        sys.exit(1)
    if not args.cwlc_zip:
        print("ERROR: --cwlc-zip is required for CWLC dataset")
        sys.exit(1)

    ds = build_cwlc_dataset_from_zip(args.cwlc_zip)
    print(f"   CWLC docs loaded: {len(ds)}")

    records = []
    all_messages = []
    for i, example in enumerate(ds):
        tags = example.get("tags", []) or []
        pairs = []
        for item in tags:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                pairs.append(f"({item[0]}, {item[1]})")
        pairs_str = ", ".join(pairs)

        messages = [
            {
                "role": "user",
                "content": f"A partir de las siguiente Entidades Médicas [{pairs_str}] debes redactar una nota clínica que las contenga",
            },
        ]

        records.append({
            "id": example.get("doc_id", i),
            "tags": tags,
            "original_text": example.get("text", ""),
        })
        all_messages.append(messages)

    return records, all_messages


def load_radgraph(args):
    """Load RadGraph from JSONL."""
    if build_radgraph_dataset_from_jsonl is None:
        print("ERROR: utils.build_radgraph_dataset_from_jsonl not available")
        sys.exit(1)
    if not args.radgraph_jsonl:
        print("ERROR: --radgraph-jsonl is required for RADGRAPH dataset")
        sys.exit(1)

    ds = build_radgraph_dataset_from_jsonl(args.radgraph_jsonl)
    print(f"   RadGraph docs loaded: {len(ds)}")

    records = []
    all_messages = []
    for i, example in enumerate(ds):
        ner_entities = example.get("ner_entities", []) or []
        pairs = []
        for entity in ner_entities:
            if isinstance(entity, dict):
                label = entity.get("label", "Unknown")
                text = entity.get("text", "")
                if text:
                    pairs.append(f"({label}, {text})")
        pairs_str = ", ".join(pairs)

        messages = [
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

        records.append({
            "id": example.get("doc_id", i),
            "dataset_source": example.get("dataset_source", ""),
            "ner_entities": ner_entities,
            "original_text": example.get("text", ""),
        })
        all_messages.append(messages)

    return records, all_messages


DATASET_LOADERS = {
    "mimic": load_mimic,
    "cares": load_cares,
    "cwlc": load_cwlc,
    "radgraph": load_radgraph,
}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic clinical notes with vLLM (multi-dataset)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset selection
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["mimic", "cares", "cwlc", "radgraph"],
                        help="Dataset to generate from")

    # Required
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to fine-tuned merged model")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Output JSON file path (auto-generated if not set)")

    # Dataset-specific inputs
    parser.add_argument("--mimic-json", type=str, default=None,
                        help="Path to MIMIC JSON file (Alpaca format)")
    parser.add_argument("--cwlc-zip", type=str, default=None,
                        help="Path to CWLC ZIP file")
    parser.add_argument("--radgraph-jsonl", type=str, nargs="+", default=None,
                        help="Path(s) to RadGraph-XL JSONL files")
    parser.add_argument("--hf-dataset", type=str, default=None,
                        help="HuggingFace dataset name (for CARES, default: chizhikchi/CARES)")
    parser.add_argument("--split", type=str, default=None,
                        help="HF dataset split (default: train)")

    # vLLM engine
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--pipeline-parallel-size", type=int, default=1,
                        help="Number of GPUs for pipeline parallelism")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                        help="Fraction of GPU memory to use")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Maximum model context length")
    parser.add_argument("--max-num-seqs", type=int, default=45,
                        help="Maximum concurrent sequences in batch")

    # Sampling
    parser.add_argument("--n", type=int, default=5,
                        help="Number of outputs to generate per prompt")
    parser.add_argument("--max-tokens", type=int, default=6000,
                        help="Maximum tokens to generate per output")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Top-p nucleus sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.5,
                        help="Repetition penalty")

    # Other
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples (for testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file (skip already done)")
    parser.add_argument("--save-every-n", type=int, default=500,
                        help="Save results to disk every N prompts (incremental saving)")

    args = parser.parse_args()

    # Auto-generate output file if not provided
    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).parent.parent / "output" / args.dataset.upper()
        args.output_file = str(out_dir / f"synth_{args.dataset}_{timestamp}.json")

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Config summary
    # =========================================================================
    print("=" * 70)
    print(f"vLLM Generation — {args.dataset.upper()}")
    print("=" * 70)
    print(f"  Start time:       {datetime.now()}")
    print(f"  Dataset:          {args.dataset}")
    print(f"  Model:            {args.model_path}")
    print(f"  Output:           {args.output_file}")
    print(f"  Tensor parallel:  {args.tensor_parallel_size}")
    print(f"  GPU mem util:     {args.gpu_memory_utilization}")
    print(f"  Max model len:    {args.max_model_len}")
    print(f"  Outputs per prompt (n): {args.n}")
    print(f"  Max tokens:       {args.max_tokens}")
    print(f"  Temperature:      {args.temperature}")
    print(f"  Top-p:            {args.top_p}")
    print(f"  Repetition pen.:  {args.repetition_penalty}")
    print(f"  Max samples:      {args.max_samples or 'all'}")
    print(f"  Save every N:     {args.save_every_n}")
    print("=" * 70)
    print()

    # =========================================================================
    # 1. Load data (returns records + raw messages, NOT formatted prompts)
    # =========================================================================
    print(f"1. Loading {args.dataset.upper()} data...")

    loader = DATASET_LOADERS[args.dataset]
    all_data, all_messages = loader(args)

    print(f"   Total records loaded: {len(all_data)}")

    # Resume support: skip already processed IDs
    done_ids = set()
    existing_docs = []
    if args.resume and Path(args.output_file).exists():
        with open(args.output_file, "r", encoding="utf-8") as f:
            existing_docs = json.load(f)
        done_ids = {doc.get("id") for doc in existing_docs if "id" in doc}
        print(f"   Resuming: {len(done_ids)} already processed")

    # Filter out already done
    if done_ids:
        indices = [i for i, rec in enumerate(all_data) if rec.get("id") not in done_ids]
        data = [all_data[i] for i in indices]
        messages_list = [all_messages[i] for i in indices]
    else:
        data = all_data
        messages_list = all_messages

    if args.max_samples and len(data) > args.max_samples:
        print(f"   Limiting to {args.max_samples} samples")
        data = data[:args.max_samples]
        messages_list = messages_list[:args.max_samples]

    print(f"   Samples to process: {len(data)}")

    if len(data) == 0:
        print("   Nothing to do!")
        return

    # =========================================================================
    # 2. Load model
    # =========================================================================
    print("\n2. Loading model with vLLM...")
    t0 = time.time()

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=True,
        dtype="auto",
    )

    load_time = time.time() - t0
    print(f"   ✓ Model loaded in {load_time:.1f}s")

    # =========================================================================
    # 3. Build prompts using the model's tokenizer chat template
    # =========================================================================
    print("\n3. Formatting prompts with tokenizer chat template...")
    tokenizer = llm.get_tokenizer()
    prompts = []
    for msgs in messages_list:
        prompt = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
    print(f"   Prompts ready: {len(prompts)}")
    print(f"   Example prompt (first 300 chars):\n   {prompts[0][:300]}...")

    # =========================================================================
    # 4. Generate in chunks with incremental saving
    # =========================================================================
    total_prompts = len(prompts)
    chunk_size = args.save_every_n
    num_chunks = (total_prompts + chunk_size - 1) // chunk_size

    print(f"\n4. Generating {total_prompts} × {args.n} = {total_prompts * args.n} total outputs...")
    print(f"   Processing in {num_chunks} chunk(s) of up to {chunk_size} prompts each")
    print(f"   Saving incrementally to: {args.output_file}")

    sampling_params = SamplingParams(
        n=args.n,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    all_docs = list(existing_docs)  # start with existing docs if resuming
    t0_total = time.time()

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total_prompts)
        chunk_prompts = prompts[start:end]
        chunk_data = data[start:end]

        print(f"\n   ── Chunk {chunk_idx + 1}/{num_chunks} "
              f"(prompts {start + 1}-{end} of {total_prompts}) ──")

        t0_chunk = time.time()
        outputs = llm.generate(chunk_prompts, sampling_params)
        chunk_time = time.time() - t0_chunk

        chunk_rate = len(chunk_prompts) / chunk_time if chunk_time > 0 else 0
        print(f"      Generated in {chunk_time:.1f}s ({chunk_rate:.2f} prompts/sec)")

        # Build docs for this chunk
        for i, output in enumerate(outputs):
            doc = dict(chunk_data[i])  # copy all metadata from the record
            doc["prompt"] = output.prompt
            for j in range(args.n):
                if j < len(output.outputs):
                    doc[f"output{j + 1}"] = output.outputs[j].text.strip()
                else:
                    doc[f"output{j + 1}"] = ""
            all_docs.append(doc)

        # Save incrementally after each chunk
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(all_docs, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - t0_total
        done_so_far = end
        remaining = total_prompts - done_so_far
        if done_so_far > 0:
            eta_sec = (elapsed / done_so_far) * remaining
            eta_min = eta_sec / 60
            eta_str = f"{eta_min:.1f}min" if eta_min > 1 else f"{eta_sec:.0f}s"
        else:
            eta_str = "?"

        print(f"      ✓ Saved {len(all_docs)} records to disk "
              f"({done_so_far}/{total_prompts} done, ETA: {eta_str})")

    gen_time = time.time() - t0_total
    samples_per_sec = total_prompts / gen_time if gen_time > 0 else 0
    print(f"\n   ✓ All generation completed in {gen_time:.1f}s ({samples_per_sec:.2f} prompts/sec)")

    # =========================================================================
    # 5. Summary
    # =========================================================================
    total_new = len(all_docs) - len(existing_docs)
    avg_len = 0
    if total_new > 0:
        new_docs = all_docs[-total_new:]
        lengths = [len(d.get("output1", "").split()) for d in new_docs]
        avg_len = sum(lengths) / len(lengths) if lengths else 0

    print("\n" + "=" * 70)
    print(f"Generation Complete — {args.dataset.upper()}")
    print("=" * 70)
    print(f"  End time:          {datetime.now()}")
    print(f"  Model load time:   {load_time:.1f}s")
    print(f"  Generation time:   {gen_time:.1f}s")
    print(f"  Throughput:        {samples_per_sec:.2f} prompts/sec")
    print(f"  Total records:     {len(all_docs)}")
    print(f"  New records:       {total_new}")
    print(f"  Total outputs:     {len(all_docs) * args.n}")
    print(f"  Avg output length: {avg_len:.0f} words (output1)")
    print(f"  Chunks processed:  {num_chunks}")
    print(f"  Output file:       {args.output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
