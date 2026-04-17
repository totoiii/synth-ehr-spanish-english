#!/usr/bin/env python3
# filepath: /home/lmiranda/ehr-synthetic-bmc-2026/scripts/run_fidelity.py
"""
Script to run fidelity evaluation metrics on synthetic clinical notes.
Supports CARES, CWLC, and RADGRAPH datasets.

Usage:
    python run_fidelity.py --input_file <path_to_jsonl> [--output_dir <path>]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from nltk.tokenize import word_tokenize, sent_tokenize

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_jsonl(filepath: str) -> list:
    """Load JSONL file and return list of documents."""
    docs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('//'):  # Skip comments
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return docs


def detect_corpus_type(docs: list) -> str:
    """Detect corpus type from the documents."""
    if not docs:
        return "unknown"
    
    first_doc = docs[0]
    mode = first_doc.get("mode", "").lower()
    
    if mode in ["cares", "cwlc", "radgraph"]:
        return mode
    
    # Fallback detection based on fields
    if "icd10" in first_doc:
        return "cares"
    elif "tags" in first_doc:
        return "cwlc"
    elif "ner_entities" in first_doc:
        return "radgraph"
    
    return "unknown"


def analyze_texts(texts: list, name: str) -> dict:
    """Analyze a list of texts and return statistics."""
    total_sentences = 0
    total_tokens = 0
    unique_words = set()
    doc_count = len(texts)

    for text in texts:
        if not text or not isinstance(text, str):
            continue
            
        sentences = sent_tokenize(text)
        sentence_count = len(sentences)
        total_sentences += sentence_count
        
        tokens = [word_tokenize(sentence) for sentence in sentences]
        token_count = sum(len(token_list) for token_list in tokens)
        total_tokens += token_count
        
        words = [word for token_list in tokens for word in token_list]
        unique_words.update(words)

    average_sentence_per_doc = total_sentences / doc_count if doc_count > 0 else 0
    average_token_per_doc = total_tokens / doc_count if doc_count > 0 else 0
    average_token_per_sentence = total_tokens / total_sentences if total_sentences > 0 else 0

    results = {
        "name": name,
        "total_documents": doc_count,
        "average_sentences_per_document": round(average_sentence_per_doc, 2),
        "average_tokens_per_document": round(average_token_per_doc, 2),
        "average_tokens_per_sentence": round(average_token_per_sentence, 2),
        "total_sentences": total_sentences,
        "total_tokens": total_tokens,
        "unique_tokens": len(unique_words),
        "unique_token_ratio": round(len(unique_words) / total_tokens, 4) if total_tokens > 0 else 0
    }
    return results


def compare_results(real_results: dict, synthetic_results: dict) -> dict:
    """Compare statistics between real and synthetic data."""
    comparison = {}
    for key in real_results.keys():
        if key == "name":
            continue
        real_val = real_results[key]
        synth_val = synthetic_results[key]
        
        if isinstance(real_val, (int, float)) and isinstance(synth_val, (int, float)):
            diff = real_val - synth_val
            pct_diff = (diff / real_val * 100) if real_val != 0 else 0
            comparison[key] = {
                "real": real_val,
                "synthetic": synth_val,
                "difference": round(diff, 4),
                "percent_difference": round(pct_diff, 2)
            }
        else:
            comparison[key] = {
                "real": real_val,
                "synthetic": synth_val,
                "difference": "N/A"
            }
    return comparison


def run_fidelity_evaluation(input_file: str, output_dir: str = None):
    """Run fidelity evaluation on a JSONL file."""
    print(f"\n{'='*60}")
    print(f"FIDELITY EVALUATION")
    print(f"{'='*60}")
    print(f"Input file: {input_file}")
    
    # Load data
    docs = load_jsonl(input_file)
    if not docs:
        print("ERROR: No documents found in the input file.")
        return
    
    print(f"Loaded {len(docs)} documents")
    
    # Detect corpus type
    corpus_type = detect_corpus_type(docs)
    print(f"Detected corpus type: {corpus_type.upper()}")
    
    # Extract texts
    original_texts = []
    generated_texts = []
    
    for doc in docs:
        original = doc.get("original_text", "")
        generated = doc.get("generated_text", "")
        
        if original and isinstance(original, str) and original.strip():
            original_texts.append(original)
        if generated and isinstance(generated, str) and generated.strip():
            generated_texts.append(generated)
    
    print(f"Original texts: {len(original_texts)}")
    print(f"Generated texts: {len(generated_texts)}")
    
    if not original_texts or not generated_texts:
        print("ERROR: Missing original or generated texts.")
        return
    
    # Analyze both sets
    print("\nAnalyzing texts...")
    original_stats = analyze_texts(original_texts, "Original")
    generated_stats = analyze_texts(generated_texts, "Generated")
    
    # Compare
    comparison = compare_results(original_stats, generated_stats)
    
    # Print results
    print(f"\n{'-'*60}")
    print("STATISTICAL COMPARISON")
    print(f"{'-'*60}")
    
    print(f"\n{'Metric':<35} {'Original':>12} {'Synthetic':>12} {'Diff %':>10}")
    print("-" * 69)
    
    for key, values in comparison.items():
        real = values['real']
        synth = values['synthetic']
        diff = values.get('percent_difference', 'N/A')
        
        if isinstance(real, float):
            real_str = f"{real:.2f}"
        else:
            real_str = str(real)
            
        if isinstance(synth, float):
            synth_str = f"{synth:.2f}"
        else:
            synth_str = str(synth)
            
        if isinstance(diff, float):
            diff_str = f"{diff:+.1f}%"
        else:
            diff_str = str(diff)
        
        print(f"{key:<35} {real_str:>12} {synth_str:>12} {diff_str:>10}")
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = Path(output_dir) / f"fidelity_results_{corpus_type}_{Path(input_file).stem}.json"
    else:
        output_file = Path(input_file).parent / f"fidelity_results_{Path(input_file).stem}.json"
    
    results = {
        "input_file": str(input_file),
        "corpus_type": corpus_type,
        "original_stats": original_stats,
        "generated_stats": generated_stats,
        "comparison": comparison
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run fidelity evaluation on synthetic clinical notes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_fidelity.py --input_file output/CARES/generated_cares_20260122_020000.jsonl
  python run_fidelity.py --input_file output/CWLC/generated_gpt_oss_20251218_113714.jsonl
  python run_fidelity.py --input_file output/RADGRAPH/generated_radgraph_20260206_140246.jsonl
        """
    )
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True, 
        help="Path to the JSONL file with original and generated texts."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Directory to save results. Defaults to same directory as input file."
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"ERROR: Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Download NLTK data if needed
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        print(f"Warning: Could not download NLTK data: {e}")
    
    run_fidelity_evaluation(args.input_file, args.output_dir)


if __name__ == "__main__":
    main()
