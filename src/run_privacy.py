#!/usr/bin/env python3
# filepath: /home/lmiranda/ehr-synthetic-bmc-2026/scripts/run_privacy.py
"""
Script to run privacy evaluation metrics on synthetic clinical notes.
Supports CARES, CWLC, and RADGRAPH datasets.

Metrics:
1. 8-gram overlap: Measures n-gram overlap between original and generated texts
2. ROUGE-5 similarity: Finds most similar training document for each synthetic document

Usage:
    python run_privacy.py --input_file <path_to_jsonl> [--output_dir <path>] [--n_jobs 4]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import namedtuple
from typing import List, Tuple, Set
import warnings

warnings.filterwarnings('ignore')

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


# ============================================================
# 8-GRAM OVERLAP METRICS
# ============================================================

def get_ngrams(text: str, n: int = 8) -> Set[str]:
    """Extract n-grams from text."""
    words = text.strip().lower().split()
    ngrams = set()
    
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i:i + n])
        ngrams.add(ngram)
    
    return ngrams


def calculate_ngram_overlap(original_texts: List[str], generated_texts: List[str], n: int = 8) -> dict:
    """Calculate n-gram overlap between original and generated texts."""
    print(f"\nCalculating {n}-gram overlap...")
    
    # Get all unique n-grams from both sets
    original_ngrams = set()
    for text in original_texts:
        if text and isinstance(text, str):
            original_ngrams.update(get_ngrams(text, n))
    
    generated_ngrams = set()
    for text in generated_texts:
        if text and isinstance(text, str):
            generated_ngrams.update(get_ngrams(text, n))
    
    # Calculate overlap
    common_ngrams = original_ngrams.intersection(generated_ngrams)
    total_ngrams = original_ngrams.union(generated_ngrams)
    
    overlap = len(common_ngrams) / len(total_ngrams) if len(total_ngrams) > 0 else 0
    
    results = {
        f"original_unique_{n}grams": len(original_ngrams),
        f"generated_unique_{n}grams": len(generated_ngrams),
        f"total_unique_{n}grams": len(total_ngrams),
        f"common_{n}grams": len(common_ngrams),
        f"{n}gram_overlap_ratio": round(overlap, 6),
        f"{n}gram_overlap_percent": round(overlap * 100, 4)
    }
    
    return results


# ============================================================
# ROUGE-5 SIMILARITY METRICS
# ============================================================

RougeMatch = namedtuple('RougeMatch', ['query_idx', 'query', 'result', 'precision', 'recall', 'fmeasure'])


def get_best_match_rouge(training_docs: List[str], synth_doc: str, synth_idx: int) -> RougeMatch:
    """Find the best matching training document for a synthetic document using ROUGE-5."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("Warning: rouge-score package not installed. Install with: pip install rouge-score")
        return RougeMatch(synth_idx, synth_doc, "", 0, 0, 0)
    
    scorer = rouge_scorer.RougeScorer(['rouge5'], use_stemmer=False)
    
    highscore = 0.0
    best_match = None
    
    for real_doc in training_docs:
        if not real_doc or not isinstance(real_doc, str):
            continue
        
        score = scorer.score(synth_doc, real_doc)
        precision, recall, fmeasure = score['rouge5']
        
        if recall > highscore:
            highscore = recall
            best_match = RougeMatch(synth_idx, synth_doc, real_doc, precision, recall, fmeasure)
    
    if best_match is None:
        best_match = RougeMatch(synth_idx, synth_doc, "", 0, 0, 0)
    
    return best_match


def calculate_rouge_similarity(original_texts: List[str], generated_texts: List[str], 
                               n_jobs: int = 1, max_samples: int = None) -> Tuple[List[RougeMatch], dict]:
    """Calculate ROUGE-5 similarity between generated and original texts."""
    print(f"\nCalculating ROUGE-5 similarity...")
    
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("ERROR: rouge-score package not installed. Install with: pip install rouge-score")
        return [], {}
    
    # Limit samples if specified
    if max_samples and len(generated_texts) > max_samples:
        print(f"Limiting to {max_samples} samples for ROUGE evaluation...")
        generated_texts = generated_texts[:max_samples]
    
    matches = []
    
    if n_jobs > 1:
        try:
            from joblib import Parallel, delayed
            from tqdm import tqdm
            
            print(f"Running with {n_jobs} parallel jobs...")
            matches = Parallel(n_jobs=n_jobs)(
                delayed(get_best_match_rouge)(original_texts, doc, idx)
                for idx, doc in enumerate(tqdm(generated_texts, desc="ROUGE-5"))
            )
        except ImportError:
            print("joblib/tqdm not available, running sequentially...")
            n_jobs = 1
    
    if n_jobs == 1:
        try:
            from tqdm import tqdm
            iterator = tqdm(enumerate(generated_texts), total=len(generated_texts), desc="ROUGE-5")
        except ImportError:
            iterator = enumerate(generated_texts)
            print(f"Processing {len(generated_texts)} documents...")
        
        for idx, doc in iterator:
            if doc and isinstance(doc, str):
                match = get_best_match_rouge(original_texts, doc, idx)
                matches.append(match)
    
    # Calculate statistics
    if matches:
        recalls = [m.recall for m in matches]
        precisions = [m.precision for m in matches]
        fmeasures = [m.fmeasure for m in matches]
        
        # Sort by recall to get top matches
        sorted_matches = sorted(matches, key=lambda x: x.recall, reverse=True)
        top_n = min(20, len(sorted_matches))
        top_recalls = [m.recall for m in sorted_matches[:top_n]]
        
        stats = {
            "total_evaluated": len(matches),
            "recall": {
                "mean": round(sum(recalls) / len(recalls), 6),
                "max": round(max(recalls), 6),
                "min": round(min(recalls), 6),
                "median": round(sorted(recalls)[len(recalls) // 2], 6)
            },
            "precision": {
                "mean": round(sum(precisions) / len(precisions), 6),
                "max": round(max(precisions), 6),
                "min": round(min(precisions), 6)
            },
            "fmeasure": {
                "mean": round(sum(fmeasures) / len(fmeasures), 6),
                "max": round(max(fmeasures), 6),
                "min": round(min(fmeasures), 6)
            },
            f"top_{top_n}_recall": {
                "mean": round(sum(top_recalls) / len(top_recalls), 6),
                "max": round(max(top_recalls), 6),
                "min": round(min(top_recalls), 6)
            }
        }
    else:
        stats = {"error": "No matches computed"}
    
    return matches, stats


# ============================================================
# LONGEST MATCHING SEQUENCE
# ============================================================

def longest_matching_sequence_words(query: str, result: str) -> List[str]:
    """Find the longest matching sequence of words between query and result."""
    import difflib
    
    query_words = query.lower().split()
    result_words = result.lower().split()
    
    seq_matcher = difflib.SequenceMatcher(None, query_words, result_words)
    match = seq_matcher.find_longest_match(0, len(query_words), 0, len(result_words))
    
    return query_words[match.a:match.a + match.size]


def analyze_longest_matches(matches: List[RougeMatch], top_n: int = 20) -> dict:
    """Analyze longest matching sequences for top matches."""
    print(f"\nAnalyzing longest matching sequences (top {top_n})...")
    
    sorted_matches = sorted(matches, key=lambda x: x.recall, reverse=True)[:top_n]
    
    word_counts = []
    for match in sorted_matches:
        if match.result:
            longest = longest_matching_sequence_words(match.query, match.result)
            word_counts.append(len(longest))
    
    if word_counts:
        stats = {
            "analyzed_count": len(word_counts),
            "avg_longest_match_words": round(sum(word_counts) / len(word_counts), 2),
            "max_longest_match_words": max(word_counts),
            "min_longest_match_words": min(word_counts),
            "word_counts": word_counts
        }
    else:
        stats = {"error": "No matches to analyze"}
    
    return stats


# ============================================================
# MAIN EVALUATION
# ============================================================

def run_privacy_evaluation(input_file: str, output_dir: str = None, 
                          n_jobs: int = 1, max_rouge_samples: int = 500,
                          skip_rouge: bool = False):
    """Run privacy evaluation on a JSONL file."""
    print(f"\n{'='*60}")
    print(f"PRIVACY EVALUATION")
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
    
    results = {
        "input_file": str(input_file),
        "corpus_type": corpus_type,
        "total_original": len(original_texts),
        "total_generated": len(generated_texts)
    }
    
    # ============================================================
    # 1. 8-gram overlap
    # ============================================================
    print(f"\n{'-'*60}")
    print("8-GRAM OVERLAP ANALYSIS")
    print(f"{'-'*60}")
    
    ngram_results = calculate_ngram_overlap(original_texts, generated_texts, n=8)
    results["8gram_overlap"] = ngram_results
    
    print(f"\n8-gram Statistics:")
    print(f"  Original unique 8-grams:  {ngram_results['original_unique_8grams']:,}")
    print(f"  Generated unique 8-grams: {ngram_results['generated_unique_8grams']:,}")
    print(f"  Common 8-grams:           {ngram_results['common_8grams']:,}")
    print(f"  Overlap ratio:            {ngram_results['8gram_overlap_ratio']:.6f}")
    print(f"  Overlap percent:          {ngram_results['8gram_overlap_percent']:.4f}%")
    
    # Also calculate 5-gram for comparison
    ngram_5_results = calculate_ngram_overlap(original_texts, generated_texts, n=5)
    results["5gram_overlap"] = ngram_5_results
    
    print(f"\n5-gram Statistics:")
    print(f"  Original unique 5-grams:  {ngram_5_results['original_unique_5grams']:,}")
    print(f"  Generated unique 5-grams: {ngram_5_results['generated_unique_5grams']:,}")
    print(f"  Common 5-grams:           {ngram_5_results['common_5grams']:,}")
    print(f"  Overlap ratio:            {ngram_5_results['5gram_overlap_ratio']:.6f}")
    print(f"  Overlap percent:          {ngram_5_results['5gram_overlap_percent']:.4f}%")
    
    # ============================================================
    # 2. ROUGE-5 similarity (optional, can be slow)
    # ============================================================
    if not skip_rouge:
        print(f"\n{'-'*60}")
        print("ROUGE-5 SIMILARITY ANALYSIS")
        print(f"{'-'*60}")
        
        try:
            matches, rouge_stats = calculate_rouge_similarity(
                original_texts, generated_texts, 
                n_jobs=n_jobs, max_samples=max_rouge_samples
            )
            
            results["rouge5_similarity"] = rouge_stats
            
            if "recall" in rouge_stats:
                print(f"\nROUGE-5 Recall Statistics:")
                print(f"  Mean recall:   {rouge_stats['recall']['mean']:.6f}")
                print(f"  Max recall:    {rouge_stats['recall']['max']:.6f}")
                print(f"  Min recall:    {rouge_stats['recall']['min']:.6f}")
                print(f"  Median recall: {rouge_stats['recall']['median']:.6f}")
            
            # Longest matching sequence analysis
            if matches:
                longest_match_stats = analyze_longest_matches(matches, top_n=20)
                results["longest_match_analysis"] = longest_match_stats
                
                if "avg_longest_match_words" in longest_match_stats:
                    print(f"\nLongest Matching Sequence (top 20):")
                    print(f"  Average words: {longest_match_stats['avg_longest_match_words']}")
                    print(f"  Max words:     {longest_match_stats['max_longest_match_words']}")
                    print(f"  Min words:     {longest_match_stats['min_longest_match_words']}")
                
                # Save detailed matches to CSV
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    csv_file = Path(output_dir) / f"rouge5_matches_{corpus_type}_{Path(input_file).stem}.csv"
                else:
                    csv_file = Path(input_file).parent / f"rouge5_matches_{Path(input_file).stem}.csv"
                
                try:
                    import pandas as pd
                    df = pd.DataFrame([
                        {
                            "idx": m.query_idx,
                            "query": m.query[:200] + "..." if len(m.query) > 200 else m.query,
                            "result": m.result[:200] + "..." if len(m.result) > 200 else m.result,
                            "precision": m.precision,
                            "recall": m.recall,
                            "fmeasure": m.fmeasure
                        }
                        for m in matches
                    ])
                    df.to_csv(csv_file, index=False)
                    print(f"\nROUGE matches saved to: {csv_file}")
                except ImportError:
                    print("pandas not available, skipping CSV export")
                    
        except Exception as e:
            print(f"ERROR in ROUGE analysis: {e}")
            results["rouge5_similarity"] = {"error": str(e)}
    else:
        print("\nSkipping ROUGE-5 analysis (use --run_rouge to enable)")
    
    # ============================================================
    # Save results
    # ============================================================
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = Path(output_dir) / f"privacy_results_{corpus_type}_{Path(input_file).stem}.json"
    else:
        output_file = Path(input_file).parent / f"privacy_results_{Path(input_file).stem}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'-'*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run privacy evaluation on synthetic clinical notes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (8-gram overlap only, fast)
  python run_privacy.py --input_file output/CARES/generated_cares_20260122_020000.jsonl --skip_rouge
  
  # Full evaluation with ROUGE-5 (slower)
  python run_privacy.py --input_file output/CWLC/generated_gpt_oss_20251218_113714.jsonl --n_jobs 4
  
  # With sample limit for ROUGE
  python run_privacy.py --input_file output/RADGRAPH/generated_radgraph_20260206_140246.jsonl --max_rouge_samples 100
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
    parser.add_argument(
        "--n_jobs", 
        type=int, 
        default=1, 
        help="Number of parallel jobs for ROUGE calculation (default: 1)."
    )
    parser.add_argument(
        "--max_rouge_samples", 
        type=int, 
        default=500, 
        help="Maximum number of samples for ROUGE evaluation (default: 500)."
    )
    parser.add_argument(
        "--skip_rouge", 
        action="store_true",
        help="Skip ROUGE-5 analysis (faster, only compute n-gram overlap)."
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"ERROR: Input file not found: {args.input_file}")
        sys.exit(1)
    
    run_privacy_evaluation(
        args.input_file, 
        args.output_dir, 
        n_jobs=args.n_jobs,
        max_rouge_samples=args.max_rouge_samples,
        skip_rouge=args.skip_rouge
    )


if __name__ == "__main__":
    main()
