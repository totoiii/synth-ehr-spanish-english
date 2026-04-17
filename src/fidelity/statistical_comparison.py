import json
import argparse
from nltk.tokenize import word_tokenize, sent_tokenize

def analyze_corpus(json_file, field):
    # Analyze the datasets regarding a range of statistical features
    with open(json_file, 'r') as file:
        docs = json.load(file)

    total_sentences = 0
    total_tokens = 0
    unique_words = set()
    doc_count = len(docs)

    for doc in docs:
        text = doc.get(field, "")
        
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
        "total_documents": doc_count,
        "average_sentences_per_document": average_sentence_per_doc,
        "average_tokens_per_document": average_token_per_doc,
        "average_tokens_per_sentence": average_token_per_sentence,
        "total_sentences": total_sentences,
        "total_tokens": total_tokens,
        "unique_tokens": len(unique_words),
        "unique_token_ratio": len(unique_words) / total_tokens if total_tokens > 0 else 0
    }
    return results

def compare_results(real_results, synthetic_results):
    # Compares the results between real and synthetic data
    comparison = {}
    for key in real_results.keys():
        comparison[key] = {
            "real": real_results[key],
            "synthetic": synthetic_results[key],
            "difference": real_results[key] - synthetic_results[key] if isinstance(real_results[key], (int, float)) else "N/A"
        }
    return comparison

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_file", type=str, required=True, help="Path to the JSON file with real data.")
    parser.add_argument("--field_real", type=str, required=True, help="The output field in the real JSON documents to analyze.")
    parser.add_argument("--synthetic_file", type=str, required=True, help="Path to the JSON file with synthetic data.")
    parser.add_argument("--field_synthetic", type=str, required=True, help="The output field in the synthetic JSON documents to analyze.")
    args = parser.parse_args()

    # Analyze both datasets
    real_results = analyze_corpus(args.real_file, args.field)
    synthetic_results = analyze_corpus(args.synthetic_file, args.field)

    # Compare results
    comparison = compare_results(real_results, synthetic_results)

    # Print the comparison
    print(json.dumps(comparison, indent=4))
