import pandas as pd
import difflib
import argparse

def longest_matching_sequence_words(query, result):
    # Function to find the longest matching sequence of words
    query_words = query.split()
    result_words = result.split()
    
    seq_matcher = difflib.SequenceMatcher(None, query_words, result_words)
    match = seq_matcher.find_longest_match(0, len(query_words), 0, len(result_words))
    
    return query_words[match.a: match.a + match.size]

def process_longest_match(csv_file, max_rows):
    # Process CSV and calculate word counts
    data = pd.read_csv(csv_file)
    # Sort by recall
    data = data.sort_values(by="recall", ascending=False)
    
    word_counts = []
    
    for index, row in data[:max_rows].iterrows(): 
        query = row['query'].strip().lower()
        result = row['result'].strip().lower()
        
        # Find the longest matching sequence
        longest_match = longest_matching_sequence_words(query, result)
        
        # Count the number of words in the longest matching sequence
        word_count = len(longest_match)
        word_counts.append(word_count) 
    
    # Print statistics
    if word_counts:
        avg_word_count = sum(word_counts) / len(word_counts)
        sorted_counts = sorted(word_counts)
        print(f"Average Word Count: {avg_word_count:.2f}")
        print(f"Minimum Word Count: {min(sorted_counts)}")
        print(f"Maximum Word Count: {max(sorted_counts)}")
    else:
        print("No matching sequences found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rouge_file", type=str, required=True, help="Path to ROUGE-5 CSV file")
    parser.add_argument("--max_rows", type=int, default=20, help="Maximum number of rows to process from the CSV file (default: 20).")
    args = parser.parse_args()
    process_longest_match(csv_file=args.rouge_file, max_rows=args.max_rows)
