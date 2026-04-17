import pandas as pd
import nltk
from collections import Counter
from nltk import ngrams
import argparse

def generate_5grams(text):
    # Generate 5-grams
    tokens = nltk.word_tokenize(text)
    return [' '.join(gram) for gram in ngrams(tokens, 5)]

def count_matching_5gram_stats(train_file, splits_file, query_file, split_name, max_queries):
    # Get overall count of overlapping 5-grams in training data

    data = pd.read_feather(train_file)
    # Sort by recall
    data = data.sort_values(by="recall", ascending=False)
    splits = pd.read_feather(splits_file)
    
    # Filter the training split
    train_ids = splits[splits["split"] == split_name]['_id']
    df_train = data[data['_id'].isin(train_ids)]
    df_query = pd.read_csv(query_file)
    
    train_5grams = []
    for doc in df_train['text']:
        train_5grams.extend(generate_5grams(doc))
    
    train_5gram_counts = Counter(train_5grams)
    
    all_counts = []
    for idx, query_row in df_query[:max_queries].iterrows():  
        query_document = query_row['query']
        result_document = query_row['result']
        
        query_5grams = generate_5grams(query_document)
        # Overlap
        result_5grams = set(generate_5grams(result_document))
        
        # Count occurrences of matching 5-grams in the training data
        for five_gram in query_5grams:
            if five_gram in result_5grams:  
                count = train_5gram_counts[five_gram]  
                all_counts.append(count)
    
    # Compute and print 
    min_count = min(all_counts)
    max_count = max(all_counts)
    avg_count = sum(all_counts) / len(all_counts)
    print(f"Minimum 5-gram count: {min_count}")
    print(f"Maximum 5-gram count: {max_count}")
    print(f"Average 5-gram count: {avg_count:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training data file (Feather format).")
    parser.add_argument("--splits_file", type=str, required=True, help="Path to the splits file (Feather format).")
    parser.add_argument("--rouge_file", type=str, required=True, help="Path to the query data file (CSV format).")
    parser.add_argument("--split_name", type=str, default="train", help="Name of the split to use for training data (default: 'train').")
    parser.add_argument("--max_queries", type=int, default=20, help="Maximum number of queries to process (default: 20).")
    
    args = parser.parse_args()
    
    count_matching_5gram_stats(
        train_file=args.train_file, 
        splits_file=args.splits_file, 
        query_file=args.roge_file, 
        split_name=args.split_name, 
        max_queries=args.max_queries
    )