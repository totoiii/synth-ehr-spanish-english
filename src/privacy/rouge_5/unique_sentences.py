import pandas as pd
import re
import nltk
from collections import Counter
from nltk import ngrams

data = pd.read_feather("/nethome/lkiefer/SynText/medical-coding-reproducibility/files/data/shortened_notes/short_notes.feather")
splits = pd.read_feather("/nethome/lkiefer/SynText/medical-coding-reproducibility/files/data/llama_31/mimiciv_icd10_split.feather")
train_ids = splits[splits["split"] == "train"]['_id']
df_train = data[data['_id'].isin(train_ids)]
df_query = pd.read_csv("/nethome/lkiefer/SynText/privacy/mdpi2021-textgen/output/train_out1_fulltest/rougescores_5gram_test_sorted.csv")


# Function to generate 5-grams from a text using NLTK
def generate_5grams(text):
    # Tokenize the text while keeping punctuation
    tokens = nltk.word_tokenize(text)
    # Generate 5-grams
    return [' '.join(gram) for gram in ngrams(tokens, 5)]

# Optimized function to count 5-gram matches that are also in the result column
def count_matching_5gram_in_results(df_train, df_query, output_file):
    # Step 1: Preprocess `df_train` to get 5-gram counts
    train_5grams = []
    for doc in df_train['text']:
        train_5grams.extend(generate_5grams(doc))
    
    # Step 2: Count occurrences of each 5-gram in the training data
    train_5gram_counts = Counter(train_5grams)
    
    # Step 3: Prepare results for each query based on matching with the 'result' column
    results = []
    for idx, query_row in df_query[:20].iterrows():
        query_document = query_row['query']
        result_document = query_row['result']
        
        # Generate 5-grams for the current query document
        query_5grams = generate_5grams(query_document)
        
        # Check which 5-grams from the query are also in the result
        result_5grams = set(generate_5grams(result_document))
        
        # Count occurrences of matching 5-grams in the training data
        for five_gram in query_5grams:
            if five_gram in result_5grams:  # Only consider if it's also in the result
                count = train_5gram_counts[five_gram]  # Fast lookup
                results.append({'query_index': idx + 1, '5-gram': five_gram, 'count': count})
    
    # Convert results to DataFrame and write to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results written to {output_file}")

# Usage
count_matching_5gram_in_results(df_train, df_query, '15gram_counts_sorted.csv')

import difflib
# Function to find the longest matching sequence
def longest_matching_sequence_words(query, result):
    # Split the strings into words
    query_words = query.split()
    result_words = result.split()
    
    # Create a sequence matcher for words
    seq_matcher = difflib.SequenceMatcher(None, query_words, result_words)
    match = seq_matcher.find_longest_match(0, len(query_words), 0, len(result_words))
    
    # Return the matching sequence of words
    return query_words[match.a: match.a + match.size]

# List to store word counts for each row
word_counts = []

# Iterate over the DataFrame and print the longest matching sequences and their word counts
for index, row in test_sorted[:20].iterrows():
    # Preprocess by lowering case and stripping whitespace
    query = row['query'].strip().lower()
    result = row['result'].strip().lower()
    
    longest_match = longest_matching_sequence_words(query, result)
    
    # Count the number of words in the longest matching sequence
    word_count = len(longest_match)
    word_counts.append(word_count)  # Store the word count in the list
    
    # Print the results
    print(f"Query: '{row['query']}'\nResult: '{row['result']}'")
    print(f"Longest Matching Sequence: '{' '.join(longest_match)}' (Word Count: {word_count})\n")

# Print the list of word counts
print("List of Word Counts:", word_counts)

word_counts = sorted(word_counts)
print(sum(word_counts) / len(word_counts))
print(word_counts)