import argparse
import json

def get_unique_eightgrams(filename, field):
    unique_ngrams = set()
    with open(filename, 'r') as file:
        data = json.load(file)
        for doc in data:
            text = doc[field]
            words = text.strip().lower().split()
            for i in range(len(words)):
                if i + 7 >= len(words):
                    break
                unique_8gram = words[i]
                for j in range(i + 1, i + 8):
                    unique_8gram += " " + words[j]

                unique_ngrams.add(unique_8gram)

    return unique_ngrams


def main(original_file, original_field, synthetic_file, synthetic_field):
    original_8grams = get_unique_eightgrams(original_file, original_field)
    synthetic_8grams = get_unique_eightgrams(synthetic_file, synthetic_field)

    print(f"Number of unique eightgrams in {original_file}: {len(original_8grams)}")
    print(f"Number of unique eightgrams in {synthetic_file}: {len(synthetic_8grams)}")

    total_8grams = set()
    total_8grams.update(original_8grams, synthetic_8grams)
    print(f"Total unique eightgrams across both files: {len(total_8grams)}")

    common_8grams = original_8grams.intersection(synthetic_8grams)
    print(f"Number of common eightgrams: {len(common_8grams)}")

    overlap = len(common_8grams) / len(total_8grams) if len(total_8grams) > 0 else 0
    print(f"8gram overlap: {overlap:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_file', dest='original_file', type=str, required=True, help="Path to the original JSON file.")
    parser.add_argument('--original_field', dest='original_field', type=str, required=True, help="Name of field in JSON file.")
    parser.add_argument('--synthetic_file', dest='synthetic_file', type=str, required=True, help="Path to the synthetic JSON file.")
    parser.add_argument('--synthetic_field', dest='synthetic_field', type=str, required=True, help="Name of field in JSON file.")

    args = parser.parse_args()

    main(args.original_file, args.original_field, args.synthetic_file, args.synthetic_field)
