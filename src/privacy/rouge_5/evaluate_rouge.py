import argparse
import pandas as pd

def get_measures(data_file):
    data = pd.read_csv(data_file)
    r_scores = data["recall"]
    
    # Overall measures
    average_r_score_overall = r_scores.mean()
    max_r_score_overall = r_scores.max()
    min_r_score_overall = r_scores.min()
    median_r_score_overall = r_scores.median()
    
    # Get the top 122 highest recall scores
    top_r_scores = r_scores.nlargest(122)
    
    # Measures for top 122 scores
    average_r_score_top = top_r_scores.mean()
    max_r_score_top = top_r_scores.max()
    min_r_score_top = top_r_scores.min()
    median_r_score_top = top_r_scores.median()

    # Print the overall results
    print("Overall Recall Scores:")
    print(f"Average recall-score: {average_r_score_overall}")
    print(f"Maximum recall-score: {max_r_score_overall}")
    print(f"Minimum recall-score: {min_r_score_overall}")
    print(f"Median recall-score: {median_r_score_overall}")

    # Print the top 122 results
    print("\nTop 122 Recall Scores:")
    print(f"Average recall-score (top 122): {average_r_score_top}")
    print(f"Maximum recall-score (top 122): {max_r_score_top}")
    print(f"Minimum recall-score (top 122): {min_r_score_top}")
    print(f"Median recall-score (top 122): {median_r_score_top}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rouge_file', dest='rouge_file', type=str, required=True, help="Path to the CSV file storing the ROUGE-5 scores.")

    args = parser.parse_args()
    get_measures(args.rouge_file)