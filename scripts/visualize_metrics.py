import os
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

def main(args):
    run_dir = Path(args.run_dir)
    
    # 1. Load the overall metrics json
    metrics_file = run_dir / "metrics_test_epoch_0.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
        
        print("="*60)
        print("          OVERALL TEST METRICS (PLM-ICD)")
        print("="*60)
        test_metrics = metrics.get("test", {})
        for metric_group, values in test_metrics.items():
            print(f"\n[{metric_group.upper()}]")
            for k, v in values.items():
                if isinstance(v, float):
                    print(f"  {k:20s}: {v:.4f}")
                else:
                    print(f"  {k:20s}: {v}")
    else:
        print(f"Metrics file not found at {metrics_file}")
    
    # 2. Calculate per-label metrics from the predictions file
    pred_file = run_dir / "predictions_test.feather"
    if pred_file.exists():
        print("\n" + "="*60)
        print("          PER-LABEL METRICS CALCULATION")
        print("="*60)
        df = pd.read_feather(pred_file)
        
        # 'df' has columns: code1, code2, ..., '_id', 'target'
        # 'target' contains arrays of strings with the actual ground truth codes
        
        # The probability columns are all the columns except '_id' and 'target'
        code_cols = [c for c in df.columns if c not in ["_id", "target"] and not c.startswith("LABEL_")]
        
        # Convert the ground truth list of strings into binary vectors
        y_true = np.zeros((len(df), len(code_cols)), dtype=int)
        y_pred = np.zeros((len(df), len(code_cols)), dtype=int)
        
        code_to_idx = {code: i for i, code in enumerate(code_cols)}
        
        print(f"Analyzing {len(code_cols)} target ICD codes across {len(df)} test samples...")
        
        for i, row in df.iterrows():
            # Ground truth
            actual_codes = row["target"]
            if hasattr(actual_codes, "tolist"):
                actual_codes = actual_codes.tolist()
                
            for code in actual_codes:
                if code in code_to_idx:
                    y_true[i, code_to_idx[code]] = 1
            
            # Predictions (thresholded at 0.5; probabilities are already sigmoided)
            for j, code in enumerate(code_cols):
                if row[code] >= 0.5:
                    y_pred[i, j] = 1
                    
        # Calculate reports
        report_data = []
        for j, code in enumerate(code_cols):
            support = int(np.sum(y_true[:, j]))
            if support > 0 or np.sum(y_pred[:, j]) > 0:
                f1 = f1_score(y_true[:, j], y_pred[:, j], zero_division=0)
                prec = precision_score(y_true[:, j], y_pred[:, j], zero_division=0)
                rec = recall_score(y_true[:, j], y_pred[:, j], zero_division=0)
                
                report_data.append({
                    "ICD_Code": code,
                    "Support": support,
                    "Precision": round(prec, 4),
                    "Recall": round(rec, 4),
                    "F1_Score": round(f1, 4),
                    "Predicted_Count": int(np.sum(y_pred[:, j]))
                })
        
        # Convert to DataFrame, sort by Support (frequency), and save
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values(by="Support", ascending=False).reset_index(drop=True)
        
        output_csv = run_dir / "per_label_metrics.csv"
        report_df.to_csv(output_csv, index=False)
        print(f"Saved per-label metrics to: {output_csv}")
        
        # Print top 15 most frequent codes
        print("\nTop 15 Most Frequent Codes Performance:")
        print(report_df.head(15).to_string(index=False))
        
    else:
        print(f"Predictions feather not found at {pred_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize PLM-ICD metrics")
    parser.add_argument("--run-dir", type=str, required=True, 
                        help="Path to the PLM-ICD output run directory")
    args = parser.parse_args()
    main(args)
