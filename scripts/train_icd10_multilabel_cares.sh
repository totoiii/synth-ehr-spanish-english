#!/usr/bin/env bash
# Train ICD-10 multi-label classifier on CARES synthetic data (converted from JSON array to JSONL).
#
# Usage:
#   bash scripts/train_icd10_multilabel_cares.sh <PCT> [TEXT_SOURCE]
#
# Examples:
#   bash scripts/train_icd10_multilabel_cares.sh 25pct
#   bash scripts/train_icd10_multilabel_cares.sh 100pct original
#
# Arguments:
#   PCT: Percentage identifier (e.g., 5pct, 25pct, 50pct, 100pct) matching the input filename.
#   TEXT_SOURCE: 'generated' (default), 'original', or 'both_concat'.
#
# The script expects input files in output/CARES/ matches 'synth_cares_${PCT}_*.json'.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <PCT> [TEXT_SOURCE]"
    echo "Example: $0 25pct generated"
    exit 1
fi

PCT="$1"
TEXT_SOURCE="${2:-generated}"
LABEL_TYPE="chapters" # Fixed as requested 

# Allow the caller to override the input JSON (used by the multi-run pipeline).
if [[ -n "${SYNTH_JSON_OVERRIDE:-}" ]]; then
    INPUT_FILE="$SYNTH_JSON_OVERRIDE"
else
    INPUT_PATTERN="output/CARES/synth_cares_${PCT}_*.json"
    INPUT_FILE=$(ls -t $INPUT_PATTERN 2>/dev/null | head -n 1)
fi

if [[ -z "$INPUT_FILE" || ! -f "$INPUT_FILE" ]]; then
    echo "Error: No file found matching $INPUT_PATTERN (or override invalid)"
    exit 1
fi

echo "Found input file: $INPUT_FILE"

# Create a temporary JSONL file
TEMP_JSONL="temp_train_${PCT}_$(date +%s).jsonl"
echo "Converting JSON array to JSONL and renaming keys..."

python3 -c "
import json
import sys

input_file = '$INPUT_FILE'
output_file = '$TEMP_JSONL'

try:
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    with open(output_file, 'w') as f:
        for entry in data:
            # Rename output1 to generated_text if present
            if 'output1' in entry:
                entry['generated_text'] = entry.pop('output1')
            
            # Ensure required keys exist (icd10 is critical)
            if 'icd10' not in entry:
                continue
                
            f.write(json.dumps(entry) + '\n')
            
    print(f'Successfully converted {len(data)} items to {output_file}')
except Exception as e:
    print(f'Error converting file: {e}')
    sys.exit(1)
"

# Set output directory (allow override)
OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-output/CARES/CARES_downstream_${LABEL_TYPE}_${PCT}_${TEXT_SOURCE}}"
echo "Output directory: $OUTPUT_DIR"

# Run the training script
echo "Starting training..."
# Pass through environment variables for hyperparameters if set
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"2"} # Default to GPU 0 if not set, script defaults to 1,2

bash scripts/train_icd10_multilabel.sh \
    --train-jsonl "$TEMP_JSONL" \
    --train-text-source "$TEXT_SOURCE" \
    --label-type "$LABEL_TYPE" \
    --output-dir "$OUTPUT_DIR" \
    ${MAX_TRAIN_SAMPLES:+--max-train-samples "$MAX_TRAIN_SAMPLES"} \
    ${MAX_EVAL_SAMPLES:+--max-eval-samples "$MAX_EVAL_SAMPLES"} \
    --num-train-epochs "${EPOCHS:-10}"

# Cleanup
echo "Cleaning up temporary file..."
rm -f "$TEMP_JSONL"

echo "Done. Results in $OUTPUT_DIR"
