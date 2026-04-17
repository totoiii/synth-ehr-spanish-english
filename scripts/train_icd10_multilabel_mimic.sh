#!/usr/bin/env bash
# Train ICD-10 multi-label classifier on MIMIC synthetic data (converted from JSON array to JSONL).
#
# Usage:
#   bash scripts/train_icd10_multilabel_mimic.sh <PCT> [TEXT_SOURCE]
#
# Examples:
#   bash scripts/train_icd10_multilabel_mimic.sh 100pct
#   bash scripts/train_icd10_multilabel_mimic.sh 100pct original
#
# Arguments:
#   PCT: Percentage identifier (e.g., 5pct, 25pct, 50pct, 100pct) matching the input filename.
#   TEXT_SOURCE: 'generated' (default), 'original', or 'both_concat'.
#
# The script expects input files in output/MIMIC/ matches 'synth_mimic_small_${PCT}_*.json'.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <PCT> [TEXT_SOURCE]"
    echo "Example: $0 100pct generated"
    exit 1
fi

PCT="$1"
TEXT_SOURCE="${2:-generated}"
LABEL_TYPE="icd10" # Use all ICD-10 codes, not chapters, as requested

# Allow the caller to override the input JSON (used by the multi-run pipeline).
if [[ -n "${SYNTH_JSON_OVERRIDE:-}" ]]; then
    INPUT_FILE="$SYNTH_JSON_OVERRIDE"
else
    INPUT_PATTERN="output/MIMIC/synth_mimic_small_${PCT}_*.json"
    INPUT_FILE=$(ls -t $INPUT_PATTERN 2>/dev/null | head -n 1)
    if [[ -z "$INPUT_FILE" ]]; then
        INPUT_PATTERN_ALT="output/MIMIC/synth_mimic_${PCT}_*.json"
        INPUT_FILE=$(ls -t $INPUT_PATTERN_ALT 2>/dev/null | head -n 1)
    fi
fi

if [[ -z "$INPUT_FILE" || ! -f "$INPUT_FILE" ]]; then
    echo "Error: No input file available (override invalid or pattern match empty)"
    exit 1
fi

echo "Found input file: $INPUT_FILE"

# Create a temporary JSONL file
TEMP_JSONL="temp_train_mimic_${PCT}_$(date +%s).jsonl"
TEMP_EVAL_JSONL="temp_eval_mimic_${PCT}_$(date +%s).jsonl"
echo "Converting JSON array to JSONL, renaming keys, and mapping ICD-10 codes..."

python3 -c "
import json
import sys
import csv

input_file = '$INPUT_FILE'
output_file = '$TEMP_JSONL'
eval_file = 'data/mimic/mimic_m.json'
eval_output_file = '$TEMP_EVAL_JSONL'
csv_file = 'data/mimic/ICD10_descriptions_mimic.csv'

def stream_json_array(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        # Read until the opening bracket
        while True:
            char = f.read(1)
            if not char or char == '[':
                break

        buffer = ''
        decoder = json.JSONDecoder()
        while True:
            chunk = f.read(2048 * 2048)  # 4MB chunks
            if not chunk:
                break
            buffer += chunk
            
            while True:
                buffer = buffer.lstrip(' \n\r\t,')
                if not buffer or buffer.startswith(']'):
                    break
                try:
                    obj, idx = decoder.raw_decode(buffer)
                    yield obj
                    buffer = buffer[idx:]
                except json.JSONDecodeError:
                    # Incomplete object, read more
                    break

def process_file(in_path, out_path, desc_to_code, sorted_descs):
    valid_entries = 0
    total_entries = 0
    with open(out_path, 'w', encoding='utf-8') as fout:
        for entry in stream_json_array(in_path):
            total_entries += 1
            if 'output' in entry:
                entry['original_text'] = entry.pop('output')
                
            if 'output1' in entry:
                entry['generated_text'] = entry.pop('output1')
            elif 'output_1' in entry:
                entry['generated_text'] = entry.pop('output_1')
            elif 'original_text' in entry and 'generated_text' not in entry:
                # If evaluating on original mimic_m.json, there's no generated_text.
                # We can just copy original_text to generated_text so that it doesn't fail if we ask for generated_text
                entry['generated_text'] = entry['original_text']
            
            input_text = entry.get('input', '')
            codes = []
            
            # Map descriptions to codes
            for desc in sorted_descs:
                if desc in input_text:
                    codes.append(desc_to_code[desc])
                    # Remove the matched description to prevent sub-matches
                    input_text = input_text.replace(desc, '')
            
            # Save mapped codes to 'icd10' field
            entry['icd10'] = codes
            
            # We must set mode 'cares' because the training script filters on 'mode' None or 'cares'
            entry['mode'] = 'cares'
            
            if len(codes) > 0:
                fout.write(json.dumps(entry) + '\n')
                valid_entries += 1
    return valid_entries, total_entries

try:
    # 1. Load ICD-10 descriptions
    desc_to_code = {}
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['icd_version'] == '10': # Use ICD-10 mapping
                desc_to_code[row['long_title']] = row['icd_code']
    
    # Sort descriptions by length descending to match longest first
    sorted_descs = sorted(desc_to_code.keys(), key=len, reverse=True)

    # 2. Parse JSON iteratively and convert Train Date
    train_v, train_t = process_file(input_file, output_file, desc_to_code, sorted_descs)
    print(f'Successfully mapped and converted {train_v} items to {output_file} (Original total: {train_t})')
    
    # 3. Parse JSON iteratively and convert Eval Data 
    eval_v, eval_t = process_file(eval_file, eval_output_file, desc_to_code, sorted_descs)
    print(f'Successfully mapped and converted {eval_v} items to {eval_output_file} (Original total: {eval_t})')
    
except Exception as e:
    print(f'Error converting file: {e}')
    sys.exit(1)
"

# Set output directory (allow override)
OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-output/MIMIC/MIMIC_downstream_${LABEL_TYPE}_${PCT}_${TEXT_SOURCE}}"
echo "Output directory: $OUTPUT_DIR"

# Run the training script
echo "Starting training..."
# Pass through environment variables for hyperparameters if set
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"2"} # Default to GPU 1 if not set

bash scripts/train_icd10_multilabel.sh \
    --train-jsonl "$TEMP_JSONL" \
    --train-text-source "$TEXT_SOURCE" \
    --eval-jsonl "$TEMP_EVAL_JSONL" \
    --eval-text-source "original" \
    --label-type "$LABEL_TYPE" \
    --output-dir "$OUTPUT_DIR" \
    ${MAX_TRAIN_SAMPLES:+--max-train-samples "$MAX_TRAIN_SAMPLES"} \
    ${MAX_EVAL_SAMPLES:+--max-eval-samples "$MAX_EVAL_SAMPLES"} \
    --num-train-epochs "${EPOCHS:-10}"

# Cleanup
echo "Cleaning up temporary files..."
rm "$TEMP_JSONL" "$TEMP_EVAL_JSONL"

echo "Done. Results in $OUTPUT_DIR"
