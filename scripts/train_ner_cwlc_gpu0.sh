#!/usr/bin/env bash
# Run NER encoder experiments on CWLC JSONL generations.

set -euo pipefail

# Argument: GPU_ID (optional, default 1)
GPU_ID=${1:-1}
export CUDA_VISIBLE_DEVICES=$GPU_ID

DATASET=${DATASET:-"CWLC"}
ENV_NAME=${ENV_NAME:-"unsloth-test"}
MODEL_NAME=${MODEL_NAME:-"dccuchile/bert-base-spanish-wwm-cased"}
SEED=${SEED:-3407}
EPOCHS=${EPOCHS:-3}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./output/logs"

# Define datasets
FILE_100="./output/CWLC/synth_cwlc_100pct_20260219_092658.json"
FILE_50="./output/CWLC/synth_cwlc_50pct_20260219_121519.json"
FILE_25="./output/CWLC/synth_cwlc_25pct_20260219_150626.json"
FILE_5="./output/CWLC/synth_cwlc_5pct_20260219_174956.json"

mkdir -p "$LOG_DIR"

echo "========================================================================"
echo "NER Encoder experiments: CWLC"
echo "GPU_ID: $GPU_ID"
echo "Env: $ENV_NAME"
echo "Model: $MODEL_NAME"
echo "Seed: $SEED"
echo "Timestamp: $TIMESTAMP"
echo "========================================================================"

# Experiment 1: Train/Eval on original_text (Run ONCE using the 100% file as source)
# We assume the 100% file contains all the original text examples we want to evaluate on.
OUT_ORIG="./output/${DATASET}/${DATASET}_downstream_original_100pct_${TIMESTAMP}"
LOG_ORIG="$LOG_DIR/${DATASET}_downstream_original_100pct_${TIMESTAMP}.log"

echo -e "\n[1/5] Training on original_text (using 100% file) -> $FILE_100"
echo "Output: $OUT_ORIG"
echo "Log: $LOG_ORIG"

conda run -n "$ENV_NAME" python -u src/train_ner_encoder.py \
  --jsonl "$FILE_100" \
  --model-name "$MODEL_NAME" \
  --text-field original_text \
  --output-dir "$OUT_ORIG" \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  2>&1 | tee "$LOG_ORIG"

# Experiment 2: Train/Eval on generated_text (Loop: 100 -> 50 -> 25 -> 5)
# Note: We use 'generated_text' field for these.

declare -a FILES=("$FILE_100" "$FILE_50" "$FILE_25" "$FILE_5")
declare -a TAGS=("100pct" "50pct" "25pct" "5pct")
# Start step counter at 2
STEP=2

for i in "${!FILES[@]}"; do
    FILE="${FILES[$i]}"
    TAG="${TAGS[$i]}"
    
    OUT_GEN="./output/${DATASET}/${DATASET}_downstream_generated_${TAG}_${TIMESTAMP}"
    LOG_GEN="$LOG_DIR/${DATASET}_downstream_generated_${TAG}_${TIMESTAMP}.log"
    
    echo -e "\n[$STEP/5] Training on generated_text ($TAG) -> $FILE"
    echo "Output: $OUT_GEN"
    echo "Log: $LOG_GEN"
    
    conda run -n "$ENV_NAME" python -u src/train_ner_encoder.py \
      --jsonl "$FILE" \
      --model-name "$MODEL_NAME" \
      --text-field output1 \
      --output-dir "$OUT_GEN" \
      --seed "$SEED" \
      --epochs "$EPOCHS" \
      2>&1 | tee "$LOG_GEN"
      
    STEP=$((STEP+1))
done

echo -e "\nAll experiments completed!"
