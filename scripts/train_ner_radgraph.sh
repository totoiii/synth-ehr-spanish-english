#!/usr/bin/env bash
# Run NER encoder experiments on RADGRAPH JSONL generations.

set -euo pipefail

# Argument: GPU_ID (optional, default 1)
# Note: The user might override this environment variable or pass a different one.
GPU_ID=${1:-1}
export CUDA_VISIBLE_DEVICES=$GPU_ID

DATASET=${DATASET:-"RADGRAPH"}
ENV_NAME=${ENV_NAME:-"unsloth-test"}
# Default model for Spanish, but maybe we should use English for MIMIC/RADGRAPH?
# The user didn't specify, but existing script uses "dccuchile/bert-base-spanish-wwm-cased".
# The dataset source is "mimic-chest-ct", which is English.
# However, the user asked to copy the script which likely implies keeping the same model unless specified.
# But "MIMIC" is English proper. Let's stick to the script default or maybe switch to roberta-base?
# The user's prompt implies adapting the script for *these texts* using *that .py*.
# The .py default is Spanish BERT.
# Given it's MIMIC data (English), using a Spanish model is suboptimal but technically "copying the script".
# I'll stick to the variable, maybe defaulting to roberta-base or bert-base-uncased if I want to be smart?
# No, let's stick to the variable so it can be overridden, and maybe default to the same as before to minimize deviation
# unless I'm sure.
# Actually, the user says "Entiende este sh... y a partir de esto crea... para que tambien ocupe ese .py...".
# I will use the same default model variable, but maybe I should comment on it.
# Wait, the dataset is MIMIC (English). The previous script was CWLC (likely Spanish given the model).
# I'll change the default model to `roberta-base` or `bert-base-cased` for English if I can, but to be safe I'll leave the variable structure
# and just let the user override it or use the script's default.
# Actually, the python script has a default `dccuchile/bert-base-spanish-wwm-cased`.
# I'll override it in the script to `roberta-base` for RADGRAPH since it's English.
MODEL_NAME=${MODEL_NAME:-"roberta-base"} 
SEED=${SEED:-3407}
EPOCHS=${EPOCHS:-3}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./output/logs"

# Define datasets
# 100%: .../synth_radgraph_100pct_20260218_133312.json
# 25%: .../synth_radgraph_25pct_20260217_232951.json
# 5%: .../synth_radgraph_5pct_20260217_200905.json
# 50%: Not provided, so I'll skip it or comment it out.

FILE_100="./output/RADGRAPH/synth_radgraph_100pct_20260218_133312.json"
# FILE_50="" # Not provided
FILE_25="./output/RADGRAPH/synth_radgraph_25pct_20260217_232951.json"
FILE_5="./output/RADGRAPH/synth_radgraph_5pct_20260217_200905.json"

mkdir -p "$LOG_DIR"

echo "========================================================================"
echo "NER Encoder experiments: RADGRAPH"
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

echo -e "\n[1/4] Training on original_text (using 100% file) -> $FILE_100"
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

# Experiment 2: Train/Eval on generated_text (Loop: 100 -> 25 -> 5)
# Note: We use 'output1' field for these (as per user request and file content).
# The user prompt mentions "output1" in the JSON example.

declare -a FILES=("$FILE_100" "$FILE_25" "$FILE_5")
declare -a TAGS=("100pct" "25pct" "5pct")
# Start step counter at 2
STEP=2

for i in "${!FILES[@]}"; do
    FILE="${FILES[$i]}"
    TAG="${TAGS[$i]}"
    
    OUT_GEN="./output/${DATASET}/${DATASET}_downstream_generated_${TAG}_${TIMESTAMP}"
    LOG_GEN="$LOG_DIR/${DATASET}_downstream_generated_${TAG}_${TIMESTAMP}.log"
    
    echo -e "\n[$STEP/4] Training on generated_text ($TAG) -> $FILE"
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
