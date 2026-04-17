#!/bin/bash
#
# Fine-tune GPT-OSS 20B on MIMIC discharge summaries
# Dataset: MIMIC (mimic_s.json, mimic_m.json, or mimic_l.json)
#
# Dataset profile:
#   - mimic_s: ~13,378 records | mimic_m: ~19K | mimic_l: ~89,098 records
#   - English discharge summaries, Median ~1,919 tokens, P95 ~4,023
#
# Usage:
#   bash scripts/finetuning_mimic.sh [SIZE] [TRAIN_PCT]
#
# Examples:
#   bash scripts/finetuning_mimic.sh small         # small dataset, 100%
#   bash scripts/finetuning_mimic.sh large 50      # large dataset, 50%
#   bash scripts/finetuning_mimic.sh medium 10     # medium dataset, 10%

set -e
set -o pipefail

# Activate conda environment
source /home/lmiranda/miniconda3/etc/profile.d/conda.sh
conda activate unsloth-test

# GPU configuration: single GPU for Unsloth LoRA
TRAIN_CUDA_DEVICES="${TRAIN_CUDA_DEVICES:-0}"

# Parse arguments
MIMIC_SIZE="${1:-small}"
TRAIN_PCT="${2:-100}"

# Data paths
MIMIC_DATA_DIR="/home/lmiranda/ehr-synthetic-bmc-2026/data/mimic"

case "$MIMIC_SIZE" in
    small|s)
        MIMIC_JSON="${MIMIC_DATA_DIR}/mimic_s.json"
        SIZE_SUFFIX="small"
        BASE_EPOCHS=3       # ~13K records
        ;;
    medium|m)
        MIMIC_JSON="${MIMIC_DATA_DIR}/mimic_m.json"
        SIZE_SUFFIX="medium"
        BASE_EPOCHS=2       # ~19K records
        ;;
    large|l)
        MIMIC_JSON="${MIMIC_DATA_DIR}/mimic_l.json"
        SIZE_SUFFIX="large"
        BASE_EPOCHS=1       # ~89K records — 1 epoch is enough
        ;;
    *)
        echo "Usage: $0 [small|medium|large] [TRAIN_PCT]"
        echo "  small (s)  - mimic_s.json (~13K records)"
        echo "  medium (m) - mimic_m.json (~19K records)"
        echo "  large (l)  - mimic_l.json (~89K records)"
        echo ""
        echo "  TRAIN_PCT  - Percentage of training set (1-100, default: 100)"
        exit 1
        ;;
esac

if [ ! -f "$MIMIC_JSON" ]; then
    echo "❌ ERROR: MIMIC JSON file not found: $MIMIC_JSON"
    echo ""
    echo "Expected location: ${MIMIC_DATA_DIR}/"
    echo "Available files:"
    ls -la "${MIMIC_DATA_DIR}/" 2>/dev/null || echo "  Directory not found"
    exit 1
fi

BATCH_SIZE=${BATCH_SIZE:-2}
MAX_SEQ_LENGTH=2048
EPOCHS=${BASE_EPOCHS}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./output/logs"
TRAIN_LOG="${LOG_DIR}/train_mimic_${SIZE_SUFFIX}_${TRAIN_PCT}pct_${TIMESTAMP}.log"

mkdir -p ${LOG_DIR}

echo "========================================================================"
echo "GPT-OSS 20B Fine-tuning - MIMIC Discharge Summaries"
echo "========================================================================"
echo "Start time: $(date)"
echo "Dataset size: ${SIZE_SUFFIX}"
echo "MIMIC JSON: ${MIMIC_JSON}"
echo "Training set: ${TRAIN_PCT}%"
echo "Batch size: ${BATCH_SIZE} | Epochs: ${EPOCHS} | Max seq length: ${MAX_SEQ_LENGTH}"
echo "Train log: ${TRAIN_LOG}"
echo "========================================================================"
echo ""

# Count records (approximate)
RECORD_COUNT=$(grep -c '"instruction"' "$MIMIC_JSON" 2>/dev/null || echo "unknown")
echo "Approximate record count: ${RECORD_COUNT}"
echo ""

echo "========================================================================"
echo "Fine-tuning GPT-OSS 20B on MIMIC dataset (${SIZE_SUFFIX}, ${TRAIN_PCT}%)"
echo "========================================================================"
echo ""

CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_DEVICES} ${PYTHON:-python} -u src/trainer_unsloth.py \
    --model "unsloth/gpt-oss-20b-unsloth-bnb-4bit" \
    --output-dir "./output/MIMIC/gpt-oss-20b_${TRAIN_PCT}pct_${TIMESTAMP}" \
    --mimic-json "${MIMIC_JSON}" \
    --train-pct ${TRAIN_PCT} \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --max-seq-length ${MAX_SEQ_LENGTH} \
    --learning-rate 2e-4 \
    --lora-r 16 \
    --lora-alpha 32 \
    --reasoning-effort medium \
    2>&1 | tee "${TRAIN_LOG}"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ Training failed with exit code ${TRAIN_EXIT_CODE}"
    echo "Check log: ${TRAIN_LOG}"
    exit $TRAIN_EXIT_CODE
fi

echo ""
echo "✓ Training completed successfully!"
echo ""
