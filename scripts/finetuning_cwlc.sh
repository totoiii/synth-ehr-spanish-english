#!/bin/bash
#
# Fine-tune GPT-OSS 20B on CWLC dataset
# Dataset: CWLC (Clinical notes with medical entity annotations, from ZIP)
#
# Dataset profile:
#   - 9,000 records (paired .txt + .ann)
#   - Spanish clinical notes, very short: Median ~54 tokens, Max ~721
#
# Data file should be at:
#   - data/cwlc.zip (or already extracted in data/cwlc/)
#
# Usage:
#   bash scripts/finetuning_cwlc.sh [TRAIN_PCT]
#
# Examples:
#   bash scripts/finetuning_cwlc.sh         # 100% of training data
#   bash scripts/finetuning_cwlc.sh 50      # 50% of training data
#   bash scripts/finetuning_cwlc.sh 10      # 10% of training data

set -e
set -o pipefail

# Activate conda environment
source /home/lmiranda/miniconda3/etc/profile.d/conda.sh
conda activate unsloth-test

# Prefer stable GPU (Ada 48GB) unless user overrides.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"1"}

# Training set percentage (default: 100)
TRAIN_PCT="${1:-100}"

# ── Adaptive hyperparameters ──
# CWLC texts are very short (median 54 tokens, max 721)
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_SEQ_LENGTH=256
EPOCHS=20

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./output/logs"
TRAIN_LOG="${LOG_DIR}/train_cwlc_${TRAIN_PCT}pct_${TIMESTAMP}.log"

# CWLC data path
CWLC_ZIP="./data/cwlc.zip"

if [ ! -f "$CWLC_ZIP" ]; then
    echo "❌ ERROR: CWLC ZIP file not found: $CWLC_ZIP"
    echo ""
    echo "Expected location: ./data/cwlc.zip"
    echo "Available files in data/:"
    ls -la ./data/ 2>/dev/null || echo "  Directory not found"
    exit 1
fi

mkdir -p ${LOG_DIR}

echo "========================================================================"
echo "GPT-OSS 20B Fine-tuning - CWLC Dataset"
echo "========================================================================"
echo "Start time: $(date)"
echo "Train log: ${TRAIN_LOG}"
echo "CWLC ZIP: ${CWLC_ZIP}"
echo "Training set: ${TRAIN_PCT}%"
echo "Batch size: ${BATCH_SIZE} | Epochs: ${EPOCHS} | Max seq length: ${MAX_SEQ_LENGTH}"
echo "========================================================================"
echo ""

echo "========================================================================"
echo "Fine-tuning GPT-OSS 20B on CWLC dataset (${TRAIN_PCT}%)"
echo "========================================================================"
echo ""

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${PYTHON:-python} -u src/trainer_unsloth.py \
    --model "unsloth/gpt-oss-20b-unsloth-bnb-4bit" \
    --output-dir "./output/CWLC/gpt-oss-20b_${TRAIN_PCT}pct_${TIMESTAMP}" \
    --dataset "local_cwlc" \
    --cwlc-zip "${CWLC_ZIP}" \
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

