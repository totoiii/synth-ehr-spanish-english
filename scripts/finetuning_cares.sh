#!/bin/bash
#
# Fine-tune GPT-OSS 20B on CARES dataset
#
# Dataset profile:
#   - 2,253 training examples (Spanish clinical notes)
#   - Median ~402 tokens, P95 ~917, Max ~1,956
#
# Usage:
#   bash scripts/finetuning_cares.sh [TRAIN_PCT]
#
# Examples:
#   bash scripts/finetuning_cares.sh          # 100% of training data
#   bash scripts/finetuning_cares.sh 50       # 50% of training data
#   bash scripts/finetuning_cares.sh 10       # 10% of training data

set -e
set -o pipefail

# Activate conda environment
source /home/lmiranda/miniconda3/etc/profile.d/conda.sh
conda activate unsloth-test

# Prefer stable GPU (Ada 48GB) unless user overrides.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}

# Training set percentage (default: 100)
TRAIN_PCT="${1:-100}"

# ── Adaptive hyperparameters based on TRAIN_PCT ──
BATCH_SIZE=${BATCH_SIZE:-2}
MAX_SEQ_LENGTH=512
EPOCHS=20

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./output/logs"
TRAIN_LOG="${LOG_DIR}/train_cares_${TRAIN_PCT}pct_${TIMESTAMP}.log"
DATASET="chizhikchi/CARES"

mkdir -p ${LOG_DIR}

echo "========================================================================"
echo "GPT-OSS 20B Fine-tuning - CARES Dataset"
echo "========================================================================"
echo "Start time: $(date)"
echo "Train log: ${TRAIN_LOG}"
echo "Training set: ${TRAIN_PCT}%"
echo "Batch size: ${BATCH_SIZE} | Epochs: ${EPOCHS} | Max seq length: ${MAX_SEQ_LENGTH}"
echo "========================================================================"
echo ""

echo "========================================================================"
echo "Fine-tuning GPT-OSS 20B on $DATASET dataset (${TRAIN_PCT}%)"
echo "========================================================================"
echo ""

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${PYTHON:-python} -u src/trainer_unsloth.py \
    --model "unsloth/gpt-oss-20b-unsloth-bnb-4bit" \
    --output-dir "./output/CARES/gpt-oss-20b_${TRAIN_PCT}pct_${TIMESTAMP}" \
    --dataset "$DATASET" \
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
