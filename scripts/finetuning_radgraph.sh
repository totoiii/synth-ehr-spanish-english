#!/bin/bash
#
# Fine-tune GPT-OSS 20B on RadGraph-XL dataset
# Dataset: RadGraph-XL (MIMIC and/or Stanford radiology reports)
#
# Dataset profile:
#   - 2,300 records (300 MIMIC + 2,000 Stanford)
#   - English radiology reports, Median ~578 tokens, P99 ~1,236
#
# Data files should be at:
#   - data/RADGRAPH/mimic-radgraph-XL.jsonl
#   - data/RADGRAPH/stanford-radgraph-XL.jsonl (optional)
#
# Usage:
#   bash scripts/finetuning_radgraph.sh [TRAIN_PCT]
#
# Examples:
#   bash scripts/finetuning_radgraph.sh         # 100% of training data
#   bash scripts/finetuning_radgraph.sh 50      # 50% of training data

set -e
set -o pipefail

# Activate conda environment
source /home/lmiranda/miniconda3/etc/profile.d/conda.sh
conda activate unsloth-test

# Prefer stable GPU (Ada 48GB) unless user overrides.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}

# Training set percentage (default: 100)
TRAIN_PCT="${1:-100}"

# ── Adaptive hyperparameters ──
# RadGraph texts are moderate length (P99 ~1,236 tokens)
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_SEQ_LENGTH=1536
EPOCHS=5

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./output/logs"
TRAIN_LOG="${LOG_DIR}/train_radgraph_${TRAIN_PCT}pct_${TIMESTAMP}.log"

# RadGraph-XL data paths
RADGRAPH_DIR="./data/RADGRAPH"
MIMIC_RADGRAPH="${RADGRAPH_DIR}/mimic-radgraph-XL.jsonl"
STANFORD_RADGRAPH="${RADGRAPH_DIR}/stanford-radgraph-XL.jsonl"

# Build list of available JSONL files
RADGRAPH_FILES=()
if [ -f "$MIMIC_RADGRAPH" ]; then
    RADGRAPH_FILES+=("$MIMIC_RADGRAPH")
    echo "✓ Found MIMIC RadGraph-XL: $MIMIC_RADGRAPH"
fi
if [ -f "$STANFORD_RADGRAPH" ]; then
    RADGRAPH_FILES+=("$STANFORD_RADGRAPH")
    echo "✓ Found Stanford RadGraph-XL: $STANFORD_RADGRAPH"
fi

if [ ${#RADGRAPH_FILES[@]} -eq 0 ]; then
    echo "❌ ERROR: No RadGraph-XL JSONL files found!"
    echo "Expected locations:"
    echo "  - $MIMIC_RADGRAPH"
    echo "  - $STANFORD_RADGRAPH"
    echo ""
    echo "Please download the RadGraph-XL dataset from PhysioNet."
    exit 1
fi

mkdir -p ${LOG_DIR}

echo "========================================================================"
echo "GPT-OSS 20B Fine-tuning - RadGraph-XL"
echo "========================================================================"
echo "Start time: $(date)"
echo "Train log: ${TRAIN_LOG}"
echo "RadGraph files: ${RADGRAPH_FILES[*]}"
echo "Training set: ${TRAIN_PCT}%"
echo "Batch size: ${BATCH_SIZE} | Epochs: ${EPOCHS} | Max seq length: ${MAX_SEQ_LENGTH}"
echo "========================================================================"
echo ""

echo "========================================================================"
echo "Fine-tuning GPT-OSS 20B on RadGraph-XL dataset (${TRAIN_PCT}%)"
echo "========================================================================"
echo ""

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${PYTHON:-python} -u src/trainer_unsloth.py \
    --model "unsloth/gpt-oss-20b-unsloth-bnb-4bit" \
    --output-dir "./output/RADGRAPH/gpt-oss-20b_${TRAIN_PCT}pct_${TIMESTAMP}" \
    --radgraph-jsonl "${RADGRAPH_FILES[@]}" \
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
