#!/bin/bash
# =============================================================================
# CONTINUED FINETUNING — Resume training from existing LoRA adapters
# Trains for additional epochs with lower learning rate to reduce loss
# =============================================================================
#
# Usage:
#   # Train all datasets on GPU 0:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/continued_finetuning.sh
#
#   # Train specific dataset:
#   DATASETS="CWLC" CUDA_VISIBLE_DEVICES=0 bash scripts/continued_finetuning.sh
#
#   # Train specific percentage:
#   PERCENTAGES="100" DATASETS="CARES" bash scripts/continued_finetuning.sh
#
#   # Background run:
#   nohup bash scripts/continued_finetuning.sh > output/logs/continued_finetuning.log 2>&1 &

set -e
set -o pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
EXTRA_EPOCHS="${EXTRA_EPOCHS:-5}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
DATASETS="${DATASETS:-CWLC CARES RADGRAPH MIMIC}"
PERCENTAGES="${PERCENTAGES:-5 25 50 100}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Activate conda
source /home/lmiranda/miniconda3/etc/profile.d/conda.sh
conda activate unsloth-test

LOG_DIR="./output/logs"
mkdir -p "${LOG_DIR}"

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  CONTINUED FINETUNING — Additional ${EXTRA_EPOCHS} epochs                       ║"
echo "║  Datasets: ${DATASETS}                                             "
echo "║  Percentages: ${PERCENTAGES}                                       "
echo "║  Learning rate: ${LEARNING_RATE} (lower for stability)             "
echo "║  GPU: ${CUDA_VISIBLE_DEVICES:-auto}                                "
echo "║  Started: $(date)                                                  "
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# ─── Dataset-specific configuration ─────────────────────────────────────────
get_dataset_config() {
    local dataset="$1"
    case "$dataset" in
        CWLC)
            DATASET_FLAG="local_cwlc"
            DATA_ARGS="--cwlc-zip ./data/cwlc.zip"
            MAX_SEQ_LENGTH=512
            ;;
        CARES)
            DATASET_FLAG="chizhikchi/CARES"
            DATA_ARGS=""
            MAX_SEQ_LENGTH=1024
            ;;
        RADGRAPH)
            DATASET_FLAG="local_radgraph"
            DATA_ARGS="--radgraph-jsonl ./data/RADGRAPH/mimic-radgraph-XL.jsonl ./data/RADGRAPH/stanford-radgraph-XL.jsonl"
            MAX_SEQ_LENGTH=1536
            ;;
        MIMIC)
            DATASET_FLAG="local_mimic"
            DATA_ARGS="--mimic-json ./data/mimic/mimic_s.json"
            MAX_SEQ_LENGTH=4096
            ;;
        *)
            echo "❌ Unknown dataset: $dataset"
            return 1
            ;;
    esac
}

# ─── Main loop ───────────────────────────────────────────────────────────────
TOTAL=0
FAILED=0

for DATASET in ${DATASETS}; do
    get_dataset_config "$DATASET"

    for PCT in ${PERCENTAGES}; do
        # Find the latest adapter for this dataset/percentage
        ADAPTER_DIR=$(ls -td output/${DATASET}/gpt-oss-20b_${PCT}pct_*/final_model 2>/dev/null | head -1)

        if [ -z "$ADAPTER_DIR" ]; then
            echo "⚠️  No adapter found for ${DATASET} ${PCT}%, skipping..."
            continue
        fi

        PARENT_DIR=$(dirname "$ADAPTER_DIR")
        MODEL_NAME=$(basename "$PARENT_DIR")
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        OUTPUT_DIR="./output/${DATASET}/gpt-oss-20b_${PCT}pct_continued_${TIMESTAMP}"
        TRAIN_LOG="${LOG_DIR}/continued_${DATASET,,}_${PCT}pct_${TIMESTAMP}.log"

        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  ${DATASET} ${PCT}% — Continuing from: ${MODEL_NAME}"
        echo "  Adapter: ${ADAPTER_DIR}"
        echo "  Output:  ${OUTPUT_DIR}"
        echo "  Epochs:  ${EXTRA_EPOCHS} | LR: ${LEARNING_RATE} | MaxSeq: ${MAX_SEQ_LENGTH}"
        echo "  Log:     ${TRAIN_LOG}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        python -u src/trainer_unsloth.py \
            --model "unsloth/gpt-oss-20b-unsloth-bnb-4bit" \
            --output-dir "${OUTPUT_DIR}" \
            --dataset "${DATASET_FLAG}" \
            ${DATA_ARGS} \
            --train-pct ${PCT} \
            --batch-size ${BATCH_SIZE} \
            --epochs ${EXTRA_EPOCHS} \
            --max-seq-length ${MAX_SEQ_LENGTH} \
            --learning-rate ${LEARNING_RATE} \
            --lora-r 16 \
            --lora-alpha 32 \
            --reasoning-effort medium \
            --resume-from-adapter "${ADAPTER_DIR}" \
            2>&1 | tee "${TRAIN_LOG}"

        TRAIN_EXIT=${PIPESTATUS[0]}
        TOTAL=$((TOTAL + 1))

        if [ $TRAIN_EXIT -ne 0 ]; then
            echo "❌ Continued training failed for ${DATASET} ${PCT}%!"
            echo "   Check: ${TRAIN_LOG}"
            FAILED=$((FAILED + 1))
            continue
        fi

        echo "✅ Continued training done: ${DATASET} ${PCT}%"
        echo "   New adapter: ${OUTPUT_DIR}/final_model"
        echo ""
    done
done

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Continued Finetuning Complete!                                    ║"
echo "║  Total: ${TOTAL}, Failed: ${FAILED}                                "
echo "║  Finished: $(date)                                                 ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Merge new adapters:  use the merge phase of mega pipeline scripts"
echo "  2. Generate with:       START_PHASE=3 bash scripts/finetuning_and_synthesis_*.sh"
echo ""

if [ $FAILED -gt 0 ]; then
    exit 1
fi
