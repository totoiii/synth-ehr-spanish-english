#!/bin/bash
# =============================================================================
# CONTINUED FINETUNING 1 — CARES 
# =============================================================================
#
# Loss analysis (100%):
#   CARES:    2.19 (3 epochs)  → +5 extra epochs (highest loss, needs most work)
#
# Usage:
#   screen -S cont_ft_1
#   bash scripts/continued_finetuning_1.sh
#   # Ctrl+A D to detach

set -e
set -o pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
GPU_ID=0
LEARNING_RATE="5e-5"
BATCH_SIZE="${BATCH_SIZE:-16}"
PERCENTAGES=(5 25 50 100)

# Extra epochs per dataset
CARES_EPOCHS=12


PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

source /home/lmiranda/miniconda3/etc/profile.d/conda.sh
conda activate unsloth-test

LOG_DIR="./output/logs"
mkdir -p "${LOG_DIR}"

#echo "╔══════════════════════════════════════════════════════════════════════╗"
#echo "║  CONTINUED FINETUNING 1 — CARES                        ║"
#echo "║  GPU: ${GPU_ID} (Ada 49GB)                                                ║"
#echo "║  CARES: +${CARES_EPOCHS} epochs                         ║"
#echo "║  Learning rate: ${LEARNING_RATE}                                           ║"
#echo "║  Percentages: ${PERCENTAGES[*]}                                            ║"
#echo "║  Started: $(date)                                                  ║"
#echo "╚══════════════════════════════════════════════════════════════════════╝"
#echo ""
#
TOTAL=0
FAILED=0
#
## ─── CARES ───────────────────────────────────────────────────────────────────
#echo ""
#echo "██████████████████████████████████████████████████████████████████████"
#echo "█  CARES — +${CARES_EPOCHS} epochs (current loss: ~2.19)                            █"
#echo "██████████████████████████████████████████████████████████████████████"
#echo ""
#
#for PCT in "${PERCENTAGES[@]}"; do
#    ADAPTER_DIR=$(ls -td output/CARES/gpt-oss-20b_${PCT}pct_continued*/final_model 2>/dev/null | head -1)
#
#    if [ -z "$ADAPTER_DIR" ]; then
#        echo "⚠️  No adapter found for CARES ${PCT}%, skipping..."
#        continue
#    fi
#
#    MODEL_NAME=$(basename "$(dirname "$ADAPTER_DIR")")
#    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
#    OUTPUT_DIR="./output/CARES/gpt-oss-20b_${PCT}pct_continued_${TIMESTAMP}"
#    TRAIN_LOG="${LOG_DIR}/continued_cares_${PCT}pct_${TIMESTAMP}.log"
#
#    echo "━━━ CARES ${PCT}% — Resuming from: ${MODEL_NAME} ━━━"
#    echo "  Adapter: ${ADAPTER_DIR}"
#    echo "  Output:  ${OUTPUT_DIR}"
#    echo "  Epochs:  ${CARES_EPOCHS} | LR: ${LEARNING_RATE} | MaxSeq: 1024"
#
#    CUDA_VISIBLE_DEVICES=${GPU_ID} python -u src/trainer_unsloth.py \
#        --model "unsloth/gpt-oss-20b-unsloth-bnb-4bit" \
#        --output-dir "${OUTPUT_DIR}" \
#        --dataset "chizhikchi/CARES" \
#        --train-pct ${PCT} \
#        --batch-size ${BATCH_SIZE} \
#        --epochs ${CARES_EPOCHS} \
#        --max-seq-length 1024 \
#        --learning-rate ${LEARNING_RATE} \
#        --lora-r 16 \
#        --lora-alpha 32 \
#        --reasoning-effort medium \
#        --resume-from-adapter "${ADAPTER_DIR}" \
#        2>&1 | tee "${TRAIN_LOG}"
#
#    TRAIN_EXIT=${PIPESTATUS[0]}
#    TOTAL=$((TOTAL + 1))
#
#    if [ $TRAIN_EXIT -ne 0 ]; then
#        echo "❌ Failed: CARES ${PCT}%! Check: ${TRAIN_LOG}"
#        FAILED=$((FAILED + 1))
#        continue
#    fi
#    echo "✅ CARES ${PCT}% done → ${OUTPUT_DIR}/final_model"
#    echo ""
#done

BATCH_SIZE_RADGRAPH="${BATCH_SIZE_RADGRAPH:-16}"
RADGRAPH_EPOCHS=12

# ─── RADGRAPH ────────────────────────────────────────────────────────────────
echo ""
echo "██████████████████████████████████████████████████████████████████████"
echo "█  RADGRAPH — +${RADGRAPH_EPOCHS} epochs (current loss: ~1.05)                          █"
echo "██████████████████████████████████████████████████████████████████████"
echo ""

for PCT in "${PERCENTAGES[@]}"; do
    ADAPTER_DIR=$(ls -td output/RADGRAPH/gpt-oss-20b_${PCT}pct_continued*/final_model 2>/dev/null | head -1)

    if [ -z "$ADAPTER_DIR" ]; then
        echo "⚠️  No adapter found for RADGRAPH ${PCT}%, skipping..."
        continue
    fi

    MODEL_NAME=$(basename "$(dirname "$ADAPTER_DIR")")
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="./output/RADGRAPH/gpt-oss-20b_${PCT}pct_continued_${TIMESTAMP}"
    TRAIN_LOG="${LOG_DIR}/continued_radgraph_${PCT}pct_${TIMESTAMP}.log"

    echo "━━━ RADGRAPH ${PCT}% — Resuming from: ${MODEL_NAME} ━━━"
    echo "  Adapter: ${ADAPTER_DIR}"
    echo "  Output:  ${OUTPUT_DIR}"
    echo "  Epochs:  ${RADGRAPH_EPOCHS} | LR: ${LEARNING_RATE} | MaxSeq: 1536"

    CUDA_VISIBLE_DEVICES=${GPU_ID} python -u src/trainer_unsloth.py \
        --model "unsloth/gpt-oss-20b-unsloth-bnb-4bit" \
        --output-dir "${OUTPUT_DIR}" \
        --dataset "local_radgraph" \
        --radgraph-jsonl "./data/RADGRAPH/mimic-radgraph-XL.jsonl" "./data/RADGRAPH/stanford-radgraph-XL.jsonl" \
        --train-pct ${PCT} \
        --batch-size ${BATCH_SIZE_RADGRAPH} \
        --epochs ${RADGRAPH_EPOCHS} \
        --max-seq-length 1536 \
        --learning-rate ${LEARNING_RATE} \
        --lora-r 16 \
        --lora-alpha 32 \
        --reasoning-effort medium \
        --resume-from-adapter "${ADAPTER_DIR}" \
        2>&1 | tee "${TRAIN_LOG}"

    TRAIN_EXIT=${PIPESTATUS[0]}
    TOTAL=$((TOTAL + 1))

    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "❌ Failed: RADGRAPH ${PCT}%! Check: ${TRAIN_LOG}"
        FAILED=$((FAILED + 1))
        continue
    fi
    echo "✅ RADGRAPH ${PCT}% done → ${OUTPUT_DIR}/final_model"
    echo ""
done
