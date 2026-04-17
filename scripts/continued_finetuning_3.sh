#!/bin/bash
# =============================================================================
# CONTINUED FINETUNING 3 — CWLC (GPU 2, Ada 49GB) + RADGRAPH (GPU 0, Ada 49GB)
# =============================================================================
#
# Loss analysis (100%):
#   CWLC: 0.96 (5 epochs) → +3 extra epochs (already lowest loss, just polish)
#   RADGRAPH: 1.05 (5 epochs)  → +3 extra epochs (already decent)
#
# Usage:
#   screen -S cont_ft_3
#   bash scripts/continued_finetuning_3.sh
#   # Ctrl+A D to detach

set -e
set -o pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
GPU_ID=1
LEARNING_RATE="5e-5"
BATCH_SIZE_CWLC="${BATCH_SIZE_CWLC:-16}"
BATCH_SIZE_RADGRAPH="${BATCH_SIZE_RADGRAPH:-16}"
RADGRAPH_EPOCHS=12
PERCENTAGES=(5 25 50 100)
CWLC_EPOCHS=12

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

source /home/lmiranda/miniconda3/etc/profile.d/conda.sh
conda activate unsloth-test

LOG_DIR="./output/logs"
mkdir -p "${LOG_DIR}"

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  CONTINUED FINETUNING 3 — CWLC                                    ║"
echo "║  GPU: ${GPU_ID} (Ada 49GB)                                                ║"
echo "║  CWLC: +${CWLC_EPOCHS} epochs (current loss: ~0.96) | RADGRAPH: +${RADGRAPH_EPOCHS} epochs    ║"
echo "║  Learning rate: ${LEARNING_RATE}                                           ║"
echo "║  Percentages: ${PERCENTAGES[*]}                                            ║"
echo "║  Started: $(date)                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

TOTAL=0
FAILED=0

for PCT in "${PERCENTAGES[@]}"; do
    ADAPTER_DIR=$(ls -td output/CWLC/gpt-oss-20b_${PCT}pct_continued*/final_model 2>/dev/null | grep -v "continued_" | head -1)

    if [ -z "$ADAPTER_DIR" ]; then
        echo "⚠️  No adapter found for CWLC ${PCT}%, skipping..."
        continue
    fi

    MODEL_NAME=$(basename "$(dirname "$ADAPTER_DIR")")
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="./output/CWLC/gpt-oss-20b_${PCT}pct_continued_${TIMESTAMP}"
    TRAIN_LOG="${LOG_DIR}/continued_cwlc_${PCT}pct_${TIMESTAMP}.log"

    echo "━━━ CWLC ${PCT}% — Resuming from: ${MODEL_NAME} ━━━"
    echo "  Adapter: ${ADAPTER_DIR}"
    echo "  Output:  ${OUTPUT_DIR}"
    echo "  Epochs:  ${CWLC_EPOCHS} | LR: ${LEARNING_RATE} | MaxSeq: 512"

    CUDA_VISIBLE_DEVICES=${GPU_ID} python -u src/trainer_unsloth.py \
        --model "unsloth/gpt-oss-20b-unsloth-bnb-4bit" \
        --output-dir "${OUTPUT_DIR}" \
        --dataset "local_cwlc" \
        --cwlc-zip "./data/cwlc.zip" \
        --train-pct ${PCT} \
        --batch-size ${BATCH_SIZE_CWLC} \
        --epochs ${CWLC_EPOCHS} \
        --max-seq-length 512 \
        --learning-rate ${LEARNING_RATE} \
        --lora-r 16 \
        --lora-alpha 32 \
        --reasoning-effort medium \
        --resume-from-adapter "${ADAPTER_DIR}" \
        2>&1 | tee "${TRAIN_LOG}"

    TRAIN_EXIT=${PIPESTATUS[0]}
    TOTAL=$((TOTAL + 1))

    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "❌ Failed: CWLC ${PCT}%! Check: ${TRAIN_LOG}"
        FAILED=$((FAILED + 1))
        continue
    fi
    echo "✅ CWLC ${PCT}% done → ${OUTPUT_DIR}/final_model"
    echo ""
done

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  🎉 Continued Finetuning 3 Complete (CWLC)                        ║"
echo "║  Total: ${TOTAL}, Failed: ${FAILED}                                        ║"
echo "║  Finished: $(date)                                                 ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

if [ $FAILED -gt 0 ]; then exit 1; fi



# ─── RADGRAPH ────────────────────────────────────────────────────────────────
echo ""
echo "██████████████████████████████████████████████████████████████████████"
echo "█  RADGRAPH — +${RADGRAPH_EPOCHS} epochs (current loss: ~1.05)                          █"
echo "██████████████████████████████████████████████████████████████████████"
echo ""

for PCT in "${PERCENTAGES[@]}"; do
    ADAPTER_DIR=$(ls -td output/RADGRAPH/gpt-oss-20b_continued${PCT}pct_*/final_model 2>/dev/null | head -1)

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

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  🎉 Continued Finetuning 1 Complete (CARES + RADGRAPH)            ║"
echo "║  Total: ${TOTAL}, Failed: ${FAILED}                                        ║"
echo "║  Finished: $(date)                                                 ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

if [ $FAILED -gt 0 ]; then exit 1; fi
