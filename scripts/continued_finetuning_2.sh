#!/bin/bash
# =============================================================================
# CONTINUED FINETUNING 2 — MIMIC small + large (GPU 1, Blackwell 98GB)
# =============================================================================
#
# Loss analysis (100%):
#   MIMIC-small: 1.91 (3 epochs)  → +3 extra epochs
#   MIMIC-large: 1.79 (1 epoch)   → +3 extra epochs (benefits most from extra training)
#
# MIMIC has 2 adapters per percentage (small=older, large=newer by timestamp).
#
# Usage:
#   screen -S cont_ft_2
#   bash scripts/continued_finetuning_2.sh
#   # Ctrl+A D to detach

set -e
set -o pipefail

# ─── Trap for cleanup on Ctrl+C ──────────────────────────────────────────────
# When the script receives SIGINT or SIGTERM, kill all spawned child processes
# (e.g., Python training loop and PyTorch dataloaders) so they don't hang.
cleanup() {
    echo -e "\n🛑 Interrupted! Cleaning up child processes..."
    # -P $$ sends the signal to all direct children of this bash script
    pkill -P $$ 2>/dev/null || true
    sleep 1
    pkill -9 -P $$ 2>/dev/null || true
    exit 1
}
trap cleanup SIGINT SIGTERM

# ─── Configuration ───────────────────────────────────────────────────────────
GPU_ID=0,1,2
LEARNING_RATE="5e-5"
BATCH_SIZE="${BATCH_SIZE:-32}"
PERCENTAGES=(100)
MIMIC_EPOCHS=3

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

source /home/lmiranda/miniconda3/etc/profile.d/conda.sh
conda activate unsloth-test

LOG_DIR="./output/logs"
mkdir -p "${LOG_DIR}"

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  CONTINUED FINETUNING 2 — MIMIC (small + large)                   ║"
echo "║  GPU: ${GPU_ID} (Blackwell 98GB)                                          ║"
echo "║  MIMIC: +${MIMIC_EPOCHS} epochs (small + large)                              ║"
echo "║  Learning rate: ${LEARNING_RATE}                                           ║"
echo "║  Percentages: ${PERCENTAGES[*]}                                            ║"
echo "║  Started: $(date)                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

TOTAL=0
FAILED=0

# For each percentage, MIMIC has 2 adapters sorted by timestamp:
#   - Older = small (trained on mimic_s.json)
#   - Newer = large (trained on mimic_l.json)
# We sort oldest-first with 'tac' to process small first, large second.

for PCT in "${PERCENTAGES[@]}"; do
    # Get all adapter dirs for this PCT, sorted oldest first
    # (ls -td gives newest first, tac reverses to oldest first)
    ALL_ADAPTERS=($(ls -td output/MIMIC/gpt-oss-20b_${PCT}pct_*/final_model 2>/dev/null | grep -v "test_\|continued_" | tac))

    if [ ${#ALL_ADAPTERS[@]} -lt 2 ]; then
        echo "⚠️  Expected 2 adapters for MIMIC ${PCT}% (small+large), found ${#ALL_ADAPTERS[@]}"
        echo "    Available: ${ALL_ADAPTERS[*]}"
    fi

    ## ── MIMIC-small (first/older adapter) ──
    #if [ ${#ALL_ADAPTERS[@]} -ge 1 ]; then
    #    ADAPTER_DIR="${ALL_ADAPTERS[0]}"
    #    MODEL_NAME=$(basename "$(dirname "$ADAPTER_DIR")")
    #    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    #    OUTPUT_DIR="./output/MIMIC/gpt-oss-20b_${PCT}pct_small_continued_${TIMESTAMP}"
    #    TRAIN_LOG="${LOG_DIR}/continued_mimic_small_${PCT}pct_${TIMESTAMP}.log"
#
    #    echo "━━━ MIMIC-small ${PCT}% — Resuming from: ${MODEL_NAME} ━━━"
    #    echo "  Adapter: ${ADAPTER_DIR}"
    #    echo "  Data:    data/mimic/mimic_s.json"
    #    echo "  Output:  ${OUTPUT_DIR}"
    #    echo "  Epochs:  ${MIMIC_EPOCHS} | LR: ${LEARNING_RATE} | MaxSeq: 4096"
#
    #    CUDA_VISIBLE_DEVICES=${GPU_ID} python -u src/trainer_unsloth.py \
    #        --model "unsloth/gpt-oss-20b-unsloth-bnb-4bit" \
    #        --output-dir "${OUTPUT_DIR}" \
    #        --dataset "local_mimic" \
    #        --mimic-json "./data/mimic/mimic_s.json" \
    #        --train-pct ${PCT} \
    #        --batch-size ${BATCH_SIZE} \
    #        --epochs ${MIMIC_EPOCHS} \
    #        --max-seq-length 4096 \
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
    #        echo "❌ Failed: MIMIC-small ${PCT}%! Check: ${TRAIN_LOG}"
    #        FAILED=$((FAILED + 1))
    #    else
    #        echo "✅ MIMIC-small ${PCT}% done → ${OUTPUT_DIR}/final_model"
    #    fi
    #    echo ""
    #fi

    # ── MIMIC-large (second/newer adapter) ──
    if [ ${#ALL_ADAPTERS[@]} -ge 2 ]; then
        ADAPTER_DIR="${ALL_ADAPTERS[1]}"
        MODEL_NAME=$(basename "$(dirname "$ADAPTER_DIR")")
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        OUTPUT_DIR="./output/MIMIC/gpt-oss-20b_${PCT}pct_large_continued_${TIMESTAMP}"
        TRAIN_LOG="${LOG_DIR}/continued_mimic_large_${PCT}pct_${TIMESTAMP}.log"

        echo "━━━ MIMIC-large ${PCT}% — Resuming from: ${MODEL_NAME} ━━━"
        echo "  Adapter: ${ADAPTER_DIR}"
        echo "  Data:    data/mimic/mimic_l.json"
        echo "  Output:  ${OUTPUT_DIR}"
        echo "  Epochs:  ${MIMIC_EPOCHS} | LR: ${LEARNING_RATE} | MaxSeq: 4096"

        CUDA_VISIBLE_DEVICES=${GPU_ID} torchrun --nproc_per_node=3 src/trainer_unsloth.py \
            --model "unsloth/gpt-oss-20b-unsloth-bnb-4bit" \
            --output-dir "${OUTPUT_DIR}" \
            --dataset "local_mimic" \
            --mimic-json "./data/mimic/mimic_l.json" \
            --train-pct ${PCT} \
            --batch-size ${BATCH_SIZE} \
            --epochs ${MIMIC_EPOCHS} \
            --max-seq-length 2048 \
            --learning-rate ${LEARNING_RATE} \
            --lora-r 16 \
            --lora-alpha 32 \
            --reasoning-effort medium \
            --resume-from-adapter "${ADAPTER_DIR}" \
            2>&1 | tee "${TRAIN_LOG}"

        TRAIN_EXIT=${PIPESTATUS[0]}
        TOTAL=$((TOTAL + 1))

        if [ $TRAIN_EXIT -ne 0 ]; then
            echo "❌ Failed: MIMIC-large ${PCT}%! Check: ${TRAIN_LOG}"
            FAILED=$((FAILED + 1))
        else
            echo "✅ MIMIC-large ${PCT}% done → ${OUTPUT_DIR}/final_model"
        fi
        echo ""
    fi
done

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  🎉 Continued Finetuning 2 Complete (MIMIC small + large)         ║"
echo "║  Total: ${TOTAL}, Failed: ${FAILED}                                        ║"
echo "║  Finished: $(date)                                                 ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

if [ $FAILED -gt 0 ]; then exit 1; fi
