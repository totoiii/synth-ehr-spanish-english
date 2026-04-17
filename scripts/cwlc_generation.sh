#!/bin/bash
# =============================================================================
# CWLC Generation — GPU 1
# Generates synthetic text for all CWLC merged models (5%, 25%, 50%, 100%)
# =============================================================================
#
# Usage:
#   bash scripts/cwlc_generation.sh
#
# Runs on GPU 1 (free Ada/Blackwell). Uses the latest merged model for each %.

set -e
set -o pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=1
GPU_ID=1
PERCENTAGES=(100)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

VENV_PYTHON="${PROJECT_DIR}/.venv-vllm/bin/python"
GEN_SCRIPT="${PROJECT_DIR}/src/generate_all_vllm.py"

LOG_DIR="./output/logs"
mkdir -p "${LOG_DIR}"

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  CWLC Generation — GPU ${GPU_ID}                                       ║"
echo "║  Percentages: ${PERCENTAGES[*]}                                   ║"
echo "║  Started: $(date)                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# ─── Generation parameters (max_tokens=512 = finetuning max_seq_length) ─────
# ┌──────────┬──────────┬───────────┬────────────┬───────────────┬──────────────┐
# │ Dataset  │ Prompts  │ N_OUTPUTS │ max_tokens │ max_model_len │ max_num_seqs │
# ├──────────┼──────────┼───────────┼────────────┼───────────────┼──────────────┤
# │ CWLC     │  9,000   │     5     │    512     │     1024      │     50       │
# └──────────┴──────────┴───────────┴────────────┴───────────────┴──────────────┘

for PCT in "${PERCENTAGES[@]}"; do
    MERGED_DIR=$(ls -td output/CWLC/gpt-oss-20b_${PCT}pct_*_merged 2>/dev/null | head -1)
    if [ -z "$MERGED_DIR" ]; then
        echo "⚠️  No merged model found for CWLC ${PCT}%, skipping..."
        continue
    fi

    TS=$(date +%Y%m%d_%H%M%S)
    OUTPUT_FILE="output/CWLC/synth_cwlc_${PCT}pct_${TS}.json"
    GEN_LOG="${LOG_DIR}/gen_cwlc_${PCT}pct_${TS}.log"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  CWLC ${PCT}% — $(date)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Model: ${MERGED_DIR}"
    echo "  Output: ${OUTPUT_FILE}"
    echo "  Log: ${GEN_LOG}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} "${VENV_PYTHON}" "${GEN_SCRIPT}" \
        --dataset cwlc \
        --model-path "${MERGED_DIR}" \
        --output-file "${OUTPUT_FILE}" \
        --cwlc-zip "./data/cwlc.zip" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.90 \
        --max-model-len 1024 \
        --max-num-seqs 80 \
        --n 5 \
        --max-tokens 256 \
        --temperature 1.0 \
        --top-p 1.0 \
        --repetition-penalty 1.5 \
        --save-every-n 500 \
        2>&1 | tee "${GEN_LOG}"

    GEN_EXIT=${PIPESTATUS[0]}
    if [ $GEN_EXIT -ne 0 ]; then
        echo "❌ Generation failed for CWLC ${PCT}%! Check: ${GEN_LOG}"
        exit $GEN_EXIT
    fi
    echo "✅ Generated CWLC ${PCT}% → ${OUTPUT_FILE}"
    echo ""
done

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  🎉 CWLC Generation Complete!                                     ║"
echo "║  Finished: $(date)                                                 ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Generated files:"
ls -lh output/CWLC/synth_cwlc_*pct_*.json 2>/dev/null
