#!/bin/bash
# =============================================================================
# MEGA PIPELINE 1 — GPU 0 (NVIDIA RTX 6000 Ada, 48 GB)
# Datasets: CARES, RADGRAPH, CWLC
# Phases:  Finetune → Merge → Generate
# =============================================================================
#
# Usage:
#  bash scripts/finetuning_and_synthesis_1.s
#
# To resume from a specific phase:
#   START_PHASE=2 bash scripts/finetuning_and_synthesis_1.sh   # skip finetuning
#   START_PHASE=3 bash scripts/finetuning_and_synthesis_1.sh   # skip to generation

set -e
set -o pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0
export TRAIN_CUDA_DEVICES=0
GPU_ID=0
PERCENTAGES=(100 50 25 5)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

START_PHASE="${START_PHASE:-1}"

# Activate conda
source /home/lmiranda/miniconda3/etc/profile.d/conda.sh
conda activate unsloth-test

VENV_PYTHON="${PROJECT_DIR}/.venv-vllm/bin/python"
GEN_SCRIPT="${PROJECT_DIR}/src/generate_all_vllm.py"

LOG_DIR="./output/logs"
mkdir -p "${LOG_DIR}"

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  MEGA PI3s: CARES, RADGRAPH, CWLC                                  ║"
echo "║  Percentages: ${PERCENTAGES[*]}                                   ║"
echo "║  Started: $(date)                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# ─── Helper to find latest model directory ───────────────────────────────────
find_latest_model() {
    local dataset_upper="$1"
    local pct="$2"
    ls -td output/${dataset_upper}/gpt-oss-20b_${pct}pct_*/final_model 2>/dev/null | head -1
}

# =============================================================================
# PHASE 1 — FINETUNING
# =============================================================================
if [ "$START_PHASE" -le 1 ]; then
echo ""
echo "██████████████████████████████████████████████████████████████████████"
echo "█  PHASE 1/3 — FINETUNING                                          █"
echo "██████████████████████████████████████████████████████████████████████"
echo ""

# ── CARES ──────────────────────────────────────────────────────────────────
for PCT in "${PERCENTAGES[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  CARES ${PCT}% — $(date)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    CUDA_VISIBLE_DEVICES=${GPU_ID} BATCH_SIZE=16 bash scripts/finetuning_cares.sh ${PCT}
    echo "✅ CARES ${PCT}% finetuning done"
    echo ""
done

## ── RADGRAPH ───────────────────────────────────────────────────────────────
#for PCT in "${PERCENTAGES[@]}"; do
#    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#    echo "  RADGRAPH ${PCT}% — $(date)"
#    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#    CUDA_VISIBLE_DEVICES=${GPU_ID} BATCH_SIZE=8 bash scripts/finetuning_radgraph.sh ${PCT}
#    echo "✅ RADGRAPH ${PCT}% finetuning done"
#    echo ""
#done
#
## ── CWLC ───────────────────────────────────────────────────────────────────
#for PCT in "${PERCENTAGES[@]}"; do
#    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#    echo "  CWLC ${PCT}% — $(date)"
#    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#    CUDA_VISIBLE_DEVICES=${GPU_ID} BATCH_SIZE=16 bash scripts/finetuning_cwlc.sh ${PCT}
#    echo "✅ CWLC ${PCT}% finetuning done"
#    echo ""
#done

echo ""
echo "✅ PHASE 1 COMPLETE — All finetuning done!"
echo ""
fi  # end PHASE 1

# =============================================================================
# PHASE 2 — MERGE LoRA MODELS
# =============================================================================
if [ "$START_PHASE" -le 2 ]; then
echo ""
echo "██████████████████████████████████████████████████████████████████████"
echo "█  PHASE 2/3 — MERGE LoRA MODELS                                   █"
echo "██████████████████████████████████████████████████████████████████████"
echo ""

for DATASET_NAME in CARES; do
    for PCT in "${PERCENTAGES[@]}"; do
        LORA_PATH=$(find_latest_model "$DATASET_NAME" "$PCT")
        if [ -z "$LORA_PATH" ]; then
            echo "⚠️  No model found for ${DATASET_NAME} ${PCT}%, skipping..."
            continue
        fi
        MODEL_DIR=$(dirname "$LORA_PATH")
        MERGED_DIR="${MODEL_DIR}_merged"
        TS=$(date +%Y%m%d_%H%M%S)
        MERGE_LOG="${LOG_DIR}/merge_${DATASET_NAME,,}_${PCT}pct_${TS}.log"

        echo "━━━ Merging ${DATASET_NAME} ${PCT}% ━━━"
        echo "  LoRA: ${LORA_PATH}"
        echo "  Output: ${MERGED_DIR}"
        echo "  Log: ${MERGE_LOG}"

        CUDA_VISIBLE_DEVICES=${GPU_ID} python -u src/merge_lora_model.py \
            --lora-path "${LORA_PATH}" \
            --output-path "${MERGED_DIR}" \
            --base-model "unsloth/gpt-oss-20b" \
            --dtype bfloat16 \
            --device-map auto \
            2>&1 | tee "${MERGE_LOG}"

        MERGE_EXIT=${PIPESTATUS[0]}
        if [ $MERGE_EXIT -ne 0 ]; then
            echo "❌ Merge failed for ${DATASET_NAME} ${PCT}%! Check: ${MERGE_LOG}"
            exit $MERGE_EXIT
        fi

        echo "✅ Merged ${DATASET_NAME} ${PCT}%"
        echo ""
    done
done

echo ""
echo "✅ PHASE 2 COMPLETE — All models merged!"
echo ""
fi  # end PHASE 2

# =============================================================================
# PHASE 3 — GENERATION (vLLM)
# =============================================================================
if [ "$START_PHASE" -le 3 ]; then
echo ""
echo "██████████████████████████████████████████████████████████████████████"
echo "█  PHASE 3/3 — TEXT GENERATION (vLLM)                               █"
echo "██████████████████████████████████████████████████████████████████████"
echo ""

# Generation parameters per dataset (n=1, max_tokens = finetuning max_seq_length):
# ┌────────────┬──────────┬───────────┬──────────────┬────────────┬───────────────┬──────────────┐
# │ Dataset    │ Prompts  │ N_OUTPUTS │ Total Gen    │ max_tokens │ max_model_len │ max_num_seqs │
# ├────────────┼──────────┼───────────┼──────────────┼────────────┼───────────────┼──────────────┤
# │ CARES      │  2,253   │     1     │    2,253     │   1024     │     2048      │     25       │
# │ RADGRAPH   │  2,300   │     1     │    2,300     │   1536     │     8192      │     16       │
# │ CWLC       │  9,000   │     1     │    9,000     │    256     │     1024      │     80       │
# └────────────┴──────────┴───────────┴──────────────┴────────────┴───────────────┴──────────────┘

for PCT in "${PERCENTAGES[@]}"; do
    # ── CARES ──
    MERGED_DIR=$(ls -td output/CARES/gpt-oss-20b_${PCT}pct_*_merged 2>/dev/null | head -1)
    if [ -n "$MERGED_DIR" ]; then
        TS=$(date +%Y%m%d_%H%M%S)
        OUTPUT_FILE="output/CARES/synth_cares_${PCT}pct_${TS}.json"
        GEN_LOG="${LOG_DIR}/gen_cares_${PCT}pct_${TS}.log"
        echo "━━━ Generating CARES ${PCT}% — $(date) ━━━"
        echo "  Model: ${MERGED_DIR}"
        echo "  Output: ${OUTPUT_FILE}"
        echo "  Log: ${GEN_LOG}"

        CUDA_VISIBLE_DEVICES=${GPU_ID} "${VENV_PYTHON}" "${GEN_SCRIPT}" \
            --dataset cares \
            --model-path "${MERGED_DIR}" \
            --output-file "${OUTPUT_FILE}" \
            --tensor-parallel-size 1 \
            --gpu-memory-utilization 0.90 \
            --max-model-len 1024 \
            --max-num-seqs 25 \
            --n 1 \
            --max-tokens 512 \
            --temperature 1.0 \
            --top-p 1.0 \
            --repetition-penalty 1.1 \
            2>&1 | tee "${GEN_LOG}"

        GEN_EXIT=${PIPESTATUS[0]}
        if [ $GEN_EXIT -ne 0 ]; then
            echo "❌ Generation failed for CARES ${PCT}%! Check: ${GEN_LOG}"
            exit $GEN_EXIT
        fi
        echo "✅ Generated CARES ${PCT}% → ${OUTPUT_FILE}"
        echo ""
    fi

    ### ── RADGRAPH ──
    ##MERGED_DIR=$(ls -td output/RADGRAPH/gpt-oss-20b_${PCT}pct_*_merged 2>/dev/null | head -1)
    ##if [ -n "$MERGED_DIR" ]; then
    ##    TS=$(date +%Y%m%d_%H%M%S)
    ##    OUTPUT_FILE="output/RADGRAPH/synth_radgraph_${PCT}pct_${TS}.json"
    ##    GEN_LOG="${LOG_DIR}/gen_radgraph_${PCT}pct_${TS}.log"
    ##    echo "━━━ Generating RADGRAPH ${PCT}% — $(date) ━━━"
    ##    echo "  Model: ${MERGED_DIR}"
    ##    echo "  Output: ${OUTPUT_FILE}"
    ##    echo "  Log: ${GEN_LOG}"
##
    ##    CUDA_VISIBLE_DEVICES=${GPU_ID} "${VENV_PYTHON}" "${GEN_SCRIPT}" \
    ##        --dataset radgraph \
    ##        --model-path "${MERGED_DIR}" \
    ##        --output-file "${OUTPUT_FILE}" \
    ##        --radgraph-jsonl "./data/RADGRAPH/mimic-radgraph-XL.jsonl" "./data/RADGRAPH/stanford-radgraph-XL.jsonl" \
    ##        --tensor-parallel-size 1 \
    ##        --gpu-memory-utilization 0.90 \
    ##        --max-model-len 8192 \
    ##        --max-num-seqs 16 \
    ##        --n 1 \
    ##        --max-tokens 1536 \
    ##        --temperature 1.0 \
    ##        --top-p 1.0 \
    ##        --repetition-penalty 1.1 \
    ##        2>&1 | tee "${GEN_LOG}"
##
    ##    GEN_EXIT=${PIPESTATUS[0]}
    ##    if [ $GEN_EXIT -ne 0 ]; then
    ##        echo "❌ Generation failed for RADGRAPH ${PCT}%! Check: ${GEN_LOG}"
    ##        exit $GEN_EXIT
    ##    fi
    ##    echo "✅ Generated RADGRAPH ${PCT}% → ${OUTPUT_FILE}"
    ##    echo ""
    ##fi
##
    ### ── CWLC ──
    ##MERGED_DIR=$(ls -td output/CWLC/gpt-oss-20b_${PCT}pct_*_merged 2>/dev/null | head -1)
    ##if [ -n "$MERGED_DIR" ]; then
    ##    TS=$(date +%Y%m%d_%H%M%S)
    ##    OUTPUT_FILE="output/CWLC/synth_cwlc_${PCT}pct_${TS}.json"
    ##    GEN_LOG="${LOG_DIR}/gen_cwlc_${PCT}pct_${TS}.log"
    ##    echo "━━━ Generating CWLC ${PCT}% — $(date) ━━━"
    ##    echo "  Model: ${MERGED_DIR}"
    ##    echo "  Output: ${OUTPUT_FILE}"
    ##    echo "  Log: ${GEN_LOG}"
##
    ##    CUDA_VISIBLE_DEVICES=${GPU_ID} "${VENV_PYTHON}" "${GEN_SCRIPT}" \
    ##        --dataset cwlc \
    ##        --model-path "${MERGED_DIR}" \
    ##        --output-file "${OUTPUT_FILE}" \
    ##        --cwlc-zip "./data/cwlc.zip" \
    ##        --tensor-parallel-size 1 \
    ##        --gpu-memory-utilization 0.90 \
    ##        --max-model-len 1024 \
    ##        --max-num-seqs 80 \
    ##        --n 1 \
    ##        --max-tokens 256 \
    ##        --temperature 1.0 \
    ##        --top-p 1.0 \
    ##        --repetition-penalty 1.1 \
    ##        2>&1 | tee "${GEN_LOG}"
##
    ##    GEN_EXIT=${PIPESTATUS[0]}
    ##    if [ $GEN_EXIT -ne 0 ]; then
    ##        echo "❌ Generation failed for CWLC ${PCT}%! Check: ${GEN_LOG}"
    ##        exit $GEN_EXIT
    ##    fi
    ##    echo "✅ Generated CWLC ${PCT}% → ${OUTPUT_FILE}"
    ##    echo ""
    ##fi
done

echo ""
echo "✅ PHASE 3 COMPLETE — All generation done!"
echo ""
fi  # end PHASE 3

# =============================================================================
# DONE
# =============================================================================
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  🎉 MEGA PIPELINE 1 COMPLETE!                                     ║"
echo "║  Finished: $(date)                                                 ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Summary of generated files:"
echo "  CARES:    $(ls output/CARES/synth_cares_*pct_*.json 2>/dev/null | wc -l) files"
#echo "  RADGRAPH: $(ls output/RADGRAPH/synth_radgraph_*pct_*.json 2>/dev/null | wc -l) files"
#echo "  CWLC:     $(ls output/CWLC/synth_cwlc_*pct_*.json 2>/dev/null | wc -l) files"
echo ""
echo "Logs in: ${LOG_DIR}/"
ls -lt ${LOG_DIR}/train_*.log ${LOG_DIR}/merge_*.log ${LOG_DIR}/gen_*.log 2>/dev/null | head -20
