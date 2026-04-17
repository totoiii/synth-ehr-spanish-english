#!/bin/bash
# =============================================================================
# MEGA PIPELINE 3 — MIMIC (large)
# Train GPU:  2 (NVIDIA RTX 6000 Ada, 48 GB)
# Gen GPU:    1 (Blackwell 96GB)
# Phases:  Finetune → Merge → Generate
# =============================================================================
#
# NOTE: Do NOT export CUDA_VISIBLE_DEVICES globally here.
# Each phase sets it inline per-command to avoid GPU memory leaks
# between phases and to prevent conflicts.
#
# Usage:
#   nohup bash scripts/finetuning_and_synthesis_3.sh > output/logs/mega_pipeline_3.log 2>&1 &
#
# To resume from a specific phase:
#   START_PHASE=2 bash scripts/finetuning_and_synthesis_3.sh   # skip finetuning
#   START_PHASE=3 bash scripts/finetuning_and_synthesis_3.sh   # skip to generation

set -e
set -o pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
TRAIN_GPU=2                        # Ada GPU for finetuning
MERGE_GPU=2                        # Ada GPU for merging
GEN_GPUS="2"                      # Blackwell 96GB for generation
GEN_TP=1                          # Tensor parallel size for generation
GEN_PP=1                          # Pipeline parallel size for generation

# NOTE: No global export of CUDA_VISIBLE_DEVICES — each command sets it inline
# to avoid GPU memory conflicts between phases.

PERCENTAGES=(100 50 25 5)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

START_PHASE="${START_PHASE:-1}"

# MIMIC variants
MIMIC_VARIANTS=("large")

# Activate conda
source /home/lmiranda/miniconda3/etc/profile.d/conda.sh
conda activate unsloth-test

VENV_PYTHON="${PROJECT_DIR}/.venv-vllm/bin/python"
GEN_SCRIPT="${PROJECT_DIR}/src/generate_all_vllm.py"

LOG_DIR="./output/logs"
mkdir -p "${LOG_DIR}"

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  MEGA PIPELINE 3 — MIMIC (large)                                  ║"
echo "║  Train GPU: ${TRAIN_GPU} (Ada 48GB)                                       ║"
echo "║  Gen GPU:   ${GEN_GPUS} (Blackwell 96GB, TP=${GEN_TP})                            ║"
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
echo "█  PHASE 1/3 — FINETUNING (GPU ${TRAIN_GPU})                                █"
echo "██████████████████████████████████████████████████████████████████████"
echo ""

for VARIANT in "${MIMIC_VARIANTS[@]}"; do
    for PCT in "${PERCENTAGES[@]}"; do
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  MIMIC-${VARIANT} ${PCT}% — $(date)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        CUDA_VISIBLE_DEVICES=${TRAIN_GPU} TRAIN_CUDA_DEVICES=${TRAIN_GPU} BATCH_SIZE=32 bash scripts/finetuning_mimic.sh ${VARIANT} ${PCT}
        echo "✅ MIMIC-${VARIANT} ${PCT}% finetuning done"
        echo ""
    done
done

echo ""
echo "✅ PHASE 1 COMPLETE — All MIMIC finetuning done!"
echo ""
fi  # end PHASE 1

# =============================================================================
# PHASE 2 — MERGE LoRA MODELS
# =============================================================================
if [ "$START_PHASE" -le 2 ]; then
echo ""
echo "██████████████████████████████████████████████████████████████████████"
echo "█  PHASE 2/3 — MERGE LoRA MODELS (GPU ${MERGE_GPU})                         █"
echo "██████████████████████████████████████████████████████████████████████"
echo ""

for PCT in "${PERCENTAGES[@]}"; do
    # MIMIC models: for each PCT there are 2 dirs (small first, large second)
    # We merge ALL of them
    ALL_LORA_DIRS=($(ls -td output/MIMIC/gpt-oss-20b_${PCT}pct_*/final_model 2>/dev/null))

    if [ ${#ALL_LORA_DIRS[@]} -eq 0 ]; then
        echo "⚠️  No models found for MIMIC ${PCT}%, skipping..."
        continue
    fi

    for LORA_PATH in "${ALL_LORA_DIRS[@]}"; do
        MODEL_DIR=$(dirname "$LORA_PATH")
        MERGED_DIR="${MODEL_DIR}_merged"
        MODEL_NAME=$(basename "$MODEL_DIR")
        TS=$(date +%Y%m%d_%H%M%S)
        MERGE_LOG="${LOG_DIR}/merge_mimic_${MODEL_NAME}_${TS}.log"

        # Skip if already merged
        if [ -d "$MERGED_DIR" ] && [ -f "${MERGED_DIR}/config.json" ]; then
            echo "⏭️  Already merged: ${MERGED_DIR}"
            continue
        fi

        echo "━━━ Merging MIMIC ${MODEL_NAME} ━━━"
        echo "  LoRA: ${LORA_PATH}"
        echo "  Output: ${MERGED_DIR}"
        echo "  Log: ${MERGE_LOG}"

        CUDA_VISIBLE_DEVICES=${MERGE_GPU} python -u src/merge_lora_model.py \
            --lora-path "${LORA_PATH}" \
            --output-path "${MERGED_DIR}" \
            --base-model "unsloth/gpt-oss-20b" \
            --dtype bfloat16 \
            --device-map auto \
            2>&1 | tee "${MERGE_LOG}"

        MERGE_EXIT=${PIPESTATUS[0]}
        if [ $MERGE_EXIT -ne 0 ]; then
            echo "❌ Merge failed for ${MODEL_NAME}! Check: ${MERGE_LOG}"
            exit $MERGE_EXIT
        fi

        echo "✅ Merged ${MODEL_NAME}"
        echo ""
    done
done

echo ""
echo "✅ PHASE 2 COMPLETE — All models merged!"
echo ""
fi  # end PHASE 2

# =============================================================================
# PHASE 3 — GENERATION (vLLM with 2 GPUs, TP=2)
# =============================================================================
if [ "$START_PHASE" -le 3 ]; then
echo ""
echo "██████████████████████████████████████████████████████████████████████"
echo "█  PHASE 3/3 — TEXT GENERATION (vLLM, PP=${GEN_PP}, TP=${GEN_TP}, GPUs: ${GEN_GPUS})          █"
echo "██████████████████████████████████████████████████████████████████████"
echo ""

# Kill any leftover Python/CUDA processes on the generation GPU(s) to free memory
echo "🧹 Clearing GPU memory before generation..."
echo "   GPU ${GEN_GPUS} memory before cleanup:"
nvidia-smi --id=${GEN_GPUS} --query-gpu=index,memory.used,memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null || true

# Give a moment for any cleanup
sleep 5

echo "   GPU ${GEN_GPUS} memory after cleanup:"
nvidia-smi --id=${GEN_GPUS} --query-gpu=index,memory.used,memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null || true
echo ""

# Generation parameters for MIMIC with 2 GPUs (TP=2), n=5:
# ┌──────────┬──────────┬───────────┬──────────────┬────────────┬───────────────┬──────────────┐
# │ Dataset  │ Prompts  │ N_OUTPUTS │ Total Gen    │ max_tokens │ max_model_len │ max_num_seqs │
# ├──────────┼──────────┼───────────┼──────────────┼────────────┼───────────────┼──────────────┤
# │ MIMIC-s  │  13,378  │     5     │   66,890     │   6000     │     8192      │     64       │
# │ MIMIC-l  │  89,098  │     5     │  445,490     │   6000     │     8192      │     64       │
# └──────────┴──────────┴───────────┴──────────────┴────────────┴───────────────┴──────────────┘

# For each PCT, find the 2 merged models (small=older, large=newer)
for PCT in "${PERCENTAGES[@]}"; do
##    # Try to find explicit 'small_continued' merged model (newest first)
    SMALL_MERGED=$(ls -td output/MIMIC/gpt-oss-20b_${PCT}pct_small_continued_*_merged 2>/dev/null | head -1)

    # Fallback to original logic (oldest merged model) if no explicit small model found
    if [ -z "$SMALL_MERGED" ]; then
        ALL_DIRS=($(ls -td output/MIMIC/gpt-oss-20b_${PCT}pct_*_merged 2>/dev/null | tac))
        if [ ${#ALL_DIRS[@]} -ge 1 ]; then
            SMALL_MERGED="${ALL_DIRS[0]}"
        fi
    fi

##    # Process MIMIC-small
##    if [ -n "$SMALL_MERGED" ]; then
##        MERGED_DIR="$SMALL_MERGED"
##        TS=$(date +%Y%m%d_%H%M%S)
##        OUTPUT_FILE="output/MIMIC/synth_mimic_small_${PCT}pct_${TS}.json"
##        GEN_LOG="${LOG_DIR}/gen_mimic_small_${PCT}pct_${TS}.log"
##        MIMIC_JSON="./data/mimic/mimic_s.json"
##
##        echo "━━━ Generating MIMIC-small ${PCT}% — $(date) ━━━"
##        echo "  Model: ${MERGED_DIR}"
##        echo "  Input: ${MIMIC_JSON}"
##        echo "  Output: ${OUTPUT_FILE}"
##        echo "  Log: ${GEN_LOG}"
##        echo "  GPUs: ${GEN_GPUS} (PP=${GEN_PP}, TP=${GEN_TP})"
##        echo ""
##
##        CUDA_VISIBLE_DEVICES=${GEN_GPUS} "${VENV_PYTHON}" "${GEN_SCRIPT}" \
##            --dataset mimic \
##            --model-path "${MERGED_DIR}" \
##            --mimic-json "${MIMIC_JSON}" \
##            --output-file "${OUTPUT_FILE}" \
##            --tensor-parallel-size ${GEN_TP} \
##            --pipeline-parallel-size ${GEN_PP} \
##            --gpu-memory-utilization 0.90 \
##            --max-model-len 4096 \
##            --max-num-seqs 256 \
##            --n 1 \
##            --max-tokens 2048 \
##            --temperature 1.0 \
##            --top-p 1.0 \
##            --repetition-penalty 1.1 \
##            2>&1 | tee "${GEN_LOG}"
##
##        GEN_EXIT=${PIPESTATUS[0]}
##        if [ $GEN_EXIT -ne 0 ]; then
##            echo "❌ Generation failed for MIMIC-small ${PCT}%! Check: ${GEN_LOG}"
##            exit $GEN_EXIT
##        fi
##        echo "✅ Generated MIMIC-small ${PCT}% → ${OUTPUT_FILE}"
##        echo ""
##    fi

    # Try to find explicit 'large_continued' merged model (newest first)
    LARGE_MERGED=$(ls -td output/MIMIC/gpt-oss-20b_${PCT}pct_large_continued_*_merged 2>/dev/null | head -1)

    # Fallback to original logic (2nd oldest merged model) if no explicit large model found
    if [ -z "$LARGE_MERGED" ]; then
        ALL_DIRS=($(ls -td output/MIMIC/gpt-oss-20b_${PCT}pct_*_merged 2>/dev/null | tac))
        if [ ${#ALL_DIRS[@]} -ge 2 ]; then
            LARGE_MERGED="${ALL_DIRS[1]}"
        fi
    fi

    # Process MIMIC-large
    if [ -n "$LARGE_MERGED" ]; then
        MERGED_DIR="$LARGE_MERGED"
        TS=$(date +%Y%m%d_%H%M%S)
        OUTPUT_FILE="output/MIMIC/synth_mimic_large_${PCT}pct_${TS}.json"
        GEN_LOG="${LOG_DIR}/gen_mimic_large_${PCT}pct_${TS}.log"
        MIMIC_JSON="./data/mimic/mimic_l.json"
#
        echo "━━━ Generating MIMIC-large ${PCT}% — $(date) ━━━"
        echo "  Model: ${MERGED_DIR}"
        echo "  Input: ${MIMIC_JSON}"
        echo "  Output: ${OUTPUT_FILE}"
        echo "  Log: ${GEN_LOG}"
        echo "  GPUs: ${GEN_GPUS} (PP=${GEN_PP}, TP=${GEN_TP})"
        echo ""
#
        CUDA_VISIBLE_DEVICES=${GEN_GPUS} "${VENV_PYTHON}" "${GEN_SCRIPT}" \
            --dataset mimic \
            --model-path "${MERGED_DIR}" \
            --mimic-json "${MIMIC_JSON}" \
            --output-file "${OUTPUT_FILE}" \
            --tensor-parallel-size ${GEN_TP} \
            --pipeline-parallel-size ${GEN_PP} \
            --gpu-memory-utilization 0.90 \
            --max-model-len 6000 \
            --max-num-seqs 64 \
            --n 1 \
            --max-tokens 4096 \
            --temperature 1.0 \
            --top-p 1.0 \
            --repetition-penalty 1.1 \
            2>&1 | tee "${GEN_LOG}"
#
        GEN_EXIT=${PIPESTATUS[0]}
        if [ $GEN_EXIT -ne 0 ]; then
            echo "❌ Generation failed for MIMIC-large ${PCT}%! Check: ${GEN_LOG}"
            exit $GEN_EXIT
        fi
        echo "✅ Generated MIMIC-large ${PCT}% → ${OUTPUT_FILE}"
        echo ""
    fi
done

echo ""
echo "✅ PHASE 3 COMPLETE — All MIMIC generation done!"
echo ""
fi  # end PHASE 3

# =============================================================================
# DONE
# =============================================================================
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  🎉 MEGA PIPELINE 3 COMPLETE!                                     ║"
echo "║  Finished: $(date)                                                 ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Summary of generated files:"
echo "  MIMIC: $(ls output/MIMIC/synth_mimic_*pct_*.json 2>/dev/null | wc -l) files"
ls -lh output/MIMIC/synth_mimic_*pct_*.json 2>/dev/null || true
echo ""
echo "Logs in: ${LOG_DIR}/"
ls -lt ${LOG_DIR}/train_mimic_*.log ${LOG_DIR}/merge_mimic_*.log ${LOG_DIR}/gen_mimic_*.log 2>/dev/null | head -20
