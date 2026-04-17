#!/bin/bash
#
# Merge LoRA adapter with base model for vLLM inference
#
# Usage:
#   bash scripts/merge_lora_model.sh <lora-path> [output-path] [base-model]
#
# Examples:
#   bash scripts/merge_lora_model.sh output/CARES/gpt-oss-20b_5pct_20260215/final_model
#   bash scripts/merge_lora_model.sh output/MIMIC/gpt-oss-20b_5pct_20260215/final_model output/gpt-oss-20b-mimic-merged
#
# The base model is auto-detected from adapter_config.json.
# 4-bit references (e.g., unsloth/gpt-oss-20b-unsloth-bnb-4bit) are
# automatically converted to their FP16 equivalents.

set -euo pipefail

# Activate conda environment
source /home/lmiranda/miniconda3/etc/profile.d/conda.sh
conda activate unsloth-test

# Parse arguments
LORA_PATH="${1:?Usage: $0 <lora-path> [output-path] [base-model]}"
OUTPUT_PATH="${2:-}"
BASE_MODEL="${3:-unsloth/gpt-oss-20b}"

# GPU: prefer Blackwell (96GB) for merging since FP16 base model is ~40GB
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Auto-generate output path if not specified
if [ -z "$OUTPUT_PATH" ]; then
    # Extract dataset dir from lora path: output/CARES/gpt-oss-20b_.../final_model → output/CARES/merged
    PARENT_DIR="$(dirname "$(dirname "$LORA_PATH")")"
    RUN_DIR="$(basename "$(dirname "$LORA_PATH")")"
    OUTPUT_PATH="${PARENT_DIR}/${RUN_DIR}_merged"
fi

echo "========================================================================"
echo "LoRA Merge → vLLM Compatible Model"
echo "========================================================================"
echo "LoRA path:   ${LORA_PATH}"
echo "Output:      ${OUTPUT_PATH}"
echo "GPU:         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "========================================================================"
echo ""

CMD=(
    python -u src/merge_lora_model.py
    --lora-path "${LORA_PATH}"
    --output-path "${OUTPUT_PATH}"
    --dtype bfloat16
    --device-map auto
)

if [ -n "$BASE_MODEL" ]; then
    CMD+=(--base-model "${BASE_MODEL}")
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} "${CMD[@]}"

echo ""
echo "✓ Merge completed! Merged model at: ${OUTPUT_PATH}"
