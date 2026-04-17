#!/bin/bash
# =============================================================================
# Generate synthetic clinical notes using vLLM (native, no Docker)
# =============================================================================
#
# Supports 4 datasets: MIMIC, CARES, CWLC, RADGRAPH
#
# Usage:
#   DATASET=mimic  ./scripts/generate_vllm.sh                    # MIMIC (10 test samples)
#   DATASET=mimic  ./scripts/generate_vllm.sh --full              # MIMIC (all 13k samples)
#   DATASET=cares  ./scripts/generate_vllm.sh --max-samples 50    # CARES (50 samples)
#   DATASET=cwlc   ./scripts/generate_vllm.sh --full              # CWLC (all)
#   DATASET=radgraph ./scripts/generate_vllm.sh --full            # RADGRAPH (all)
#
# Environment variables (override defaults):
#   DATASET           Dataset name       (default: mimic)
#   MODEL_PATH        Path to model      (required or auto-detected)
#   INPUT_FILE        Input file/path    (dataset-specific default)
#   OUTPUT_FILE       Output JSON file   (auto-generated with timestamp)
#   CUDA_DEVICES      GPUs to use        (default: 0)
#   TENSOR_PARALLEL   TP size            (default: 1)
#   MAX_SAMPLES       Limit samples      (default: 10, empty = all)
#
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${PROJECT_DIR}/.venv-vllm/bin/python"
SCRIPT="${PROJECT_DIR}/src/generate_all_vllm.py"

# -----------------------------------------------------------------------------
# Dataset selection
# -----------------------------------------------------------------------------
DATASET="${DATASET:-CARES}"
DATASET_LOWER="$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')"
DATASET_UPPER="$(echo "${DATASET}" | tr '[:lower:]' '[:upper:]')"

# -----------------------------------------------------------------------------
# Per-dataset defaults
# -----------------------------------------------------------------------------
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

case "${DATASET_LOWER}" in
    mimic)
        MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/output/gpt-oss-20b-mimic-merged}"
        INPUT_FILE="${INPUT_FILE:-${PROJECT_DIR}/data/mimic/mimic_s.json}"
        OUTPUT_FILE="${OUTPUT_FILE:-${PROJECT_DIR}/output/MIMIC/synth_mimic_${TIMESTAMP}.json}"
        MAX_TOKENS="${MAX_TOKENS:-6000}"
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
        ;;
    cares)
        MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/output/gpt-oss-20b-cares-merged}"
        INPUT_FILE="${INPUT_FILE:-}"  # loaded from HuggingFace
        OUTPUT_FILE="${OUTPUT_FILE:-${PROJECT_DIR}/output/CARES/synth_cares_${TIMESTAMP}.json}"
        MAX_TOKENS="${MAX_TOKENS:-4000}"
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
        ;;
    cwlc)
        MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/output/gpt-oss-20b-cwlc-merged}"
        INPUT_FILE="${INPUT_FILE:-${PROJECT_DIR}/data/cwlc.zip}"
        OUTPUT_FILE="${OUTPUT_FILE:-${PROJECT_DIR}/output/CWLC/synth_cwlc_${TIMESTAMP}.json}"
        MAX_TOKENS="${MAX_TOKENS:-4000}"
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
        ;;
    radgraph)
        MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/output/gpt-oss-20b-radgraph-merged}"
        INPUT_FILE="${INPUT_FILE:-}"  # must be set via RADGRAPH_JSONL
        OUTPUT_FILE="${OUTPUT_FILE:-${PROJECT_DIR}/output/RADGRAPH/synth_radgraph_${TIMESTAMP}.json}"
        MAX_TOKENS="${MAX_TOKENS:-4000}"
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
        ;;
    *)
        echo "ERROR: Unknown dataset '${DATASET}'"
        echo "Supported: mimic, cares, cwlc, radgraph"
        exit 1
        ;;
esac

# -----------------------------------------------------------------------------
# Common defaults (can be overridden via env vars)
# -----------------------------------------------------------------------------
CUDA_DEVICES="${CUDA_DEVICES:-1}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
N_OUTPUTS="${N_OUTPUTS:-5}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.5}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-45}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"

# HF dataset override (for CARES)
HF_DATASET="${HF_DATASET:-}"
SPLIT="${SPLIT:-}"

# RadGraph JSONL paths (space-separated)
RADGRAPH_JSONL="${RADGRAPH_JSONL:-}"

# Parse special flags
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --full)
            MAX_SAMPLES=""
            ;;
        *)
            EXTRA_ARGS+=("$arg")
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Validate
# -----------------------------------------------------------------------------
if [[ ! -f "${VENV_PYTHON}" ]]; then
    echo "ERROR: vLLM venv not found at ${VENV_PYTHON}"
    echo "Create it with: uv venv .venv-vllm && uv pip install vllm"
    exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: Model not found: ${MODEL_PATH}"
    exit 1
fi

# Validate dataset-specific inputs
if [[ "${DATASET_LOWER}" == "mimic" && ! -f "${INPUT_FILE}" ]]; then
    echo "ERROR: MIMIC JSON not found: ${INPUT_FILE}"
    echo "Set INPUT_FILE= to the correct path"
    exit 1
fi

if [[ "${DATASET_LOWER}" == "cwlc" && -n "${INPUT_FILE}" && ! -f "${INPUT_FILE}" ]]; then
    echo "ERROR: CWLC ZIP not found: ${INPUT_FILE}"
    echo "Set INPUT_FILE= to the correct path"
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT_FILE}")"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo "======================================================================"
echo "vLLM Generation — ${DATASET_UPPER}"
echo "======================================================================"
echo "  Dataset:          ${DATASET_LOWER}"
echo "  Model:            ${MODEL_PATH}"
if [[ -n "${INPUT_FILE}" ]]; then
echo "  Input:            ${INPUT_FILE}"
fi
echo "  Output:           ${OUTPUT_FILE}"
echo "  CUDA devices:     ${CUDA_DEVICES}"
echo "  Tensor parallel:  ${TENSOR_PARALLEL}"
echo "  Outputs per prompt: ${N_OUTPUTS}"
echo "  Max tokens:       ${MAX_TOKENS}"
echo "  Temperature:      ${TEMPERATURE}"
echo "  Repetition pen.:  ${REPETITION_PENALTY}"
echo "  Max samples:      ${MAX_SAMPLES:-all}"
echo "  vLLM Python:      ${VENV_PYTHON}"
echo "======================================================================"
echo ""



# -----------------------------------------------------------------------------
# Build command
# -----------------------------------------------------------------------------
CMD=(
     "${VENV_PYTHON}" "${SCRIPT}"
    --dataset "${DATASET_LOWER}"
    --model-path "${MODEL_PATH}"
    --output-file "${OUTPUT_FILE}"
    --tensor-parallel-size "${TENSOR_PARALLEL}"
    --gpu-memory-utilization "${GPU_MEMORY_UTIL}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --n "${N_OUTPUTS}"
    --max-tokens "${MAX_TOKENS}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --repetition-penalty "${REPETITION_PENALTY}"
)

# Dataset-specific args
case "${DATASET_LOWER}" in
    mimic)
        CMD+=(--mimic-json "${INPUT_FILE}")
        ;;
    cwlc)
        if [[ -n "${INPUT_FILE}" ]]; then
            CMD+=(--cwlc-zip "${INPUT_FILE}")
        fi
        ;;
    cares)
        if [[ -n "${HF_DATASET}" ]]; then
            CMD+=(--hf-dataset "${HF_DATASET}")
        fi
        if [[ -n "${SPLIT}" ]]; then
            CMD+=(--split "${SPLIT}")
        fi
        ;;
    radgraph)
        if [[ -n "${RADGRAPH_JSONL}" ]]; then
            CMD+=(--radgraph-jsonl ${RADGRAPH_JSONL})  # intentionally unquoted for splitting
        fi
        ;;
esac

if [[ -n "${MAX_SAMPLES}" ]]; then
    CMD+=(--max-samples "${MAX_SAMPLES}")
fi

# Append any extra args passed to this script
CMD+=("${EXTRA_ARGS[@]}")

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
echo "Running: CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} ${CMD[*]}"
echo ""

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" CUDA_DEVICE_ORDER=PCI_BUS_ID \
    "${CMD[@]}"

EXIT_CODE=$?

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "✓ Generation finished successfully!"
    echo "  Output: ${OUTPUT_FILE}"
    if [[ -f "${OUTPUT_FILE}" ]]; then
        SIZE=$(du -h "${OUTPUT_FILE}" | cut -f1)
        echo "  Size: ${SIZE}"
    fi
else
    echo "✗ Generation failed with exit code ${EXIT_CODE}"
fi

exit $EXIT_CODE
