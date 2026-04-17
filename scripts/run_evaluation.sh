#!/bin/bash
# =============================================================================
# Evaluation Pipeline — Fidelity + Privacy
# Runs run_fidelity.py and run_privacy.py on all generated synthetic texts.
# =============================================================================
#
# Usage:
#   bash scripts/run_evaluation.sh                          # all generated files
#   bash scripts/run_evaluation.sh output/CWLC              # only CWLC files
#   bash scripts/run_evaluation.sh output/CWLC/synth_cwlc_5pct_20260217_152604.json  # single file
#
# Options (env vars):
#   SKIP_ROUGE=1        Skip slow ROUGE-5 analysis
#   N_JOBS=4            Parallel jobs for ROUGE (default: 4)
#   MAX_ROUGE=500       Max samples for ROUGE   (default: 500)
#   PYTHON=python3      Python interpreter

set -e
set -o pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

DATASET_NAME="${DATASET_NAME:-MIMIC}"    

PYTHON="${PYTHON:-python3}"
FIDELITY_SCRIPT="src/run_fidelity.py"
PRIVACY_SCRIPT="src/run_privacy.py"

SKIP_ROUGE="${SKIP_ROUGE:-0}"
N_JOBS="${N_JOBS:-4}"
MAX_ROUGE="${MAX_ROUGE:-500}"

EVAL_DIR="output/${DATASET_NAME}/evaluation"
mkdir -p "$EVAL_DIR"

TMPDIR_EVAL=$(mktemp -d)
trap "rm -rf $TMPDIR_EVAL" EXIT

# ─── Helper: convert our JSON format to JSONL with generated_text ────────────
convert_json_to_jsonl() {
    local input_json="$1"
    local output_jsonl="$2"

    ${PYTHON} -c "
import json, sys

with open('${input_json}', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Handle both list (our format) and single-object
if not isinstance(data, list):
    data = [data]

with open('${output_jsonl}', 'w', encoding='utf-8') as out:
    for doc in data:
        # Map output1 -> generated_text (what the eval scripts expect)
        if 'output1' in doc and 'generated_text' not in doc:
            doc['generated_text'] = doc['output1']
        out.write(json.dumps(doc, ensure_ascii=False) + '\n')

print(f'  Converted {len(data)} records: {\"${output_jsonl}\"}')
"
}

# ─── Collect files to evaluate ───────────────────────────────────────────────
declare -a FILES=()

if [ $# -eq 0 ]; then
    # No args: find all synth_*.json across all dataset dirs
    while IFS= read -r f; do
        FILES+=("$f")
    done < <(find output -name "synth_*pct_*.json" -type f | sort)
elif [ -d "$1" ]; then
    # Arg is a directory: find synth files within
    while IFS= read -r f; do
        FILES+=("$f")
    done < <(find "$1" -name "synth_*pct_*.json" -type f | sort)
elif [ -f "$1" ]; then
    # Arg is a single file
    FILES=("$1")
else
    echo "ERROR: '$1' is not a valid file or directory."
    exit 1
fi

if [ ${#FILES[@]} -eq 0 ]; then
    echo "No synth_*pct_*.json files found to evaluate."
    exit 0
fi

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Evaluation Pipeline — Fidelity + Privacy                         ║"
echo "║  Files to evaluate: ${#FILES[@]}                                          ║"
echo "║  Output dir: ${EVAL_DIR}                                    ║"
echo "║  ROUGE: $([ "$SKIP_ROUGE" = "1" ] && echo "SKIP" || echo "ON (n_jobs=${N_JOBS}, max=${MAX_ROUGE})")                                            ║"
echo "║  Started: $(date)                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

TOTAL=${#FILES[@]}
CURRENT=0
FAILED=0

for INPUT_FILE in "${FILES[@]}"; do
    CURRENT=$((CURRENT + 1))
    BASENAME=$(basename "$INPUT_FILE" .json)
    DATASET_DIR=$(basename "$(dirname "$INPUT_FILE")")

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [${CURRENT}/${TOTAL}] ${DATASET_DIR}/${BASENAME}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # 1. Convert JSON → JSONL (with generated_text field)
    JSONL_FILE="${TMPDIR_EVAL}/${BASENAME}.jsonl"
    echo "  Converting to JSONL..."
    convert_json_to_jsonl "$INPUT_FILE" "$JSONL_FILE"

    # Subdirectory for this file's results
    FILE_EVAL_DIR="${EVAL_DIR}/${DATASET_DIR}"
    mkdir -p "$FILE_EVAL_DIR"

    # 2. Run fidelity evaluation
    echo ""
    echo "  ── Fidelity Evaluation ──"
    if ${PYTHON} "$FIDELITY_SCRIPT" \
        --input_file "$JSONL_FILE" \
        --output_dir "$FILE_EVAL_DIR" 2>&1; then
        echo "  ✅ Fidelity done"
    else
        echo "  ❌ Fidelity failed for ${BASENAME}"
        FAILED=$((FAILED + 1))
    fi

    # 3. Run privacy evaluation
    echo ""
    echo "  ── Privacy Evaluation ──"
    PRIVACY_ARGS="--input_file $JSONL_FILE --output_dir $FILE_EVAL_DIR --n_jobs ${N_JOBS} --max_rouge_samples ${MAX_ROUGE}"
    if [ "$SKIP_ROUGE" = "1" ]; then
        PRIVACY_ARGS="$PRIVACY_ARGS --skip_rouge"
    fi

    if ${PYTHON} "$PRIVACY_SCRIPT" $PRIVACY_ARGS 2>&1; then
        echo "  ✅ Privacy done"
    else
        echo "  ❌ Privacy failed for ${BASENAME}"
        FAILED=$((FAILED + 1))
    fi

    echo ""
done

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  🎉 Evaluation Complete!                                          ║"
echo "║  Processed: ${TOTAL} files, Failed: ${FAILED}                              ║"
echo "║  Finished: $(date)                                                 ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: ${EVAL_DIR}/"
ls -lh ${EVAL_DIR}/*/*.json 2>/dev/null
