#!/usr/bin/env bash
# Train ICD-10 multi-label classifier on synthetic/original texts and evaluate on CARES test.
#
# Usage (chapters - como paper CARES, ~88% F1):
#   bash scripts/train_icd10_multilabel.sh \
#     --train-jsonl output/CARES/generated_cares_20260122_020000.jsonl \
#     --train-text-source original \
#     --label-type chapters \
#     --output-dir output/CARES/clf_chapters
#
# Usage (icd10 codes - 156 clases, más difícil):
#   bash scripts/train_icd10_multilabel.sh \
#     --train-jsonl output/CARES/generated_cares_20260122_020000.jsonl \
#     --label-type icd10 \
#     --output-dir output/CARES/clf_icd10

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}

# Defaults optimizados para CARES (como paper original)
CUDA_VISIBLE_DEVICES_DEFAULT=${CUDA_VISIBLE_DEVICES_DEFAULT:-"1,2"}
HF_DATASET=${HF_DATASET:-chizhikchi/CARES}
MODEL=${MODEL:-emilyalsentzer/Bio_ClinicalBERT}  # RoBERTa biomédico-clínico español
TRAIN_TEXT_SOURCE=${TRAIN_TEXT_SOURCE:-original}
LABEL_TYPE=${LABEL_TYPE:-chapters}  # chapters (16 clases) o icd10 (156 clases)
MAX_LENGTH=${MAX_LENGTH:-512}
LR=${LR:-4e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.006}
EPOCHS=${EPOCHS:-40}
TRAIN_BS=${TRAIN_BS:-32}
EVAL_BS=${EVAL_BS:-32}
WARMUP_RATIO=${WARMUP_RATIO:-0.0}
WARMUP_STEPS=${WARMUP_STEPS:-300}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-5}
THRESHOLD=${THRESHOLD:-0.5}
SEED=${SEED:-42}
LOGGING_STEPS=${LOGGING_STEPS:-50}
FP16=${FP16:-false}
BF16=${BF16:-false}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-}
EVAL_STRATEGY=${EVAL_STRATEGY:-steps}
SAVE_STRATEGY=${SAVE_STRATEGY:-steps}
SAVE_STEPS=${SAVE_STEPS:-200}
EVAL_STEPS=${EVAL_STEPS:-200}

# Nuevos flags (diagnóstico / dumps)
SAVE_EVAL_PREDICTIONS=${SAVE_EVAL_PREDICTIONS:-true}
MAX_EVAL_PREDICTIONS=${MAX_EVAL_PREDICTIONS:-}

TRAIN_JSONL=""
OUTPUT_DIR=""
CUDA_VISIBLE_DEVICES_ARG=""
FORCE_SINGLE_GPU=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda-visible-devices)
      CUDA_VISIBLE_DEVICES_ARG="$2"; shift 2;;
    --force-single-gpu)
      FORCE_SINGLE_GPU=true; shift 1;;

    --train-jsonl)
      TRAIN_JSONL="$2"; shift 2;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2;;

    --eval-jsonl)
      EVAL_JSONL="$2"; shift 2;;
    --eval-text-source)
      EVAL_TEXT_SOURCE="$2"; shift 2;;
    --hf-dataset)
      HF_DATASET="$2"; shift 2;;
    --model)
      MODEL="$2"; shift 2;;
    --train-text-source)
      TRAIN_TEXT_SOURCE="$2"; shift 2;;
    --label-type)
      LABEL_TYPE="$2"; shift 2;;
    --max-length)
      MAX_LENGTH="$2"; shift 2;;
    --learning-rate)
      LR="$2"; shift 2;;
    --weight-decay)
      WEIGHT_DECAY="$2"; shift 2;;
    --num-train-epochs)
      EPOCHS="$2"; shift 2;;
    --per-device-train-batch-size)
      TRAIN_BS="$2"; shift 2;;
    --per-device-eval-batch-size)
      EVAL_BS="$2"; shift 2;;
    --warmup-ratio)
      WARMUP_RATIO="$2"; shift 2;;
    --warmup-steps)
      WARMUP_STEPS="$2"; shift 2;;
    --early-stopping-patience)
      EARLY_STOPPING_PATIENCE="$2"; shift 2;;
    --threshold)
      THRESHOLD="$2"; shift 2;;
    --seed)
      SEED="$2"; shift 2;;
    --logging-steps)
      LOGGING_STEPS="$2"; shift 2;;

    --max-train-samples)
      MAX_TRAIN_SAMPLES="$2"; shift 2;;
    --max-eval-samples)
      MAX_EVAL_SAMPLES="$2"; shift 2;;

    --eval-strategy)
      EVAL_STRATEGY="$2"; shift 2;;
    --save-strategy)
      SAVE_STRATEGY="$2"; shift 2;;
    --save-steps)
      SAVE_STEPS="$2"; shift 2;;
    --eval-steps)
      EVAL_STEPS="$2"; shift 2;;

    --fp16)
      FP16=true; shift 1;;
    --bf16)
      BF16=true; shift 1;;

    --save-eval-predictions)
      SAVE_EVAL_PREDICTIONS=true; shift 1;;
    --max-eval-predictions)
      MAX_EVAL_PREDICTIONS="$2"; shift 2;;

    -h|--help)
      sed -n '1,120p' "$0"; exit 0;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1;;
  esac
done

# Si no se setea por fuera ni por flag, default (1,2)
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -z "$CUDA_VISIBLE_DEVICES_ARG" ]]; then
  export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_DEFAULT"
fi
if [[ -n "$CUDA_VISIBLE_DEVICES_ARG" ]]; then
  export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_ARG"
fi

if [[ -z "$TRAIN_JSONL" ]]; then
  echo "ERROR: --train-jsonl is required" >&2
  exit 1
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  echo "ERROR: --output-dir is required" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "${EVAL_JSONL:-}" ]]; then
  EXTRA_ARGS+=(--eval-jsonl "$EVAL_JSONL")
fi
if [[ -n "${EVAL_TEXT_SOURCE:-}" ]]; then
  EXTRA_ARGS+=(--eval-text-source "$EVAL_TEXT_SOURCE")
fi

if [[ -n "$MAX_TRAIN_SAMPLES" ]]; then
  EXTRA_ARGS+=(--max-train-samples "$MAX_TRAIN_SAMPLES")
fi
if [[ -n "$MAX_EVAL_SAMPLES" ]]; then
  EXTRA_ARGS+=(--max-eval-samples "$MAX_EVAL_SAMPLES")
fi
if [[ "$FP16" == "true" ]]; then
  EXTRA_ARGS+=(--fp16)
fi
if [[ "$BF16" == "true" ]]; then
  EXTRA_ARGS+=(--bf16)
fi
if [[ "$FORCE_SINGLE_GPU" == "true" ]]; then
  EXTRA_ARGS+=(--force-single-gpu)
fi
if [[ "$SAVE_EVAL_PREDICTIONS" == "true" ]]; then
  EXTRA_ARGS+=(--save-eval-predictions)
fi
if [[ -n "$MAX_EVAL_PREDICTIONS" ]]; then
  EXTRA_ARGS+=(--max-eval-predictions "$MAX_EVAL_PREDICTIONS")
fi

$PYTHON_BIN -u src/train_multilabel_classification.py \
  --train-jsonl "$TRAIN_JSONL" \
  --train-text-source "$TRAIN_TEXT_SOURCE" \
  --label-type "$LABEL_TYPE" \
  --hf-dataset "$HF_DATASET" \
  --model "$MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --max-length "$MAX_LENGTH" \
  --learning-rate "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --num-train-epochs "$EPOCHS" \
  --per-device-train-batch-size "$TRAIN_BS" \
  --per-device-eval-batch-size "$EVAL_BS" \
  --warmup-ratio "$WARMUP_RATIO" \
  --warmup-steps "$WARMUP_STEPS" \
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
  --threshold "$THRESHOLD" \
  --seed "$SEED" \
  --logging-steps "$LOGGING_STEPS" \
  --eval-strategy "$EVAL_STRATEGY" \
  --save-strategy "$SAVE_STRATEGY" \
  --save-steps "$SAVE_STEPS" \
  --eval-steps "$EVAL_STEPS" \
  "${EXTRA_ARGS[@]}"
