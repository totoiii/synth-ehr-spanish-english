#!/bin/bash
# run_plm_icd_mimic.sh
# Run PLM-ICD downstream task on synthetic MIMIC data
set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <PCT> [TEXT_SOURCE] [--smoke-test]"
    echo "Example: $0 5pct generated --smoke-test"
    exit 1
fi

PCT=$1
TEXT_SOURCE=${2:-generated}
SMOKE_TEST=$3

# Paths
ROOT_DIR="/home/lmiranda/ehr-synthetic-bmc-2026"
PLM_ICD_DIR="/home/lmiranda/Repos/synthetic-ehr-notes/utility/plm_icd"

# Allow the multi-run pipeline to override these.
#   SYNTH_JSON_OVERRIDE : path to a synth file where `output1` holds the run-k text
#   OUTPUT_DIR_OVERRIDE : destination for metrics / checkpoints
#   RUN_TAG             : short tag appended to data filenames (e.g. run1)
OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-${ROOT_DIR}/output/MIMIC/MIMIC_downstream_plmicd_${PCT}_${TEXT_SOURCE}}"
SYNTH_JSON="${SYNTH_JSON_OVERRIDE:-${ROOT_DIR}/output/MIMIC/synth_mimic_small_${PCT}_20260305_041044.json}"
EVAL_JSON="${ROOT_DIR}/data/mimic/mimic_m.json"
ICD_CSV="${ROOT_DIR}/data/mimic/ICD10_descriptions_mimic.csv"
RUN_TAG="${RUN_TAG:-default}"

PLM_DATA_DIR="${PLM_ICD_DIR}/files/data/synth_mimic"

echo "=================================================="
echo " PLM-ICD Downstream Task Pipeline"
echo " PCT: ${PCT}"
echo " TEXT_SOURCE: ${TEXT_SOURCE}"
echo "=================================================="

# 1. Prepare Data
echo "[1/3] Preparing data into feather format..."
mkdir -p "${PLM_DATA_DIR}"

PREP_TAG="${PCT}_${TEXT_SOURCE}_${RUN_TAG}"
PREP_CMD="python3 ${ROOT_DIR}/scripts/prepare_synthetic_mimic.py \
    --synth-json ${SYNTH_JSON} \
    --eval-json ${EVAL_JSON} \
    --icd-csv ${ICD_CSV} \
    --text-source ${TEXT_SOURCE} \
    --text-field output1 \
    --output-dir ${PLM_DATA_DIR} \
    --output-tag ${PREP_TAG}"

if [ "${SMOKE_TEST}" == "--smoke-test" ]; then
    echo "Running in SMOKE TEST mode (200 train / 50 eval, no rare-code filter)"
    # - max-train-samples 200: cap generated-text training set
    # - max-eval-samples 50:   cap mimic_m.json eval split. This is the
    #   single biggest smoke speedup — the stock eval is ~20k rows so
    #   every validation pass takes ~45s × (train-val + val + test per
    #   epoch) × epochs, i.e. minutes per run. 50 eval rows finish in ~1s.
    # - min-target-count 1:    don't drop codes when the sample budget
    #   leaves fewer than 10 occurrences of any given code.
    PREP_CMD="${PREP_CMD} --max-train-samples 200 --max-eval-samples 50 --min-target-count 1"
fi

eval $PREP_CMD

# Copy the generated data config to PLM-ICD configs
cp "${PLM_DATA_DIR}/data_config_${PREP_TAG}.yaml" "${PLM_ICD_DIR}/configs/data/synth_mimic.yaml"

# 2. Run Training
echo "[2/3] Setting up environment and running PLM-ICD training..."
cd "${PLM_ICD_DIR}"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment and installing dependencies..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install pip setuptools wheel --upgrade
    # Install core packages needed for the script (instead of full requirements.txt which fails on some deps)
    pip install hydra-core==1.3.1 omegaconf vaex pyarrow transformers accelerate datasets wandb evaluate scikit-learn rich gensim
else
    source .venv/bin/activate
fi

TRAIN_OVERRIDES="trainer.epochs=${EPOCHS:-10}"
# Optional extra hydra overrides injected by the pipeline driver (e.g.
# `seed=43`). Appended verbatim.
if [ -n "${EXTRA_TRAIN_OVERRIDES:-}" ]; then
    TRAIN_OVERRIDES="${TRAIN_OVERRIDES} ${EXTRA_TRAIN_OVERRIDES}"
fi
if [ "${SMOKE_TEST}" == "--smoke-test" ]; then
    # Stock `metrics=defaults` includes Precision_K / Recall_K up to k=15,
    # which crashes on a 200-sample smoke set when the label vocabulary is
    # smaller than 15. We ship `metrics=smoke` (only k=1) and
    # `callbacks=smoke` (SaveBestModel/EarlyStopping watching precision@1)
    # so the callback lookup still resolves. One epoch is enough for a
    # plumbing test; bigger batch size keeps the tiny eval pass O(1).
    TRAIN_OVERRIDES="metrics=smoke callbacks=smoke trainer.epochs=1 trainer.threshold_tuning=false dataloader.batch_size=8"
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python3 main.py \
    experiment=synth_mimic \
    data.data_filename="synth_mimic_${PREP_TAG}.feather" \
    data.split_filename="synth_mimic_${PREP_TAG}_split.feather" \
    +output_dir="${OUTPUT_DIR}" \
    ${TRAIN_OVERRIDES}

# 3. Save Results
echo "[3/3] Saving results..."
mkdir -p "${OUTPUT_DIR}"

# The metrics and model checkpoints will now be saved directly to $OUTPUT_DIR

echo "Pipeline completed successfully!"
echo "Check results in: ${OUTPUT_DIR}"
