#!/usr/bin/env bash
# =============================================================================
# End-to-end 5-run synthetic EHR pipeline (generation → fidelity/privacy →
# downstream task) with per-run error analysis.
#
# For each (dataset, fraction) combination the script
#   1. Discovers the fine-tuned merged model under output/<DATASET>/
#   2. Generates 5 completions per prompt with vLLM (output1..output5)
#   3. Runs fidelity + privacy on the 5 runs and aggregates mean ± std
#   4. Runs the downstream task once per run (5 trainings) and aggregates
#      mean ± std of the headline F1 metric
#
# Supported datasets (pick one or more via DATASETS env var):
#   cares      CARES HuggingFace corpus (ICD-10 chapters classification)
#   cwlc       CWLC ZIP corpus          (Spanish clinical NER)
#   radgraph   RadGraph-XL JSONL corpus (English radiology NER)
#   mimic_s    MIMIC-IV  (small — data/mimic/mimic_s.json, PLM-ICD downstream)
#   mimic_l    MIMIC-IV  (large — data/mimic/mimic_l.json, PLM-ICD downstream)
#   all        ≡ cares cwlc radgraph mimic_s mimic_l
#
# Typical invocations ---------------------------------------------------------
#
#   # Full run, all datasets, all fractions, all stages
#   DATASETS=all bash scripts/run_full_pipeline.sh
#
#   # Only CARES at 100% and 50%, skip downstream (fast iteration)
#   DATASETS=cares FRACTIONS="100pct 50pct" STAGES="generate eval" \
#       bash scripts/run_full_pipeline.sh
#
#   # Smoke test — 2 samples, 5 generations each, epochs≈0.05, no ROUGE
#   SMOKE=1 DATASETS=all bash scripts/run_full_pipeline.sh
#
# Environment knobs (override before invocation) ------------------------------
#   DATASETS     Space-separated dataset tags               (default: all)
#   FRACTIONS    Space-separated fraction tags              (default: 5pct 25pct 50pct 100pct)
#   STAGES       Subset of {generate eval downstream}       (default: all three)
#   N_RUNS       Outputs per prompt                         (default: 5)
#   GPU_GEN      GPUs for vLLM                              (default: 1)      — large GPU
#   GPU_DOWN     GPU for downstream training                (default: 2)
#   SMOKE        1 to enable minimal smoke-test settings    (default: 0)
#   MAX_SAMPLES  Extra cap on input records (overrides SMOKE)
#   RUN_ROUGE    1 to compute ROUGE-5 in eval (slow)        (default: 0)
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Paths + env
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PY_VLLM="${PROJECT_DIR}/.venv-vllm/bin/python"
# The fidelity/privacy eval needs nltk + rouge_score, which live in the
# miniconda `base` env. Collect + materialize are stdlib-only so they also
# work under base. Override with PY_BASE=... to point at a different
# interpreter.
PY_BASE="${PY_BASE:-/home/lmiranda/miniconda3/bin/python}"
GENERATE_SCRIPT="${PROJECT_DIR}/src/generate_all_vllm.py"
EVAL_SCRIPT="${PROJECT_DIR}/src/run_multirun_eval.py"
MATERIALIZE_SCRIPT="${PROJECT_DIR}/src/materialize_runk.py"
COLLECT_SCRIPT="${PROJECT_DIR}/src/collect_downstream_metrics.py"

if [[ ! -x "$PY_VLLM" ]]; then
    echo "ERROR: vLLM venv python not found at $PY_VLLM" >&2
    exit 1
fi

CONDA_SH="/home/lmiranda/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV_DOWN="${CONDA_ENV_DOWN:-unsloth-test}"
# Absolute path to the downstream-training python (has torch/transformers).
# train_icd10_multilabel.sh honours the PYTHON_BIN env var — we point it
# here so the ICD-10 classifier runs in the right environment without
# activating conda just for that subprocess.
PY_DOWN="${PY_DOWN:-/home/lmiranda/miniconda3/envs/${CONDA_ENV_DOWN}/bin/python}"
if [[ ! -x "$PY_DOWN" ]]; then
    echo "ERROR: downstream python not found at $PY_DOWN" >&2
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Configuration with sensible defaults
# ─────────────────────────────────────────────────────────────────────────────
DATASETS="${DATASETS:-all}"
if [[ "$DATASETS" == "all" ]]; then
    # MIMIC large is intentionally excluded from the default set because
    # generating 89k prompts × n=1 with gpt-oss-20b dominates the wall time
    # (~50h). Pass DATASETS="... mimic_l" explicitly if you really want it.
    DATASETS="cares cwlc radgraph mimic_s"
fi
FRACTIONS="${FRACTIONS:-5pct 25pct 50pct 100pct}"
STAGES="${STAGES:-generate eval downstream}"

# N_RUNS is the number of downstream trainings per (dataset, fraction). We
# vary the training seed (42, 43, 44, ...) and aggregate mean ± std across
# those runs, which gives training-noise error bars without needing
# multiple generations.
N_RUNS="${N_RUNS:-3}"

# Outputs per prompt (vLLM) — kept at 1 since the error analysis now lives
# on the downstream-seed axis, not on the generation axis.
N_GEN="${N_GEN:-1}"

# When REUSE_SYNTH=1 (default), the pipeline scans output/<DS>/ for an
# existing synth_<ds>_<pct>_*.json and skips the generate stage if a
# suitable file is found. Set REUSE_SYNTH=0 to force regeneration.
REUSE_SYNTH="${REUSE_SYNTH:-1}"

GPU_GEN="${GPU_GEN:-1}"
GPU_DOWN="${GPU_DOWN:-2}"
SMOKE="${SMOKE:-0}"
RUN_ROUGE="${RUN_ROUGE:-0}"
# TEXT_SOURCE selects what the downstream model is TRAINED on:
#   generated = synth_*.json output1 field (synthetic)
#   original  = synth_*.json original_text field (real data, same rows)
# The eval test set is always the real held-out split.
TEXT_SOURCE="${TEXT_SOURCE:-generated}"
# Downstream seeds — N_RUNS entries consumed by the 3× training loop.
DOWNSTREAM_SEEDS="${DOWNSTREAM_SEEDS:-42 43 44 45 46 47 48 49 50 51}"

# Smoke-test knobs — minimal samples, tiny epochs, no ROUGE.
# We need at least ~10 prompts so that the NER encoder's train/val/test
# split math produces non-empty splits. 10 prompts × 5 generations is still
# fast enough for a full end-to-end smoke run. For the ICD-10 classifier we
# use 1 epoch (effectively <1 step over 10 samples) because a fractional
# epoch over 10 records rounds down to 0 training steps and crashes.
if [[ "$SMOKE" == "1" ]]; then
    MAX_SAMPLES="${MAX_SAMPLES:-10}"
    SMOKE_EPOCHS_ICD10="1"            # 1 epoch × 10 samples ≈ a few steps
    SMOKE_EPOCHS_NER="1"              # NER script takes int epochs; 1 is the floor
    SMOKE_NER_TEST_SIZE="0.2"
    SMOKE_NER_VAL_SIZE="0.1"
    SMOKE_PLMICD_FLAG="--smoke-test"
    RUN_ROUGE=0
else
    MAX_SAMPLES="${MAX_SAMPLES:-}"
    SMOKE_EPOCHS_ICD10=""
    SMOKE_EPOCHS_NER=""
    SMOKE_NER_TEST_SIZE=""
    SMOKE_NER_VAL_SIZE=""
    SMOKE_PLMICD_FLAG=""
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
# Include PID so parallel workers launched in the same second (via the
# 3-GPU orchestrator) don't collide on the same artifacts directory and
# end up clobbering each other's SUMMARY.md.
RUN_ROOT="${PROJECT_DIR}/output/multirun_${TIMESTAMP}_$$"
mkdir -p "$RUN_ROOT" "${PROJECT_DIR}/output/logs"

# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing helpers
# ─────────────────────────────────────────────────────────────────────────────
hr() { printf '━%.0s' {1..70}; echo; }
banner() {
    hr
    printf "  %s\n" "$*"
    hr
}
log() { printf "  [%s] %s\n" "$(date +%H:%M:%S)" "$*"; }

# Duration formatter (mm:ss) for human-readable ETAs
fmt_dur() {
    local s=$1
    printf "%dm%02ds" $((s/60)) $((s%60))
}

# ─────────────────────────────────────────────────────────────────────────────
# Model discovery — returns path to merged model via stdout
# ─────────────────────────────────────────────────────────────────────────────
discover_existing_synth() {
    # $1: dataset tag   $2: fraction tag
    # The output/<DS>/ directories are expected to contain exactly one
    # synth file per fraction (duplicates and partial runs were pruned).
    # We still guard with `ls -1 | head -n 1` in case a new regeneration
    # lands alongside the current keeper.
    local ds="$1" pct="$2"
    local dir pat
    case "$ds" in
        cares)    dir="${PROJECT_DIR}/output/CARES"    ; pat="synth_cares_${pct}_*.json" ;;
        cwlc)     dir="${PROJECT_DIR}/output/CWLC"     ; pat="synth_cwlc_${pct}_*.json" ;;
        radgraph) dir="${PROJECT_DIR}/output/RADGRAPH" ; pat="synth_radgraph_${pct}_*.json" ;;
        mimic_s)  dir="${PROJECT_DIR}/output/MIMIC"    ; pat="synth_mimic_small_${pct}_*.json" ;;
        mimic_l)  dir="${PROJECT_DIR}/output/MIMIC"    ; pat="synth_mimic_large_${pct}_*.json" ;;
        *) return 1 ;;
    esac
    local match
    match=$(ls -1 $dir/$pat 2>/dev/null | head -n 1 || true)
    [[ -z "$match" || ! -f "$match" ]] && return 1
    printf "%s" "$match"
}

discover_model() {
    # $1: dataset tag  $2: fraction tag
    local ds="$1" pct="$2"
    local glob=""
    case "$ds" in
        cares)    glob="${PROJECT_DIR}/output/CARES/gpt-oss-20b_${pct}_*_merged" ;;
        cwlc)     glob="${PROJECT_DIR}/output/CWLC/gpt-oss-20b_${pct}_*_merged" ;;
        radgraph) glob="${PROJECT_DIR}/output/RADGRAPH/gpt-oss-20b_${pct}_*continued*_merged" ;;
        mimic_s)  glob="${PROJECT_DIR}/output/MIMIC/gpt-oss-20b_${pct}_small_continued_*_merged" ;;
        mimic_l)  glob="${PROJECT_DIR}/output/MIMIC/gpt-oss-20b_${pct}_large_continued_*_merged" ;;
        *) echo "" ; return 1 ;;
    esac
    # Use ls -1dt to pick the most recently modified merged directory when
    # multiple timestamps exist for the same fraction.
    local match
    match=$(ls -1dt $glob 2>/dev/null | head -n 1 || true)
    if [[ -z "$match" || ! -d "$match" ]]; then
        return 1
    fi
    printf "%s" "$match"
}

# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset generation runner
# ─────────────────────────────────────────────────────────────────────────────
run_generation() {
    # $1: ds  $2: pct  $3: model_path  $4: synth_out_file
    local ds="$1" pct="$2" model="$3" out_file="$4"
    local gen_ds max_len max_tok args
    args=(--model-path "$model"
          --output-file "$out_file"
          --n "$N_GEN"
          --tensor-parallel-size 1
          --gpu-memory-utilization 0.90
          --max-num-seqs 45)
    case "$ds" in
        cares)
            gen_ds="cares"; max_tok=4000; max_len=4096
            ;;
        cwlc)
            gen_ds="cwlc";  max_tok=4000; max_len=4096
            args+=(--cwlc-zip "${PROJECT_DIR}/data/cwlc.zip")
            ;;
        radgraph)
            gen_ds="radgraph"; max_tok=4000; max_len=4096
            args+=(--radgraph-jsonl
                   "${PROJECT_DIR}/data/RADGRAPH/mimic-radgraph-XL.jsonl"
                   "${PROJECT_DIR}/data/RADGRAPH/stanford-radgraph-XL.jsonl")
            ;;
        mimic_s)
            gen_ds="mimic"; max_tok=6000; max_len=4096
            args+=(--mimic-json "${PROJECT_DIR}/data/mimic/mimic_s.json")
            ;;
        mimic_l)
            gen_ds="mimic"; max_tok=6000; max_len=4096
            args+=(--mimic-json "${PROJECT_DIR}/data/mimic/mimic_l.json")
            ;;
    esac
    args=(--dataset "$gen_ds" --max-tokens "$max_tok" --max-model-len "$max_len" "${args[@]}")
    if [[ -n "$MAX_SAMPLES" ]]; then
        args+=(--max-samples "$MAX_SAMPLES")
    fi

    log "  vLLM command: CUDA_VISIBLE_DEVICES=$GPU_GEN $PY_VLLM $GENERATE_SCRIPT ${args[*]}"
    CUDA_VISIBLE_DEVICES="$GPU_GEN" CUDA_DEVICE_ORDER=PCI_BUS_ID \
        "$PY_VLLM" "$GENERATE_SCRIPT" "${args[@]}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset downstream runner (expects a materialised run-k JSON)
# ─────────────────────────────────────────────────────────────────────────────
run_downstream_one() {
    # $1: ds  $2: pct  $3: run_idx  $4: synth_json  $5: run_out_dir  $6: seed
    local ds="$1" pct="$2" k="$3" synth_json="$4" run_out="$5" seed="$6"
    mkdir -p "$run_out"

    case "$ds" in
        cares)
            local epochs_arg=""
            if [[ -n "$SMOKE_EPOCHS_ICD10" ]]; then
                epochs_arg="$SMOKE_EPOCHS_ICD10"
            fi
            # Match the reference run that achieved f1_micro_flat≈0.92:
            #   model=PlanTL-GOB-ES/bsc-bio-ehr-es (Spanish biomedical),
            #   epochs=40 (early-stopping patience=5 kicks in ≈ epoch 25),
            #   lr=4e-5, warmup_steps=300, batch=32.
            # Bio_ClinicalBERT is English-only and collapses on CARES.
            env \
                PYTHON_BIN="$PY_DOWN" \
                MODEL="PlanTL-GOB-ES/bsc-bio-ehr-es" \
                EPOCHS="${epochs_arg:-${EPOCHS:-40}}" \
                SEED="$seed" \
                CUDA_VISIBLE_DEVICES="$GPU_DOWN" \
                SYNTH_JSON_OVERRIDE="$synth_json" \
                OUTPUT_DIR_OVERRIDE="$run_out" \
                MAX_TRAIN_SAMPLES="${MAX_SAMPLES:-}" \
                MAX_EVAL_SAMPLES="${MAX_SAMPLES:-}" \
                bash "${PROJECT_DIR}/scripts/train_icd10_multilabel_cares.sh" "$pct" "$TEXT_SOURCE"
            ;;

        cwlc|radgraph)
            local model_name
            if [[ "$ds" == "cwlc" ]]; then
                # bsc-bio-ehr-es: Spanish biomedical EHR model (RoBERTa-based),
                # trained on clinical text — matches CWLC domain much better than
                # the general-purpose bert-base-spanish-wwm-cased.
                model_name="PlanTL-GOB-ES/bsc-bio-ehr-es"
            else
                model_name="roberta-base"
            fi
            # 20 epochs with load_best_model_at_end + early stopping (patience=3
            # via best-model tracking) — safe to set high since the trainer saves
            # and restores the best checkpoint automatically.
            local epochs="${EPOCHS:-20}"
            if [[ -n "$SMOKE_EPOCHS_NER" ]]; then epochs="$SMOKE_EPOCHS_NER"; fi
            local extra=()
            if [[ -n "$SMOKE_NER_TEST_SIZE" ]]; then extra+=(--test-size "$SMOKE_NER_TEST_SIZE"); fi
            if [[ -n "$SMOKE_NER_VAL_SIZE" ]];  then extra+=(--val-size  "$SMOKE_NER_VAL_SIZE"); fi
            # Map pipeline TEXT_SOURCE to the NER encoder's --text-field:
            #   generated → output1 (synthetic)
            #   original  → original_text (real, eval test is always real anyway)
            local text_field="output1"
            if [[ "$TEXT_SOURCE" == "original" ]]; then text_field="original_text"; fi
            source "$CONDA_SH" && conda activate "$CONDA_ENV_DOWN"
            CUDA_VISIBLE_DEVICES="$GPU_DOWN" \
                python -u "${PROJECT_DIR}/src/train_ner_encoder.py" \
                    --jsonl "$synth_json" \
                    --model-name "$model_name" \
                    --text-field "$text_field" \
                    --output-dir "$run_out" \
                    --seed "$seed" \
                    --epochs "$epochs" \
                    "${extra[@]}"
            conda deactivate
            ;;

        mimic_s|mimic_l)
            # run_plm_icd_mimic.sh activates its own .venv inside the
            # plm_icd repo, so we pass data/output overrides + SEED env.
            # Arg 2 is text-source; prepare_synthetic_mimic.py honours
            # "original" by reading the raw `output` field.
            CUDA_VISIBLE_DEVICES="$GPU_DOWN" \
            SYNTH_JSON_OVERRIDE="$synth_json" \
            OUTPUT_DIR_OVERRIDE="$run_out" \
            RUN_TAG="run${k}_${ds}_${TEXT_SOURCE}" \
            EXTRA_TRAIN_OVERRIDES="seed=$seed" \
                bash "${PROJECT_DIR}/scripts/run_plm_icd_mimic.sh" "$pct" "$TEXT_SOURCE" $SMOKE_PLMICD_FLAG
            ;;
    esac
}

# ─────────────────────────────────────────────────────────────────────────────
# ETA estimator — prints a rough budget table before starting
# ─────────────────────────────────────────────────────────────────────────────
print_eta_budget() {
    local num_frac num_ds total
    num_frac=$(echo "$FRACTIONS" | wc -w)
    num_ds=$(echo "$DATASETS" | wc -w)
    total=$((num_frac * num_ds))
    hr
    echo "  Pipeline budget"
    hr
    printf "  Datasets  : %s\n" "$DATASETS"
    printf "  Fractions : %s\n" "$FRACTIONS"
    printf "  TextSrc   : %s\n" "$TEXT_SOURCE"
    printf "  Stages    : %s\n" "$STAGES"
    printf "  N runs    : %s\n" "$N_RUNS"
    printf "  Combos    : %d dataset × %d fraction = %d jobs\n" "$num_ds" "$num_frac" "$total"
    printf "  GPU gen   : %s      GPU downstream: %s\n" "$GPU_GEN" "$GPU_DOWN"
    printf "  SMOKE     : %s\n" "$SMOKE"
    hr
    cat <<'EOF'
  Rough wall-clock estimates (single RTX 6000 Ada, full mode):
    generate step :  ~15–40 min per fraction per dataset (vLLM, n=5)
    fidelity/priv :  ~1–3  min per fraction per dataset (5-run aggregate)
    downstream    :  ~5–20 min per run × 5 runs ≈ 25–100 min per fraction
  Full all-datasets × 4 fractions budget: ~18–36 hours.
  The smoke profile takes ~3–8 min total across all datasets.
EOF
    hr
}

# ─────────────────────────────────────────────────────────────────────────────
# Top-level loop
# ─────────────────────────────────────────────────────────────────────────────
print_eta_budget

T_START=$(date +%s)
SUMMARY_FILE="${RUN_ROOT}/SUMMARY.md"
{
    echo "# Full pipeline summary — ${TIMESTAMP}"
    echo
    echo "Datasets: ${DATASETS}"
    echo "Fractions: ${FRACTIONS}"
    echo "TextSource: ${TEXT_SOURCE}"
    echo "Stages: ${STAGES}"
    echo "N runs: ${N_RUNS}"
    echo "Smoke: ${SMOKE}"
    echo
    echo "| Dataset | Fraction | TextSrc | Generate | tok/doc | Downstream F1 |"
    echo "|---------|----------|---------|----------|---------|---------------|"
} > "$SUMMARY_FILE"

stage_includes() { [[ " $STAGES " == *" $1 "* ]]; }

for ds in $DATASETS; do
    for pct in $FRACTIONS; do
        banner "Dataset=$ds   Fraction=$pct   TextSource=$TEXT_SOURCE"
        combo_out="${RUN_ROOT}/${ds}_${pct}_${TEXT_SOURCE}"
        mkdir -p "$combo_out"

        model_path=""
        if model_path=$(discover_model "$ds" "$pct"); then
            log "Model: $model_path"
        else
            log "WARN: no merged model found for $ds $pct — skipping."
            echo "| $ds | $pct | $TEXT_SOURCE | ❌ no-model | — | — |" >> "$SUMMARY_FILE"
            continue
        fi

        # --- STAGE 1: generate (or reuse an existing synth file) ------------
        synth_json="${combo_out}/synth_${ds}_${pct}.json"
        reused=""
        if [[ "$REUSE_SYNTH" == "1" ]] && existing_synth=$(discover_existing_synth "$ds" "$pct"); then
            log "Reusing existing synth: $existing_synth"
            synth_json="$existing_synth"
            reused="$existing_synth"
            gen_cell="♻ reused"
        elif stage_includes generate; then
            t0=$(date +%s)
            log "[1/3] vLLM generation (n=$N_GEN)"
            if run_generation "$ds" "$pct" "$model_path" "$synth_json" 2>&1 \
                | tee "${combo_out}/generate.log"; then
                dt=$(( $(date +%s) - t0 ))
                log "  ✓ generate complete in $(fmt_dur $dt)"
                gen_cell="✓ $(fmt_dur $dt)"
            else
                log "  ✗ generate FAILED"
                gen_cell="❌"
                echo "| $ds | $pct | $TEXT_SOURCE | $gen_cell | — | — |" >> "$SUMMARY_FILE"
                continue
            fi
        else
            existing=$(ls -1t "${combo_out}"/synth_*.json 2>/dev/null | head -n 1 || true)
            [[ -n "$existing" ]] && synth_json="$existing"
            gen_cell="⏭ skipped"
        fi

        if [[ ! -f "$synth_json" ]]; then
            log "  ✗ no synth json available — skipping remaining stages"
            echo "| $ds | $pct | $TEXT_SOURCE | $gen_cell | — | — |" >> "$SUMMARY_FILE"
            continue
        fi

        # --- STAGE 2: fidelity + privacy multirun eval -----------------------
        tok_cell="—"
        if stage_includes eval; then
            t0=$(date +%s)
            log "[2/3] Multi-run fidelity+privacy eval"
            eval_out="${combo_out}/multirun_eval.json"
            eval_args=(--input-file "$synth_json"
                       --output-file "$eval_out"
                       --dataset "$ds")
            if [[ -n "$MAX_SAMPLES" ]]; then eval_args+=(--max-samples "$MAX_SAMPLES"); fi
            if [[ "$RUN_ROUGE" == "1" ]]; then eval_args+=(--run-rouge); fi
            if "$PY_BASE" "$EVAL_SCRIPT" "${eval_args[@]}" 2>&1 | tee "${combo_out}/eval.log"; then
                dt=$(( $(date +%s) - t0 ))
                log "  ✓ eval complete in $(fmt_dur $dt)"
                # Single tok/doc value: for generated → synth avg; for original → real avg.
                tok_val=$("$PY_BASE" -c "
import json, sys
d = json.load(open(sys.argv[1]))
text_source = '${TEXT_SOURCE}'
if text_source == 'original':
    v = d.get('aggregate', {}).get('fidelity_real', {}).get('avg_tokens_per_document', {}).get('mean')
else:
    v = d.get('aggregate', {}).get('fidelity_generated', {}).get('avg_tokens_per_document', {}).get('mean')
print(f'{v:.1f}' if v is not None else '-')
" "$eval_out")
                tok_cell="${tok_val} ($(fmt_dur $dt))"
            else
                log "  ✗ eval FAILED"
                tok_cell="❌"
            fi
        fi

        # --- STAGE 3: downstream × N_RUNS (different seeds, same synth) -----
        down_cell="—"
        if stage_includes downstream; then
            t0=$(date +%s)
            log "[3/3] Downstream task × $N_RUNS runs (seed variance)"
            down_root="${combo_out}/downstream_runs"
            mkdir -p "$down_root"
            downstream_failed=0
            collect_dirs=()

            # Slice N_RUNS seeds out of the DOWNSTREAM_SEEDS list.
            read -r -a _seeds_arr <<< "$DOWNSTREAM_SEEDS"
            for k in $(seq 1 "$N_RUNS"); do
                run_out="${down_root}/run${k}"
                seed="${_seeds_arr[$((k-1))]:-$((41 + k))}"
                log "  → training run ${k}/${N_RUNS}  (seed=$seed)"
                if run_downstream_one "$ds" "$pct" "$k" "$synth_json" "$run_out" "$seed" \
                        2>&1 | tee -a "${combo_out}/downstream.log"; then
                    collect_dirs+=("$run_out")
                    log "    ✓ run${k} done"
                else
                    log "    ✗ run${k} failed"
                    downstream_failed=$((downstream_failed+1))
                fi
            done

            # Collect metrics across the successful runs.
            if [[ ${#collect_dirs[@]} -gt 0 ]]; then
                task_kind=""
                case "$ds" in
                    cwlc|radgraph)  task_kind="ner" ;;
                    cares)          task_kind="icd10" ;;
                    mimic_s|mimic_l) task_kind="plmicd" ;;
                esac
                collect_args=(--task "$task_kind"
                              --output-file "${combo_out}/downstream_aggregate.json"
                              --dataset "$ds"
                              --fraction "$pct")
                for d in "${collect_dirs[@]}"; do
                    collect_args+=(--run-dir "$d")
                done
                "$PY_BASE" "$COLLECT_SCRIPT" "${collect_args[@]}" \
                    2>&1 | tee -a "${combo_out}/downstream.log" || true
                # Extract headline F1 for the summary table.
                f1_cell=$("$PY_BASE" -c "
import json, sys
p = sys.argv[1]
try:
    d = json.load(open(p))
except Exception:
    print('—'); sys.exit(0)
agg = d.get('aggregate', {})
key = None
for cand in ('f1', 'f1_sample', 'f1_micro', 'f1_macro'):
    if cand in agg and agg[cand].get('mean') is not None:
        key = cand; break
if key is None:
    print('—')
else:
    m, s = agg[key]['mean'], agg[key]['std']
    print(f'{key}={m:.4f} ± {s:.4f}')
" "${combo_out}/downstream_aggregate.json")
            else
                f1_cell="❌ all runs failed"
            fi
            dt=$(( $(date +%s) - t0 ))
            down_cell="$f1_cell ($(fmt_dur $dt))"
        fi

        echo "| $ds | $pct | $TEXT_SOURCE | $gen_cell | $tok_cell | $down_cell |" >> "$SUMMARY_FILE"
    done
done

T_END=$(date +%s)
TOTAL=$(( T_END - T_START ))

{
    echo
    echo "Total wall clock: $(fmt_dur $TOTAL)"
    echo "Artifacts under: $RUN_ROOT"
} >> "$SUMMARY_FILE"

banner "Pipeline finished in $(fmt_dur $TOTAL)"
echo "  Summary: $SUMMARY_FILE"
echo "  Artifacts: $RUN_ROOT"
