#!/usr/bin/env bash
# =============================================================================
# Parallel orchestrator — distributes the (dataset × fraction) combos across
# 3 GPUs and runs `run_full_pipeline.sh` in parallel on each, one combo at a
# time per worker. Each worker uses ONE GPU for both generation and
# downstream (they run sequentially inside a combo), which is why we can
# launch three workers concurrently on three distinct GPUs.
#
# Defaults reflect the "error analysis via seed variance" setup:
#   * generation n=1 (reuses existing synth_*.json when present)
#   * downstream task repeated N_RUNS=3 times with different training seeds
#   * mimic_l is intentionally excluded (too slow to be worthwhile)
#
# Usage:
#   bash scripts/run_pipeline_3gpu.sh                      # everything
#   DATASETS="cares cwlc" bash scripts/run_pipeline_3gpu.sh
#   FRACTIONS="100pct 50pct" bash scripts/run_pipeline_3gpu.sh
#   STAGES="downstream" bash scripts/run_pipeline_3gpu.sh  # skip generate+eval
#
# Env overrides:
#   DATASETS         default: "cares cwlc radgraph mimic_s"
#   FRACTIONS        default: "5pct 25pct 50pct 100pct"
#   STAGES           default: "generate eval downstream"
#   N_RUNS           downstream seed-variance replicates (default 3)
#   N_GEN            outputs per prompt in generation (default 1)
#   GPU_LIST         space-separated GPU indices (default "0 1 2")
#   REUSE_SYNTH      1 to reuse existing synth files  (default 1)
#   RUN_ROUGE        1 to add ROUGE-5 in privacy eval (default 0)
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# ─── Defaults ─────────────────────────────────────────────────────────────────
: "${DATASETS:=cares cwlc radgraph mimic_s}"
: "${FRACTIONS:=5pct 25pct 50pct 100pct}"
: "${STAGES:=generate eval downstream}"
: "${N_RUNS:=3}"
: "${N_GEN:=1}"
: "${GPU_LIST:=0 1 2}"
: "${REUSE_SYNTH:=1}"
: "${RUN_ROUGE:=0}"
# TEXT_SOURCES is a space-separated list (e.g. "generated original"). Each
# entry spawns its own set of (dataset × fraction) combos so real vs synth
# downstream comparisons can be run in a single orchestrator invocation.
: "${TEXT_SOURCES:=generated}"

read -r -a GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}
if (( NUM_GPUS == 0 )); then
    echo "ERROR: GPU_LIST is empty" >&2
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PARALLEL_ROOT="${PROJECT_DIR}/output/parallel_${TIMESTAMP}"
mkdir -p "$PARALLEL_ROOT"

# ─── Build (ds, pct, text_source) combo list ─────────────────────────────────
# For TEXT_SOURCE=original the fraction doesn't matter (all original rows are
# used regardless). We therefore run it only once per dataset at 100pct to
# avoid redundant identical runs.
declare -a COMBOS=()
for ds in $DATASETS; do
    for ts in $TEXT_SOURCES; do
        if [[ "$ts" == "original" ]]; then
            COMBOS+=("${ds}:100pct:${ts}")
        else
            for pct in $FRACTIONS; do
                COMBOS+=("${ds}:${pct}:${ts}")
            done
        fi
    done
done

if (( ${#COMBOS[@]} == 0 )); then
    echo "ERROR: no combos to run" >&2
    exit 1
fi

# Per-GPU shard lists.
declare -A SHARD_DS SHARD_PCT
for i in "${!GPUS[@]}"; do
    SHARD_DS[$i]=""
    SHARD_PCT[$i]=""
done
for i in "${!COMBOS[@]}"; do
    gpu_idx=$(( i % NUM_GPUS ))
    combo="${COMBOS[$i]}"
    ds="${combo%%:*}"
    pct="${combo##*:}"
    # Accumulate unique (ds, pct) per shard; we concatenate and let the
    # child pipeline iterate. Because each shard calls run_full_pipeline.sh
    # once, we need to give it a DATASETS and FRACTIONS product that covers
    # exactly the combos assigned to this shard. Easiest: launch one child
    # per combo rather than trying to pack per-shard.
    :
done

# ─── Launch: one child per combo, GPU assigned round-robin ───────────────────
printf "╭────────────────────────────────────────────────────────────────────╮\n"
printf "│  Parallel pipeline orchestrator                                   │\n"
printf "├────────────────────────────────────────────────────────────────────┤\n"
printf "│  Combos       : %-50s │\n" "${#COMBOS[@]} total"
printf "│  GPUs         : %-50s │\n" "${GPU_LIST}"
printf "│  Stages       : %-50s │\n" "${STAGES}"
printf "│  N_RUNS       : %-50s │\n" "${N_RUNS}"
printf "│  N_GEN        : %-50s │\n" "${N_GEN}"
printf "│  REUSE_SYNTH  : %-50s │\n" "${REUSE_SYNTH}"
printf "│  Artifacts    : %-50s │\n" "${PARALLEL_ROOT}"
printf "╰────────────────────────────────────────────────────────────────────╯\n"

LOG_DIR="${PARALLEL_ROOT}/worker_logs"
mkdir -p "$LOG_DIR"

declare -a CHILD_PIDS=()
declare -A GPU_BUSY
for g in "${GPUS[@]}"; do GPU_BUSY[$g]=""; done

# Simple scheduler: iterate over combos; assign each to the first free GPU
# (blocking if all are busy). Each worker runs run_full_pipeline.sh once for
# that single combo.
pick_free_gpu() {
    while true; do
        for g in "${GPUS[@]}"; do
            pid="${GPU_BUSY[$g]}"
            if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
                # If previous job ended, reap it.
                if [[ -n "$pid" ]]; then
                    wait "$pid" 2>/dev/null || true
                fi
                GPU_BUSY[$g]=""
                echo "$g"
                return
            fi
        done
        sleep 3
    done
}

launch_combo() {
    local ds="$1" pct="$2" ts="$3" gpu="$4"
    local log="${LOG_DIR}/${ds}_${pct}_${ts}_gpu${gpu}.log"
    echo "[launch]  $(date +%H:%M:%S)  $ds $pct $ts → GPU $gpu  (log: $log)"
    (
        env \
            DATASETS="$ds" \
            FRACTIONS="$pct" \
            TEXT_SOURCE="$ts" \
            STAGES="$STAGES" \
            N_RUNS="$N_RUNS" \
            N_GEN="$N_GEN" \
            GPU_GEN="$gpu" \
            GPU_DOWN="$gpu" \
            REUSE_SYNTH="$REUSE_SYNTH" \
            RUN_ROUGE="$RUN_ROUGE" \
            bash "${PROJECT_DIR}/scripts/run_full_pipeline.sh"
    ) > "$log" 2>&1 &
    local pid=$!
    GPU_BUSY[$gpu]=$pid
    CHILD_PIDS+=("$pid")
}

T_START=$(date +%s)
for combo in "${COMBOS[@]}"; do
    IFS=':' read -r ds pct ts <<< "$combo"
    free_gpu=$(pick_free_gpu)
    launch_combo "$ds" "$pct" "$ts" "$free_gpu"
done

# Wait for the last batch to drain.
echo
echo "[orchestrator] waiting for remaining workers to finish..."
for pid in "${CHILD_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

T_END=$(date +%s)
ELAPSED=$(( T_END - T_START ))

echo
echo "╭────────────────────────────────────────────────────────────────────╮"
printf "│  Parallel run complete in %02dh%02dm                                  │\n" \
    $(( ELAPSED/3600 )) $(( (ELAPSED%3600)/60 ))
echo "╰────────────────────────────────────────────────────────────────────╯"

# Aggregate per-combo SUMMARY.md blobs into one master table.
MASTER="${PARALLEL_ROOT}/MASTER_SUMMARY.md"
{
    echo "# Parallel pipeline master summary — ${TIMESTAMP}"
    echo
    echo "Datasets: ${DATASETS}"
    echo "Fractions: ${FRACTIONS}"
    echo "Text sources: ${TEXT_SOURCES}"
    echo "Stages: ${STAGES}"
    echo "N_RUNS (seeds): ${N_RUNS}"
    echo "N_GEN: ${N_GEN}"
    echo "GPUs used: ${GPU_LIST}"
    echo "Wall clock: $((ELAPSED/3600))h$(( (ELAPSED%3600)/60 ))m"
    echo
    echo "| Dataset | Fraction | TextSrc | Generate | tok/doc | Downstream F1 |"
    echo "|---------|----------|---------|----------|---------|---------------|"
} > "$MASTER"

# Scrape the per-worker SUMMARY.md entries. Each run_full_pipeline invocation
# creates a fresh output/multirun_<ts>/SUMMARY.md — we glob them from this
# parallel run's time window.
for log in "$LOG_DIR"/*.log; do
    # Extract the multirun_<ts> artifacts dir that the child printed.
    artif=$(grep -oE "Artifacts: [^ ]*" "$log" | head -n 1 | awk '{print $2}')
    [[ -z "$artif" || ! -f "$artif/SUMMARY.md" ]] && continue
    # Grab only table rows (lines that start with "| " and are not headers).
    awk '/^\| [a-z]/ {print}' "$artif/SUMMARY.md" >> "$MASTER"
done

echo "Master summary: $MASTER"
echo "Per-worker logs: $LOG_DIR/"
