#!/usr/bin/env bash
# =============================================================================
# Smoke test for the full pipeline.
#
# Forces MAX_SAMPLES=2 prompts × N_RUNS=5 outputs per prompt for each
# (dataset, fraction) combination, and uses a tiny epoch budget for all
# downstream tasks so the whole thing finishes in a few minutes. This does
# *not* try to produce meaningful metrics — it only verifies the plumbing.
#
# Usage:
#   bash scripts/run_smoke_tests.sh                 # all datasets, 5pct only
#   DATASETS="cares cwlc" bash scripts/run_smoke_tests.sh
#   FRACTIONS="5pct 100pct" bash scripts/run_smoke_tests.sh
# =============================================================================

set -euo pipefail

: "${DATASETS:=cares cwlc radgraph mimic_s mimic_l}"
: "${FRACTIONS:=5pct}"
: "${GPU_GEN:=1}"
: "${GPU_DOWN:=2}"

export SMOKE=1 MAX_SAMPLES=10 N_RUNS=5 STAGES="generate eval downstream"
export DATASETS FRACTIONS GPU_GEN GPU_DOWN

bash "$(dirname "$0")/run_full_pipeline.sh"
