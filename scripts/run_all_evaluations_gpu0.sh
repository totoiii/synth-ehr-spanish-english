#!/bin/bash
# =============================================================================
# RUN ALL EVALUATIONS (Fidelity, Privacy, NER) — GPU 0
# =============================================================================
# Usage:
#   screen -S eval_gpu0
#   bash scripts/run_all_evaluations_gpu0.sh
# =============================================================================

set -e
set -o pipefail

## Configuration
GPU_ID=0
export CUDA_VISIBLE_DEVICES=${GPU_ID}
#|
## Activate conda
#
#
#echo "╔══════════════════════════════════════════════════════════════════════╗"
#echo "║  RUNNING ALL EVALUATIONS — GPU ${GPU_ID}                                   ║"
#echo "║  1. Fidelity & Privacy (run_evaluation.sh)                         ║"
#echo "║  2. NER Training (CWLC & RADGRAPH)                                 ║"
#echo "║  Started: $(date)                                                  ║"
#echo "╚══════════════════════════════════════════════════════════════════════╝"
#echo ""
#
## ─────────────────────────────────────────────────────────────────────────────
## 1. Fidelity & Privacy Evaluation (CPU/Light GPU)
## ─────────────────────────────────────────────────────────────────────────────
#echo "██████████████████████████████████████████████████████████████████████"
#echo "█  STEP 1: Fidelity & Privacy Evaluation                           █"
#echo "██████████████████████████████████████████████████████████████████████"
#
## Run on all files found in output directories (automatically finds new ones)
## SKIP_ROUGE=1 to speed up (privacy check is slow otherwise)
#bash scripts/run_evaluation.sh
#
#echo "✅ Fidelity & Privacy Evaluation Complete"
#echo ""
#
#source /home/lmiranda/miniconda3/etc/profile.d/conda.sh
#conda activate unsloth-test
#
## ─────────────────────────────────────────────────────────────────────────────
## 2. NER Experiments — CWLC
## ─────────────────────────────────────────────────────────────────────────────
#echo "██████████████████████████████████████████████████████████████████████"
#echo "█  STEP 2: NER Experiments — CWLC                                  █"
#echo "██████████████████████████████████████████████████████████████████████"
#
## Explicitly defining the NEW files (based on Feb 19 timestamps)
## 100%: .../synth_cwlc_100pct_20260219_092658.json
## 50%:  .../synth_cwlc_50pct_20260219_121519.json
## 25%:  .../synth_cwlc_25pct_20260219_150626.json
## 5%:   .../synth_cwlc_5pct_20260219_174956.json
#
## Create a temporary copy of the script to modify file paths
#cp scripts/train_ner_cwlc_.sh scripts/train_ner_cwlc_gpu0.sh
#
## Update file paths in the script using sed
#sed -i 's|FILE_100=.*|FILE_100="./output/CWLC/synth_cwlc_100pct_20260219_092658.json"|' scripts/train_ner_cwlc_gpu0.sh
#sed -i 's|FILE_50=.*|FILE_50="./output/CWLC/synth_cwlc_50pct_20260219_121519.json"|' scripts/train_ner_cwlc_gpu0.sh
#sed -i 's|FILE_25=.*|FILE_25="./output/CWLC/synth_cwlc_25pct_20260219_150626.json"|' scripts/train_ner_cwlc_gpu0.sh
#sed -i 's|FILE_5=.*|FILE_5="./output/CWLC/synth_cwlc_5pct_20260219_174956.json"|' scripts/train_ner_cwlc_gpu0.sh
#
## Run the modified script
#bash scripts/train_ner_cwlc_gpu0.sh ${GPU_ID}
#
#echo "✅ NER CWLC Complete"
#echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 3. NER Experiments — RADGRAPH
# ─────────────────────────────────────────────────────────────────────────────
echo "██████████████████████████████████████████████████████████████████████"
echo "█  STEP 3: NER Experiments — RADGRAPH                              █"
echo "██████████████████████████████████████████████████████████████████████"

# Explicitly defining the NEW files (based on Feb 19 timestamps)
# 100%: .../synth_radgraph_100pct_20260219_075046.json
# 50%:  .../synth_radgraph_50pct_20260219_103351.json
# 25%:  .../synth_radgraph_25pct_20260219_132552.json
# 5%:   .../synth_radgraph_5pct_20260219_161326.json

# Create a temporary copy of the script
cp scripts/train_ner_radgraph.sh scripts/train_ner_radgraph_gpu0.sh

# Update file paths
sed -i 's|FILE_100=.*|FILE_100="./output/RADGRAPH/synth_radgraph_100pct_20260219_075046.json"|' scripts/train_ner_radgraph_gpu0.sh
# Also need to add/update 50% line since original script commented it out or didn't have it
# Assuming original script had variable definitions we can replace.
# If FILE_50 was missing/commented, we append it or replace a placeholder.
# Let's just robustly replace lines or append if needed? 
# The simplest is to replace the known existing lines.
sed -i 's|FILE_25=.*|FILE_25="./output/RADGRAPH/synth_radgraph_25pct_20260219_132552.json"|' scripts/train_ner_radgraph_gpu0.sh
sed -i 's|FILE_5=.*|FILE_5="./output/RADGRAPH/synth_radgraph_5pct_20260219_161326.json"|' scripts/train_ner_radgraph_gpu0.sh

# Special handling for 50%: The original script might not have it active.
# We'll just define it and ensure the FILES array includes it if possible.
# But for safety/simplicity, I will just stick to 100, 25, 5 if that's what the script supports,
# OR force update the FILES array definition in the script via sed.

# Updating FILES array to include 50pct if desired:
# Original: declare -a FILES=("$FILE_100" "$FILE_25" "$FILE_5")
# Let's try to add FILE_50 to variable defs and array.
sed -i '/FILE_100=/a FILE_50="./output/RADGRAPH/synth_radgraph_50pct_20260219_103351.json"' scripts/train_ner_radgraph_gpu0.sh
sed -i 's|declare -a FILES=(".*"|declare -a FILES=("$FILE_100" "$FILE_50" "$FILE_25" "$FILE_5")|' scripts/train_ner_radgraph_gpu0.sh
sed -i 's|declare -a TAGS=(".*"|declare -a TAGS=("100pct" "50pct" "25pct" "5pct")|' scripts/train_ner_radgraph_gpu0.sh

# Run the modified script
bash scripts/train_ner_radgraph_gpu0.sh ${GPU_ID}

echo "✅ NER RADGRAPH Complete"
echo ""

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  🎉 ALL EVALUATIONS COMPLETE (GPU ${GPU_ID})                               ║"
echo "║  Finished: $(date)                                                 ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
