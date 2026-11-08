#!/bin/bash
# Paper-protocol replication: thresholds fitted on VALIDATION, then ONE pass over TEST.
# Paper subset IDs + frozen sensitivity exclusions. Refuses to rerun a prompt that already has
# a result (no repeated test peeking). Submit from the repo root:
#   sbatch --export=ALL,PROMPT=BINARY_BASELINE scripts/slurm/paper_eval.sbatch
#SBATCH --job-name=ss-papereval
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:45:00
#SBATCH --output=results/slurm-%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# Study hardware is fixed (decision 2026-11-02, revised): H100 NVL, bfloat16. Refuse anything else.
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
case "$GPU_NAME" in *H100*NVL*) ;; *) echo "Refusing: study GPU is H100 NVL, got $GPU_NAME" >&2; exit 2;; esac
source .venv/bin/activate
DTYPE=bfloat16                                 # fixed for the whole study
: "${PROMPT:?set PROMPT, e.g. BINARY_BASELINE}"
REV="${REV:-0e9e39f249a16976918f6564b8830bc894c89659}"
export HF_HUB_OFFLINE=1                        # never download inside a job
if ls results/paper_eval_"${PROMPT}"_"${DTYPE}"_*.json >/dev/null 2>&1; then
  echo "Refusing: a test-set result for $PROMPT already exists (evaluate once)." >&2; exit 3
fi
OUT="results/paper_eval_${PROMPT}_${DTYPE}_${SLURM_JOB_ID}.json"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
safety-study evaluate --protocol paper --paper-prompt "$PROMPT" \
  --schema decomposedharm --index-base 1 \
  --validation data/raw/decomposedharm/combined_val.csv \
  --test data/raw/decomposedharm/combined_test.csv \
  --ids configs/paper_eval_split.json --exclude configs/sensitivity_exclusions.json \
  --backend hf --model-id meta-llama/Llama-3.1-8B-Instruct --model-revision "$REV" --dtype "$DTYPE" \
  --mode full --out "$OUT.partial" > /dev/null
mv "$OUT.partial" "$OUT"
echo "wrote $OUT"
