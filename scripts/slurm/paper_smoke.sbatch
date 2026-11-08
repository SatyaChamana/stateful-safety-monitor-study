#!/bin/bash
# Validation-only paper-mode reply check. Model must already be downloaded on the login node.
# Submit from the repo root, e.g.:
#   sbatch --export=ALL,PROMPT=BINARY_BASELINE scripts/slurm/paper_smoke.sbatch
#SBATCH --job-name=ss-papersmoke
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm-%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# Study hardware is fixed (decision 2026-11-02, revised): H100 NVL, bfloat16. Refuse anything else.
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
case "$GPU_NAME" in *H100*NVL*) ;; *) echo "Refusing: study GPU is H100 NVL, got $GPU_NAME" >&2; exit 2;; esac
source .venv/bin/activate
DTYPE=bfloat16                                 # fixed for the whole study
PROMPT="${PROMPT:-BINARY_BASELINE}"
N="${N:-20}"
REV="${REV:-0e9e39f249a16976918f6564b8830bc894c89659}"
export HF_HUB_OFFLINE=1                       # never download inside a job
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
safety-study paper-smoke --schema decomposedharm --index-base 1 \
  --validation data/raw/decomposedharm/combined_val.csv --ids configs/paper_eval_split.json \
  --paper-prompt "$PROMPT" --n "$N" --backend hf \
  --model-id meta-llama/Llama-3.1-8B-Instruct --model-revision "$REV" --dtype "$DTYPE" \
  --out "results/paper_smoke_${PROMPT}_${DTYPE}_${SLURM_JOB_ID}.json" > /dev/null
echo "wrote results/paper_smoke_${PROMPT}_${DTYPE}_${SLURM_JOB_ID}.json"
