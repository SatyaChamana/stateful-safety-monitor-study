#!/bin/bash
# Phase 3: one history condition per array task, for ONE stage.
#   Validation (fits thresholds):  sbatch --array=0-9 --export=ALL,STAGE=validation scripts/slurm/phase3.sbatch
#   Test (needs configs/phase3_thresholds.json, frozen after validation):
#                                  sbatch --array=0-9 --export=ALL,STAGE=test scripts/slurm/phase3.sbatch
# Task i runs pilot_conditions[i] from configs/phase3_plan.json. Refuses to repeat a condition+stage.
#SBATCH --job-name=ss-phase3
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm-%x-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# Study hardware is fixed (decision 2026-11-02, revised): H100 NVL, bfloat16. Refuse anything else.
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
case "$GPU_NAME" in *H100*NVL*) ;; *) echo "Refusing: study GPU is H100 NVL, got $GPU_NAME" >&2; exit 2;; esac
source .venv/bin/activate
DTYPE=bfloat16
: "${STAGE:?set STAGE=validation or STAGE=test}"
LIST="${LIST:-pilot_conditions}"
export HF_HUB_OFFLINE=1
read -r COND MODE K MODEL REV DMODEL DREV PROMPT < <(python - "$SLURM_ARRAY_TASK_ID" "$LIST" <<'PY'
import json, sys
plan = json.load(open("configs/phase3_plan.json"))
cond = plan[sys.argv[2]][int(sys.argv[1])]
mode, _, k = cond.partition("_k")
j, d = plan["judge"], plan["dense_encoder"]
print(cond, mode, k or "-", j["model_id"], j["model_revision"], d["model"], d["revision"], j["paper_prompt"])
PY
)
if ls results/phase3a1_"${STAGE}"_"${COND}"_*.json >/dev/null 2>&1; then
  echo "Refusing: $STAGE result for $COND already exists." >&2; exit 3
fi
case "$STAGE" in
  validation) DATA=data/raw/decomposedharm/combined_val.csv;  EXTRA=() ;;
  test)       DATA=data/raw/decomposedharm/combined_test.csv; EXTRA=(--thresholds configs/phase3_thresholds.json) ;;
  *) echo "STAGE must be validation or test" >&2; exit 4 ;;
esac
KARGS=(); [ "$K" != "-" ] && KARGS=(--k "$K")
DARGS=(); case "$MODE" in dense|hybrid) DARGS=(--dense-model "$DMODEL" --dense-revision "$DREV");; esac
OUT="results/phase3a1_${STAGE}_${COND}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"  # protocol amendment 1
echo "condition=$COND stage=$STAGE gpu=$GPU_NAME"
safety-study history-eval --stage "$STAGE" --mode "$MODE" "${KARGS[@]}" "${DARGS[@]}" "${EXTRA[@]}" \
  --schema decomposedharm --index-base 1 --data "$DATA" --ids configs/paper_eval_split.json \
  --paper-prompt "$PROMPT" --backend hf --model-id "$MODEL" --model-revision "$REV" --dtype "$DTYPE" \
  --out "$OUT.partial" > /dev/null
mv "$OUT.partial" "$OUT"
echo "wrote $OUT"
