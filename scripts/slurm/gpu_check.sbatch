#!/bin/bash
# GPU sanity check: driver, torch CUDA build, bf16 support. Submit from the repo root:
#   sbatch scripts/slurm/gpu_check.sbatch
#SBATCH --job-name=ss-gpucheck
#SBATCH --partition=debug-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=results/slurm-%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# Study hardware is fixed (decision 2026-11-02, revised): H100 NVL, bfloat16. Refuse anything else.
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
case "$GPU_NAME" in *H100*NVL*) ;; *) echo "Refusing: study GPU is H100 NVL, got $GPU_NAME" >&2; exit 2;; esac
source .venv/bin/activate
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python scripts/system_report.py --out "results/system_report_${SLURM_JOB_ID}.json" >/dev/null
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda build", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "bf16_supported", torch.cuda.is_bf16_supported())
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    print("bf16 matmul ok", float((x @ x).float().abs().mean()) > 0)
PY
