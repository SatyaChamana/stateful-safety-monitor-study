# Running on a SLURM GPU cluster (GVSU Clipper)

Observed 2026-11-02 from the login node (user-run probe): Ubuntu 24.04, x86_64, no GPU on the login node, SLURM. Modules: `python/3.12.3` (default), `cuda/12.9`, `cuda/13.3`, `miniconda3`. Home quota 46 GB soft / 51 GB hard, about 4.7 GB used. Hugging Face is reachable.

| Partition | GPUs (2 per node) | Time limit |
| --- | --- | --- |
| `debug-gpu` | RTX 8000 (2 nodes), V100S (4), H100 NVL (3) | 30 min |
| `short-gpu` | RTX 8000 (2), V100S (4), H100 NVL (3) | 4 h |
| `gpu` | RTX 8000 (2), V100S (4), H100 NVL (3) | 5 days |
| `scavenger` | RTX 8000 (2), V100S (4) | 5 days (preemptible, by name) |
| `class` | RTX 8000 (1, sharded) | 2 days |

Full `sinfo` pasted 2026-11-02. An earlier truncated listing wrongly suggested H100 was 30-minute only; the study briefly locked V100S on that premise, then switched to H100.

**Study hardware (fixed 2026-11-02, revised):** NVIDIA H100 NVL, `bfloat16`, for every run. The SLURM scripts request `gpu:nvidia_h100_nvl:1`, hard-code `DTYPE=bfloat16`, and exit if the allocated GPU is not an H100 NVL. bf16 is Llama 3.1's native precision; the remaining serving deviation is local Transformers vs the paper's Together "Turbo".

## Steps

1. **Mac → clipper, code only.** Run this in a terminal on the **Mac**, not inside an SSH session (on clipper, `$HOME` points to clipper's home and the copy finds nothing). Leaves out data, venv, upstream clone and results:
   ```bash
   rsync -av --exclude .venv --exclude data --exclude third_party --exclude results \
     --exclude __pycache__ --exclude .pytest_cache --exclude .ruff_cache --exclude '*.egg-info' \
     "$HOME/My Projects/Knowledge Base/phd/Experiment/stateful-safety-experiment/" \
     <user>@<cluster-host>:~/stateful-safety-experiment/
   ```
2. **Login node setup:** `cd ~/stateful-safety-experiment && bash scripts/clipper/setup.sh` (uv + Python 3.11, model extras, pinned upstream clone, CI, Linux lock file).
3. **Your own HF login, then data** (login node):
   ```bash
   source .venv/bin/activate && hf auth login
   hf download YuehHanChen/DecomposedHarm --repo-type dataset \
     --revision 25e1d5910c2cccbdcae11799193595cea999bda8 --local-dir data/raw/decomposedharm
   sha256sum data/raw/decomposedharm/*.csv   # must match docs/dataset_manifest.md §1
   ```
4. **SLURM account (required by clipper):** find yours with `sacctmgr -n show assoc user=$USER format=account%30,partition%20`, then `echo 'export SBATCH_ACCOUNT=<account>' >> ~/.bashrc && source ~/.bashrc` (or pass `--account=<account>` to each `sbatch`). Scripts do not hard-code an account.
5. **GPU check** (no model needed): `sbatch scripts/slurm/gpu_check.sbatch`, then read `results/slurm-ss-gpucheck-*.out`.
6. **After Meta approves Llama access**, download the pinned model on the login node, skipping the duplicate `original/` weights (~16 GB) to stay within quota:
   ```bash
   hf download meta-llama/Llama-3.1-8B-Instruct --revision 0e9e39f249a16976918f6564b8830bc894c89659 \
     --exclude "original/*"
   ```
7. **Validation smoke:** `sbatch --export=ALL,PROMPT=BINARY_BASELINE scripts/slurm/paper_smoke.sbatch`. Jobs run with `HF_HUB_OFFLINE=1`, so they cannot fetch a different revision.

Data handling: data and model are fetched on clipper with your own login. `setup.sh` sets `umask 077` and `chmod 700` on `data/`, `third_party/`, `results/` and the HF cache. Results contain aggregates only. Copy back only `results/*.json` after checking them.

## Phase 3 (history-selection comparison)

1. Re-sync code from the Mac (step 1), then on clipper re-run `bash scripts/clipper/setup.sh` (adds the `retrieval` extra: sentence-transformers).
2. Download the pinned encoder on the login node:
   ```bash
   hf download sentence-transformers/all-MiniLM-L6-v2 --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41
   ```
3. **Validation stage** (10 conditions, one array task each; fits readout A thresholds; outputs `results/phase3a1_validation_*`; the older `phase3_validation_*` files are superseded):
   `sbatch --array=0-9 --export=ALL,STAGE=validation scripts/slurm/phase3.sbatch`
4. Copy `results/phase3a1_validation_*.json` to the Mac. Decide the margin and primary k from validation only, then freeze on the Mac:
   `python scripts/freeze_phase3_thresholds.py --margin <m> --primary-k <k> results/phase3a1_validation_*.json`
   and re-sync so clipper has `configs/phase3_thresholds.json`.
5. **Test stage** (one pass per condition, frozen thresholds): `sbatch --array=0-9 --export=ALL,STAGE=test scripts/slurm/phase3.sbatch`
