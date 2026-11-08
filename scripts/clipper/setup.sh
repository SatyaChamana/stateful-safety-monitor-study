#!/usr/bin/env bash
# One-time setup on the clipper LOGIN node (no GPU needed). Run from the repo root.
# Data and model are fetched with YOUR Hugging Face login; nothing is copied from another machine.
set -euo pipefail
cd "$(dirname "$0")/../.."
umask 077                                   # new files readable by you only
UPSTREAM_COMMIT=79f5b4109aec8ff88f92ffe2d7a94eb419699d68
DATA_REV=25e1d5910c2cccbdcae11799193595cea999bda8

if ! command -v uv >/dev/null; then         # no curl|sh: install uv into a private venv
  python3 -m venv "$HOME/.uv-tool" && "$HOME/.uv-tool/bin/pip" install -q uv
  export PATH="$HOME/.uv-tool/bin:$PATH"
fi
[ -d .venv ] || UV_PYTHON_PREFERENCE=only-managed uv venv --python 3.11 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -q -e '.[dev,science,data,model,retrieval]'

mkdir -p data/raw third_party results
mkdir -p "$HOME/.cache/huggingface"
chmod 700 data third_party results "$HOME/.cache/huggingface"
if [ ! -d third_party/Monitoring-Decomposition-Attack ]; then
  git clone -q https://github.com/YuehHanChen/Monitoring-Decomposition-Attack third_party/Monitoring-Decomposition-Attack
fi
git -C third_party/Monitoring-Decomposition-Attack checkout -q "$UPSTREAM_COMMIT"

bash scripts/ci.sh
uv pip freeze | grep -v '^-e \|stateful-safety-experiment' > constraints/clipper-py311-linux-x86_64.txt
echo "Setup OK. Next: hf auth login, then the data download in docs/clipper.md."
echo "Dataset revision to download: $DATA_REV"
