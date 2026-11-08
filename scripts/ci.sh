#!/usr/bin/env bash
# Local CI: CPU-only, no credentials, no network. Run from the repository root.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m ruff check --select E9,F src tests scripts
python -m pytest -q
# Smoke CLI on fictional fixtures only; outputs are SMOKE_TEST_ONLY.
out="$(mktemp -d)"
python -m stateful_safety audit --data fixtures/synthetic_fixture.jsonl --index-base 0 --out "$out/a.json" >/dev/null
python -m stateful_safety audit --schema decomposedharm --index-base 1 --revision fixture \
  --data fixtures/decomposedharm_shape_fixture_val.csv fixtures/decomposedharm_shape_fixture_test.csv \
  --out "$out/b.json" >/dev/null
python -m stateful_safety evaluate --validation fixtures/synthetic_validation_fixture.jsonl \
  --test fixtures/synthetic_test_fixture.jsonl --index-base 0 --mode full --backend mock \
  --out "$out/c.json" >/dev/null
grep -q SMOKE_TEST_ONLY "$out/c.json"
rm -rf "$out"
echo "CI OK"
