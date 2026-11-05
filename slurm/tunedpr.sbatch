#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
# Tuned-config PR validation (PROTOCOL.md): unit tests, then the serving benchmark.
# Usage: sbatch --export=ALL,PY=...,OVERLAY=<patched vllm copy>,TESTS=<tests dir with the new test>,OUT=<dir> slurm/tunedpr.sbatch
set -u
export OPENSSL_CONF=/dev/null HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 VLLM_USE_FLASHINFER_SAMPLER=0
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
S=${SRC:-$(pwd)/src}
echo "== unit tests"
(cd "$TESTS/v1/determinism" && PYTHONPATH="$OVERLAY:${PLUG:-}" "$PY" -m pytest -q -p vllm_preimport -p no:cacheprovider --confcutdir=. \
   test_matmul_batch_invariant.py -rA > "$OUT/unit.log" 2>&1); grep -E "qwen25_7b|passed|failed" "$OUT/unit.log" | tail -8
echo "== serving benchmark"
bash "$S/serve_bench.sh"
