#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2:00:00
# Runs vLLM's own determinism suite (tests/v1/determinism/test_batch_invariance.py) in MODE=on (stock) or plain (cuBLAS routing).
# Usage: sbatch --export=ALL,PY=<python with vllm+pytest>,TESTS=<vllm checkout>/tests/v1/determinism,MODE=on|plain,OUT=<dir> slurm/vllm_suite.sbatch
set -uo pipefail
export HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 VLLM_USE_FLASHINFER_SAMPLER=0
S=${SRC:-$(pwd)/src}
[ "$MODE" = plain ] && export VBI_PLAIN=1 PYTHONPATH="$S/plainpatch${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi --query-gpu=name,driver_version --format=csv
cd "$TESTS"
"$PY" -m pytest test_batch_invariance.py -v -p no:cacheprovider --confcutdir=. -rA 2>&1 | tee "$OUT/suite_$MODE.log"
echo "MARKERS $(grep -c 'patched linear_batch_invariant' "$OUT/suite_$MODE.log")"
