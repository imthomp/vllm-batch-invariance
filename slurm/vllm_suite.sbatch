#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2:00:00
# Runs vLLM's own determinism suite (tests/v1/determinism/test_batch_invariance.py) in MODE=on (stock) or plain (cuBLAS routing).
# Usage: sbatch --export=ALL,PY=<python with vllm+pytest>,TESTS=<vllm checkout at the installed commit>/tests/v1/determinism,PLUGINS=<dir>,MODE=on|plain|pr,OUT=<dir> slurm/vllm_suite.sbatch
set -uo pipefail
export HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 VLLM_USE_FLASHINFER_SAMPLER=0
S=${SRC:-$(pwd)/src}
[ "$MODE" = plain ] && export VBI_PLAIN=1 PYTHONPATH="$S/plainpatch${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi --query-gpu=name,driver_version --format=csv
# PLUGINS: a directory with vllm_preimport.py ("import vllm.config"), standing in for the import order that
# vLLM's repository-wide conftest.py provides; without it the test helper hits a circular import.
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PLUGINS"
cd "$TESTS"
"$PY" -m pytest test_batch_invariance.py -v -p vllm_preimport -p no:cacheprovider --confcutdir=. -rA 2>&1 | tee "$OUT/suite_$MODE.log"
echo "MARKERS $(grep -c 'patched linear_batch_invariant' "$OUT/suite_$MODE.log")"
