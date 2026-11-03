#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
# Usage: sbatch --partition=<h100 partition> --export=ALL,KIND=cost|cross,MODE=off|on|plain,MODEL=<hf id>,TAG=<name>,PY=<python>,DATA=<problems.json>,OUT=<dir> slurm/run.sbatch
set -eu
export HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
# On FIPS-mode hosts, also set OPENSSL_CONF=/dev/null and uninstall opencv-python-headless.
S=$(cd "$(dirname "$0")/.." && pwd)/src
$PY -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
case "$MODE" in
  off) export VLLM_BATCH_INVARIANT=0; RUN=("$PY") ;;
  on) export VLLM_BATCH_INVARIANT=1; RUN=("$PY") ;;
  plain) export VLLM_BATCH_INVARIANT=1; RUN=("$PY" "$S/hopper_wrap.py") ;;
  *) echo "bad MODE"; exit 2 ;;
esac
case "$KIND" in
  cost) "${RUN[@]}" "$S/decode_throughput.py" --model "$MODEL" --out "$OUT/main_${TAG}_$MODE.json" ;;
  cross) "${RUN[@]}" "$S/crossover.py" --data "$DATA" --out "$OUT/main_${TAG}_$MODE.jsonl" --model "$MODEL" --dtype bfloat16 --n 100 ;;
  *) echo "bad KIND"; exit 2 ;;
esac
