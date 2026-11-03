#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2:00:00
# Phase 2. Usage: sbatch --gres=gpu:<TP> --export=ALL,PY=<python with vllm>,KIND=cross|decode|prefill,MODE=off|on|plain,
#   MODEL=<hf id>,TAG=<name>,TP=<n>,OUT=<dir>[,DATA=<problems.json>,ARGS="<extra crossover2 args>"] slurm/phase2.sbatch
set -euo pipefail
export HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 VLLM_USE_FLASHINFER_SAMPLER=0
# FlashInfer all-reduce JIT-compiles with nvcc and ninja; batch-invariant mode disables it anyway, so this only affects off.
export VLLM_ALLREDUCE_USE_FLASHINFER=0
# On FIPS-mode hosts, also set OPENSSL_CONF=/dev/null and uninstall opencv-python-headless.
S=${SRC:-$(pwd)/src}
"$PY" -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__)"
nvidia-smi --query-gpu=name,driver_version --format=csv
case "$MODE" in
  off) export VLLM_BATCH_INVARIANT=0 ;;
  on) export VLLM_BATCH_INVARIANT=1 ;;
  plain) export VLLM_BATCH_INVARIANT=1 VBI_PLAIN=1 PYTHONPATH="$S/plainpatch${PYTHONPATH:+:$PYTHONPATH}" ;;
  *) echo "bad MODE"; exit 2 ;;
esac
NAME=${KIND}_${TAG}_${MODE}${REP:+_r$REP}
LOG=$OUT/$NAME.log
case "$KIND" in
  cross) "$PY" "$S/crossover2.py" --data "$DATA" --out "$OUT/$NAME.jsonl" --model "$MODEL" --tp "$TP" ${ARGS:-} 2>&1 | tee "$LOG" ;;
  decode) "$PY" "$S/decode_throughput.py" --model "$MODEL" --tp "$TP" --dtype bfloat16 --out "$OUT/$NAME.json" 2>&1 | tee "$LOG" ;;
  prefill) "$PY" "$S/prefill_throughput.py" --model "$MODEL" --tp "$TP" --out "$OUT/$NAME.json" 2>&1 | tee "$LOG" ;;
  *) echo "bad KIND"; exit 2 ;;
esac
echo "MARKERS $(grep -c 'patched linear_batch_invariant' "$LOG" || true)"
