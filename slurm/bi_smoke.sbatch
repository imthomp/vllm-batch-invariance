#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1:00:00
# Batch-invariant-mode compatibility check (src/bi_smoke.py) for large models before booking a full node.
# Usage: sbatch --export=ALL,PY=...,MODELS="<path>:<path>",LIBPATH=... slurm/bi_smoke.sbatch
export OPENSSL_CONF=/dev/null HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_BATCH_INVARIANT=1
[ -n "${LIBPATH:-}" ] && export LD_LIBRARY_PATH=$LIBPATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
S=${SRC:-$(pwd)/src}
nvidia-smi --query-gpu=name --format=csv,noheader
IFS=: read -ra MS <<< "$MODELS"
for m in "${MS[@]}"; do
  echo "== $m"
  timeout 1500 "$PY" "$S/bi_smoke.py" "$m" 2>&1 | grep -E "SMOKE OK|MoE backend|Error|error:|not support|Traceback|batch.invariant|BATCH_INVARIANT" | grep -v "^\s*$" | tail -15
done
