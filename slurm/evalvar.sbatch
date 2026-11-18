#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --export=ALL
# Eval-variance runs (PROTOCOL.md "Do serving conditions change NLP evaluation conclusions?").
# One job = one model x one batch-invariant mode: C2, C3, C4, C1, each twice, in separate processes.
# Usage: sbatch --job-name=VL_ev_<short>_bi<0|1> --export=ALL,MODEL=<hf id>,SHORT=<short>,BI=<0|1>[,LIMIT=N][,OUT_ROOT=..] slurm/evalvar.sbatch
set -euo pipefail
ROOT=${ROOT:-/home/it238/projects/vllm-batch-invariance}
VENV=${VENV:-/nobackup/autodelete/usr/it238/research_lab/vllm-pr-check/.venv}
SRC=${SRC:-/home/it238/vllm-generic-default}   # PR #67376 tree (generic Hopper fallback; includes #65627)
MODEL=${MODEL:?}; SHORT=${SHORT:?}; BI=${BI:?}; LIMIT=${LIMIT:-0}
OUT_ROOT=${OUT_ROOT:-/nobackup/autodelete/usr/it238/research_lab/data/evalvar}
OUT=$OUT_ROOT/$SHORT/bi$BI
mkdir -p "$OUT"
export OPENSSL_CONF=/dev/null HF_HOME=/nobackup/autodelete/usr/it238/hf_lmeval HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=$VENV/lib/python3.12/site-packages/nvidia/cu13/lib   # humming NVRTC workaround, every arm
unset VLLM_USE_BREAKABLE_CUDAGRAPH || true
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "src $(git -C "$SRC" rev-parse HEAD) model $MODEL bi $BI limit $LIMIT job ${SLURM_JOB_ID:-local}"
for run in C2_r1 C3_r1 C4_r1 C2_r2 C3_r2 C4_r2 C1_r1 C1_r2; do
  f=$OUT/$run.jsonl
  if [ -s "$f" ] && [ -s "${f%.jsonl}.meta.json" ]; then echo "skip $run (done)"; continue; fi
  rm -f "$f" "${f%.jsonl}.meta.json"
  VLLM_BATCH_INVARIANT=$BI PYTHONPATH=$SRC "$VENV/bin/python" "$ROOT/src/evalvar_gen.py" --model "$MODEL" \
    --cond "${run%_r*}" --out "$f" --limit "$LIMIT" > "$OUT/$run.log" 2>&1 || { echo "FAILED $run"; tail -40 "$OUT/$run.log"; exit 1; }
  grep -c "auto-enabling VLLM_USE_BREAKABLE_CUDAGRAPH" "$OUT/$run.log" | sed "s/^/$run breakable_marker=/"
  tail -1 "$OUT/$run.log" | cut -c1-300
done
echo "DONE $SHORT bi$BI"
