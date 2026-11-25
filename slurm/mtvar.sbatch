#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=cs2
#SBATCH --qos=cs
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --export=ALL
# Cluster rules for agents: see BYU_ORC_AGENTS.md at the repo root. Partition cs2 is required: it is the only partition
# with H100s (partitions.conf), and H100 is a hardware requirement
# (the generic batch-invariant configuration is Hopper-only and matches the math study's GPU).
# MT eval-variance runs (PROTOCOL.md "Do serving conditions change MT evaluation conclusions?").
# One job = one model x one task x one batch-invariant mode: C2, C3, C4, C1, each twice, in separate processes.
# Usage: sbatch --job-name=VL_mt_<short>_bi<0|1> --export=ALL,TASK=translate|judge,MODEL=<hf id>,SHORT=<short>,BI=<0|1>[,ITEMS=f][,LIMIT=N][,OUT_ROOT=..] slurm/mtvar.sbatch
set -euo pipefail
ROOT=${ROOT:-/home/it238/projects/vllm-batch-invariance}
VENV=${VENV:-/nobackup/autodelete/usr/it238/research_lab/vllm-pr-check/.venv}
SRC=${SRC:-/home/it238/vllm-generic-default}
TASK=${TASK:?}; MODEL=${MODEL:?}; SHORT=${SHORT:?}; BI=${BI:?}; LIMIT=${LIMIT:-0}; ITEMS=${ITEMS:-}
OUT_ROOT=${OUT_ROOT:-/nobackup/autodelete/usr/it238/research_lab/data/mtvar}
OUT=$OUT_ROOT/$TASK/$SHORT/bi$BI
mkdir -p "$OUT"
export OPENSSL_CONF=/dev/null HF_HOME=/nobackup/autodelete/usr/it238/hf_lmeval HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_CACHE=/nobackup/autodelete/usr/it238/hf_lmeval/hub
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=$VENV/lib/python3.12/site-packages/nvidia/cu13/lib   # humming NVRTC workaround, every arm
unset VLLM_USE_BREAKABLE_CUDAGRAPH || true
# Judge items are built from the systems' BI-on C2 r1 translations on first use (atomic rename; parallel jobs agree)
if [ "$TASK" = judge ] && [ -n "${ITEMS_FROM:-}" ] && [ ! -s "$ITEMS" ]; then
  python3 "$ROOT/src/mtvar_items.py" --root "$ITEMS_FROM" --out "$ITEMS.tmp.$$" && mv -n "$ITEMS.tmp.$$" "$ITEMS"; rm -f "$ITEMS.tmp.$$"
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "src $(git -C "$SRC" rev-parse HEAD) task $TASK model $MODEL bi $BI limit $LIMIT job ${SLURM_JOB_ID:-local}"
RUNS=${RUNS:-"C2_r1 C3_r1 C4_r1 C2_r2 C3_r2 C4_r2 C1_r1 C1_r2"}   # subset per job: the serial C1 runs take hours
for run in $RUNS; do
  f=$OUT/$run.jsonl
  if [ -s "$f" ] && [ -s "${f%.jsonl}.meta.json" ]; then echo "skip $run (done)"; continue; fi
  rm -f "$f" "${f%.jsonl}.meta.json"
  VLLM_BATCH_INVARIANT=$BI PYTHONPATH=$SRC "$VENV/bin/python" "$ROOT/src/mtvar_gen.py" --task "$TASK" --model "$MODEL" \
    --cond "${run%_r*}" --out "$f" --limit "$LIMIT" ${ITEMS:+--items "$ITEMS"} > "$OUT/$run.log" 2>&1 \
    || { echo "FAILED $run"; tail -40 "$OUT/$run.log"; exit 1; }
  echo "$run breakable_marker=$(grep -c "auto-enabling VLLM_USE_BREAKABLE_CUDAGRAPH" "$OUT/$run.log" || true)"
  tail -1 "$OUT/$run.log" | cut -c1-300
done
echo "DONE $TASK $SHORT bi$BI"
