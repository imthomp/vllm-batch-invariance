#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --export=ALL
# gpt-oss follow-up to vLLM #65931 (PROTOCOL.md "gpt-oss follow-up to #65931"), part (A).
# Arms: name=<source tree>:<VLLM_BATCH_INVARIANT>[:<VLLM_USE_BREAKABLE_CUDAGRAPH>] (third field optional: unset = vLLM decides);
# WORKLOADS (optional) = space-separated subset of decode32 prefill32 decode1 longdecode32; ORDERS gives one space-separated arm order per round (';'-separated).
# Usage: sbatch --job-name=VL_<desc> --export=ALL,TAG=<tag>,ARMS="off=<src>:0 base=<src>:1 ...",ORDERS="a b;b a" slurm/oss_followup_serve.sbatch
set -euo pipefail

ROOT=${ROOT:-/home/it238/projects/vllm-batch-invariance}
VENV=${VENV:-/nobackup/autodelete/usr/it238/research_lab/vllm-pr-check/.venv}
PY=$VENV/bin/python
VLLM=$VENV/bin/vllm
MODEL=${MODEL:-openai/gpt-oss-20b}
SERVE_EXTRA=${SERVE_EXTRA:-}
DATA_ROOT=${DATA_ROOT:-/nobackup/autodelete/usr/it238/research_lab/data/oss_followup}
TAG=${TAG:?set TAG}
ARMS=${ARMS:?set ARMS}
ORDERS=${ORDERS:?set ORDERS}
OUT=$DATA_ROOT/$TAG
PORT=$((24000 + ${SLURM_JOB_ID:-1} % 20000))
MARKER="VLLM_BATCH_INVARIANT=1: auto-enabling VLLM_USE_BREAKABLE_CUDAGRAPH=1"
mkdir -p "$OUT"
export OPENSSL_CONF=/dev/null HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
# Humming NVRTC workaround (humming PR #202 not yet released), identical in every arm
export LD_LIBRARY_PATH=$VENV/lib/python3.12/site-packages/nvidia/cu13/lib
unset VLLM_USE_BREAKABLE_CUDAGRAPH || true

declare -A SRC BI BRK
for a in $ARMS; do name=${a%%=*}; IFS=: read -r SRC[$name] BI[$name] BRK[$name] <<< "${a#*=}"; done

SERVER_PID=
cleanup_server() { if [ -n "${SERVER_PID:-}" ]; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; SERVER_PID=; fi; }
trap cleanup_server EXIT
trap 'cleanup_server; exit 143' TERM

{
  echo "{\"tag\": \"$TAG\", \"model\": \"$MODEL\", \"job_id\": \"${SLURM_JOB_ID:-local}\","
  echo " \"gpu\": \"$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)\", \"venv\": \"$VENV\", \"arms\": {"
  first=1; for n in "${!SRC[@]}"; do [ $first = 1 ] || echo ","; first=0
    echo -n "  \"$n\": {\"src\": \"${SRC[$n]}\", \"sha\": \"$(git -C "${SRC[$n]}" rev-parse HEAD)\", \"bi\": ${BI[$n]}, \"breakable_env\": \"${BRK[$n]:-unset}\"}"; done
  echo "}}"
} > "$OUT/metadata.json"
cat "$OUT/metadata.json"
"$PY" -c 'import torch,vllm; print("runtime", torch.__version__, vllm.__version__)'

all_workloads=("decode32:512:256:32:600" "prefill32:1024:4:32:600" "decode1:512:64:1:200" "longdecode32:4096:128:32:400")
workloads=()
for w in "${all_workloads[@]}"; do
  if [ -z "${WORKLOADS:-}" ] || [[ " $WORKLOADS " == *" ${w%%:*} "* ]]; then workloads+=("$w"); fi
done

run_arm() {
  local arm=$1 round=$2 src=${SRC[$1]} bi=${BI[$1]} brk=${BRK[$1]:-}
  local log=$OUT/server_${arm}_r${round}.log
  local brkenv=(-u VLLM_USE_BREAKABLE_CUDAGRAPH)
  [ -n "$brk" ] && brkenv=(VLLM_USE_BREAKABLE_CUDAGRAPH="$brk")
  env "${brkenv[@]}" VLLM_BATCH_INVARIANT="$bi" PYTHONPATH="$src" \
    "$VLLM" serve "$MODEL" --served-model-name "$MODEL" --port "$PORT" --max-model-len 8192 \
    --gpu-memory-utilization 0.9 --no-enable-prefix-caching --seed 0 $SERVE_EXTRA > "$log" 2>&1 &
  SERVER_PID=$!
  local ready=0
  for _ in $(seq 1 240); do
    if curl -sf "localhost:$PORT/v1/models" 2>/dev/null | grep -Fq "\"id\":\"$MODEL\"" && kill -0 "$SERVER_PID" 2>/dev/null; then ready=1; break; fi
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 5
  done
  if [ "$ready" != 1 ]; then echo "SERVER FAILED arm=$arm round=$round"; tail -60 "$log"; return 1; fi
  echo "SERVER READY arm=$arm round=$round src=$src bi=$bi breakable_env=${brk:-unset} breakable_marker=$(grep -cF "$MARKER" "$log" || true)"
  for w in "${workloads[@]}"; do
    IFS=: read -r name input output conc prompts <<< "$w"
    local res="bench_${name}_${arm}_r${round}.json"
    env -u VLLM_USE_BREAKABLE_CUDAGRAPH PYTHONPATH="$src" "$VLLM" bench serve --backend vllm --model "$MODEL" --port "$PORT" \
      --dataset-name random --random-input-len "$input" --random-output-len "$output" --num-prompts "$prompts" \
      --max-concurrency "$conc" --request-rate inf --ignore-eos --seed 20261112 --num-warmups 20 --save-result \
      --result-dir "$OUT" --result-filename "$res" > "$OUT/${res%.json}.log" 2>&1
    grep -E "Output token throughput|Request throughput" "$OUT/${res%.json}.log" | tail -2 || true
  done
  cleanup_server
  sleep 5
}

round=0
IFS=';' read -ra ORDER_LIST <<< "$ORDERS"
for order in "${ORDER_LIST[@]}"; do
  round=$((round + 1))
  for arm in $order; do run_arm "$arm" "$round"; done
done
touch "$OUT/DONE"
DEST=$ROOT/results/validation/oss_followup/$TAG
mkdir -p "$DEST"
cp "$OUT"/*.json "$OUT"/*.log "$OUT/DONE" "$DEST/"
echo "DONE tag=$TAG results=$DEST"
