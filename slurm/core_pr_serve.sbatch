#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --export=ALL

set -euo pipefail

ROOT=${ROOT:-/home/it238/projects/vllm-batch-invariance}
BASE_SRC=${BASE_SRC:-/home/it238/vllm-generic-base}
PR_SRC=${PR_SRC:-/home/it238/vllm-generic-default}
PY=${PY:-/nobackup/autodelete/usr/it238/research_lab/vllm-pr-check/.venv/bin/python}
VLLM=${VLLM:-/nobackup/autodelete/usr/it238/research_lab/vllm-pr-check/.venv/bin/vllm}
HF_HOME=${HF_HOME:-/nobackup/autodelete/usr/it238/hf_lmeval}
DATA_ROOT=${DATA_ROOT:-/nobackup/autodelete/usr/it238/research_lab/data/core_pr_serving}
RESULT_ROOT=$ROOT/results/validation/core_pr_serving
TAG=${TAG:?set TAG}
MODEL=${MODEL:?set MODEL}
TP=${TP:?set TP}
OUT=$DATA_ROOT/$TAG
PORT=$((24000 + ${SLURM_JOB_ID:-1} % 20000))

mkdir -p "$OUT"
export OPENSSL_CONF=/dev/null HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLREDUCE_USE_FLASHINFER=0 TOKENIZERS_PARALLELISM=false
unset VLLM_USE_BREAKABLE_CUDAGRAPH || true
SERVER_PID=
cleanup_server() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=
  fi
}
trap cleanup_server EXIT
trap 'cleanup_server; exit 130' INT
trap 'cleanup_server; exit 143' TERM

BASE_SHA=$(git -C "$BASE_SRC" rev-parse HEAD)
PR_SHA=$(git -C "$PR_SRC" rev-parse HEAD)
cat > "$OUT/metadata.json" <<EOF
{
  "tag": "$TAG",
  "model": "$MODEL",
  "tp": $TP,
  "base_sha": "$BASE_SHA",
  "pr_sha": "$PR_SHA",
  "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/"/\\"/g')",
  "job_id": "${SLURM_JOB_ID:-local}",
  "python": "$PY"
}
EOF

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "BASE $BASE_SHA ($BASE_SRC)"
echo "PR   $PR_SHA ($PR_SRC)"
"$PY" -c 'import torch,vllm; print("runtime", torch.__version__, vllm.__file__, vllm.__version__)'

# Exercise the actual PR implementation on this GPU before measuring. The
# shape is an untuned Qwen2.5-7B bf16 projection; only the generic fallback
# should return BLOCK_SIZE_K=128 here.
env -u VLLM_USE_BREAKABLE_CUDAGRAPH OPENSSL_CONF=/dev/null HF_HOME="$HF_HOME" \
  HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1 PYTHONPATH="$PR_SRC" "$PY" - <<'PY' | tee "$OUT/pr_preflight.log"
import torch
from vllm.model_executor.determinism import batch_invariant_configs as bi
from vllm.platforms import current_platform

family = bi._get_tuned_matmul_arch_family(current_platform.get_device_capability())
assert family == "hopper", family
bi.resolve_tuned_matmul_configs()
default = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
           "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3}
cfg = bi._get_matmul_config(4, 4608, 3584, torch.bfloat16, default)
assert cfg is not default and cfg["BLOCK_SIZE_K"] == 128, cfg
print("PASS PR generic fallback", cfg, flush=True)
PY

workloads=(
  "decode32:512:256:32:600"
  "prefill32:1024:4:32:600"
  "decode1:512:64:1:200"
  "longdecode32:4096:128:32:400"
)

run_arm() {
  local arm=$1 round=$2 source=$3 bi=$4
  local server_log=$OUT/server_${arm}_r${round}.log
  env -u VLLM_USE_BREAKABLE_CUDAGRAPH OPENSSL_CONF=/dev/null HF_HOME="$HF_HOME" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 \
    VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_FLASHINFER=0 \
    VLLM_BATCH_INVARIANT="$bi" PYTHONPATH="$source" \
    "$VLLM" serve "$MODEL" --served-model-name "$MODEL" --port "$PORT" \
    --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.9 \
    --no-enable-prefix-caching --seed 0 > "$server_log" 2>&1 &
  SERVER_PID=$!
  local ready=0
  for _ in $(seq 1 180); do
    if curl -sf "localhost:$PORT/v1/models" 2>/dev/null | grep -Fq "\"id\":\"$MODEL\"" && kill -0 "$SERVER_PID" 2>/dev/null; then
      ready=1
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
    sleep 5
  done
  if [ "$ready" != 1 ]; then
    echo "SERVER FAILED arm=$arm round=$round model=$MODEL"
    tail -60 "$server_log"
    return 1
  fi
  echo "SERVER READY arm=$arm round=$round model=$MODEL source=$source bi=$bi"

  for workload in "${workloads[@]}"; do
    IFS=: read -r name input output concurrency prompts <<< "$workload"
    local result="bench_${name}_${arm}_r${round}.json"
    env -u VLLM_USE_BREAKABLE_CUDAGRAPH OPENSSL_CONF=/dev/null HF_HOME="$HF_HOME" \
      HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 PYTHONPATH="$source" \
      "$VLLM" bench serve --backend vllm --model "$MODEL" --port "$PORT" \
      --dataset-name random --random-input-len "$input" --random-output-len "$output" \
      --num-prompts "$prompts" --max-concurrency "$concurrency" --request-rate inf \
      --ignore-eos --seed 20261112 --num-warmups 20 --save-result \
      --result-dir "$OUT" --result-filename "$result" \
      > "$OUT/${result%.json}.log" 2>&1
    grep -E "Output token throughput|Request throughput" "$OUT/${result%.json}.log" | tail -2 || true
  done
  cleanup_server
  sleep 5
}

for round in 1 2 3; do
  case "$round" in
    1) order="base pr off" ;;
    2) order="pr off base" ;;
    3) order="off base pr" ;;
  esac
  for arm in $order; do
    case "$arm" in
      off) run_arm off "$round" "$BASE_SRC" 0 ;;
      base) run_arm base "$round" "$BASE_SRC" 1 ;;
      pr) run_arm pr "$round" "$PR_SRC" 1 ;;
    esac
  done
done

touch "$OUT/DONE"
DEST=$RESULT_ROOT/$TAG
mkdir -p "$DEST"
cp "$OUT"/*.json "$OUT"/*.log "$OUT/DONE" "$DEST/"
echo "DONE tag=$TAG results=$DEST"
