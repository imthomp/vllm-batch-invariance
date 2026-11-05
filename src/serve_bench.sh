#!/bin/bash
# Serving benchmark for the tuned-config PR (PROTOCOL.md "Tuned-config PR validation").
# Usage: PY=<python with vllm> OVERLAY=<patched vllm copy> OUT=<dir> [MODEL=...] [ROUNDS=3] bash src/serve_bench.sh
set -u
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}; ROUNDS=${ROUNDS:-3}; PORT=${PORT:-8123}
VLLM="$(dirname "$PY")/vllm"
mkdir -p "$OUT"
run_arm() {  # $1 arm (off|stock|tuned), $2 round
  local arm=$1 r=$2 env=()
  case $arm in
    off) env=(VLLM_BATCH_INVARIANT=0) ;;
    stock) env=(VLLM_BATCH_INVARIANT=1) ;;
    tuned) env=(VLLM_BATCH_INVARIANT=1 PYTHONPATH="$OVERLAY${PYTHONPATH:+:$PYTHONPATH}") ;;
  esac
  env "${env[@]}" "$VLLM" serve "$MODEL" --port $PORT --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.9 \
      --no-enable-prefix-caching --seed 0 > "$OUT/server_${arm}_r$r.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 180); do curl -sf localhost:$PORT/health >/dev/null && break; sleep 5; done
  if ! curl -sf localhost:$PORT/health >/dev/null; then echo "SERVER FAILED $arm r$r"; kill $pid; return 1; fi
  env "${env[@]}" "$PY" -c "import vllm.model_executor.determinism.batch_invariant_configs as c; print('ARM $arm hopper entries', len(c._BATCH_INVARIANT_MATMUL_TUNED_CONFIGS['hopper']))" 2>/dev/null | tail -1
  for w in decode:512:256 prefill:1024:4; do
    IFS=: read name il ol <<< "$w"
    "$VLLM" bench serve --backend vllm --model "$MODEL" --port $PORT --dataset-name random --random-input-len $il \
        --random-output-len $ol --num-prompts ${NPROMPTS:-2000} --max-concurrency 32 --request-rate inf --ignore-eos --seed 20261105 \
        --num-warmups 20 --save-result --result-dir "$OUT" --result-filename "bench_${name}_${arm}_r$r.json" > "$OUT/bench_${name}_${arm}_r$r.log" 2>&1
    grep -E "Output token throughput|Request throughput" "$OUT/bench_${name}_${arm}_r$r.log" | sed "s/^/  $arm r$r $name: /"
  done
  kill $pid; wait $pid 2>/dev/null; sleep 5
}
for r in $(seq 1 $ROUNDS); do
  if [ $((r % 2)) = 1 ]; then order="stock tuned off"; else order="tuned stock off"; fi
  for arm in $order; do run_arm $arm $r; done
done
echo DONE
