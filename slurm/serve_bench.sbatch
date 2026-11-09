#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
# Serving benchmark only (src/serve_bench.sh). The tuned arm loads a tune_bi.py output through src/tunedhook.
# Usage: sbatch --export=ALL,PY=...,MODEL=...,VBI_TUNED=<tuned json>,OUT=<dir>[,LIBPATH=<dir for LD_LIBRARY_PATH>] slurm/serve_bench.sbatch
set -u
export OPENSSL_CONF=/dev/null HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 VLLM_USE_FLASHINFER_SAMPLER=0 VBI_TUNED
[ -n "${LIBPATH:-}" ] && export LD_LIBRARY_PATH=$LIBPATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
S=${SRC:-$(pwd)/src}
# Optional: wait for a download to finish (a log line starting with DONE), and merge several tune_bi.py outputs.
if [ -n "${WAIT_LOG:-}" ]; then until grep -q "^DONE" "$WAIT_LOG"; do sleep 60; done; fi
if [ -n "${TUNED_FILES:-}" ]; then
  VBI_TUNED=$OUT/tuned_merged.json; export VBI_TUNED
  python3 -c "import json,sys; fs=sys.argv[2].split(':'); e=[x for f in fs for x in json.load(open(f))['entries']]; json.dump({'entries': e, 'sources': fs}, open(sys.argv[1],'w'))" "$VBI_TUNED" "$TUNED_FILES"
  echo "merged $(python3 -c "import json;print(len(json.load(open('$VBI_TUNED'))['entries']))") tuned entries"
fi
OVERLAY=$S/tunedhook bash "$S/serve_bench.sh"
grep -h "patched tuned table" "$OUT"/server_tuned_r*.log | head -3
