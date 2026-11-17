#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --export=ALL
# PROTOCOL.md "gpt-oss follow-up to #65931", part (B): tune buckets above 2048 for the #65931 gpt-oss entries,
# with each entry's BLOCK_K fixed and its current 2048-bucket config timed as the baseline ("reuse").
# Usage: sbatch --job-name=VL_<desc> --export=ALL,SRC_TREE=<vllm source with the #65931 table>,OUT=<dir> slurm/tune_extend.sbatch
set -eu
ROOT=${ROOT:-/home/it238/projects/vllm-batch-invariance}
VENV=${VENV:-/nobackup/autodelete/usr/it238/research_lab/vllm-pr-check/.venv}
export OPENSSL_CONF=/dev/null PYTHONPATH=$SRC_TREE
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,driver_version --format=csv
git -C "$SRC_TREE" rev-parse HEAD
# NxK  BLOCK_K  2048-bucket config (from #65931)
while read -r shape bk reuse name; do
  "$VENV/bin/python" "$ROOT/src/tune_bi.py" --shapes "$shape" --buckets 4096,8192,16384 --block-k "$bk" --reuse "$reuse" \
    --out "$OUT/extend_${name}.json"
done <<'EOF'
5120x2880 128 128,128,4,3 qkv
2880x4096 128 128,128,4,3 o
32x2880 128 16,32,4,5 router20b
128x2880 128 64,32,4,5 router120b
EOF
echo DONE
