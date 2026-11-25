#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --export=ALL
# Cluster rules for agents: see BYU_ORC_AGENTS.md at the repo root. Any GPU type works here, so no partition is set.
# Scores all MT eval-variance translations (PROTOCOL.md MT study) with MetricX-24, COMET-22 and CometKiwi-23 XL, plus
# the metric batch-size side measure. Usage: sbatch --job-name=VL_mt_metrics --export=ALL,ROOT_IN=<translate dir> slurm/mtvar_metrics.sbatch
set -euo pipefail
ROOT=${ROOT:-/home/it238/projects/vllm-batch-invariance}
MQM=/home/it238/projects/multilingual-mqm-benchmark/multilingual-mqm-benchmark
ROOT_IN=${ROOT_IN:?}
OUT=${OUT:-$ROOT/results/validation/mtvar}
mkdir -p "$OUT"
export OPENSSL_CONF=/dev/null HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name --format=csv,noheader
"$MQM/.venv/bin/python" "$ROOT/src/mtvar_metrics.py" --root "$ROOT_IN" --out "$OUT/metric_scores.jsonl.gz" \
  --side-run "$ROOT_IN/gemma2_9b/bi0/C2_r1.jsonl" --side-out "$OUT/metric_batch_side.json"
echo DONE
