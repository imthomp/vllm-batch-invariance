#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
# Better default, vLLM evaluation (PROTOCOL.md): Qwen2.5-7B crossover with the generic default; decode off/on/tuned/generic
# x3 on Qwen2.5-7B (also the tuning-pilot repeats), off/on/generic x3 on Gemma-2-9B. Rounds rotate the arm order.
# Usage: sbatch --export=ALL,PY=...,OUT=...,DATA=<problems.json>,VBI_GENERIC=...,VBI_TUNED=<tuned_q25.json> slurm/generic_eval.sbatch
set -u
D=${SLURM_SUBMIT_DIR:-$(pwd)}/slurm
export VBI_GENERIC VBI_TUNED
( export KIND=cross MODE=generic TP=1 MODEL=Qwen/Qwen2.5-7B-Instruct TAG=q25gen ARGS="--n 30"; bash "$D/phase2.sbatch" )
for r in 1 2 3; do
  case $r in 1) order="off on tuned generic";; 2) order="generic tuned on off";; 3) order="tuned off generic on";; esac
  for m in $order; do ( export KIND=decode MODE=$m TP=1 REP=$r MODEL=Qwen/Qwen2.5-7B-Instruct TAG=q25gen; bash "$D/phase2.sbatch" ); done
  for m in $(echo $order | sed 's/tuned//'); do ( export KIND=decode MODE=$m TP=1 REP=$r MODEL=google/gemma-2-9b-it TAG=g2gen; bash "$D/phase2.sbatch" ); done
done
