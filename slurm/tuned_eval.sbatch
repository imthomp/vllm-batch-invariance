#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2:30:00
# Tuning pilot (2) and (3): cross-condition check with the tuned table, then decode off/on/tuned in one job.
# Usage: sbatch --export=ALL,PY=...,OUT=...,DATA=<problems.json>,MODEL=...,VBI_TUNED=<tuned.json>,TAG=<name> slurm/tuned_eval.sbatch
set -u
D=$(dirname "$(readlink -f "$0")"); D=${SLURM_SUBMIT_DIR:-$D}/slurm
( export KIND=cross MODE=tuned TP=1 ARGS="--n 30"; bash "$D/phase2.sbatch" )
for m in off on tuned; do ( export KIND=decode MODE=$m TP=1 REP=1; bash "$D/phase2.sbatch" ); done
