#!/bin/bash
#SBATCH --partition=m9
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0:20:00
# CPU job run after a GPU job (sbatch --dependency=afterany:<id>): score it into results/, re-render macros and README.
# Usage: sbatch --dependency=afterany:<id> --export=ALL,SCORER=<src/score_*.py>,FROM=<job output dir>,REPO=<repo> slurm/score_after.sbatch
set -u
cd "$REPO"
exec 9>results/.stats.lock; flock 9
python3 "$SCORER" --from "$FROM" --merge
python3 src/make_macros.py && python3 src/render.py
python3 tests/check.py | tail -1; python3 tests/check_paper.py | tail -1
