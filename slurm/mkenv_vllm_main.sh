#!/bin/bash
# Builds the vLLM nightly environment used for every vLLM-main result (recovered 2026-11-04; originally run from /tmp).
# Usage: T=<dir for the venv> bash slurm/mkenv_vllm_main.sh   (needs uv). On FIPS hosts opencv must be removed, as below.
set -e
T=${T:?set T}
uv venv --python 3.12 $T/vllm_main
uv pip install --python $T/vllm_main/bin/python --pre "vllm==0.33.0rc1.dev454+gb8c04dd3b" --index-url https://pypi.org/simple --find-links https://wheels.vllm.ai/nightly/vllm/ --index-strategy unsafe-best-match
uv pip uninstall --python $T/vllm_main/bin/python opencv-python-headless opencv-python || true
$T/vllm_main/bin/python -m pip --version 2>/dev/null | head -1 || true
uv pip list --python $T/vllm_main/bin/python 2>/dev/null | grep -iE "^(vllm|torch|flashinfer|transformers|opencv)"
