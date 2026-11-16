# Humming NVRTC repro (gpt-oss-20b, H200, humming-kernels 0.1.18, pip-only CUDA 13)

Job VL_repro_humming3 (15443239), script `src/repro_humming_nvrtc.py`, 2x2 matrix:

| VLLM_BATCH_INVARIANT | LD_LIBRARY_PATH | MoE backend | result |
|---|---|---|---|
| 1 | unset | HUMMING | NVRTC fails: `failed to open libnvrtc-builtins.so.13.0` |
| 1 | pip `nvidia/cu13/lib` | HUMMING | generates |
| 0 | unset | TRITON | generates |
| 0 | pip `nvidia/cu13/lib` | TRITON | generates |

Raw logs: `bi{1,0}_lp{none,set}.log`; summary: `VL_repro_humming3_15443239.out` (the sbatch exit code 1 comes from the loop's final cache-dir test, not a run).
Not filed upstream: the same root cause is fixed by vllm-project/humming PR #202 (merged 2026-11-11, after v0.1.19).
