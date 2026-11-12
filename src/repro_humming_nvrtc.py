#!/usr/bin/env python3
"""Minimal repro: Humming's NVRTC JIT fails in a pip-only CUDA environment (no system CUDA toolkit on the loader path)
because its compiler subprocess loads libnvrtc by full path, and libnvrtc then cannot find libnvrtc-builtins by soname."""
import os, sys, platform
import torch, vllm
from vllm import LLM, SamplingParams
import importlib.metadata as md
from humming.utils.nvrtc import get_nvrtc_library_path
print("python", platform.python_version(), "torch", torch.__version__, "vllm", vllm.__version__,
      "humming-kernels", md.version("humming-kernels"), "gpu", torch.cuda.get_device_name(), flush=True)
print("LD_LIBRARY_PATH =", os.environ.get("LD_LIBRARY_PATH", "<unset>"), flush=True)
print("nvrtc library Humming uses:", get_nvrtc_library_path(), flush=True)
llm = LLM("openai/gpt-oss-20b", max_model_len=1024, gpu_memory_utilization=0.85)
out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=8, temperature=0))
print("OK:", repr(out[0].outputs[0].text), flush=True)
