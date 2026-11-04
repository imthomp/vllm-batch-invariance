"""Make vLLM pick the Marlin MXFP4 MoE backend instead of Humming on SM90, in every process.

Humming's NVRTC compile fails on our image, so gpt-oss-20b cannot start with the default MoE backend.
Active when VBI_NO_HUMMING=1 and this directory is on PYTHONPATH. Each patched process prints
"patched humming priority pid=<pid>". If VBI_PLAIN=1, the Linear hook in ../plainpatch is chained too
(only one sitecustomize module can load per process).
"""
import importlib.abc
import importlib.machinery
import os
import runpy
import sys

TARGET = "vllm.model_executor.layers.quantization.utils.humming.priority"


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, mod):
        self.inner.exec_module(mod)
        assert hasattr(mod, "_HUMMING_PREFERRED_CAPABILITIES"), "vLLM layout changed"
        mod._HUMMING_PREFERRED_CAPABILITIES = frozenset()
        print(f"patched humming priority pid={os.getpid()}", flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is not None:
            spec.loader = _Loader(spec.loader)
        return spec


if os.environ.get("VBI_NO_HUMMING") == "1":
    sys.meta_path.insert(0, _Finder())
if os.environ.get("VBI_PLAIN") == "1":
    runpy.run_path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plainpatch", "sitecustomize.py"))
