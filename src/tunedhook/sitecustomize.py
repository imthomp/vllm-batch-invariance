"""Add tuned Hopper entries (from src/tune_bi.py) to vLLM's batch-invariant matmul table, in every process.

Active when VBI_TUNED=<tuned.json> and this directory is on PYTHONPATH. Each patched process prints
"patched tuned table pid=<pid> shapes=<n>".
"""
import importlib.abc
import importlib.machinery
import json
import os
import sys

TARGET = "vllm.model_executor.determinism.batch_invariant_configs"


def _patch(mod):
    entries = json.load(open(os.environ["VBI_TUNED"]))["entries"]
    table = mod._BATCH_INVARIANT_MATMUL_TUNED_CONFIGS.setdefault("hopper", {})
    for e in entries:
        table[(e["n"], e["k"])] = mod._MatmulShapeConfig(
            block_k=e["block_k"], m_buckets=tuple((b[0], mod._MatmulMConfig(*b[1:])) for b in e["m_buckets"]))
    print(f"patched tuned table pid={os.getpid()} shapes={len(entries)}", flush=True)


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, mod):
        self.inner.exec_module(mod)
        _patch(mod)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is not None:
            spec.loader = _Loader(spec.loader)
        return spec


if os.environ.get("VBI_TUNED"):
    sys.meta_path.insert(0, _Finder())
