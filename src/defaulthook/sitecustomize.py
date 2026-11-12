"""Replace vLLM's batch-invariant matmul default (used for shapes without a tuned-table entry) with a generic one.

Active when VBI_GENERIC=<tune_generic.py output> and this directory is on PYTHONPATH. Patches
batch_invariant_configs._get_matmul_config before batch_invariant imports it. Tuned entries and non-bf16 dtypes are
unchanged. Each patched process prints "patched generic default pid=<pid> block_k=<k>".
"""
import importlib.abc
import importlib.machinery
import json
import os
import sys

TARGET = "vllm.model_executor.determinism.batch_invariant_configs"


def _patch(mod):
    g = json.load(open(os.environ["VBI_GENERIC"]))["generic"]
    bk, rows = g["block_k"], g["m_buckets"]
    orig = mod._get_matmul_config

    def _get_matmul_config(M, N, K, dtype, default):
        cfg = orig(M, N, K, dtype, default)
        if cfg is not default or str(dtype) != "torch.bfloat16":
            return cfg  # a tuned entry (or a non-bf16 dtype): unchanged
        r = next((r for r in rows if M <= r[0]), rows[-1])
        return {"BLOCK_SIZE_M": r[1], "BLOCK_SIZE_N": r[2], "BLOCK_SIZE_K": bk, "GROUP_SIZE_M": 8,
                "num_warps": r[3], "num_stages": r[4]}

    if os.environ.get("VBI_GENERIC_TRACE"):
        # diagnostic: count which configuration each call gets, printed at process exit
        import atexit, collections
        counts = collections.Counter()
        inner = _get_matmul_config

        def _get_matmul_config(M, N, K, dtype, default):
            cfg = inner(M, N, K, dtype, default)
            counts[(N, K, "generic" if cfg.get("BLOCK_SIZE_K") == bk and cfg is not default else "other")] += 1
            return cfg

        atexit.register(lambda: print(f"generic trace pid={os.getpid()} {dict(counts)}", flush=True))
    mod._get_matmul_config = _get_matmul_config
    print(f"patched generic default pid={os.getpid()} block_k={bk}", flush=True)


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


if os.environ.get("VBI_GENERIC"):
    sys.meta_path.insert(0, _Finder())
