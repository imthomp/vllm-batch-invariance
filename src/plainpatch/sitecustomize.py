"""Route vLLM Linear layers to torch's linear in batch-invariant mode, in every process, including TP workers.

Active when VBI_PLAIN=1 and this directory is on PYTHONPATH. Each patched process prints
"patched linear_batch_invariant pid=<pid>", with "inherited" for forked children.
"""
import importlib.abc
import importlib.machinery
import os
import sys

TARGET = "vllm.model_executor.layers.linear"


def _patch(mod):
    assert hasattr(mod, "linear_batch_invariant"), "vLLM layout changed: linear_batch_invariant not found"
    # Defined in the vLLM module so torch.compile guards can resolve torch.
    exec("def _plain_linear(x, w, bias=None):\n    return torch.nn.functional.linear(x, w, bias)\n", mod.__dict__)
    mod.linear_batch_invariant = mod._plain_linear
    print(f"patched linear_batch_invariant pid={os.getpid()}", flush=True)


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


def _after_fork():
    mod = sys.modules.get(TARGET)
    if mod is not None and getattr(mod, "linear_batch_invariant", None) is getattr(mod, "_plain_linear", 0):
        print(f"patched linear_batch_invariant pid={os.getpid()} inherited", flush=True)


if os.environ.get("VBI_PLAIN") == "1":
    sys.meta_path.insert(0, _Finder())
    # A forked child inherits the patched module without re-importing it; report it too.
    os.register_at_fork(after_in_child=_after_fork)
