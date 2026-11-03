#!/usr/bin/env python3
"""Run a script with Linear layers using torch's linear in batch-invariant mode, as the Hopper branch intends."""
import runpy
import sys

import vllm.model_executor.layers.linear as L

assert hasattr(L, "linear_batch_invariant")
# Defined in the vLLM module so torch.compile guards can resolve torch.
exec("def _plain_linear(x, w, bias=None):\n    return torch.nn.functional.linear(x, w, bias)\n", L.__dict__)
L.linear_batch_invariant = L._plain_linear
print("patched linear_batch_invariant", flush=True)
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
