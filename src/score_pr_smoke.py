#!/usr/bin/env python3
"""Record the H100 core-PR integration smoke from its raw log in the stats SSOT."""
import ast
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = Path(sys.argv[1])
dest = ROOT / "results" / "validation" / "pr_generic_smoke_h100" / source.name
dest.parent.mkdir(parents=True, exist_ok=True)
if source.resolve() != dest.resolve():
    shutil.copy2(source, dest)
log = dest.read_text(errors="replace")
config_match = re.search(r"generic qkv config: (\{[^\n]+\})", log)
assert config_match, "generic QKV config was not logged"
config = ast.literal_eval(config_match.group(1))
assert "Breakable CUDA graph enabled" in log
assert config["BLOCK_SIZE_K"] == 128
assert "PASS: target tokens match alone and in a mixed batch" in log
assert "NVIDIA H100" in log

stats = {
    "pr_smoke.generic_core.h100.breakable_graph_enabled": 1,
    "pr_smoke.generic_core.h100.qkv_block_k": config["BLOCK_SIZE_K"],
    "pr_smoke.generic_core.h100.target_tokens_match": 1,
}
sys.path.insert(0, str(ROOT / "src"))
from analyze import merge

merge(stats)
print(json.dumps(stats, indent=2))
