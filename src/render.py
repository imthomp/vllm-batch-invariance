#!/usr/bin/env python3
"""Render README.md from README.tmpl.md and results/stats.json. Placeholders: {{key}}, {{key:pct}}, {{key:int}}."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "results/stats.json").read_text())


def fmt(m):
    key, _, kind = m.group(1).partition(":")
    v = S[key]
    if kind == "pct":
        return f"{v * 100:.1f}%".replace("-", "−")
    if kind == "list":
        v = [str(int(x)) for x in v]
        return v[0] if len(v) == 1 else ", ".join(v[:-1]) + " and " + v[-1]
    if kind == "int":
        return f"{int(v):,}"
    return str(v)


def render():
    return re.sub(r"\{\{([^}]+)\}\}", fmt, (ROOT / "README.tmpl.md").read_text())


if __name__ == "__main__":
    (ROOT / "README.md").write_text(render())
    print("README.md rendered")
