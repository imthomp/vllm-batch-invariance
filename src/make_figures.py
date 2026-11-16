#!/usr/bin/env python3
"""Render the paper's figures from results/ (stats.json and shape maps). Build artifacts: never edit the PDFs by hand.

fig_cost.pdf:   Qwen2.5-7B decode loss against batch-invariant mode off, by batch size: stock mode, tuned table, cuBLAS routing.
fig_shapes.pdf: which (N, K) shapes are batch-invariant under batch-invariant mode's settings, H100 vs B200 (cuBLAS 13.1).
Palette: categorical slots 1-3 of the dataviz reference palette (validated with its validate_palette.js, light mode).
"""
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "results/stats.json").read_text())
OUT = ROOT / "paper"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
PURPLE, RED = "#7651b5", "#b84040"
INK, INK2, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#ecebe7"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})

# Figure 1: decode loss by batch size, with run-to-run ranges for the three-repeat generic-default rerun.
bs = sorted(int(k.split(".")[2][1:]) for k in S if k.startswith("generic2.q25gen2.b") and k.endswith(".tuned_loss"))
series = [("Stock mode", ORANGE, "o", "on", [S[f"generic2.q25gen2.b{b}.on_loss"] for b in bs]),
          ("Stock + breakable graphs", PURPLE, "D", "onbreak", [S[f"generic2.q25gen2.b{b}.onbreak_loss"] for b in bs]),
          ("Generic default", BLUE, "s", "generic", [S[f"generic2.q25gen2.b{b}.generic_loss"] for b in bs]),
          ("Per-model tuned table", AQUA, "^", "tuned", [S[f"generic2.q25gen2.b{b}.tuned_loss"] for b in bs]),
          ("cuBLAS routing", RED, "v", None, [S[f"main.qwen25.b{b}.plain_loss"] for b in bs])]


def paired_run_losses(arm, batch):
    """Paired per-run losses; min/max are descriptive ranges, not confidence intervals."""
    losses = []
    for run in (1, 2, 3):
        vals = {}
        for mode in ("off", arm):
            path = ROOT / "results" / "generic2" / f"decode_q25gen2_{mode}_r{run}.json"
            rows = json.loads(path.read_text())["rows"]
            vals[mode] = st.median(r["step_s"] for r in rows if r["n"] == batch)
        losses.append(1 - vals["off"] / vals[arm])
    return losses


fig, ax = plt.subplots(figsize=(4.1, 2.5))
ax.grid(axis="y", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for label, color, marker, arm, ys in series:
    ys = [100 * y for y in ys]
    ax.plot(bs, ys, color=color, linewidth=2, marker=marker, markersize=4.5, label=label)
    if arm is not None:
        ranges = [paired_run_losses(arm, b) for b in bs]
        low = [max(0, y - 100 * min(run_values)) for y, run_values in zip(ys, ranges)]
        high = [max(0, 100 * max(run_values) - y) for y, run_values in zip(ys, ranges)]
        ax.errorbar(bs, ys, yerr=[low, high], fmt="none", ecolor=color, elinewidth=0.7,
                    capsize=1.8, capthick=0.7, zorder=1)
ax.set_xscale("log", base=2)
ax.set_xticks(bs)
ax.set_xticklabels([str(b) for b in bs])
ax.set_xlabel("Batch size")
ax.set_ylabel("Decode throughput loss vs. mode off (%)")
ax.axhline(0, color=INK2, linewidth=0.6)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=3, fontsize=6.2, handlelength=1.5)
ax.set_xlim(bs[0] / 1.2, bs[-1] * 1.25)
fig.savefig(OUT / "fig_cost.pdf", bbox_inches="tight")
plt.close(fig)


# Figure 2: shape map, H100 vs B200.
def cells(d):
    t = {}
    for key, v in json.loads((ROOT / "results" / d / "shape_map_bi.json").read_text()).items():
        nk = tuple(int(x) for x in key.split("|")[0].split("x"))
        t[nk] = t.get(nk, False) or bool(v["bad"])
    return t


maps = [("H100 (cuBLAS 13.1)", cells("phase1d")), ("B200 (cuBLAS 13.1)", cells("phase1d_b200"))]
assert set(maps[0][1]) == set(maps[1][1])
ns = sorted({n for n, _ in maps[0][1]})
ks = sorted({k for _, k in maps[0][1]})
fig, axes = plt.subplots(1, 2, figsize=(6.8, 4.2), sharey=True)
for ax, (title, t) in zip(axes, maps):
    for i, n in enumerate(ns):
        for j, k in enumerate(ks):
            if (n, k) not in t:
                continue
            fail = t[(n, k)]
            ax.add_patch(plt.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, color=ORANGE if fail else NEUTRAL, linewidth=0))
            if fail:
                ax.text(j, i, "×", ha="center", va="center", color="white", fontsize=6, fontweight="bold")
    ax.set_xlim(-0.6, len(ks) - 0.4)
    ax.set_ylim(-0.6, len(ns) - 0.4)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks], rotation=60)
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels([str(n) for n in ns], fontsize=5.5)
    ax.set_title(title, fontsize=8, color=INK)
    ax.set_xlabel("Inner dimension K")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
axes[0].set_ylabel("Output width N")
handles = [plt.Rectangle((0, 0), 1, 1, color=NEUTRAL), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
fig.legend(handles, ["batch-invariant", "×  not batch-invariant (or error)"], frameon=False, loc="upper center", ncol=2,
           bbox_to_anchor=(0.5, 1.02), fontsize=7)
fig.savefig(OUT / "fig_shapes.pdf", bbox_inches="tight")
plt.close(fig)
print("wrote fig_cost.pdf, fig_shapes.pdf")
