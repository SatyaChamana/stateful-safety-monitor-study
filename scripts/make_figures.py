#!/usr/bin/env python3
"""Manuscript figures from local result files (aggregates only; no request text).

Reads results/paper_eval_*.json (Phase 2) and results/phase3a1_test_*.json (Phase 3) and
writes PNGs to docs/figures/. Exploratory panels are titled as such.
"""
from __future__ import annotations
import argparse
import glob
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_here = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("sp3", _here / "summarize_phase3.py")
sp3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp3)
_spec = importlib.util.spec_from_file_location("p3x", _here / "phase3_exploratory.py")
p3x = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p3x)

A = "A_continuous_fpr_calibrated"
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PAPER_PROMPTS = (("BINARY_BASELINE", "Binary"), ("BINARY_INTENTION_EVAL_ORIGINAL", "Probability"),
                 ("BINARY_INTENTION_EVAL_W_8_AGENT_ICL", "Agent-ICL"))
# Published reference values (paper tables / authors' saved files); quoted for comparison only.
PAPER_REF = {"Binary": [("Table 2", 0.719, 0.026), ("Table 4", 0.690, 0.028)],
             "Probability": [], "Agent-ICL": [("Table 4", 0.881, 0.014)]}


def _style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2)


def _save(fig, out: Path, name: str):
    fig.tight_layout()
    fig.savefig(out / name, dpi=200, facecolor="white")
    plt.close(fig)


def _f1_ci(test: dict) -> list[float]:
    from stateful_safety.statistics import bootstrap_outcome_counts
    return bootstrap_outcome_counts(test["TP"], test["FP"], test["FN"], test["TN"])["F1"]["ci95"]


def fig_reproduction(results: Path, out: Path):
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for y, (prompt, label) in enumerate(PAPER_PROMPTS):
        path = glob.glob(str(results / f"paper_eval_{prompt}_*.json"))[0]
        test = json.loads(Path(path).read_text())["primary_all_rows"]["upstream_sweep"]["test"]
        f1, (lo, hi) = test["F1"], _f1_ci(test)
        ax.errorbar(f1, y, xerr=[[f1 - lo], [hi - f1]], fmt="o", color=BLUE, ms=8, capsize=4, lw=2)
        ax.annotate(f"ours {f1:.3f}", (f1, y), xytext=(0, 9), textcoords="offset points",
                    ha="center", color=INK, fontsize=8)
        for j, (name, v, s) in enumerate(PAPER_REF[label]):
            yy = y - 0.28 - 0.18 * j
            ax.errorbar(v, yy, xerr=s, fmt="s", color=ORANGE, mfc="white", ms=7, capsize=3, lw=1.5)
            ax.annotate(f"paper {name} {v:.3f}", (v, yy), xytext=(8, -3), textcoords="offset points",
                        color=INK2, fontsize=7.5)
    ax.set_yticks(range(len(PAPER_PROMPTS)), [p[1] for p in PAPER_PROMPTS])
    ax.set_xlim(0.6, 1.0)
    ax.set_ylim(-0.8, 2.5)
    _style(ax, "Test F1 (paper protocol; ours: 95% CI, paper: ±1 SE)", "")
    ax.set_title("Figure 1. Reproduction of the published monitor (528 test sequences)",
                 loc="left", fontsize=9.5, color=INK)
    _save(fig, out, "fig1_reproduction.png")


def _load_test(results: Path) -> dict:
    runs = {}
    for path in glob.glob(str(results / "phase3a1_test_*.json")):
        d = json.loads(Path(path).read_text())
        runs[d["condition"]] = d
    return runs


def _family_color(cond: str) -> str:
    if cond == "current":
        return ORANGE
    if cond == "full":
        return BLUE
    if cond.startswith("recent"):
        return YELLOW
    return AQUA


def fig_operating_points(runs: dict, out: Path):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    offsets = {"current": (8, -3), "full": (8, -3), "bm25_k3": (-48, -3), "dense_k3": (8, 2),
               "hybrid_k3": (-58, 2), "recent_k3": (8, -3)}
    for cond, d in sorted(runs.items(), key=lambda kv: (not kv[0].endswith("k5"), kv[0])):
        s = d["readouts"][A]["summary"]
        k5 = cond.endswith("k5")
        ax.scatter(s["sequence_fpr"], s["timely_recall"], s=36 if k5 else 70,
                   color=GRAY if k5 else _family_color(cond), edgecolor="white", linewidth=1.5, zorder=3)
        if not k5:
            ax.annotate(cond, (s["sequence_fpr"], s["timely_recall"]), xytext=offsets.get(cond, (6, 0)),
                        textcoords="offset points", fontsize=8, color=INK)
    ax.axvline(0.05, color=INK2, ls="--", lw=1)
    ax.annotate("validation FPR budget 0.05", (0.05, 0.665), xytext=(4, 0), textcoords="offset points",
                fontsize=7.5, color=INK2)
    ax.set_xlim(0.015, 0.076)
    ax.annotate("grey: k = 5 selectors", (0.0, 1.0), xycoords="axes fraction", xytext=(4, -12),
                textcoords="offset points", fontsize=7.5, color=INK2)
    _style(ax, "Benign sequence FPR on test (frozen threshold)", "Timely recall on test")
    ax.set_title("Figure 2. Test operating points, readout A (263 harmful / 265 benign)",
                 loc="left", fontsize=9.5, color=INK)
    _save(fig, out, "fig2_operating_points.png")


def fig_forest(runs: dict, out: Path):
    ref = sp3._outcomes(runs["full"], A)
    order = ["current", "recent_k3", "bm25_k3", "dense_k3", "hybrid_k3",
             "recent_k5", "bm25_k5", "dense_k5", "hybrid_k5"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for y, cond in enumerate(reversed(order)):
        r = sp3.paired_diff(ref, sp3._outcomes(runs[cond], A))["timely_recall_diff"]
        lo, hi = r["ci95"]
        color = AQUA if cond in ("bm25_k3", "dense_k3", "hybrid_k3") else GRAY
        ax.plot([lo, hi], [y, y], color=color, lw=2.5, solid_capstyle="round")
        ax.scatter(r["point"], y, color=color, s=50, zorder=3, edgecolor="white", linewidth=1.5)
        ax.annotate(f"{r['point']:+.3f} [{lo:+.3f}, {hi:+.3f}]", (max(hi, 0.0), y), xytext=(8, -3),
                    textcoords="offset points", fontsize=7.5, color=INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.axvline(-0.05, color=ORANGE, ls="--", lw=1.2)
    ax.annotate("non-inferiority\nmargin −0.05", (-0.05, -0.3), xytext=(-4, 0),
                textcoords="offset points", ha="right", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(order)), list(reversed(order)))
    ax.set_xlim(-0.2, 0.12)
    _style(ax, "Timely recall, condition minus full (paired bootstrap, 95% CI)", "")
    ax.set_title("Figure 3. Paired differences vs full history (test, readout A)",
                 loc="left", fontsize=9.5, color=INK)
    _save(fig, out, "fig3_forest_vs_full.png")


def fig_matched(runs: dict, out: Path):
    series = (("current", ORANGE), ("full", BLUE), ("hybrid_k3", AQUA), ("recent_k3", YELLOW))
    flags = list(range(0, 41))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for cond, color in series:
        pts = [p3x.matched_flag_recall(runs[cond]["per_sequence"], f) for f in flags]
        xs = [p["benign_flags"] for p in pts]
        ys = [p["timely_recall"] for p in pts]
        ax.step(xs, ys, where="post", color=color, lw=2, label=cond)
        s = runs[cond]["readouts"][A]["summary"]
        ax.scatter(round(s["sequence_fpr"] * s["n_benign"]), s["timely_recall"], color=color, s=60,
                   edgecolor=INK, linewidth=1, zorder=4)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xlim(0, 40)
    _style(ax, "Benign sequences flagged (of 265); threshold tuned on test", "Timely recall")
    ax.set_title("Figure 4. EXPLORATORY: recall at matched false alarms (dots = frozen thresholds)",
                 loc="left", fontsize=9.5, color=INK)
    _save(fig, out, "fig4_matched_false_alarms.png")


def fig_by_type(runs: dict, out: Path):
    conds = (("current", ORANGE), ("full", BLUE), ("hybrid_k3", AQUA))
    types = ("QA", "agent", "image")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2))
    width = 0.26
    for panel, (key, label) in enumerate((("timely_recall", "Timely recall"),
                                          ("sequence_fpr", "Benign sequence FPR"))):
        ax = axes[panel]
        for j, (cond, color) in enumerate(conds):
            by = runs[cond]["readouts"][A]["summary_by_task_type"]
            vals = [by[t][key] for t in types]
            xs = [i + (j - 1) * (width + 0.02) for i in range(len(types))]
            ax.bar(xs, vals, width=width, color=color, label=cond, edgecolor="white", linewidth=1)
            for x, v in zip(xs, vals):
                ax.annotate(f"{v:.2f}", (x, v), xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=6.5, color=INK2)
        ax.set_xticks(range(len(types)), types)
        _style(ax, "", label)
        ax.set_ylim(0, 1.1 if key == "timely_recall" else 0.16)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("Figure 5. By task type at frozen thresholds (test, readout A)", x=0.02, ha="left",
                 fontsize=9.5, color=INK)
    _save(fig, out, "fig5_by_task_type.png")


def fig_tokens(runs: dict, out: Path):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for cond, d in sorted(runs.items()):
        u, s = d["usage"], d["readouts"][A]["summary"]
        tokens = u["input_tokens_total"] / u["decisions"]
        k5 = cond.endswith("k5")
        ax.scatter(tokens, s["timely_recall"], color=GRAY if k5 else _family_color(cond),
                   s=36 if k5 else 70, edgecolor="white", linewidth=1.5, zorder=3)
        if not k5:
            off = {"hybrid_k3": (7, 4), "dense_k3": (-52, -2), "bm25_k3": (7, -8)}.get(cond, (6, 3))
            ax.annotate(cond, (tokens, s["timely_recall"]), xytext=off, textcoords="offset points",
                        fontsize=8, color=INK)
    _style(ax, "Mean judge input tokens per decision", "Timely recall on test")
    ax.set_title("Figure 6. Recall vs input cost (fixed few-shot prompt dominates)",
                 loc="left", fontsize=9.5, color=INK)
    _save(fig, out, "fig6_recall_vs_tokens.png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("docs/figures"))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
    fig_reproduction(a.results, a.out)
    runs = _load_test(a.results)
    fig_operating_points(runs, a.out)
    fig_forest(runs, a.out)
    fig_matched(runs, a.out)
    fig_by_type(runs, a.out)
    fig_tokens(runs, a.out)
    print(f"wrote 6 figures to {a.out}")


if __name__ == "__main__":
    main()
