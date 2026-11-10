"""
rts14/achilles_figs.py
======================
Figures for the Achilles "Calf & Ankle" page. Same look as helpers/figures.py
(shared _style, COLORS; surgical = orange, non-surgical = blue).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import COLORS, LSI_GREEN, LSI_YELLOW
from helpers.figures import _style, _lsi_color

JOINT_COLORS = {"Hip": "#C9D1DA", "Knee": "#7A8A9C", "Ankle": COLORS["accent"]}


def joint_work_stacked(tests, title="Push-off work by joint", figsize=(7.5, 2.8), unit="J"):
    """
    tests : list of (test_label, {"s": {hip, knee, ankle}, "n": {...}})
    Horizontal stacked bars, two per test (Surgical / Non-surgical), ankle
    segment in brand orange with its share of the total printed on it.
    """
    rows, labels = [], []
    for name, jw in tests:
        for tag, lim in (("s", "Surgical"), ("n", "Non-Surg")):
            d = jw.get(tag) or {}
            rows.append([max(0.0, float(d.get(j, np.nan) or 0)) if d.get(j) == d.get(j) else 0.0
                         for j in ("hip", "knee", "ankle")])
            labels.append(f"{name} · {lim}")
    rows = np.array(rows) if rows else np.zeros((0, 3))
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    _style(ax, title=title, xlabel=f"Work ({unit})")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.5, linestyle="--")
    ax.grid(axis="y", visible=False)
    # two bars per test (surgical on top), with a gap between tests
    y = np.array([-(i // 2) * 2.6 - (i % 2) * 0.95 for i in range(len(rows))], dtype=float)
    left = np.zeros(len(rows))
    for ji, j in enumerate(("Hip", "Knee", "Ankle")):
        vals = rows[:, ji] if len(rows) else []
        ax.barh(y, vals, left=left, height=0.72, color=JOINT_COLORS[j],
                edgecolor="white", linewidth=0.8, label=j)
        left = left + vals
    tot = rows.sum(axis=1) if len(rows) else []
    for yi, r, t, lab in zip(y, rows, tot, labels):
        if t > 0:
            share = 100 * r[2] / t
            ax.text(t + max(tot) * 0.01, yi, f"{t:.0f} {unit} · ankle {share:.0f}%",
                    va="center", fontsize=7, color=COLORS["text"],
                    fontweight="bold" if "Surgical" in lab else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    for tl, lab in zip(ax.get_yticklabels(), labels):
        tl.set_color(COLORS["surgical"] if "Surgical" in lab else COLORS["non_surg"])
    if len(tot):
        ax.set_xlim(0, max(tot) * 1.28)
    ax.legend(fontsize=7, ncol=3, loc="lower right", framealpha=0.6)
    fig.tight_layout(pad=0.9)
    return fig


def aligned_curves(s, n, rate, to_s, to_n, title, ylabel, figsize=(3.6, 2.3),
                   window=(-0.6, 0.1), fill_positive=True, unit=""):
    """Surgical vs non-surgical traces aligned to take-off (t = 0)."""
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    _style(ax, title=title, ylabel=ylabel, xlabel="Time to take-off (s)")
    peaks = []
    for y, to, col, lab in ((s, to_s, COLORS["surgical"], "Surgical"),
                            (n, to_n, COLORS["non_surg"], "Non-Surgical")):
        if y is None or to is None or len(y) == 0:
            continue
        t = (np.arange(len(y)) - to) / rate
        m = (t >= window[0]) & (t <= window[1])
        if not m.any():
            continue
        ax.plot(t[m], y[m], color=col, linewidth=1.8, label=lab, alpha=0.95)
        if fill_positive:
            ax.fill_between(t[m], np.clip(y[m], 0, None), alpha=0.12, color=col)
        pk = np.nanmax(y[m])
        peaks.append((lab, pk, col))
    ax.axvline(0, color="#999999", linewidth=0.8, linestyle=":")
    ax.text(0, ax.get_ylim()[1], " take-off", fontsize=6.5, color="#888888", va="top")
    ax.axhline(0, color="#CCCCCC", linewidth=0.6)
    for i, (lab, pk, col) in enumerate(peaks):
        ax.text(0.02, 0.95 - i * 0.11, f"{lab} peak {pk:,.0f}{unit}", transform=ax.transAxes,
                fontsize=6.8, color=col, fontweight="bold", va="top")
    fig.tight_layout(pad=0.9)
    return fig


def ankle_lsi_by_test(data, title="Surgical ankle vs non-surgical (LSI)", figsize=(7.5, 2.4)):
    """
    data : {metric_label: {test_label: lsi}}  e.g. {"Peak PF moment": {"SLJ": 112, ...}}
    Grouped horizontal dot-and-bar chart with the 75 / 90 % bands.
    """
    metrics = [m for m in data if any(v == v and v is not None for v in data[m].values())]
    tests = []
    for m in metrics:
        for t in data[m]:
            if t not in tests:
                tests.append(t)
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    _style(ax, title=title, xlabel="LSI (%)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.5, linestyle="--")
    ax.axvspan(0, LSI_YELLOW, color=COLORS["red"], alpha=0.05)
    ax.axvspan(LSI_YELLOW, LSI_GREEN, color=COLORS["yellow"], alpha=0.06)
    ax.axvline(LSI_GREEN, color=COLORS["green"], linewidth=1.2, linestyle="--", alpha=0.7)
    ax.axvline(100, color="#BBBBBB", linewidth=0.8)
    ys, labs = [], []
    y = 0.0
    markers = ["o", "s", "D", "^"]
    for mi, m in enumerate(metrics):
        for ti, t in enumerate(tests):
            v = data[m].get(t)
            if v is None or v != v:
                continue
            ax.plot([100, v], [y, y], color=_lsi_color(v), linewidth=2.2, alpha=0.55)
            ax.plot(v, y, marker=markers[mi % 4], color=_lsi_color(v), markersize=6,
                    markeredgecolor="white")
            ax.text(min(v, 158) + 2, y, f"{v:.0f}%", va="center", fontsize=7, color=COLORS["text"])
            ys.append(y)
            labs.append(f"{m} — {t}")
            y -= 1
        y -= 0.5
    ax.set_yticks(ys)
    ax.set_yticklabels(labs, fontsize=7.2)
    ax.set_xlim(0, 165)
    fig.tight_layout(pad=0.9)
    return fig
