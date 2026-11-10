"""
rts14/mechanics_figs.py
=======================
Per-movement joint mechanics figures (same look as helpers/figures.py).

mechanics_grid : 3×3 small multiples — rows = angle / moment / power,
                 columns = hip / knee / ankle. Mean ± SD of every analysed
                 trial, time-normalised (0–100 % of the phase), surgical
                 orange vs non-surgical blue. The report's focus joint
                 column (ankle for Achilles, knee for ACL) is highlighted.
frontal_strip  : hip adduction, knee valgus, ankle frontal motion (change
                 from the start of the phase) — the out-of-plane picture.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import COLORS
from helpers.figures import _style

ROWS = [("ang", "Angle (°)"), ("mom", "Moment (Nm/kg)"), ("pow", "Power (W/kg)")]
COLS = [("hip", "Hip"), ("knee", "Knee"), ("ankle", "Ankle")]
ROW_NOTES = {
    ("ang", "hip"): "+ flexion", ("ang", "knee"): "+ flexion", ("ang", "ankle"): "+ dorsiflexion",
    ("mom", "hip"): "+ extensor", ("mom", "knee"): "+ extensor", ("mom", "ankle"): "+ plantarflexor",
    ("pow", "hip"): "+ generating", ("pow", "knee"): "+ generating", ("pow", "ankle"): "+ generating",
}


def _band(ax, x, m, sd, color, label):
    if m is None or np.all(np.isnan(m)):
        return None
    ax.plot(x, m, color=color, linewidth=1.6, label=label)
    if sd is not None and not np.all(np.isnan(sd)):
        ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.16, linewidth=0)
    return float(np.nanmax(m)), float(np.nanmin(m))


def mechanics_grid(curves, title, x_label, focus="ankle", figsize=(7.5, 6.2)):
    s, ssd = curves.get("s") or {}, curves.get("s_sd") or {}
    n, nsd = curves.get("n") or {}, curves.get("n_sd") or {}
    ev = curves.get("event_pct")
    x = np.linspace(0, 100, 101)
    fig, axes = plt.subplots(3, 3, figsize=figsize, facecolor="white", sharex=True)
    for ri, (rk, rlab) in enumerate(ROWS):
        for ci, (ck, clab) in enumerate(COLS):
            ax = axes[ri, ci]
            key = f"{ck}_{rk}"
            _style(ax, title=f"{clab} {rlab.split(' (')[0].lower()}",
                   ylabel=rlab if ci == 0 else "")
            ax.title.set_fontsize(8.5)
            if ck == focus:
                ax.set_facecolor("#FFF6EF")
                ax.title.set_color(COLORS["accent"])
            ps = _band(ax, x, s.get(key), ssd.get(key), COLORS["surgical"], "Surgical")
            pn = _band(ax, x, n.get(key), nsd.get(key), COLORS["non_surg"], "Non-Surgical")
            ax.axhline(0, color="#CCCCCC", linewidth=0.6)
            if ev is not None and ev == ev:
                ax.axvline(ev, color="#999999", linewidth=0.8, linestyle=":")
            # peak labels
            lines = []
            if ps:
                lines.append((f"S {ps[0]:.1f}" if rk != "ang" else f"S {ps[0]:.0f}°", COLORS["surgical"]))
            if pn:
                lines.append((f"NS {pn[0]:.1f}" if rk != "ang" else f"NS {pn[0]:.0f}°", COLORS["non_surg"]))
            for li, (txt, col) in enumerate(lines):
                ax.text(0.98, 0.95 - li * 0.13, txt, transform=ax.transAxes, ha="right", va="top",
                        fontsize=6.5, color=col, fontweight="bold")
            ax.text(0.02, 0.04, ROW_NOTES.get((rk, ck), ""), transform=ax.transAxes,
                    fontsize=6, color="#999999", style="italic")
            ax.tick_params(labelsize=6.5)
            if ri == 2:
                ax.set_xlabel(x_label if ci == 1 else "", fontsize=7)
    axes[0, 0].legend(fontsize=6.5, loc="upper left", framealpha=0.5)
    fig.suptitle(title, fontsize=10, fontweight="bold", color=COLORS["text"], y=0.995)
    fig.tight_layout(pad=0.6, h_pad=0.6, w_pad=0.6)
    return fig


def frontal_strip(curves, title, x_label, figsize=(7.5, 2.05)):
    s, ssd = curves.get("s") or {}, curves.get("s_sd") or {}
    n, nsd = curves.get("n") or {}, curves.get("n_sd") or {}
    ev = curves.get("event_pct")
    x = np.linspace(0, 100, 101)
    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor="white", sharex=True)
    for ax, (key, lab) in zip(axes, (("hip_add", "Hip adduction / abduction"),
                                     ("knee_val", "Knee valgus / varus"),
                                     ("ankle_fr", "Ankle inversion / eversion"))):
        _style(ax, title=lab, ylabel="Δ from start (°)" if key == "hip_add" else "")
        ax.title.set_fontsize(8.5)
        ps = _band(ax, x, s.get(key), ssd.get(key), COLORS["surgical"], "Surgical")
        pn = _band(ax, x, n.get(key), nsd.get(key), COLORS["non_surg"], "Non-Surgical")
        ax.axhline(0, color="#CCCCCC", linewidth=0.6)
        if ev is not None and ev == ev:
            ax.axvline(ev, color="#999999", linewidth=0.8, linestyle=":")
        for li, (p, lab2, col) in enumerate(((ps, "S", COLORS["surgical"]), (pn, "NS", COLORS["non_surg"]))):
            if p:
                ax.text(0.98, 0.95 - li * 0.14, f"{lab2} range {p[0] - p[1]:.0f}°", transform=ax.transAxes,
                        ha="right", va="top", fontsize=6.5, color=col, fontweight="bold")
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel(x_label if key == "knee_val" else "", fontsize=7)
    fig.suptitle(title, fontsize=9.5, fontweight="bold", color=COLORS["text"], y=1.0)
    fig.tight_layout(pad=0.6, w_pad=0.6)
    return fig


def mech_findings(curves, focus="ankle"):
    """Short descriptive sentences from the averaged curves (no recommendations)."""
    s, n = curves.get("s") or {}, curves.get("n") or {}
    out = []

    def pk(d, k):
        y = d.get(k)
        return (float(np.nanmax(y)), int(np.nanargmax(y))) if y is not None and not np.all(np.isnan(y)) else (None, None)

    joints = [focus] + [j for j in ("ankle", "knee", "hip") if j != focus]
    for j in joints:
        for kind, word, unit in (("mom", "moment", "Nm/kg"), ("pow", "power", "W/kg")):
            a, ta = pk(s, f"{j}_{kind}")
            b, tb = pk(n, f"{j}_{kind}")
            if a is None or b is None or b <= 0:
                continue
            l = 100 * a / b
            if l < 80 or l > 125:
                out.append(f"<b>Peak {j} {word}</b> {a:.1f} vs {b:.1f} {unit} (LSI {l:.0f}%).")
            if kind == "pow" and abs(ta - tb) >= 12:
                out.append(f"<b>Peak {j} power timing</b>: surgical at {ta}% vs non-surgical at {tb}% of the phase "
                           f"— {'later' if ta > tb else 'earlier'} on the surgical side.")
    for key, lab in (("ankle_fr", "Ankle frontal-plane"), ("hip_add", "Hip frontal-plane"), ("knee_val", "Knee frontal-plane")):
        ys, yn = s.get(key), n.get(key)
        if ys is None or yn is None or np.all(np.isnan(ys)) or np.all(np.isnan(yn)):
            continue
        rs, rn = float(np.nanmax(ys) - np.nanmin(ys)), float(np.nanmax(yn) - np.nanmin(yn))
        if abs(rs - rn) >= 5:
            out.append(f"<b>{lab} motion</b>: {rs:.0f}° range surgical vs {rn:.0f}° non-surgical.")
    return out[:4]
