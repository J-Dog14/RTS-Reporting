"""
helpers/figures.py
===================
All matplotlib figures used in the report.
Every function returns a plt.Figure — call _fig_to_image() in report.py
to embed them in the PDF.

Adding a new figure
-------------------
1. Write a function that returns a plt.Figure.
2. Call it from helpers/report.py and embed with _fig_to_image().
3. That's it — no other files need to change.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COLORS, LSI_GREEN, LSI_YELLOW


# ─── Shared style helpers ─────────────────────────────────────────────────────

def _style(ax, title="", ylabel="", xlabel=""):
    """Apply consistent styling to an axes object."""
    ax.set_facecolor("#FAFAFA")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#CCCCCC")
    ax.tick_params(colors=COLORS["text"], labelsize=8)
    ax.yaxis.label.set_color(COLORS["text"])
    ax.xaxis.label.set_color(COLORS["text"])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5, linestyle="--")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold",
                     color=COLORS["text"], pad=6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)


def _lsi_color(lsi_val) -> str:
    """Map an LSI value to a hex colour string."""
    try:
        v = float(lsi_val)
        if v != v:
            return COLORS["grid"]
        if v >= LSI_GREEN:
            return COLORS["green"]
        if v >= LSI_YELLOW:
            return COLORS["yellow"]
        return COLORS["red"]
    except (TypeError, ValueError):
        return COLORS["grid"]


# ─── Force-time curve ─────────────────────────────────────────────────────────

def force_time_curve(
    fz_surg: np.ndarray,
    fz_non_surg: np.ndarray,
    rate: float,
    title: str = "Vertical GRF",
    bw_n: float = 1.0,
    figsize: tuple = (7, 3),
) -> plt.Figure:
    """
    Bilateral force-time overlay.
    Surgical = warm orange, Non-Surgical = cool blue.
    Y-axis normalised to body weight multiples.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    _style(ax, title=title, ylabel="Force (× BW)", xlabel="Time (s)")

    def _plot(fz, color, label):
        if fz is None or len(fz) == 0:
            return
        t    = np.arange(len(fz)) / rate
        norm = fz / bw_n if bw_n > 0 else fz
        ax.plot(t, norm, color=color, linewidth=1.8, label=label, alpha=0.9)
        ax.fill_between(t, norm, alpha=0.08, color=color)

    _plot(fz_surg,     COLORS["surgical"], "Surgical")
    _plot(fz_non_surg, COLORS["non_surg"], "Non-Surgical")
    ax.axhline(1.0, color="#AAAAAA", linewidth=0.8, linestyle=":", label="1× BW")
    ax.legend(fontsize=8, framealpha=0.4, loc="upper right")

    fig.tight_layout(pad=1.5)
    return fig


# ─── Joint angle overlay ──────────────────────────────────────────────────────

def joint_angle_overlay(
    angle_surg: np.ndarray,
    angle_ns: np.ndarray,
    rate: float,
    title: str = "Knee Flexion",
    ylabel: str = "Angle (°)",
    figsize: tuple = (7, 3),
    dir_low: str = "",    # label at low end of y-axis (e.g. "← More Extended")
    dir_high: str = "",   # label at high end of y-axis (e.g. "More Flexed →")
    show_legend: bool = False,  # legend off by default for kinematic panels
) -> plt.Figure:
    """Bilateral joint angle overlay, surgical vs. non-surgical."""
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    _style(ax, title=title, ylabel=ylabel, xlabel="Time (s)")

    def _plot(ang, color, label):
        if ang is None or len(ang) == 0:
            return
        t = np.arange(len(ang)) / rate
        ax.plot(t, ang, color=color, linewidth=1.8, label=label, alpha=0.9)
        ax.fill_between(t, ang, alpha=0.07, color=color)

    _plot(angle_surg, COLORS["surgical"], "Surgical")
    _plot(angle_ns,   COLORS["non_surg"], "Non-Surgical")
    ax.axhline(0, color="#CCCCCC", linewidth=0.6)
    if show_legend:
        ax.legend(fontsize=8, framealpha=0.4)

    # Directional cue annotations on the y-axis
    if dir_low or dir_high:
        ymin, ymax = ax.get_ylim()
        xmin = ax.get_xlim()[0]
        pad = (ymax - ymin) * 0.03
        if dir_high:
            ax.text(xmin, ymax - pad, dir_high,
                    ha="left", va="top", fontsize=6.5,
                    color="#888888", style="italic")
        if dir_low:
            ax.text(xmin, ymin + pad, dir_low,
                    ha="left", va="bottom", fontsize=6.5,
                    color="#888888", style="italic")

    fig.tight_layout(pad=1.5)
    return fig


# ─── LSI bar chart ────────────────────────────────────────────────────────────

def lsi_bar_chart(
    metrics: dict,    # {label: lsi_value}
    title: str = "Limb Symmetry Index Summary",
    figsize: tuple = (7, 3.5),
) -> plt.Figure:
    """
    Horizontal bar chart of LSI values.
    Bars are color-coded green / yellow / red against the 90% and 75% thresholds.
    """
    labels = list(metrics.keys())
    values = [metrics[k] for k in labels]

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    _style(ax, title=title, xlabel="LSI (%)")

    bar_colors = [_lsi_color(v) for v in values]
    bars = ax.barh(labels, values, color=bar_colors, alpha=0.85,
                   edgecolor="white", height=0.55)

    # Reference lines — labeled via annotate so they don't crowd the legend
    ax.axvline(LSI_GREEN,  color=COLORS["green"],  linewidth=1.5, linestyle="--", alpha=0.7)
    ax.axvline(LSI_YELLOW, color=COLORS["yellow"], linewidth=1.0, linestyle=":",  alpha=0.7)
    ax.annotate(f"{LSI_GREEN}%", xy=(LSI_GREEN, 0), xytext=(LSI_GREEN + 0.5, -0.55),
                fontsize=6.5, color=COLORS["green"], fontweight="bold",
                ha="left", va="top")
    ax.annotate(f"{LSI_YELLOW}%", xy=(LSI_YELLOW, 0), xytext=(LSI_YELLOW + 0.5, -0.55),
                fontsize=6.5, color=COLORS["yellow"], fontweight="bold",
                ha="left", va="top")

    # Cap display at 160% — bars beyond that get an arrow + clipped label
    X_MAX = 160
    for bar, val in zip(bars, values):
        try:
            v = float(val)
            if v != v:
                continue
            clipped = v > X_MAX
            # value label — place just inside the right edge if clipped
            label_x = min(v + 1.5, X_MAX - 2) if not clipped else X_MAX - 3
            ha = "left" if not clipped else "right"
            label_txt = f"{v:.1f}%" if not clipped else f"▶ {v:.0f}%"
            ax.text(label_x, bar.get_y() + bar.get_height() / 2,
                    label_txt, va="center", ha=ha,
                    fontsize=7.5, color=COLORS["text"], fontweight="bold")
        except (TypeError, ValueError):
            pass

    ax.set_xlim(0, X_MAX)
    ax.invert_yaxis()

    fig.tight_layout(pad=1.5)
    return fig


# ─── COP scatter plot ──────────────────────────────────────────────────────────

def cop_scatter(
    cop_x_surg: np.ndarray, cop_y_surg: np.ndarray,
    cop_x_ns:   np.ndarray, cop_y_ns:   np.ndarray,
    title: str = "Centre of Pressure — Eyes Closed",
    figsize: tuple = (6, 3.5),
) -> plt.Figure:
    """
    Side-by-side COP scatter for surgical and non-surgical limbs.
    Trajectory fades from light to dark chronologically.
    95% confidence ellipse is drawn for each limb.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor="white")

    def _draw(ax, cop_x, cop_y, color, label):
        if cop_x is None or len(cop_x) == 0:
            ax.text(0.5, 0.5, "No Data", transform=ax.transAxes,
                    ha="center", va="center", color="#AAAAAA", fontsize=9)
            ax.set_title(label, fontsize=9, fontweight="bold", color=COLORS["text"])
            return
        _style(ax, title=label, ylabel="AP (mm)", xlabel="ML (mm)")

        # Fade trajectory from early (light) to late (full color)
        n = len(cop_x)
        step = max(1, n // 300)
        for i in range(0, n - step, step):
            alpha = 0.15 + 0.8 * (i / n)
            ax.plot(cop_x[i:i + step + 1], cop_y[i:i + step + 1],
                    color=color, alpha=alpha, linewidth=0.8)

        # 95% confidence ellipse
        try:
            cx, cy = cop_x - np.mean(cop_x), cop_y - np.mean(cop_y)
            cov = np.cov(cx, cy)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            angle_deg = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
            chi2_95 = 5.991
            w = 2 * np.sqrt(chi2_95 * abs(eigenvalues[1]))
            h = 2 * np.sqrt(chi2_95 * abs(eigenvalues[0]))
            ell = Ellipse((np.mean(cop_x), np.mean(cop_y)),
                          width=w, height=h, angle=angle_deg,
                          edgecolor=color, facecolor="none",
                          linewidth=1.5, linestyle="--", alpha=0.75)
            ax.add_patch(ell)
        except Exception:
            pass

        ax.axhline(0, color="#DDDDDD", linewidth=0.5)
        ax.axvline(0, color="#DDDDDD", linewidth=0.5)
        ax.set_aspect("equal", adjustable="box")

    _draw(ax1, cop_x_surg, cop_y_surg, COLORS["surgical"], "Surgical")
    _draw(ax2, cop_x_ns,   cop_y_ns,   COLORS["non_surg"], "Non-Surgical")

    fig.suptitle(title, fontsize=10, fontweight="bold", color=COLORS["text"])
    fig.tight_layout(pad=1.5)
    return fig


# ─── Bilateral COP AP trace (movement tests) ─────────────────────────────────

def cop_bilateral_ap_trace(
    cop_y_surg: np.ndarray,
    cop_y_ns:   np.ndarray,
    fz_surg:    np.ndarray = None,
    fz_ns:      np.ndarray = None,
    rate: float = 1000.0,
    title: str  = "COP Anterior-Posterior Displacement",
    figsize: tuple = (7.5, 2.0),
) -> plt.Figure:
    """
    Overlay AP COP displacement for surgical and non-surgical limbs.

    V3D exports COP in global lab coordinates, so raw values are large absolute
    numbers (e.g. 1900–2200 mm from lab origin).  We zero-centre each limb by
    subtracting its own contact-phase mean, converting the trace to *relative
    displacement* from the foot's neutral position.  This reveals:
      • How far the COP travels fore-aft during the movement (excursion)
      • Whether one limb's COP is biased anterior or posterior vs its midpoint
      • Asymmetric excursion = restricted / guarded loading strategy
    """
    GRF_THRESH = 30.0   # N — below this = no meaningful plate contact

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    def _prep(cop_y, fz):
        """Return (time, zero-centred displacement) masking non-contact frames."""
        if cop_y is None or len(cop_y) == 0:
            return None, None
        t = np.arange(len(cop_y)) / rate
        y = cop_y.astype(float).copy()
        # build contact mask
        if fz is not None and len(fz) == len(y):
            contact = fz >= GRF_THRESH
        else:
            contact = np.abs(y) > 1e-9
        # zero-centre using the mean of contact frames
        contact_mean = np.nanmean(y[contact]) if contact.any() else np.nanmean(y)
        y -= contact_mean
        # mask non-contact to NaN so gaps appear in the plot
        y[~contact] = np.nan
        return t, y

    t_s, y_s = _prep(cop_y_surg, fz_surg)
    t_n, y_n = _prep(cop_y_ns,   fz_ns)

    for t, y, color, label in [
        (t_s, y_s, COLORS["surgical"], "Surgical"),
        (t_n, y_n, COLORS["non_surg"], "Non-Surg"),
    ]:
        if t is None:
            continue
        ax.plot(t, y, color=color, linewidth=1.0, alpha=0.85, label=label)
        excursion = np.nanmax(y) - np.nanmin(y)
        if not np.isnan(excursion):
            ax.annotate(f"{label}: {excursion:.0f} mm range",
                        xy=(0.02 if label == "Surgical" else 0.02,
                            0.92 if label == "Surgical" else 0.78),
                        xycoords="axes fraction",
                        fontsize=7, color=color,
                        fontweight="bold")

    ax.axhline(0, color="#AAAAAA", linewidth=0.8, linestyle=":", zorder=0,
               label="Neutral (contact mean)")
    _style(ax, title=title,
           ylabel="AP Displacement (mm)\n← Posterior   Anterior →",
           xlabel="Time (s)")
    ax.legend(fontsize=6.5, loc="upper right")
    fig.tight_layout(pad=0.8)
    return fig


# ─── Endurance squat: LSI over time ──────────────────────────────────────────

def endurance_lsi_over_time(
    lsi_values: np.ndarray,
    timestamps: np.ndarray,
    title: str = "Endurance Squat — LSI Over Time",
    figsize: tuple = (7, 3),
) -> plt.Figure:
    """
    Bar chart of per-rep LSI with a linear trend line.
    Color-coded bars reveal when fatigue causes symmetry breakdown.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    _style(ax, title=title, ylabel="LSI (%)", xlabel="Time (s)")

    if lsi_values is not None and len(lsi_values) > 1:
        bar_colors = [_lsi_color(v) for v in lsi_values]
        width = max(0.5, (timestamps[-1] - timestamps[0]) / len(timestamps) * 0.7)
        ax.bar(timestamps, lsi_values, color=bar_colors,
               width=width, alpha=0.85, edgecolor="white")

        # Trend line
        try:
            z = np.polyfit(timestamps, lsi_values, 1)
            p = np.poly1d(z)
            t_line = np.linspace(timestamps[0], timestamps[-1], 100)
            ax.plot(t_line, p(t_line), color=COLORS["text"], linewidth=1.5,
                    linestyle="--", alpha=0.6, label=f"Trend ({z[0]:+.2f}%/s)")
            ax.legend(fontsize=7, framealpha=0.4, loc="lower left")
        except Exception:
            pass

    ax.axhline(LSI_GREEN,  color=COLORS["green"],  linewidth=1.2,
               linestyle="--", alpha=0.7, label=f"{LSI_GREEN}%")
    ax.axhline(LSI_YELLOW, color=COLORS["yellow"], linewidth=0.8,
               linestyle=":",  alpha=0.7)
    ax.set_ylim(0, 115)

    fig.tight_layout(pad=1.5)
    return fig


# ─── RTR radar chart ──────────────────────────────────────────────────────────

def rtr_radar(
    domain_scores: dict,    # {domain_name: score_0_to_100}
    title: str = "Return to Sport Profile",
    figsize: tuple = (5, 5),
) -> plt.Figure:
    """Spider chart summarising performance across domains (0–100)."""
    cats   = list(domain_scores.keys())
    vals   = [min(float(domain_scores[k]), 100.0) for k in cats]   # cap at 100
    n = len(cats)

    if n < 3:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Need ≥ 3 domains for radar",
                ha="center", va="center", color="#AAAAAA")
        return fig

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    vals_c  = vals  + [vals[0]]
    angles_c = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=figsize,
                           subplot_kw={"polar": True}, facecolor="white")
    ax.set_facecolor("#FAFAFA")

    # Threshold rings
    for thresh, color in [(LSI_GREEN, COLORS["green"]), (LSI_YELLOW, COLORS["yellow"])]:
        ring = [thresh] * (n + 1)
        ax.plot(angles_c, ring, color=color, linewidth=0.8, linestyle="--", alpha=0.5)
        ax.fill(angles_c, ring, color=color, alpha=0.04)

    # Data polygon
    ax.plot(angles_c, vals_c, color=COLORS["surgical"], linewidth=2.5,
            marker="o", markersize=5)
    ax.fill(angles_c, vals_c, alpha=0.18, color=COLORS["surgical"])

    ax.set_xticks(angles)
    ax.set_xticklabels(cats, size=8, color=COLORS["text"])
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 90])
    ax.set_yticklabels(["25", "50", "75", "90"], size=6, color="#AAAAAA")
    ax.spines["polar"].set_visible(False)
    ax.set_title(title, size=11, fontweight="bold", color=COLORS["text"], pad=20)

    fig.tight_layout()
    return fig


# ─── Endurance squat: bilateral COP drift across 30 s ────────────────────────

def endurance_cop_drift(
    cop_y_surg: np.ndarray,
    cop_y_ns:   np.ndarray,
    fz_surg:    np.ndarray = None,
    fz_ns:      np.ndarray = None,
    rate: float = 1000.0,
    title: str  = "Endurance Squat — Bilateral AP COP Drift (30 s)",
    figsize: tuple = (7.5, 2.0),
) -> plt.Figure:
    """
    Shows how each limb's AP COP evolves across the full 30-second endurance squat.

    During fatigue, patients typically shift load away from the surgical limb —
    this appears as the surgical-side COP amplitude *shrinking* or trending
    posterior while the non-surgical side increases.  A smoothed envelope (running
    RMS per 1-second window) is overlaid so the fatigue trajectory is visible
    through the high-frequency squat oscillations.

    The raw per-frame trace is plotted lightly in the background; a 1 s rolling
    RMS envelope (solid, bolder line) highlights the trend.
    """
    GRF_THRESH = 30.0

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    def _prep(cop_y, fz):
        if cop_y is None or len(cop_y) == 0:
            return None, None
        t = np.arange(len(cop_y)) / rate
        y = cop_y.astype(float).copy()
        if fz is not None and len(fz) == len(y):
            contact = fz >= GRF_THRESH
        else:
            contact = np.abs(y) > 1e-9
        if not contact.any():
            return None, None
        # zero-centre using contact-phase mean of entire trial
        contact_mean = np.nanmean(y[contact])
        y -= contact_mean
        y[~contact] = np.nan
        return t, y

    def _rolling_rms(y, window_frames):
        """1-D rolling RMS — ignores NaN."""
        n = len(y)
        out = np.full(n, np.nan)
        hw  = window_frames // 2
        for i in range(n):
            chunk = y[max(0, i - hw): min(n, i + hw)]
            valid = chunk[~np.isnan(chunk)]
            if len(valid) > 0:
                out[i] = float(np.sqrt(np.mean(valid ** 2)))
        return out

    window = int(rate)   # 1-second rolling window

    for cop_y, fz, color, label in [
        (cop_y_surg, fz_surg, COLORS["surgical"], "Surgical"),
        (cop_y_ns,   fz_ns,   COLORS["non_surg"], "Non-Surg"),
    ]:
        t, y = _prep(cop_y, fz)
        if t is None:
            continue
        # Raw trace (very light)
        ax.plot(t, y, color=color, linewidth=0.5, alpha=0.25)
        # RMS envelope
        rms = _rolling_rms(y, window)
        ax.plot(t, rms, color=color, linewidth=1.8, alpha=0.90,
                label=f"{label} (RMS envelope)")
        # Annotate early vs late RMS to quantify fatigue shift
        early = rms[: int(rate * 5)]   # first 5 s
        late  = rms[-int(rate * 5):]   # last 5 s
        early_m = np.nanmean(early)
        late_m  = np.nanmean(late)
        if not np.isnan(early_m) and not np.isnan(late_m):
            delta_pct = (late_m - early_m) / (early_m + 1e-9) * 100
            sign = "+" if delta_pct >= 0 else ""
            ax.annotate(
                f"{label}: {sign}{delta_pct:.0f}% shift",
                xy=(t[-1], rms[-1] if not np.isnan(rms[-1]) else 0),
                xycoords="data", fontsize=6.5, color=color, fontweight="bold",
                ha="right", va="bottom",
            )

    ax.axhline(0, color="#AAAAAA", linewidth=0.7, linestyle=":", zorder=0)
    _style(ax, title=title,
           ylabel="AP Displacement RMS (mm)",
           xlabel="Time (s)")
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout(pad=0.8)
    return fig
