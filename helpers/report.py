"""
helpers/report.py
==================
PDF report builder — compact, clinic-ready layout.

Summary page  : RTR badge · LSI bar chart · radar · clinical flags — ONE page.
Test pages    : metrics table · interpretation box · figures, ONE page per test.
                Side-by-side figure pairs keep vertical space tight.
"""

import io
import os
import sys
import datetime
from typing import Optional, List, Tuple

import numpy as np

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image, KeepTogether,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLINIC_NAME, CLINIC_SUBTITLE, COLORS, PAGE_SIZE, LSI_GREEN, LSI_YELLOW
from helpers.norms import NORMS, compute_rtr_score, time_factor as _time_factor
from helpers import figures as fig_lib


# ─── Utility helpers ─────────────────────────────────────────────────────────

def _isnan(v) -> bool:
    """Return True if v is None or a float NaN."""
    if v is None:
        return True
    try:
        import math
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


# ─── Layout constants ─────────────────────────────────────────────────────────
PAGE    = letter if PAGE_SIZE == "letter" else A4
MARGIN  = 0.5 * inch
W_FULL  = 7.5 * inch          # usable page width (8.5 - 2×0.5)
W_HALF  = (W_FULL - 0.15 * inch) / 2   # half-width for side-by-side figures
H_PAGE  = 10.0 * inch         # usable page height (11 - 2×0.5)

# ─── Colours ──────────────────────────────────────────────────────────────────
C_HEADER = colors.HexColor(COLORS["header_bg"])
C_TABLE_HEADER_FG = colors.HexColor(COLORS.get("table_header_fg", COLORS.get("header_fg", "#FFFFFF")))
C_ACCENT = colors.HexColor(COLORS["accent"])
C_GREEN  = colors.HexColor(COLORS["green"])
C_YELLOW = colors.HexColor(COLORS["yellow"])
C_RED    = colors.HexColor(COLORS["red"])
C_TEXT   = colors.HexColor(COLORS["text"])
C_GRID   = colors.HexColor(COLORS["grid"])
C_WHITE  = colors.white
C_LTGREY = colors.HexColor(COLORS["lt_grey"])


# ─── Figure helpers ───────────────────────────────────────────────────────────

def _fig_img(fig: plt.Figure, width_in: float, dpi: int = 130) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    aspect = fig.get_size_inches()[1] / fig.get_size_inches()[0]
    return Image(buf, width=width_in * inch, height=width_in * aspect * inch)


def _fig_full(fig: plt.Figure, caption: str, s: dict) -> List:
    """One figure spanning full page width with caption below."""
    img = _fig_img(fig, width_in=W_FULL / inch)
    cap = Paragraph(caption, s["cap"])
    return [img, cap, Spacer(1, 4)]


def _fig_pair(fig1: plt.Figure, cap1: str,
              fig2: plt.Figure, cap2: str, s: dict) -> List:
    """Two figures side-by-side with captions."""
    w = W_HALF / inch
    i1 = _fig_img(fig1, width_in=w)
    i2 = _fig_img(fig2, width_in=w)
    c1 = Paragraph(cap1, s["cap"])
    c2 = Paragraph(cap2, s["cap"])
    tbl = Table([[i1, i2], [c1, c2]],
                colWidths=[W_HALF, W_HALF])
    tbl.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",   (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 2),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return [tbl, Spacer(1, 3)]


# ─── Clinical one-liner helper ───────────────────────────────────────────────

def _lsi_phrase(lsi, metric: str = "loading", units: str = "") -> str:
    """
    Return a short clinical sentence describing the LSI value.

    lsi    : float (%), or None
    metric : short label used in the sentence (e.g. "peak GRF", "knee flexion")
    units  : optional unit string appended to the context (e.g. "N", "°")
    """
    if lsi is None or lsi != lsi:
        return ""
    v = float(lsi)
    if v >= 95:
        return (f"Surgical limb {metric} is symmetric with the non-surgical side "
                f"(LSI {v:.0f}%) — meets return-to-sport threshold.")
    elif v >= 90:
        return (f"Surgical limb {metric} near-symmetric (LSI {v:.0f}%) — "
                f"approaching return-to-sport target; minor deficit remains.")
    elif v >= 75:
        return (f"Surgical limb {metric} shows a moderate asymmetry (LSI {v:.0f}%) — "
                f"continued loading progression recommended before clearance.")
    else:
        return (f"Surgical limb {metric} demonstrates a significant deficit "
                f"(LSI {v:.0f}%) — targeted strengthening and re-testing indicated.")


def _delta_phrase(sv, nv, metric, unit="°", warn=None, err=None) -> str:
    """One-sentence clinical summary for a delta-style (non-LSI) kinematic metric."""
    try:
        sv_f, nv_f = float(sv), float(nv)
        delta = sv_f - nv_f
        sign  = "+" if delta >= 0 else ""
        base  = (f"Surgical {metric}: {sv_f:.1f}{unit} vs non-surgical {nv_f:.1f}{unit} "
                 f"(Δ {sign}{delta:.1f}{unit})")
        if err and abs(delta) > err:
            return f"{base} — exceeds {err:.0f}° clinical threshold; targeted intervention recommended."
        elif warn and abs(delta) > warn:
            return f"{base} — mild asymmetry; monitor closely."
        else:
            return f"{base} — within acceptable range."
    except (TypeError, ValueError):
        return ""


# ─── Formatting helpers ───────────────────────────────────────────────────────

def _fmt(val, decimals: int = 1, suffix: str = "") -> str:
    if val is None:
        return "—"
    if isinstance(val, str):
        return val
    try:
        fv = float(val)
        return "—" if fv != fv else f"{fv:.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return str(val)


def _badge_color(lsi_val):
    try:
        v = float(lsi_val)
        if v != v: return C_GRID
        if v >= LSI_GREEN:  return C_GREEN
        if v >= LSI_YELLOW: return C_YELLOW
        return C_RED
    except (TypeError, ValueError):
        return C_GRID


def _row_bg(lsi_val):
    try:
        v = float(lsi_val)
        if v != v: return None
        if v >= LSI_GREEN:  return colors.HexColor("#E8F8EF")
        if v >= LSI_YELLOW: return colors.HexColor("#FFF8E7")
        return colors.HexColor("#FDE8E8")
    except (TypeError, ValueError):
        return None


def _delta_color(delta_val, warn_deg: float, err_deg: float):
    """
    Colour for an absolute side-to-side difference (degrees).
    Green  = within warn threshold (symmetric enough)
    Yellow = between warn and err (mild asymmetry)
    Red    = beyond err threshold (significant asymmetry)
    """
    try:
        v = abs(float(delta_val))
        if v <= warn_deg: return C_GREEN
        if v <= err_deg:  return C_YELLOW
        return C_RED
    except (TypeError, ValueError):
        return C_GRID


def _delta_row_bg(delta_val, warn_deg: float, err_deg: float):
    """Row background matching _delta_color."""
    try:
        v = abs(float(delta_val))
        if v <= warn_deg: return colors.HexColor("#E8F8EF")
        if v <= err_deg:  return colors.HexColor("#FFF8E7")
        return colors.HexColor("#FDE8E8")
    except (TypeError, ValueError):
        return None


def _delta_row(label: str, sv, nv,
               warn_deg: float = 3.0, err_deg: float = 6.0):
    """
    Build a metrics-table row using absolute side-to-side delta instead of LSI.
    4th column = ("delta", Δvalue, warn_deg, err_deg)
    Returns None if both values are unavailable.
    """
    try:
        s = float(sv)
        n = float(nv)
        if s != s or n != n:
            raise ValueError
        delta = round(s - n, 1)   # signed: positive = surgical > non-surgical
    except (TypeError, ValueError):
        delta = None
    return (label, sv, nv, ("delta", delta, warn_deg, err_deg))


# ─── Styles ───────────────────────────────────────────────────────────────────

def _styles() -> dict:
    return {
        "h1": ParagraphStyle("h1",
            fontSize=20, textColor=C_WHITE, fontName="Helvetica-Bold",
            leading=24, spaceAfter=1),
        "h1sub": ParagraphStyle("h1sub",
            fontSize=10, textColor=colors.HexColor("#AACCEE"),
            fontName="Helvetica", spaceAfter=1),
        "pt": ParagraphStyle("pt",
            fontSize=8.5, textColor=C_WHITE, fontName="Helvetica", leading=13),
        "sec": ParagraphStyle("sec",
            fontSize=11, textColor=C_WHITE, fontName="Helvetica-Bold",
            backColor=C_HEADER, spaceBefore=4, spaceAfter=3,
            leftIndent=5, leading=16),
        "body": ParagraphStyle("body",
            fontSize=8.5, textColor=C_TEXT, fontName="Helvetica",
            leading=12, spaceAfter=1),
        "interp": ParagraphStyle("interp",
            fontSize=8, textColor=colors.HexColor("#1A1A1A"),
            fontName="Helvetica", leading=11.5, spaceAfter=2,
            alignment=TA_JUSTIFY),
        "interp_hd": ParagraphStyle("interp_hd",
            fontSize=8, textColor=colors.HexColor("#39414a"),
            fontName="Helvetica-Bold", spaceAfter=2),
        "small": ParagraphStyle("small",
            fontSize=7, textColor=colors.HexColor("#777777"),
            fontName="Helvetica", leading=9, spaceAfter=1),
        "cap": ParagraphStyle("cap",
            fontSize=7, textColor=colors.HexColor("#888888"),
            fontName="Helvetica-Oblique", alignment=TA_CENTER,
            spaceAfter=2, leading=9),
    }


# ─── Page header ──────────────────────────────────────────────────────────────

def _surgery_date_display(surgery_date: str) -> str:
    """Show YYYY-MM-DD as MM/DD/YYYY; pass through MM/YY and other forms."""
    if not surgery_date:
        return surgery_date
    try:
        d = datetime.datetime.strptime(surgery_date.strip(), "%Y-%m-%d").date()
        return d.strftime("%m/%d/%Y")
    except ValueError:
        return surgery_date


def _header(s: dict, name: str, dob: str, side: str,
            date: str, clinician: str = "",
            surgery_date: str = "", months_post_op: float = None,
            triple_hop: dict = None) -> Table:
    """
    Header layout — two-column inner block:
      LEFT  (65%): CLINIC_NAME (bold), then patient / date / surgery / triple hop info lines
      RIGHT (35%): "Return to Sport Assessment" — larger font, right-aligned, vertically centred
    """
    # ── Styles ────────────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "hdr_title", fontSize=14, fontName="Helvetica-Bold",
        textColor=colors.white, leading=17,
    )
    subtitle_style = ParagraphStyle(
        "hdr_subtitle_r", fontSize=12, fontName="Helvetica",
        textColor=colors.HexColor("#A8BCCC"), alignment=TA_RIGHT, leading=15,
    )
    info_style = ParagraphStyle(
        "hdr_info", fontSize=8.5, fontName="Helvetica",
        textColor=colors.HexColor("#D0D5DC"), leading=12,
    )

    # ── Patient info lines ────────────────────────────────────────────────────
    name_parts = [f"<b>Patient:</b> {name}"]
    if dob and dob != "—":
        name_parts.append(f"<b>DOB:</b> {dob}")
    name_parts.append(f"<b>Date:</b> {date}")
    line2 = Paragraph("  &nbsp;•&nbsp;  ".join(name_parts), info_style)

    side_label = "Left" if str(side).upper() == "L" else "Right" if str(side).upper() == "R" else str(side)
    side_parts = [f"<b>Surgical Side:</b> {side_label}"]
    if months_post_op is not None:
        mo = months_post_op
        time_str = f"{mo:.1f} mo post-op" if mo < 12 else f"{mo/12:.1f} yr post-op"
        if surgery_date:
            sd = _surgery_date_display(surgery_date)
            side_parts.append(f"<b>Surgery:</b> {sd} ({time_str})")
        else:
            side_parts.append(f"<b>Post-op:</b> {time_str}")
    elif surgery_date:
        side_parts.append(f"<b>Surgery:</b> {_surgery_date_display(surgery_date)}")

    th = triple_hop or {}
    th_surg = th.get("surg_in")
    th_ns   = th.get("ns_in")
    th_lsi  = th.get("lsi")
    if th_surg is not None or th_ns is not None:
        def _fmt_hop(v):
            if v is None:
                return "—"
            ft  = int(v) // 12
            ins = int(round(v)) % 12
            return f"{ft}'{ins}\""
        hop_parts = []
        if th_surg is not None:
            hop_parts.append(f"Surg {_fmt_hop(th_surg)}")
        if th_ns is not None:
            hop_parts.append(f"NS {_fmt_hop(th_ns)}")
        if th_lsi is not None:
            hop_parts.append(f"LSI {th_lsi:.1f}%")
        side_parts.append(f"<b>Triple Hop:</b> {' / '.join(hop_parts)}")
    line3 = Paragraph("  &nbsp;•&nbsp;  ".join(side_parts), info_style)

    # ── Left cell ─────────────────────────────────────────────────────────────
    left_content = [
        Paragraph(CLINIC_NAME, title_style),
        Spacer(1, 4),
        line2,
        line3,
    ]
    if clinician:
        left_content.append(Paragraph(f"<b>Clinician:</b> {clinician}",
            ParagraphStyle("cl", fontSize=7.5,
                           textColor=colors.HexColor("#AACCEE"),
                           fontName="Helvetica")))

    # ── Right cell: subtitle right-aligned, vertically centred ────────────────
    right_content = [Paragraph(CLINIC_SUBTITLE, subtitle_style)]

    LEFT_COL  = W_FULL * 0.62
    RIGHT_COL = W_FULL * 0.38
    inner = Table([[left_content, right_content]],
                  colWidths=[LEFT_COL, RIGHT_COL])
    inner.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",         (1, 0), (1,  -1), "RIGHT"),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
    ]))

    tbl = Table([[inner]], colWidths=[W_FULL])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_HEADER),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
    ]))
    return tbl


def _sec(text: str, s: dict) -> List:
    return [Paragraph(f"  {text}", s["sec"]), Spacer(1, 3)]


# ─── Compact metrics table ────────────────────────────────────────────────────

def _metrics_table(rows: list, s: dict) -> Table:
    """
    rows: list of (label, surg_val, ns_val, lsi_or_delta)

    lsi_or_delta can be:
      • a float/int  → treated as LSI (%), colour-coded against LSI thresholds,
                        displayed as "95%"
      • None         → shown as "—" (grey), no row tinting
      • ("delta", Δ, warn_deg, err_deg)
                     → absolute side-to-side difference in degrees,
                        colour-coded against absolute thresholds,
                        displayed as "Δ +3.0°"

    Compact version — tight padding, colour-coded rows, no separate status column.
    """
    _hdr_style  = ParagraphStyle("tblhdr", parent=s["body"],
                                  textColor=colors.HexColor("#FFFFFF"))
    data = [[
        Paragraph("<b>Metric</b>", _hdr_style),
        Paragraph("<b>Surgical</b>", _hdr_style),
        Paragraph("<b>Non-Surg</b>", _hdr_style),
        Paragraph("<b>LSI / Δ</b>", _hdr_style),
    ]]
    _dash_style = ParagraphStyle("greydash", fontSize=8.5, alignment=TA_CENTER,
                                  textColor=colors.HexColor("#CCCCCC"))

    for row in rows:
        name, sv, nv, lv = row[:4]
        is_bilateral = (sv == "—" and nv == "—")

        # ── Determine 4th-column display mode ────────────────────────────────
        is_delta = isinstance(lv, tuple) and len(lv) == 4 and lv[0] == "delta"

        if is_delta:
            _, delta_val, warn_deg, err_deg = lv
            badge_color = _delta_color(delta_val, warn_deg, err_deg)
            try:
                bhex = badge_color.hexval()[2:]
            except Exception:
                bhex = "AAAAAA"
            if delta_val is None:
                fourth_cell = Paragraph("—", _dash_style)
            else:
                sign = "+" if delta_val > 0 else ""
                fourth_cell = Paragraph(
                    f"<font color='#{bhex}'><b>Δ {sign}{delta_val:.1f}°</b></font>",
                    ParagraphStyle("dv", fontSize=8.5, alignment=TA_CENTER))
        else:
            try:
                bhex = _badge_color(lv).hexval()[2:]
            except Exception:
                bhex = "AAAAAA"
            fourth_cell = Paragraph(
                f"<font color='#{bhex}'><b>{_fmt(lv, suffix='%')}</b></font>",
                ParagraphStyle("lv", fontSize=8.5, alignment=TA_CENTER))

        if is_bilateral:
            name_cell = Paragraph(
                f'{name} <font size="6" color="#AAAAAA"><i>(bilateral)</i></font>',
                s["body"])
            sv_cell = Paragraph("—", _dash_style)
            nv_cell = Paragraph("—", _dash_style)
        else:
            name_cell = Paragraph(name, s["body"])
            sv_cell   = _fmt(sv)
            nv_cell   = _fmt(nv)

        data.append([name_cell, sv_cell, nv_cell, fourth_cell])

    col_w = [3.0*inch, 1.1*inch, 1.1*inch, 1.0*inch]
    tbl   = Table(data, colWidths=col_w)
    cmds  = [
        ("BACKGROUND",    (0,0), (-1,0),  C_HEADER),
        ("TEXTCOLOR",     (0,0), (-1,0),  C_TABLE_HEADER_FG),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("ALIGN",         (1,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [C_WHITE, C_LTGREY]),
        ("LINEBELOW",     (0,0), (-1,0),  0.5, C_ACCENT),
        ("LINEBELOW",     (0,1), (-1,-1), 0.3, C_GRID),
        ("TOPPADDING",    (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ("LEFTPADDING",   (0,0), (0,-1),  5),
        ("LEFTPADDING",   (1,0), (-1,-1), 3),
    ]
    for i, row in enumerate(rows, start=1):
        lv = row[3]
        if isinstance(lv, tuple) and len(lv) == 4 and lv[0] == "delta":
            _, delta_val, warn_deg, err_deg = lv
            bg = _delta_row_bg(delta_val, warn_deg, err_deg)
        else:
            bg = _row_bg(lv)
        if bg:
            cmds.append(("BACKGROUND", (0,i), (-1,i), bg))
    tbl.setStyle(TableStyle(cmds))
    return tbl


# ─── Interpretation box ───────────────────────────────────────────────────────

def _interp_box(lines: List[str], s: dict) -> Table:
    """Compact blue interpretation box — 2-3 short lines maximum."""
    content = [Paragraph("CLINICAL INTERPRETATION", s["interp_hd"])]
    for line in lines:
        content.append(Paragraph(line, s["interp"]))
    tbl = Table([[content]], colWidths=[W_FULL])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), colors.HexColor("#F2F3F4")),
        ("LINEABOVE",    (0,0), (-1,0),  1.5, colors.HexColor("#fc6c0f")),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
    ]))
    return tbl


# ─── LSI threshold legend ─────────────────────────────────────────────────────

def _legend(s: dict) -> Paragraph:
    return Paragraph(
        f"<font color='{COLORS['green']}'>●</font> ≥{LSI_GREEN}% Symmetric &nbsp;&nbsp;"
        f"<font color='{COLORS['yellow']}'>●</font> ≥{LSI_YELLOW}% Progressing &nbsp;&nbsp;"
        f"<font color='{COLORS['red']}'>●</font> &lt;{LSI_YELLOW}% Deficit &nbsp;&nbsp;"
        "LSI = Surgical ÷ Non-Surgical × 100 &nbsp;|&nbsp; "
        "<b>Δ</b> = Surgical − Non-Surgical (absolute difference; used for small-angle kinematics)",
        s["small"])


# ─── Score influence helper ───────────────────────────────────────────────────

# Human-readable labels for RTR breakdown keys
_INFLUENCE_NAMES = {
    "drop_jump_rsi":    "RSI (reactive strength)",
    "landing_lsi":      "Landing symmetry",
    "rfd_lsi":          "Rate of force dev.",
    "peak_grf_lsi":     "DJ Peak GRF sym.",
    "dl_peak_grf_lsi":  "DL Peak GRF sym.",
    "dl_load_rate_lsi": "DL Load rate sym.",
    "sl_jump_lsi":      "Single-leg jump",
    "knee_valgus_surg": "Knee valgus (surgical)",
    "cop_velocity_lsi": "Balance symmetry",
    "endurance_lsi":    "Endurance symmetry",
    "fatigue_drift":    "Fatigue drift",
}


def _score_influence_box(rtr_result: dict, s: dict,
                         width: float, tf: float,
                         perf_score: float,
                         months_post_op: float = None) -> list:
    """
    Stacked Strengths → Detractors panel (single column, full width).
    Returns a list of flowables sized to `width` inches.
    """
    breakdown  = rtr_result.get("breakdown", {})
    red_factor = rtr_result.get("red_factor", 1.0)
    score      = rtr_result.get("score", 0)

    # Split into strengths (green) and detractors (yellow/red), sorted by score
    strengths  = sorted(
        [(k, v) for k, v in breakdown.items() if v["color"] == "green"],
        key=lambda x: -x[1]["score"])[:4]
    detractors = sorted(
        [(k, v) for k, v in breakdown.items() if v["color"] in ("red", "yellow")],
        key=lambda x: x[1]["score"])[:5]

    C_BOX   = colors.HexColor("#f8f8f8")
    C_STR_H = colors.HexColor("#1a5c34")   # dark green header bg
    C_DET_H = colors.HexColor("#7a1c10")   # dark red header bg
    C_STR   = colors.HexColor("#1a5c34")   # dark green item text
    C_DET   = colors.HexColor("#a02010")   # dark red item text
    C_ORG   = colors.HexColor("#fc6c0f")
    C_SUB   = colors.HexColor("#555555")

    box_w = width * inch

    s_hd_str = ParagraphStyle("sdr_hd_s", fontSize=7.5, fontName="Helvetica-Bold",
                               textColor=C_WHITE, leading=10, leftIndent=4)
    s_hd_det = ParagraphStyle("sdr_hd_d", fontSize=7.5, fontName="Helvetica-Bold",
                               textColor=C_WHITE, leading=10, leftIndent=4)
    s_item_str = ParagraphStyle("sdr_is", fontSize=7.2, fontName="Helvetica",
                                textColor=C_STR, leading=10, leftIndent=8, spaceAfter=1)
    s_item_det = ParagraphStyle("sdr_id", fontSize=7.2, fontName="Helvetica",
                                textColor=C_DET, leading=10, leftIndent=8, spaceAfter=1)
    s_eqn  = ParagraphStyle("sdr_eqn", fontSize=6.5, fontName="Helvetica",
                             textColor=C_SUB, leading=9, leftIndent=4)
    s_none = ParagraphStyle("sdr_none", fontSize=7, fontName="Helvetica",
                             textColor=C_SUB, leading=9, leftIndent=8)

    rows = []

    # ── STRENGTHS header ──────────────────────────────────────────────────────
    rows.append([Paragraph("▲  STRENGTHS", s_hd_str)])
    if strengths:
        for k, v in strengths:
            name  = _INFLUENCE_NAMES.get(k, k)
            score_val = v.get("score", 0)
            rows.append([Paragraph(f"  {name}  ({score_val:.0f}/100)", s_item_str)])
    else:
        rows.append([Paragraph("  No metrics reached green threshold yet.", s_none)])

    # ── DETRACTORS header ─────────────────────────────────────────────────────
    rows.append([Paragraph("▼  DETRACTORS", s_hd_det)])
    if detractors:
        for k, v in detractors:
            name      = _INFLUENCE_NAMES.get(k, k)
            score_val = v.get("score", 0)
            rows.append([Paragraph(f"  {name}  ({score_val:.0f}/100)", s_item_det)])
    else:
        rows.append([Paragraph("  No significant deficits detected.", s_none)])

    # ── Score equation ────────────────────────────────────────────────────────
    eqn_parts = []
    if perf_score != score:
        eqn_parts.append(f"Performance {perf_score:.0f}/100")
        if red_factor < 1.0:
            eqn_parts.append(f"×{red_factor:.2f} red-flag penalty")
        if tf < 0.99:
            eqn_parts.append(f"×{tf:.2f} time modifier")
        eqn_parts.append(f"= {score:.0f}/100")
    if eqn_parts:
        rows.append([Paragraph("  →  ".join(eqn_parts), s_eqn)])

    tbl = Table(rows, colWidths=[box_w])

    # Style: alternate header rows with coloured backgrounds
    str_hdr_row  = 0
    det_hdr_row  = 1 + len(strengths) if strengths else 2

    style = [
        ("BACKGROUND",    (0, str_hdr_row), (-1, str_hdr_row), C_STR_H),
        ("BACKGROUND",    (0, det_hdr_row), (-1, det_hdr_row), C_DET_H),
        ("BACKGROUND",    (0, 0),           (-1, -1),           C_BOX),
        ("TOPPADDING",    (0, 0),           (-1, -1),           2),
        ("BOTTOMPADDING", (0, 0),           (-1, -1),           2),
        ("LEFTPADDING",   (0, 0),           (-1, -1),           0),
        ("RIGHTPADDING",  (0, 0),           (-1, -1),           0),
        ("VALIGN",        (0, 0),           (-1, -1),           "MIDDLE"),
        ("LINEABOVE",     (0, 0),           (-1, 0),            1.0, C_ORG),
        # Header rows override background
        ("BACKGROUND",    (0, str_hdr_row), (-1, str_hdr_row), C_STR_H),
        ("BACKGROUND",    (0, det_hdr_row), (-1, det_hdr_row), C_DET_H),
    ]
    tbl.setStyle(TableStyle(style))
    return [tbl]


# ─── Executive summary (one page) ─────────────────────────────────────────────

def _exec_summary(s: dict, patient_data: dict, lsi_summary: dict,
                  domain_scores: dict, clinical_notes: list,
                  rtr_result: dict) -> list:
    story = []

    months_post_op = patient_data.get("months_since_surgery")
    surgery_date   = patient_data.get("surgery_date", "")

    # Header
    story.append(_header(
        s,
        patient_data.get("name", "Unknown"),
        patient_data.get("dob", "—"),
        patient_data.get("surgical_side", "—"),
        patient_data.get("test_date", str(datetime.date.today())),
        patient_data.get("clinician", ""),
        surgery_date=surgery_date,
        months_post_op=months_post_op,
        triple_hop=patient_data.get("triple_hop"),
    ))
    story.append(Spacer(1, 5))

    # ── Score / badge / radar / S&D ──────────────────────────────────────────
    score       = rtr_result.get("score", 0)
    perf_score  = rtr_result.get("perf_score", score)
    grade       = rtr_result.get("grade", "—")
    red_count   = rtr_result.get("red_count", 0)
    tf          = rtr_result.get("time_factor", 1.0)
    time_note   = rtr_result.get("time_note")

    gc = (colors.HexColor("#fc6c0f") if grade == "Ready"       else
          colors.HexColor("#39414a") if grade == "Progressing" else
          colors.HexColor("#4a3030") if grade == "Caution"     else
          colors.HexColor("#3a2828"))

    sub_parts = []
    if red_count > 0:
        sub_parts.append(f"{red_count} red flag{'s' if red_count > 1 else ''}")
    if months_post_op is not None and tf < 0.99:
        sub_parts.append(f"{months_post_op:.0f} mo post-op")
    sub_line = " · ".join(sub_parts) if sub_parts else ""

    score_cell = Table(
        [[Paragraph(f"<b>{score:.0f}</b>",
                    ParagraphStyle("sc", fontSize=38, textColor=C_WHITE,
                                   fontName="Helvetica-Bold",
                                   alignment=TA_CENTER, leading=44))]],
        colWidths=[1.0 * inch])
    score_cell.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
    ]))

    # ── Column widths ─────────────────────────────────────────────────────────
    # Left column: badge + radar + S&D at original 3.15" width
    # Bar chart: starts 2" left of where right column begins (using negative
    #            LEFTPADDING), so its left edge sits at 3.15-2 = 1.15" from
    #            margin.  Figure is 2" wider to reach the right page edge.
    LEFT_W    = 3.15   # inches — badge, radar, S&D
    BAR_SHIFT = 0.75    # inches — bar chart moves this far left into left margin
    COL_R_W   = W_FULL / inch - LEFT_W   # ~4.35" — right column table allocation
    BAR_W     = COL_R_W + BAR_SHIFT      # ~6.35" — actual matplotlib figure width

    # Badge — fits within left column
    BADGE_W    = LEFT_W - 0.1   # 3.05"
    badge_text = (f"<b>RTS Score</b><br/>{grade}"
                  + (f"<br/><font size='7'>{sub_line}</font>" if sub_line else ""))
    badge = Table([[
        score_cell,
        Paragraph(badge_text,
                  ParagraphStyle("gr", fontSize=11, textColor=C_WHITE,
                                 fontName="Helvetica-Bold", leading=15,
                                 spaceAfter=0)),
    ]], colWidths=[1.05 * inch, (BADGE_W - 1.05) * inch])
    badge.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), gc),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
    ]))

    # Radar — full left column width, square aspect
    radar_img = Spacer(1, 1)
    if len(domain_scores) >= 3:
        radar_fig = fig_lib.rtr_radar(
            domain_scores, figsize=(LEFT_W, LEFT_W * 0.92))
        radar_img = _fig_img(radar_fig, width_in=LEFT_W)

    # S&D — full left column width
    influence_items = _score_influence_box(
        rtr_result, s, width=LEFT_W,
        tf=tf, perf_score=perf_score,
        months_post_op=months_post_op)

    # Assemble left column rows
    left_rows = [[badge], [Spacer(1, 6)], [radar_img], [Spacer(1, 4)]]
    for item in influence_items:
        left_rows.append([item])
    left_col = Table(left_rows, colWidths=[LEFT_W * inch])
    left_col.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
    ]))

    # LSI bar chart — figure is BAR_W wide; negative left padding shifts it
    # 2" left so its left edge sits at 1.15" from margin, right edge at 7.5"
    lsi_vals = {label: vals[2] for label, vals in lsi_summary.items()
                if vals[2] is not None and vals[2] == vals[2]}
    n      = len(lsi_vals)
    BAR_H  = max(4.0, min(n * 0.38, 7.0))   # inches
    right_cell = Spacer(1, 1)
    if lsi_vals:
        bar_fig = fig_lib.lsi_bar_chart(
            lsi_vals,
            title="Limb Symmetry Index — All Tests",
            figsize=(BAR_W, BAR_H))
        right_cell = _fig_img(bar_fig, width_in=BAR_W)

    # ── Z-order: bar chart in col 0 (drawn first = BEHIND),
    #            left content in col 1 (drawn second = IN FRONT).
    #
    # col 0 width = COL_R_W (4.35").  Bar chart (BAR_W = 6.35") sits in col 0
    #   with LEFTPADDING = BAR_SHIFT - BAR_SHIFT = ... let's work it out:
    #   we want the bar chart left edge at page x = LEFT_W - BAR_SHIFT = 1.15"
    #   col 0 starts at page x = 0, so LEFTPADDING on col 0 = 1.15" = (LEFT_W - BAR_SHIFT).
    # col 1 width = LEFT_W (3.15").  Left content (also 3.15") sits in col 1
    #   but col 1 starts at page x = COL_R_W = 4.35", so we need
    #   LEFTPADDING = -COL_R_W to shift it back to page x = 0.
    two_col = Table([[right_cell, left_col]],
                    colWidths=[COL_R_W * inch, LEFT_W * inch])
    two_col.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (0, -1),  (LEFT_W - BAR_SHIFT + 0.5) * inch),
        ("RIGHTPADDING",  (0, 0), (0, -1),  0),
        ("LEFTPADDING",   (1, 0), (1, -1),  (-COL_R_W - 0.18) * inch),
        ("RIGHTPADDING",  (1, 0), (1, -1),  0),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(two_col)

    # Spacer to push clinical flags below the bar chart
    CLINICAL_EST = 1.2 * inch
    FOOTER_H     = 0.35 * inch
    HEADER_EST   = 0.95 * inch
    FILL_H = (H_PAGE
              - HEADER_EST
              - 5
              - BAR_H * inch
              - CLINICAL_EST
              - FOOTER_H
              - 0.5 * inch)
    if FILL_H > 0:
        story.append(Spacer(1, FILL_H))

    # ── Clinical flags ────────────────────────────────────────────────────────
    story += _sec("CLINICAL FLAGS", s)
    if clinical_notes:
        for note in clinical_notes[:6]:
            story.append(Paragraph(f"• {note}", s["body"]))
    else:
        story.append(Paragraph(
            "No significant asymmetries flagged. Review individual test pages.",
            s["body"]))
    story.append(Spacer(1, 3))

    # Footer
    story.append(_legend(s))
    story.append(Spacer(1, 3))
    story.append(HRFlowable(width="100%", thickness=0.4, color=C_GRID))
    story.append(Paragraph(
        f"Generated {datetime.datetime.now().strftime('%B %d, %Y %H:%M')}  |  "
        f"{CLINIC_NAME}  |  Confidential — for clinical use only.",
        s["small"]))

    return story


# ─── Build report ─────────────────────────────────────────────────────────────

def build_report(output_path: str, patient_data: dict,
                 test_results: dict, all_signals: dict = None,
                 rtr_metrics: dict = None) -> str:
    if all_signals is None:
        all_signals = {}

    s   = _styles()
    doc = SimpleDocTemplate(
        output_path, pagesize=PAGE,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
    )

    bw_n     = patient_data.get("bw_kg", 70) * 9.81
    rate_grf = patient_data.get("rate_grf", 1000)
    rate_kin = patient_data.get("rate_kin", 200)

    lsi_summary   = {}
    domain_scores = {}
    clinical_notes= []
    _rtr          = {}

    def _reg(label, sv, nv, lv, domain=None, rtr_key=None):
        lsi_summary[label] = (sv, nv, lv)
        if domain and lv is not None and lv == lv:
            domain_scores.setdefault(domain, []).append(lv)
        if rtr_key and lv is not None and lv == lv:
            _rtr[rtr_key] = lv
        if lv is not None and lv == lv and lv < LSI_YELLOW:
            clinical_notes.append(
                f"{label}: LSI {lv:.1f}% — significant asymmetry")

    def _reg_delta(label, sv, nv, warn_deg: float, err_deg: float):
        """
        Register a metric compared by absolute side-to-side delta (degrees).
        Stores None as LSI so the table renders in delta mode.
        Flags clinically only when the absolute difference exceeds err_deg.
        Does NOT contribute to domain_scores — LSI-based radar scoring only.
        """
        lsi_summary[label] = (sv, nv, None)
        try:
            delta = float(sv) - float(nv)   # signed: + means surgical > non-surg
            if abs(delta) > err_deg:
                sign = "+" if delta >= 0 else ""
                clinical_notes.append(
                    f"{label}: Δ {sign}{delta:.1f}° — exceeds {err_deg:.0f}° threshold")
        except (TypeError, ValueError):
            pass

    # ── helper: AP COP excursion (fore-aft range) during contact phase ──────────
    def _ap_excursion(cop_y, fz, thresh=30.0):
        """
        Returns fore-aft COP range (mm) during contact frames.
        V3D COP is in global lab coords — raw absolute values are meaningless.
        Excursion (max-min) is coordinate-system independent and clinically useful:
        smaller excursion on surgical side = restricted / guarded loading.
        """
        import numpy as _np
        if cop_y is None:
            return None
        y = cop_y.astype(float).copy()
        if fz is not None and len(fz) == len(y):
            contact = fz >= thresh
        else:
            contact = _np.abs(y) > 1e-9
        yc = y[contact]
        if len(yc) < 5:
            return None
        ex = float(_np.nanmax(yc) - _np.nanmin(yc))
        return round(ex, 1) if not _np.isnan(ex) else None

    # ── helper: build a compact one-page test section ─────────────────────────
    def _test_page(title: str, rows: list, interp_lines: list,
                   fig_full=None,  cap_full="",  sum_full="",
                   fig_l=None,     cap_l="",     sum_l="",
                   fig_r=None,     cap_r="",     sum_r="",
                   fig_l2=None,    cap_l2="",    sum_l2="",
                   fig_r2=None,    cap_r2="",    sum_r2="",
                   fig_cop=None,   cap_cop="",   sum_cop="",
                   description: str = "",
                   summary: str = "") -> list:
        """
        Lay out one test section:
          title bar → metrics table → interpretation box
          optional GRF full-width figure
          optional pair of kinematic figures (row 1)
          optional pair of kinematic figures (row 2)
          optional full-width COP AP trace (fig_cop)
          optional movement description box at bottom
        Always ends with PageBreak.
        """
        pg = []
        pg += _sec(title, s)
        pg.append(_metrics_table(rows, s))
        pg.append(Spacer(1, 3))
        pg.append(_interp_box(interp_lines, s))
        pg.append(Spacer(1, 3))

        # ── inline one-liner style ────────────────────────────────────────────
        _oneliner_s = ParagraphStyle(
            "_ol", fontSize=7.5, fontName="Helvetica-Oblique",
            textColor=colors.HexColor("#3A4A5A"),
            leading=10, leftIndent=0, rightIndent=0)

        def _oneliner(text: str):
            """Render a small italic clinical note below a figure."""
            if not text:
                return []
            tbl = Table([[Paragraph(text, _oneliner_s)]], colWidths=[W_FULL])
            tbl.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (-1,-1), colors.HexColor("#EDF1F6")),
                ("TOPPADDING",    (0,0), (-1,-1), 3),
                ("BOTTOMPADDING", (0,0), (-1,-1), 3),
                ("LEFTPADDING",   (0,0), (-1,-1), 6),
                ("RIGHTPADDING",  (0,0), (-1,-1), 6),
                ("LINEABOVE",     (0,0), (-1,0),  0.5, colors.HexColor("#A8BDD0")),
            ]))
            return [tbl, Spacer(1, 4)]

        if fig_full is not None:
            pg += _fig_full(fig_full, cap_full, s)
            pg += _oneliner(sum_full)

        if fig_l is not None and fig_r is not None:
            pg += _fig_pair(fig_l, cap_l, fig_r, cap_r, s)
            if sum_l or sum_r:
                pg += _oneliner(f"<b>Left panel:</b> {sum_l}" if sum_l else "")
                if sum_r:
                    pg += _oneliner(f"<b>Right panel:</b> {sum_r}")
        elif fig_l is not None:
            img = _fig_img(fig_l, width_in=W_HALF / inch)
            cap = Paragraph(cap_l, s["cap"])
            pg += [img, cap, Spacer(1, 4)]
            pg += _oneliner(sum_l)
        elif fig_r is not None:
            img = _fig_img(fig_r, width_in=W_HALF / inch)
            cap = Paragraph(cap_r, s["cap"])
            pg += [img, cap, Spacer(1, 4)]
            pg += _oneliner(sum_r)

        if fig_l2 is not None and fig_r2 is not None:
            pg += _fig_pair(fig_l2, cap_l2, fig_r2, cap_r2, s)
            if sum_l2 or sum_r2:
                pg += _oneliner(f"<b>Left panel:</b> {sum_l2}" if sum_l2 else "")
                if sum_r2:
                    pg += _oneliner(f"<b>Right panel:</b> {sum_r2}")
        elif fig_l2 is not None:
            img = _fig_img(fig_l2, width_in=W_HALF / inch)
            cap = Paragraph(cap_l2, s["cap"])
            pg += [img, cap, Spacer(1, 4)]
            pg += _oneliner(sum_l2)

        if fig_cop is not None:
            pg += _fig_full(fig_cop, cap_cop, s)
            pg += _oneliner(sum_cop)

        # ── Page-level clinical synthesis ─────────────────────────────────────
        if summary:
            s_sum = ParagraphStyle(
                "fig_summary", fontSize=7.5, fontName="Helvetica-Oblique",
                textColor=colors.HexColor("#444444"),
                leading=10, leftIndent=6, rightIndent=6,
                borderPad=4)
            sum_tbl = Table([[Paragraph(summary, s_sum)]], colWidths=[W_FULL])
            sum_tbl.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#EFF3F7")),
                ("TOPPADDING",    (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING",   (0, 0), (-1, -1), 8),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
                ("LINEABOVE",     (0, 0), (-1, 0),  0.75, colors.HexColor("#B0BFCC")),
            ]))
            pg.append(sum_tbl)
            pg.append(Spacer(1, 3))

        # ── Movement description box ──────────────────────────────────────────
        if description:
            pg.append(Spacer(1, 4))
            s_desc_lbl = ParagraphStyle(
                "desc_lbl", fontSize=7, fontName="Helvetica-Bold",
                textColor=colors.HexColor(COLORS["header_fg"]),
                leading=9, leftIndent=4)
            s_desc_txt = ParagraphStyle(
                "desc_txt", fontSize=7, fontName="Helvetica",
                textColor=colors.HexColor(COLORS["text"]),
                leading=10, leftIndent=4, rightIndent=4,
                spaceAfter=2)
            desc_rows = [
                [Paragraph("ABOUT THIS TEST", s_desc_lbl)],
                [Paragraph(description, s_desc_txt)],
            ]
            desc_tbl = Table(desc_rows, colWidths=[W_FULL])
            desc_tbl.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor(COLORS["header_bg"])),
                ("BACKGROUND",    (0, 1), (-1, -1), colors.HexColor(COLORS["lt_grey"])),
                ("TOPPADDING",    (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING",   (0, 0), (-1, -1), 0),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
                ("LINEABOVE",     (0, 0), (-1, 0),  1.0, colors.HexColor(COLORS["accent"])),
            ]))
            pg.append(desc_tbl)

        pg.append(PageBreak())
        return pg

    test_story = []

    # ── DROP JUMP ─────────────────────────────────────────────────────────────
    dj = test_results.get("drop_jump")
    if dj:
        sigs = all_signals.get("drop_jump", {})
        rows = []

        # ── Loading ───────────────────────────────────────────────────────────
        if dj.grf_overall:
            o = dj.grf_overall
            _reg("DJ Peak GRF",   o.surgical.peak_force_N,   o.non_surgical.peak_force_N,   o.lsi_peak,    "Force", "peak_grf_lsi")
            _reg("DJ RFD 100ms",  o.surgical.rfd_Ns,         o.non_surgical.rfd_Ns,         o.lsi_rfd,     "Force", "rfd_lsi")
            rows += [
                ("Peak GRF (N)",       o.surgical.peak_force_N,  o.non_surgical.peak_force_N,  o.lsi_peak),
                ("Impulse (N·s)",      o.surgical.impulse_Ns,    o.non_surgical.impulse_Ns,    o.lsi_impulse),
                ("RFD 100ms (N/s)",    o.surgical.rfd_Ns,        o.non_surgical.rfd_Ns,        o.lsi_rfd),
            ]
        _reg("DJ Load Rate", dj.loading_rate_surg_ns, dj.loading_rate_ns_ns, dj.loading_rate_lsi, "Force")
        _reg("DJ Land Sym",  None, None, dj.landing_lsi_200ms, "Force", "landing_lsi")
        rows += [
            ("Loading Rate 50ms (N/s)", dj.loading_rate_surg_ns,  dj.loading_rate_ns_ns,  dj.loading_rate_lsi),
            ("Landing Sym 200ms (%)",   "—",                      "—",                    dj.landing_lsi_200ms),
            ("Impact Transient (N)",    dj.impact_transient_surg, dj.impact_transient_ns, dj.impact_transient_lsi),
            ("Time-to-Peak (ms)",       dj.time_to_peak_surg_ms,  dj.time_to_peak_ns_ms,  None),
            ("RSI",                     _fmt(dj.rsi, 3),          "—",                    None),
            ("Contact Time (s)",        _fmt(dj.contact_time_s, 3), "—",                  None),
        ]
        if dj.rsi == dj.rsi:
            _rtr["drop_jump_rsi"] = dj.rsi

        # ── Kinematics ────────────────────────────────────────────────────────
        if dj.kinematics:
            k = dj.kinematics
            _reg("DJ Knee Flexion", k.surg_knee.peak_flexion_deg, k.non_surg_knee.peak_flexion_deg, k.lsi_peak_flexion, "Kinematics")
            _reg_delta("DJ Knee Valgus", k.surg_knee.peak_valgus_deg, k.non_surg_knee.peak_valgus_deg, warn_deg=3.0, err_deg=6.0)
            rows += [
                ("Peak Knee Flexion (°)",    k.surg_knee.peak_flexion_deg,   k.non_surg_knee.peak_flexion_deg,   k.lsi_peak_flexion),
                _delta_row("Peak Knee Valgus (°)",   k.surg_knee.peak_valgus_deg,    k.non_surg_knee.peak_valgus_deg,    warn_deg=3.0, err_deg=6.0),
                _delta_row("Peak Tibial IR (°)",     k.surg_knee.peak_tibial_ir_deg, k.non_surg_knee.peak_tibial_ir_deg, warn_deg=5.0, err_deg=8.0),
                _delta_row("Tibial Rot Excursion (°)", k.surg_knee.tibial_rot_excursion, k.non_surg_knee.tibial_rot_excursion, warn_deg=5.0, err_deg=8.0),
            ]
            if k.surg_hip and k.non_surg_hip:
                sh, nh = k.surg_hip, k.non_surg_hip
                rows += [
                    _delta_row("Peak Hip Flexion (°)",   sh.peak_flexion_deg,   nh.peak_flexion_deg,   warn_deg=5.0, err_deg=10.0),
                    _delta_row("Peak Hip Adduction (°)", sh.peak_adduction_deg, nh.peak_adduction_deg, warn_deg=3.0, err_deg=6.0),
                    _delta_row("Peak Hip IR (°)",        sh.peak_ir_deg,        nh.peak_ir_deg,        warn_deg=5.0, err_deg=8.0),
                ]
        if sigs.get("ankle_flex_surg") is not None:
            ank_s = float(np.nanmax(sigs["ankle_flex_surg"])) if len(sigs["ankle_flex_surg"]) > 0 else None
            ank_n = float(np.nanmax(sigs["ankle_flex_ns"])) if sigs.get("ankle_flex_ns") is not None and len(sigs["ankle_flex_ns"]) > 0 else None
            _reg_delta("DJ Ankle Dorsiflexion", ank_s, ank_n, warn_deg=3.0, err_deg=6.0)
            rows += [_delta_row("Peak Ankle Dorsiflexion (°)", ank_s, ank_n, warn_deg=3.0, err_deg=6.0)]

        # ── Kinetics ──────────────────────────────────────────────────────────
        km_s = sigs.get("knee_moment_surg")
        km_n = sigs.get("knee_moment_ns")
        if not _isnan(km_s) or not _isnan(km_n):
            km_lsi = round(km_s / km_n * 100, 1) if (not _isnan(km_s) and not _isnan(km_n) and km_n > 0) else None
            _reg("DJ Knee Ext Moment", km_s, km_n, km_lsi, "Force")
            rows += [("Peak Knee Ext Moment (Nm)", _fmt(km_s, 1), _fmt(km_n, 1), km_lsi)]

        # ── COP ───────────────────────────────────────────────────────────────
        ap_s = _ap_excursion(sigs.get("cop_y_surg"), sigs.get("fz_surg"))
        ap_n = _ap_excursion(sigs.get("cop_y_ns"),   sigs.get("fz_ns"))
        if ap_s is not None or ap_n is not None:
            rows += [("AP COP Excursion (mm)", ap_s, ap_n, None)]

        # ── Interpretation ────────────────────────────────────────────────────
        rsi = dj.rsi if dj.rsi == dj.rsi else None
        ll  = dj.landing_lsi_200ms if dj.landing_lsi_200ms == dj.landing_lsi_200ms else None
        interp = []
        if rsi:
            interp.append(
                f"<b>RSI {rsi:.2f}</b> ({'✓ meets' if rsi >= 1.3 else '⚠ below'} RTS target ≥ 1.30). "
                + ("Reactive capacity is adequate." if rsi >= 1.3
                   else "Stiff-landing pattern likely — progressive plyometric loading indicated."))
        if ll:
            interp.append(
                f"<b>Landing symmetry {ll:.0f}% LSI</b>: "
                + ("Good bilateral loading at contact." if 90 <= ll <= 110
                   else f"Surgical limb loading {abs(ll-100):.0f}% {'more' if ll>100 else 'less'} than non-surgical — {'protective offloading of NS side detected' if ll>100 else 'surgical guarding detected'}."))
        interp.append("<b>Targets:</b> RSI ≥ 1.30 · Landing LSI 90–110% · Load rate LSI ≥ 90% · Knee flexion ≥ 60° · Valgus < 5° · Tibial IR < 10°.")

        # ── Figures: GRF → sagittal pair → frontal/transverse pair → COP ─────
        grf_fig   = fig_lib.force_time_curve(sigs.get("fz_surg"), sigs.get("fz_ns"), rate_grf,
                       "Drop Jump — Bilateral Vertical GRF", bw_n, figsize=(7.5, 2.2)) \
                    if sigs.get("fz_surg") is not None else None
        flex_fig  = fig_lib.joint_angle_overlay(sigs.get("knee_flex_surg"), sigs.get("knee_flex_ns"),
                       rate_kin, "Knee Flexion", "Flexion (°)", figsize=(3.5, 2.2),
                       dir_low="← More Extended", dir_high="More Flexed →") \
                    if sigs.get("knee_flex_surg") is not None else None
        ank_fig   = fig_lib.joint_angle_overlay(sigs.get("ankle_flex_surg"), sigs.get("ankle_flex_ns"),
                       rate_kin, "Ankle Dorsiflexion", "Dorsiflexion (°)", figsize=(3.5, 2.2),
                       dir_low="← More Plantarflexed", dir_high="More Dorsiflexed →") \
                    if sigs.get("ankle_flex_surg") is not None else None
        valg_fig  = fig_lib.joint_angle_overlay(sigs.get("knee_valgus_surg"), sigs.get("knee_valgus_ns"),
                       rate_kin, "Knee Valgus", "Valgus (°)", figsize=(3.5, 2.2),
                       dir_low="← More Varus", dir_high="More Valgus →") \
                    if sigs.get("knee_valgus_surg") is not None else None
        tibr_fig  = fig_lib.joint_angle_overlay(sigs.get("tib_rot_surg"), sigs.get("tib_rot_ns"),
                       rate_kin, "Tibial Internal Rotation", "Int Rotation (°)", figsize=(3.5, 2.2),
                       dir_low="← More External", dir_high="More Internal →") \
                    if sigs.get("tib_rot_surg") is not None else None
        cop_fig   = fig_lib.cop_bilateral_ap_trace(
                       sigs.get("cop_y_surg"), sigs.get("cop_y_ns"),
                       fz_surg=sigs.get("fz_surg"), fz_ns=sigs.get("fz_ns"),
                       rate=rate_grf, title="Drop Jump — COP Anterior-Posterior Trace",
                       figsize=(7.5, 1.8)) \
                    if sigs.get("cop_y_surg") is not None else None

        # ── Per-graph one-liner summaries ─────────────────────────────────────
        _k_dj = dj.kinematics  # may be None
        _dj_grf_sum  = (
            f"Landing symmetry {dj.landing_lsi_200ms:.0f}% at 200ms — "
            + ("symmetric bilateral loading." if 90 <= dj.landing_lsi_200ms <= 110
               else f"surgical limb {'overloading' if dj.landing_lsi_200ms > 110 else 'offloading'} detected.")
        ) if dj.landing_lsi_200ms == dj.landing_lsi_200ms else ""

        _dj_flex_sum = (
            _lsi_phrase(_k_dj.lsi_peak_flexion, "knee flexion depth")
        ) if _k_dj else ""

        _dj_valg_sum = (
            _delta_phrase(_k_dj.surg_knee.peak_valgus_deg, _k_dj.non_surg_knee.peak_valgus_deg,
                          "knee valgus", "°", warn=3.0, err=6.0)
        ) if _k_dj else ""

        _dj_tib_sum = (
            _delta_phrase(_k_dj.surg_knee.peak_tibial_ir_deg, _k_dj.non_surg_knee.peak_tibial_ir_deg,
                          "tibial IR", "°", warn=5.0, err=8.0)
        ) if _k_dj else ""

        _dj_cop_sum = (
            f"AP COP excursion: surgical {ap_s:.0f}mm vs non-surgical {ap_n:.0f}mm — "
            + ("symmetric forward weight shift." if ap_s is not None and ap_n is not None and abs(ap_s - ap_n) < 5
               else "reduced anterior shift on surgical side — guarded or restricted loading strategy.")
        ) if (ap_s is not None and ap_n is not None) else ""

        _dj_sum = _lsi_phrase(dj.landing_lsi_200ms, "bilateral landing symmetry")
        test_story += _test_page("DROP JUMP", rows, interp,
            fig_full=grf_fig,  cap_full="Bilateral GRF (normalised to BW). Orange = surgical, Blue = non-surgical.",
            sum_full=_dj_grf_sum,
            fig_l=flex_fig,    cap_l="Knee flexion. Target ≥ 60° at contact. Reduced depth = stiff landing strategy.",
            sum_l=_dj_flex_sum,
            fig_r=ank_fig,     cap_r="Ankle dorsiflexion. Restricted dorsiflexion → shifts load to quadriceps / increases ACL stress.",
            fig_l2=valg_fig,   cap_l2="Knee valgus. Target < 5°. Elevated valgus = medial collapse / ACL re-injury risk.",
            sum_l2=_dj_valg_sum,
            fig_r2=tibr_fig,   cap_r2="Tibial internal rotation. IR > 10° at landing = increased ACL rotational load.",
            sum_r2=_dj_tib_sum,
            fig_cop=cop_fig,   cap_cop="AP COP displacement (zero-centred). Smaller excursion on surgical limb = restricted fore-aft weight transfer / guarded landing strategy.",
            sum_cop=_dj_cop_sum,
            summary=_dj_sum,
            description=(
                "The Drop Jump is a reactive bilateral landing task performed from a standard box height. "
                "The athlete drops off the box, absorbs the impact with both limbs simultaneously, then immediately "
                "jumps for maximum height. It tests the athlete's ability to rapidly store and release elastic energy "
                "(reactive strength) and to absorb high-impact loads symmetrically. Key concerns after ACL reconstruction "
                "include stiff-knee landing (reduced flexion, elevated impact transient), medial knee collapse (valgus), "
                "and side-to-side force asymmetries that indicate ongoing limb offloading. The Reactive Strength Index "
                "(RSI = jump height ÷ contact time) summarises the explosive reactive capacity of the neuromuscular system."
            ))

    # ── DROP LANDING ──────────────────────────────────────────────────────────
    dl = test_results.get("drop_landing")
    if dl:
        sigs = all_signals.get("drop_landing", {})
        rows = []

        # ── Loading ───────────────────────────────────────────────────────────
        if dl.grf_overall:
            o = dl.grf_overall
            _reg("DL Peak GRF", o.surgical.peak_force_N, o.non_surgical.peak_force_N, o.lsi_peak, "Force")
            rows += [
                ("Peak GRF (N)",       o.surgical.peak_force_N,  o.non_surgical.peak_force_N,  o.lsi_peak),
                ("Impulse (N·s)",      o.surgical.impulse_Ns,    o.non_surgical.impulse_Ns,    o.lsi_impulse),
            ]
        _reg("DL Load Rate", dl.loading_rate_surg_ns, dl.loading_rate_ns_ns, dl.loading_rate_lsi, "Force")
        _reg("DL Impact",    dl.impact_transient_surg, dl.impact_transient_ns, dl.impact_transient_lsi, "Force")
        rows += [
            ("Loading Rate 50ms (N/s)", dl.loading_rate_surg_ns,  dl.loading_rate_ns_ns,  dl.loading_rate_lsi),
            ("Impact Transient (N)",    dl.impact_transient_surg, dl.impact_transient_ns, dl.impact_transient_lsi),
        ]

        # ── Kinematics ────────────────────────────────────────────────────────
        if dl.kinematics:
            k = dl.kinematics
            _reg("DL Knee Flexion", k.surg_knee.peak_flexion_deg, k.non_surg_knee.peak_flexion_deg, k.lsi_peak_flexion, "Kinematics")
            _reg_delta("DL Knee Valgus", k.surg_knee.peak_valgus_deg, k.non_surg_knee.peak_valgus_deg, warn_deg=3.0, err_deg=6.0)
            rows += [
                ("Peak Knee Flexion (°)",    k.surg_knee.peak_flexion_deg,   k.non_surg_knee.peak_flexion_deg,   k.lsi_peak_flexion),
                _delta_row("Peak Knee Valgus (°)",     k.surg_knee.peak_valgus_deg,    k.non_surg_knee.peak_valgus_deg,    warn_deg=3.0, err_deg=6.0),
                _delta_row("Peak Tibial IR (°)",       k.surg_knee.peak_tibial_ir_deg, k.non_surg_knee.peak_tibial_ir_deg, warn_deg=5.0, err_deg=8.0),
                _delta_row("Tibial Rot Excursion (°)", k.surg_knee.tibial_rot_excursion, k.non_surg_knee.tibial_rot_excursion, warn_deg=5.0, err_deg=8.0),
            ]
            if k.surg_hip and k.non_surg_hip:
                sh, nh = k.surg_hip, k.non_surg_hip
                rows += [
                    _delta_row("Peak Hip Flexion (°)",   sh.peak_flexion_deg,   nh.peak_flexion_deg,   warn_deg=5.0, err_deg=10.0),
                    _delta_row("Peak Hip Adduction (°)", sh.peak_adduction_deg, nh.peak_adduction_deg, warn_deg=3.0, err_deg=6.0),
                    _delta_row("Peak Hip IR (°)",        sh.peak_ir_deg,        nh.peak_ir_deg,        warn_deg=5.0, err_deg=8.0),
                ]
        if sigs.get("ankle_flex_surg") is not None:
            ank_s = float(np.nanmax(sigs["ankle_flex_surg"])) if len(sigs["ankle_flex_surg"]) > 0 else None
            ank_n = float(np.nanmax(sigs["ankle_flex_ns"])) if sigs.get("ankle_flex_ns") is not None and len(sigs["ankle_flex_ns"]) > 0 else None
            rows += [_delta_row("Peak Ankle Dorsiflexion (°)", ank_s, ank_n, warn_deg=3.0, err_deg=6.0)]

        # ── Kinetics ──────────────────────────────────────────────────────────
        km_s = sigs.get("knee_moment_surg")
        km_n = sigs.get("knee_moment_ns")
        if not _isnan(km_s) or not _isnan(km_n):
            km_lsi = round(km_s / km_n * 100, 1) if (not _isnan(km_s) and not _isnan(km_n) and km_n > 0) else None
            _reg("DL Knee Ext Moment", km_s, km_n, km_lsi, "Force")
            rows += [("Peak Knee Ext Moment (Nm)", _fmt(km_s, 1), _fmt(km_n, 1), km_lsi)]

        # ── COP ───────────────────────────────────────────────────────────────
        ap_s = _ap_excursion(sigs.get("cop_y_surg"), sigs.get("fz_surg"))
        ap_n = _ap_excursion(sigs.get("cop_y_ns"),   sigs.get("fz_ns"))
        if ap_s is not None or ap_n is not None:
            rows += [("AP COP Excursion (mm)", ap_s, ap_n, None)]

        # ── Interpretation ────────────────────────────────────────────────────
        lr  = dl.loading_rate_lsi  if dl.loading_rate_lsi  == dl.loading_rate_lsi  else None
        imp = dl.impact_transient_lsi if dl.impact_transient_lsi == dl.impact_transient_lsi else None
        interp = ["Absorptive landing — assesses shock attenuation without a re-jump. Lower, more gradual force = better control."]
        if lr:
            interp.append(
                f"<b>Load rate LSI {lr:.0f}%</b>: "
                + ("Symmetric loading strategy." if 90 <= lr <= 110
                   else f"Surgical limb loading rate {abs(lr-100):.0f}% {'higher' if lr>100 else 'lower'} — {'NS side protective offloading' if lr>100 else 'surgical inhibition'}."))
        if imp:
            interp.append(f"<b>Impact transient LSI {imp:.0f}%</b>. Target: < 2× BW peak, LSI 90–110%, no sharp spike.")
        interp.append("<b>Targets:</b> Peak GRF < 3× BW · Load rate LSI 90–110% · Knee flexion ≥ 60° · Valgus < 5° · Tibial IR < 10°.")

        # ── Figures ───────────────────────────────────────────────────────────
        grf_fig  = fig_lib.force_time_curve(sigs.get("fz_surg"), sigs.get("fz_ns"), rate_grf,
                      "Drop Landing — Bilateral GRF", bw_n, figsize=(7.5, 2.2)) \
                   if sigs.get("fz_surg") is not None else None
        flex_fig = fig_lib.joint_angle_overlay(sigs.get("knee_flex_surg"), sigs.get("knee_flex_ns"),
                      rate_kin, "Knee Flexion", "Flexion (°)", figsize=(3.5, 2.2),
                      dir_low="← More Extended", dir_high="More Flexed →") \
                   if sigs.get("knee_flex_surg") is not None else None
        ank_fig  = fig_lib.joint_angle_overlay(sigs.get("ankle_flex_surg"), sigs.get("ankle_flex_ns"),
                      rate_kin, "Ankle Dorsiflexion", "Dorsiflexion (°)", figsize=(3.5, 2.2),
                      dir_low="← More Plantarflexed", dir_high="More Dorsiflexed →") \
                   if sigs.get("ankle_flex_surg") is not None else None
        valg_fig = fig_lib.joint_angle_overlay(sigs.get("knee_valgus_surg"), sigs.get("knee_valgus_ns"),
                      rate_kin, "Knee Valgus", "Valgus (°)", figsize=(3.5, 2.2),
                      dir_low="← More Varus", dir_high="More Valgus →") \
                   if sigs.get("knee_valgus_surg") is not None else None
        tibr_fig = fig_lib.joint_angle_overlay(sigs.get("tib_rot_surg"), sigs.get("tib_rot_ns"),
                      rate_kin, "Tibial Internal Rotation", "Int Rotation (°)", figsize=(3.5, 2.2),
                      dir_low="← More External", dir_high="More Internal →") \
                   if sigs.get("tib_rot_surg") is not None else None
        cop_fig  = fig_lib.cop_bilateral_ap_trace(
                      sigs.get("cop_y_surg"), sigs.get("cop_y_ns"),
                      fz_surg=sigs.get("fz_surg"), fz_ns=sigs.get("fz_ns"),
                      rate=rate_grf, title="Drop Landing — COP Anterior-Posterior Trace",
                      figsize=(7.5, 1.8)) \
                   if sigs.get("cop_y_surg") is not None else None

        _k_dl = dl.kinematics
        _dl_grf_sum = (
            f"Peak GRF symmetry {dl.peak_force_lsi:.0f}% — "
            + _lsi_phrase(dl.peak_force_lsi, "shock absorption").split(" — ", 1)[-1]
        ) if dl.peak_force_lsi == dl.peak_force_lsi else ""

        _dl_flex_sum = (
            _lsi_phrase(_k_dl.lsi_peak_flexion, "landing knee flexion depth")
        ) if _k_dl else ""

        _dl_valg_sum = (
            _delta_phrase(_k_dl.surg_knee.peak_valgus_deg, _k_dl.non_surg_knee.peak_valgus_deg,
                          "knee valgus", "°", warn=3.0, err=6.0)
        ) if _k_dl else ""

        _dl_tib_sum = (
            _delta_phrase(_k_dl.surg_knee.peak_tibial_ir_deg, _k_dl.non_surg_knee.peak_tibial_ir_deg,
                          "tibial IR", "°", warn=5.0, err=8.0)
        ) if _k_dl else ""

        _dl_sum = _lsi_phrase(dl.peak_force_lsi, "peak landing force absorption")
        test_story += _test_page("DROP LANDING", rows, interp,
            fig_full=grf_fig,  cap_full="Bilateral GRF. Smooth progressive rise = good shock absorption. Sharp impact spike = stiff landing.",
            sum_full=_dl_grf_sum,
            fig_l=flex_fig,    cap_l="Knee flexion at landing. Reduced depth = stiff strategy / quad inhibition.",
            sum_l=_dl_flex_sum,
            fig_r=ank_fig,     cap_r="Ankle dorsiflexion. Restricted ROM limits knee flexion depth and increases ACL load.",
            fig_l2=valg_fig,   cap_l2="Knee valgus at landing. > 8° surgical = elevated ACL stress.",
            sum_l2=_dl_valg_sum,
            fig_r2=tibr_fig,   cap_r2="Tibial IR at landing. Combined with valgus = high ACL injury-risk pattern.",
            sum_r2=_dl_tib_sum,
            fig_cop=cop_fig,   cap_cop="AP COP (zero-centred). Posterior bias on surgical side = heel-dominant / reduced forefoot loading.",
            summary=_dl_sum,
            description=(
                "The Drop Landing isolates the deceleration phase of a jump: the athlete drops from a box and sticks "
                "the landing bilaterally without a subsequent jump. This removes the propulsive demand and focuses entirely "
                "on the athlete's ability to safely absorb and dissipate ground reaction forces. After ACL reconstruction, "
                "common compensations include a stiff-knee strategy (reduced flexion, sharp impact spike), excessive valgus "
                "collapse, and asymmetric weight acceptance—all of which increase mechanical stress at the reconstructed "
                "ligament. A smooth, progressive force curve with symmetric knee flexion ≥ 60° and valgus < 8° is the "
                "clinical target. Loading rate and impact transient LSI reflect side-to-side shock absorption capacity."
            ))

    # ── MAX VERTICAL JUMP ─────────────────────────────────────────────────────
    mvj = test_results.get("max_vertical")
    if mvj:
        sigs = all_signals.get("max_vertical", {})
        rows = []

        # ── Loading ───────────────────────────────────────────────────────────
        if mvj.grf_concentric:
            c = mvj.grf_concentric
            _reg("CMJ Concentric", c.surgical.peak_force_N, c.non_surgical.peak_force_N, c.lsi_peak,    "Force")
            _reg("CMJ Impulse",    c.surgical.impulse_Ns,   c.non_surgical.impulse_Ns,   c.lsi_impulse, "Force")
            rows += [
                ("Peak Concentric GRF (N)",  c.surgical.peak_force_N, c.non_surgical.peak_force_N, c.lsi_peak),
                ("Propulsion Impulse (N·s)", c.surgical.impulse_Ns,   c.non_surgical.impulse_Ns,   c.lsi_impulse),
            ]
        _reg("CMJ Peak Force", mvj.peak_force_surg_N, mvj.peak_force_ns_N, mvj.peak_force_lsi, "Force", "peak_grf_lsi")
        rows += [
            ("Peak Force Surg / NS (N)",   mvj.peak_force_surg_N, mvj.peak_force_ns_N, mvj.peak_force_lsi),
            ("Propulsion LSI (%)",         "—",                   "—",                 mvj.propulsion_lsi),
            ("Jump Height",                _fmt(mvj.jump_height_cm, 1), "—",           None),
            ("Flight Time (s)",            _fmt(mvj.flight_time_s, 3),  "—",           None),
            ("Unweighting Impulse Surg/NS",_fmt(mvj.unweighting_impulse_surg, 1),
                                           _fmt(mvj.unweighting_impulse_ns, 1), None),
        ]

        # ── Kinematics ────────────────────────────────────────────────────────
        if mvj.kinematics:
            k = mvj.kinematics
            _reg("CMJ Knee Flexion", k.surg_knee.peak_flexion_deg, k.non_surg_knee.peak_flexion_deg, k.lsi_peak_flexion, "Kinematics")
            _reg_delta("CMJ Knee Valgus", k.surg_knee.peak_valgus_deg, k.non_surg_knee.peak_valgus_deg, warn_deg=3.0, err_deg=6.0)
            rows += [
                ("Peak Knee Flexion (°)",    k.surg_knee.peak_flexion_deg,   k.non_surg_knee.peak_flexion_deg,   k.lsi_peak_flexion),
                _delta_row("Peak Knee Valgus (°)",     k.surg_knee.peak_valgus_deg,    k.non_surg_knee.peak_valgus_deg,    warn_deg=3.0, err_deg=6.0),
                _delta_row("Peak Tibial IR (°)",       k.surg_knee.peak_tibial_ir_deg, k.non_surg_knee.peak_tibial_ir_deg, warn_deg=5.0, err_deg=8.0),
                _delta_row("Tibial Rot Excursion (°)", k.surg_knee.tibial_rot_excursion, k.non_surg_knee.tibial_rot_excursion, warn_deg=5.0, err_deg=8.0),
            ]
            if k.surg_hip and k.non_surg_hip:
                sh, nh = k.surg_hip, k.non_surg_hip
                rows += [
                    _delta_row("Peak Hip Flexion (°)",   sh.peak_flexion_deg,   nh.peak_flexion_deg,   warn_deg=5.0, err_deg=10.0),
                    _delta_row("Peak Hip Adduction (°)", sh.peak_adduction_deg, nh.peak_adduction_deg, warn_deg=3.0, err_deg=6.0),
                    _delta_row("Peak Hip IR (°)",        sh.peak_ir_deg,        nh.peak_ir_deg,        warn_deg=5.0, err_deg=8.0),
                ]
        if sigs.get("ankle_flex_surg") is not None:
            ank_s = float(np.nanmax(sigs["ankle_flex_surg"])) if len(sigs["ankle_flex_surg"]) > 0 else None
            ank_n = float(np.nanmax(sigs["ankle_flex_ns"])) if sigs.get("ankle_flex_ns") is not None and len(sigs["ankle_flex_ns"]) > 0 else None
            rows += [_delta_row("Peak Ankle Dorsiflexion (°)", ank_s, ank_n, warn_deg=3.0, err_deg=6.0)]

        # ── Kinetics ──────────────────────────────────────────────────────────
        km_s = sigs.get("knee_moment_surg")
        km_n = sigs.get("knee_moment_ns")
        if not _isnan(km_s) or not _isnan(km_n):
            km_lsi = round(km_s / km_n * 100, 1) if (not _isnan(km_s) and not _isnan(km_n) and km_n > 0) else None
            _reg("CMJ Knee Ext Moment", km_s, km_n, km_lsi, "Force")
            rows += [("Peak Knee Ext Moment (Nm)", _fmt(km_s, 1), _fmt(km_n, 1), km_lsi)]

        # ── COP ───────────────────────────────────────────────────────────────
        ap_s = _ap_excursion(sigs.get("cop_y_surg"), sigs.get("fz_surg"))
        ap_n = _ap_excursion(sigs.get("cop_y_ns"),   sigs.get("fz_ns"))
        if ap_s is not None or ap_n is not None:
            rows += [("AP COP Excursion (mm)", ap_s, ap_n, None)]

        # ── Interpretation ────────────────────────────────────────────────────
        pk  = mvj.peak_force_lsi if mvj.peak_force_lsi == mvj.peak_force_lsi else None
        pro = mvj.propulsion_lsi if mvj.propulsion_lsi == mvj.propulsion_lsi else None
        interp = ["Bilateral CMJ — sensitive to subtle propulsion deficits masked at lower intensities."]
        if pk:
            interp.append(
                f"<b>Peak force LSI {pk:.0f}%</b>: "
                + ("Symmetric bilateral output." if 90 <= pk <= 110
                   else f"Surgical limb generating {abs(pk-100):.0f}% {'more' if pk>100 else 'less'} force — {'possible NS compensation' if pk>100 else 'quad/hip extensor deficit; strength work required'}."))
        if pro:
            interp.append(f"<b>Propulsion LSI {pro:.0f}%</b>. Targets: ≥ 90% propulsion, ≥ 90% peak force, matched unweighting impulse.")

        # ── Figures ───────────────────────────────────────────────────────────
        grf_fig  = fig_lib.force_time_curve(sigs.get("fz_surg"), sigs.get("fz_ns"), rate_grf,
                      "Max Vertical Jump — Bilateral GRF", bw_n, figsize=(7.5, 2.2)) \
                   if sigs.get("fz_surg") is not None else None
        flex_fig = fig_lib.joint_angle_overlay(sigs.get("knee_flex_surg"), sigs.get("knee_flex_ns"),
                      rate_kin, "Knee Flexion", "Flexion (°)", figsize=(3.5, 2.2),
                      dir_low="← More Extended", dir_high="More Flexed →") \
                   if sigs.get("knee_flex_surg") is not None else None
        ank_fig  = fig_lib.joint_angle_overlay(sigs.get("ankle_flex_surg"), sigs.get("ankle_flex_ns"),
                      rate_kin, "Ankle Dorsiflexion", "Dorsiflexion (°)", figsize=(3.5, 2.2),
                      dir_low="← More Plantarflexed", dir_high="More Dorsiflexed →") \
                   if sigs.get("ankle_flex_surg") is not None else None
        valg_fig = fig_lib.joint_angle_overlay(sigs.get("knee_valgus_surg"), sigs.get("knee_valgus_ns"),
                      rate_kin, "Knee Valgus", "Valgus (°)", figsize=(3.5, 2.2),
                      dir_low="← More Varus", dir_high="More Valgus →") \
                   if sigs.get("knee_valgus_surg") is not None else None
        tibr_fig = fig_lib.joint_angle_overlay(sigs.get("tib_rot_surg"), sigs.get("tib_rot_ns"),
                      rate_kin, "Tibial Internal Rotation", "Int Rotation (°)", figsize=(3.5, 2.2),
                      dir_low="← More External", dir_high="More Internal →") \
                   if sigs.get("tib_rot_surg") is not None else None
        cop_fig  = fig_lib.cop_bilateral_ap_trace(
                      sigs.get("cop_y_surg"), sigs.get("cop_y_ns"),
                      fz_surg=sigs.get("fz_surg"), fz_ns=sigs.get("fz_ns"),
                      rate=rate_grf, title="Max Vertical Jump — COP Anterior-Posterior Trace",
                      figsize=(7.5, 1.8)) \
                   if sigs.get("cop_y_surg") is not None else None

        _k_cmj = mvj.kinematics if hasattr(mvj, "kinematics") else None
        _cmj_grf_sum = _lsi_phrase(mvj.propulsion_lsi, "propulsive peak force")
        _cmj_flex_sum = (
            _lsi_phrase(_k_cmj.lsi_peak_flexion, "countermovement knee flexion depth")
        ) if _k_cmj else ""
        _cmj_valg_sum = (
            _delta_phrase(_k_cmj.surg_knee.peak_valgus_deg, _k_cmj.non_surg_knee.peak_valgus_deg,
                          "knee valgus", "°", warn=3.0, err=6.0)
        ) if _k_cmj else ""
        _cmj_tib_sum = (
            _delta_phrase(_k_cmj.surg_knee.peak_tibial_ir_deg, _k_cmj.non_surg_knee.peak_tibial_ir_deg,
                          "tibial IR", "°", warn=5.0, err=8.0)
        ) if _k_cmj else ""

        _cmj_sum = _lsi_phrase(mvj.peak_force_lsi, "propulsive force")
        test_story += _test_page("MAX VERTICAL JUMP (CMJ)", rows, interp,
            fig_full=grf_fig,  cap_full="Full CMJ: unweighting → eccentric load → propulsion → flight. Propulsive peaks should be symmetric.",
            sum_full=_cmj_grf_sum,
            fig_l=flex_fig,    cap_l="Knee flexion depth. Asymmetric countermovement depth = compensatory loading strategy.",
            sum_l=_cmj_flex_sum,
            fig_r=ank_fig,     cap_r="Ankle dorsiflexion. Restricted ROM reduces jump height and shifts load away from the knee.",
            fig_l2=valg_fig,   cap_l2="Knee valgus during propulsion. Dynamic valgus under load = ACL stress.",
            sum_l2=_cmj_valg_sum,
            fig_r2=tibr_fig,   cap_r2="Tibial IR during propulsion. Coupled with valgus = high re-injury risk pattern.",
            sum_r2=_cmj_tib_sum,
            fig_cop=cop_fig,   cap_cop="AP COP displacement during CMJ (zero-centred). Anterior shift during propulsion drives jump height — asymmetric or reduced excursion on surgical side = limb offloading.",
            summary=_cmj_sum,
            description=(
                "The Countermovement Jump (CMJ) is the gold-standard bilateral power test. The athlete begins in quiet "
                "stance, rapidly descends into a squat (countermovement), then drives upward for maximum jump height. "
                "The full GRF waveform reveals each phase: unweighting, eccentric loading, propulsion, and flight. "
                "After ACL reconstruction, typical deficits include an asymmetric countermovement depth (surgical limb "
                "offloading during the eccentric phase), reduced propulsive impulse, and side-to-side jump height "
                "asymmetry. Propulsive LSI < 90% is a common criterion for RTS decisions. Kinematic analysis identifies "
                "compensatory strategies—valgus, reduced knee flexion, and tibial rotation—that indicate the surgical "
                "limb is not being loaded with full confidence or mechanical efficiency."
            ))

    # ── ENDURANCE SQUAT ───────────────────────────────────────────────────────
    esq = test_results.get("endurance_squat")
    if esq:
        sigs = all_signals.get("endurance_squat", {})
        _reg("Endurance LSI", esq.mean_lsi_peak, None, esq.mean_lsi_peak, "Endurance", "endurance_lsi")
        if esq.fatigue_drift_pct == esq.fatigue_drift_pct:
            _rtr["fatigue_drift"] = esq.fatigue_drift_pct
        rows = [
            ("Mean Peak Force LSI (%)",  "—",                   "—", esq.mean_lsi_peak),
            ("LSI First 10s (%)",        "—",                   "—", esq.lsi_first_third),
            ("LSI Last 10s (%)",         "—",                   "—", esq.lsi_last_third),
            ("Fatigue Drift (Δ%)",       esq.fatigue_drift_pct, "—", None),
            ("Total Reps Detected",      esq.n_cycles,          "—", None),
            ("Mean Peak Force Surg (N)", esq.peak_force_surg_N, "—", None),
            ("Mean Peak Force NS (N)",   esq.peak_force_ns_N,   "—", None),
        ]
        drift = esq.fatigue_drift_pct
        dv    = drift if drift is not None and drift == drift else None
        interp = ["30s bilateral squat — reveals neuromuscular fatigue-induced asymmetry missed by single-rep testing."]
        if dv is not None:
            interp.append(
                f"<b>Fatigue drift {dv:.1f}%</b>: "
                + ("✓ Symmetry maintained under fatigue." if dv > -5
                   else f"{'⚠' if dv > -10 else '✗'} LSI declined {abs(dv):.0f}% — surgical limb offloads under prolonged effort. Endurance-specific rehab required."))
        interp.append("<b>Targets:</b> Mean LSI ≥ 90% · Fatigue drift < 5% · No progressive offloading across 30s.")

        lsi_fig = fig_lib.endurance_lsi_over_time(esq.lsi_over_time, esq.time_axis,
                      figsize=(4.0, 2.2)) \
                  if esq.lsi_over_time is not None and len(esq.lsi_over_time) > 1 else None
        grf_fig = fig_lib.force_time_curve(sigs.get("fz_surg"), sigs.get("fz_ns"), rate_grf,
                      "Endurance Squat — Full 30s GRF", bw_n, figsize=(3.5, 2.2)) \
                  if sigs.get("fz_surg") is not None else None
        cop_drift_fig = fig_lib.endurance_cop_drift(
                      sigs.get("cop_y_surg"), sigs.get("cop_y_ns"),
                      fz_surg=sigs.get("fz_surg"), fz_ns=sigs.get("fz_ns"),
                      rate=rate_grf, figsize=(7.5, 1.8)) \
                  if sigs.get("cop_y_surg") is not None else None

        ap_s = _ap_excursion(sigs.get("cop_y_surg"), sigs.get("fz_surg"))
        ap_n = _ap_excursion(sigs.get("cop_y_ns"),   sigs.get("fz_ns"))
        if ap_s is not None or ap_n is not None:
            rows += [("AP COP Excursion — Surg / NS (mm)", ap_s, ap_n, None)]
        km_s = sigs.get("knee_moment_surg")
        km_n = sigs.get("knee_moment_ns")
        if not _isnan(km_s) or not _isnan(km_n):
            km_lsi = round(km_s / km_n * 100, 1) if (not _isnan(km_s) and not _isnan(km_n) and km_n > 0) else None
            _reg("Endurance Knee Ext Moment", km_s, km_n, km_lsi, "Force")
            rows += [("Peak Knee Ext Moment (Nm)", _fmt(km_s, 1), _fmt(km_n, 1), km_lsi)]

        _esq_lsi_sum = _lsi_phrase(esq.mean_lsi_peak, "mean 30s force symmetry")
        dv = esq.fatigue_drift_pct if esq.fatigue_drift_pct == esq.fatigue_drift_pct else None
        _esq_cop_sum = (
            f"Fatigue drift: {dv:.1f}% LSI change across 30s — "
            + ("symmetry maintained under sustained effort." if dv is not None and dv > -5
               else f"surgical limb offloads progressively; {abs(dv):.0f}% decline under fatigue.")
        ) if dv is not None else ""

        _esq_sum = _lsi_phrase(esq.mean_lsi_peak, "mean force symmetry over 30 s")
        test_story += _test_page("ENDURANCE SQUAT (30s)", rows, interp,
            fig_l=lsi_fig, cap_l="Per-rep LSI across 30s. Declining trend = neuromuscular fatigue breakdown.",
            sum_l=_esq_lsi_sum,
            fig_r=grf_fig, cap_r="Full 30s GRF trace. Force magnitude and rhythm across the set.",
            fig_cop=cop_drift_fig,
            sum_cop=_esq_cop_sum,
            summary=_esq_sum,
            cap_cop="Bilateral AP COP drift across 30s (zero-centred, 1s RMS envelope). "
                    "Surgical-side amplitude shrinking or trending posterior = offloading under fatigue. "
                    "Diverging traces = progressive asymmetry with sustained effort.",
            description=(
                "The 30-Second Endurance Squat assesses neuromuscular fatigue and limb symmetry under sustained "
                "bilateral loading. The athlete performs continuous bilateral squats to a standardised depth for a "
                "full 30 seconds. Unlike single-repetition tests, this task reveals how well the neuromuscular system "
                "maintains symmetry as it fatigues. Athletes post-ACL reconstruction frequently show an initial LSI "
                "within acceptable range that deteriorates progressively across the set as the nervous system "
                "disinhibits protective guarding. The COP drift trace exposes this compensation: surgical-limb weight "
                "transfer decreases, traces diverge, and loading shifts toward the uninvolved side. This test is "
                "particularly sensitive to residual quadriceps inhibition that is not apparent during brief, "
                "high-energy tasks."
            ))

    # ── SINGLE-LEG VERTICAL JUMP ──────────────────────────────────────────────
    slj = test_results.get("single_leg_jump")
    if slj and slj.surgical and slj.non_surgical:
        sv, nv = slj.surgical, slj.non_surgical
        sigs   = all_signals.get("single_leg_jump", {})

        # ── Loading ───────────────────────────────────────────────────────────
        _reg("SL Jump Height", sv.jump_height_cm,        nv.jump_height_cm,        slj.lsi_jump_height, "Force", "sl_jump_lsi")
        _reg("SL Peak GRF",    sv.peak_force_N,          nv.peak_force_N,          slj.lsi_peak_force,  "Force")
        _reg("SL Impulse",     sv.propulsion_impulse_Ns, nv.propulsion_impulse_Ns, slj.lsi_impulse,     "Force")
        rows = [
            ("Jump Height",              sv.jump_height_cm,        nv.jump_height_cm,        slj.lsi_jump_height),
            ("Peak GRF (N)",             sv.peak_force_N,          nv.peak_force_N,          slj.lsi_peak_force),
            ("Propulsion Impulse (N·s)", sv.propulsion_impulse_Ns, nv.propulsion_impulse_Ns, slj.lsi_impulse),
            ("RFD 100ms (N/s)",          sv.rfd_100ms,             nv.rfd_100ms,             None),
            ("Flight Time (s)",          sv.flight_time_s,         nv.flight_time_s,         None),
            ("Contact Time (s)",         sv.contact_time_s,        nv.contact_time_s,        None),
        ]

        # ── Kinematics ────────────────────────────────────────────────────────
        rows += [
            ("Peak Knee Flexion (°)",     sv.peak_knee_flexion_deg, nv.peak_knee_flexion_deg, None),
            _delta_row("Peak Knee Valgus (°)", sv.peak_valgus_deg,  nv.peak_valgus_deg,       warn_deg=3.0, err_deg=6.0),
        ]
        if sigs.get("tib_rot_surg") is not None:
            tib_s = float(np.nanmax(sigs["tib_rot_surg"])) if len(sigs["tib_rot_surg"]) > 0 else None
            tib_n = float(np.nanmax(sigs["tib_rot_ns"])) if sigs.get("tib_rot_ns") is not None and len(sigs["tib_rot_ns"]) > 0 else None
            rows += [_delta_row("Peak Tibial IR (°)", tib_s, tib_n, warn_deg=5.0, err_deg=8.0)]
        if sigs.get("ankle_flex_surg") is not None:
            ank_s = float(np.nanmax(sigs["ankle_flex_surg"])) if len(sigs["ankle_flex_surg"]) > 0 else None
            ank_n = float(np.nanmax(sigs["ankle_flex_ns"])) if sigs.get("ankle_flex_ns") is not None and len(sigs["ankle_flex_ns"]) > 0 else None
            rows += [_delta_row("Peak Ankle Dorsiflexion (°)", ank_s, ank_n, warn_deg=3.0, err_deg=6.0)]
        if sigs.get("hip_add_surg") is not None:
            hadd_s = float(np.nanmax(sigs["hip_add_surg"])) if len(sigs["hip_add_surg"]) > 0 else None
            hadd_n = float(np.nanmax(sigs["hip_add_ns"])) if sigs.get("hip_add_ns") is not None and len(sigs["hip_add_ns"]) > 0 else None
            rows += [_delta_row("Peak Hip Adduction (°)", hadd_s, hadd_n, warn_deg=3.0, err_deg=6.0)]

        # ── Kinetics ──────────────────────────────────────────────────────────
        km_s = sigs.get("knee_moment_surg")
        km_n = sigs.get("knee_moment_ns")
        if not _isnan(km_s) or not _isnan(km_n):
            km_lsi = round(km_s / km_n * 100, 1) if (not _isnan(km_s) and not _isnan(km_n) and km_n > 0) else None
            _reg("SLJ Knee Ext Moment", km_s, km_n, km_lsi, "Force")
            rows += [("Peak Knee Ext Moment (Nm)", _fmt(km_s, 1), _fmt(km_n, 1), km_lsi)]

        # ── COP ───────────────────────────────────────────────────────────────
        ap_s = _ap_excursion(sigs.get("cop_y_surg"), sigs.get("fz_surg"))
        ap_n = _ap_excursion(sigs.get("cop_y_ns"),   sigs.get("fz_ns"))
        if ap_s is not None or ap_n is not None:
            rows += [("AP COP Excursion (mm)", ap_s, ap_n, None)]

        # ── Interpretation ────────────────────────────────────────────────────
        jh = slj.lsi_jump_height if slj.lsi_jump_height == slj.lsi_jump_height else None
        pk = slj.lsi_peak_force  if slj.lsi_peak_force  == slj.lsi_peak_force  else None
        interp = ["Highest-demand test — isolates each limb, removing bilateral compensation. Strongest re-injury risk predictor."]
        if jh:
            interp.append(
                f"<b>Jump height LSI {jh:.0f}%</b>: "
                + ("✓ Meets RTS threshold (≥ 90%)." if jh >= 90
                   else f"{'⚠' if jh >= 75 else '✗'} {100-jh:.0f}% below non-surgical. {'Continued plyometric loading required' if jh >= 75 else 'RTS not recommended — significant power deficit'}."))
        if pk and pk < 90:
            interp.append(f"<b>Peak force LSI {pk:.0f}%</b>: Force deficit on surgical side — prioritise quad/hip extensor strengthening.")
        interp.append("<b>Targets:</b> Jump height LSI ≥ 90% · Peak GRF LSI ≥ 90% · Valgus < 5° · Tibial IR < 10° on surgical limb.")

        # ── Figures ───────────────────────────────────────────────────────────
        grf_fig  = fig_lib.force_time_curve(sigs.get("fz_surg"), sigs.get("fz_ns"), rate_grf,
                      "Single-Leg Jump — Surgical vs Non-Surgical GRF", bw_n, figsize=(7.5, 2.2)) \
                   if sigs.get("fz_surg") is not None or sigs.get("fz_ns") is not None else None
        flex_fig = fig_lib.joint_angle_overlay(sigs.get("knee_flex_surg"), sigs.get("knee_flex_ns"),
                      rate_kin, "Knee Flexion", "Flexion (°)", figsize=(3.5, 2.2),
                      dir_low="← More Extended", dir_high="More Flexed →") \
                   if sigs.get("knee_flex_surg") is not None else None
        ank_fig  = fig_lib.joint_angle_overlay(sigs.get("ankle_flex_surg"), sigs.get("ankle_flex_ns"),
                      rate_kin, "Ankle Dorsiflexion", "Dorsiflexion (°)", figsize=(3.5, 2.2),
                      dir_low="← More Plantarflexed", dir_high="More Dorsiflexed →") \
                   if sigs.get("ankle_flex_surg") is not None else None
        valg_fig = fig_lib.joint_angle_overlay(sigs.get("knee_valgus_surg"), sigs.get("knee_valgus_ns"),
                      rate_kin, "Knee Valgus", "Valgus (°)", figsize=(3.5, 2.2),
                      dir_low="← More Varus", dir_high="More Valgus →") \
                   if sigs.get("knee_valgus_surg") is not None else None
        tibr_fig = fig_lib.joint_angle_overlay(sigs.get("tib_rot_surg"), sigs.get("tib_rot_ns"),
                      rate_kin, "Tibial Internal Rotation", "Int Rotation (°)", figsize=(3.5, 2.2),
                      dir_low="← More External", dir_high="More Internal →") \
                   if sigs.get("tib_rot_surg") is not None else None
        cop_fig  = fig_lib.cop_bilateral_ap_trace(
                      sigs.get("cop_y_surg"), sigs.get("cop_y_ns"),
                      fz_surg=sigs.get("fz_surg"), fz_ns=sigs.get("fz_ns"),
                      rate=rate_grf, title="Single-Leg Jump — COP AP Trace (Per Limb Trial)",
                      figsize=(7.5, 1.8)) \
                   if sigs.get("cop_y_surg") is not None else None

        _k_slj = None
        if hasattr(slj, "surgical") and hasattr(slj.surgical, "kinematics"):
            _k_slj = slj.surgical.kinematics
        _slj_grf_sum = _lsi_phrase(slj.lsi_jump_height, "single-leg jump height")
        _slj_flex_sum = (
            _lsi_phrase(_k_slj.lsi_peak_flexion, "single-leg knee flexion depth")
        ) if _k_slj and hasattr(_k_slj, "lsi_peak_flexion") else ""
        _slj_valg_sum = (
            _delta_phrase(_k_slj.surg_knee.peak_valgus_deg, _k_slj.non_surg_knee.peak_valgus_deg,
                          "knee valgus", "°", warn=3.0, err=6.0)
        ) if _k_slj and hasattr(_k_slj, "surg_knee") else ""
        _slj_tib_sum = (
            _delta_phrase(_k_slj.surg_knee.peak_tibial_ir_deg, _k_slj.non_surg_knee.peak_tibial_ir_deg,
                          "tibial IR", "°", warn=5.0, err=8.0)
        ) if _k_slj and hasattr(_k_slj, "surg_knee") else ""

        _slj_sum = _lsi_phrase(slj.lsi_jump_height, "single-leg jump height")
        test_story += _test_page("SINGLE-LEG VERTICAL JUMP", rows, interp,
            fig_full=grf_fig,  cap_full="Per-limb GRF (separate trials). Propulsive peak height and shape should match.",
            sum_full=_slj_grf_sum,
            fig_l=flex_fig,    cap_l="Knee flexion per limb. Reduced surgical depth = power deficit or guarded strategy.",
            sum_l=_slj_flex_sum,
            fig_r=ank_fig,     cap_r="Ankle dorsiflexion per limb. Restricted ROM = reduced propulsive range / increased ACL load.",
            fig_l2=valg_fig,   cap_l2="Knee valgus per limb. Elevated surgical valgus under single-leg load = re-injury risk.",
            sum_l2=_slj_valg_sum,
            fig_r2=tibr_fig,   cap_r2="Tibial IR per limb. Surgical-side IR > 10° = high rotational ACL stress pattern.",
            sum_r2=_slj_tib_sum,
            fig_cop=cop_fig,   cap_cop="AP COP displacement (zero-centred, per limb trial). Reduced anterior excursion on surgical side = guarded takeoff / limited push-off.",
            summary=_slj_sum,
            description=(
                "The Single-Leg Vertical Jump isolates each limb independently, removing the ability of the uninvolved "
                "leg to compensate. The athlete performs a maximal countermovement jump landing on the same leg. "
                "Jump height LSI (surgical vs. non-surgical) is among the most sensitive power-based RTS criteria—"
                "a threshold of ≥ 90% is widely cited in ACL literature. Single-leg testing also reveals subtle "
                "compensations masked in bilateral tasks: reduced propulsive depth, hip drop (Trendelenburg pattern), "
                "and increased tibial rotation are common markers of residual neuromuscular asymmetry. Pelvis drop "
                "depth quantifies hip abductor control on the stance limb, which is an independent predictor of "
                "re-injury risk following return to sport."
            ))

    # ── PROPRIOCEPTION ────────────────────────────────────────────────────────
    prop = test_results.get("proprioception")
    if prop:
        sigs_all = all_signals.get("proprioception", {})

        def _prop_page(cond, sigs_key, title,
                       norm_vel, lsi_note):
            """Build one balance condition page."""
            if cond is None:
                return []
            sc = cond.surgical
            nc = cond.non_surgical
            if sc is None or nc is None:
                return []
            sigs = sigs_all.get(sigs_key, {})

            lsi_vel   = cond.lsi_cop_velocity
            lsi_ell   = cond.lsi_ellipse_area
            _reg(f"Balance COP Vel ({sigs_key.capitalize()})",
                 sc.mean_velocity_mm_s, nc.mean_velocity_mm_s, lsi_vel,
                 "Balance", "cop_velocity_lsi")
            _reg(f"Balance Ellipse ({sigs_key.capitalize()})",
                 sc.ellipse_area_mm2,   nc.ellipse_area_mm2,   lsi_ell, "Balance")

            rows = [
                ("COP Mean Velocity (mm/s)", sc.mean_velocity_mm_s,  nc.mean_velocity_mm_s,  lsi_vel),
                ("COP Ellipse Area (mm²)",   sc.ellipse_area_mm2,    nc.ellipse_area_mm2,    lsi_ell),
                ("Total Excursion (mm)",     sc.total_excursion_mm,  nc.total_excursion_mm,  None),
                ("AP Range (mm)",            sc.range_ap_mm,         nc.range_ap_mm,         None),
                ("ML Range (mm)",            sc.range_ml_mm,         nc.range_ml_mm,         None),
                ("RMS Displacement (mm)",    sc.rms_displacement_mm, nc.rms_displacement_mm, None),
            ]

            sv  = sc.mean_velocity_mm_s if sc.mean_velocity_mm_s == sc.mean_velocity_mm_s else None
            nv  = nc.mean_velocity_mm_s if nc.mean_velocity_mm_s == nc.mean_velocity_mm_s else None
            vlsi = lsi_vel if lsi_vel == lsi_vel else None

            interp = [lsi_note]
            if sv and nv:
                interp.append(
                    f"Surgical {sv:.0f} mm/s vs Non-Surgical {nv:.0f} mm/s — "
                    + ("Surgical limb less stable. Proprioceptive retraining indicated."
                       if sv > nv else "COP velocity comparable between limbs."))
            if vlsi:
                interp.append(
                    f"<b>COP Velocity LSI {vlsi:.0f}%</b> (NS÷Surg×100 — lower = better). "
                    f"Normative target: &lt; {norm_vel} mm/s. LSI target ≥ 90%.")

            cop_fig = fig_lib.cop_scatter(
                          sigs.get("cop_x_surg"), sigs.get("cop_y_surg"),
                          sigs.get("cop_x_ns"),   sigs.get("cop_y_ns"),
                          figsize=(7.5, 3.5)) \
                      if sigs.get("cop_x_surg") is not None else None

            return _test_page(title, rows, interp,
                fig_full=cop_fig,
                cap_full="COP trajectory — eyes closed. Tighter ellipse = better joint position sense. 95% confidence ellipse shown.",
                description=(
                    "The Balance / Proprioception test assesses joint position sense and neuromuscular control under "
                    "sensory challenge. The athlete stands on a single leg with eyes closed, removing visual "
                    "compensation and isolating the mechanoreceptor feedback system of the ankle and knee. Center of "
                    "Pressure (COP) velocity is the most sensitive metric: a higher velocity indicates greater "
                    "corrective activity, meaning the neuromuscular system is working harder to maintain stability. "
                    "After ACL reconstruction, deficits in mechanoreceptor density within the graft and capsular "
                    "tissue result in elevated COP velocity and a larger sway ellipse on the surgical limb. The Airex "
                    "pad condition adds an unstable surface to further stress the proprioceptive system, revealing "
                    "deficits that may be masked on firm ground. Symmetry between limbs (LSI ≥ 85–90%) is the "
                    "primary RTS criterion for this test."
                ))

        # Standard condition (PCTL1 / PCTR1 — flat firm surface)
        test_story += _prop_page(
            prop.standard, "standard",
            title="BALANCE — STANDARD (EYES CLOSED, FIRM SURFACE)",
            norm_vel=12,
            lsi_note="Eyes-closed single-leg stance on firm surface. "
                     "COP velocity is the most sensitive metric — "
                     "higher speed = more corrective activity = reduced proprioception. "
                     "Target: &lt; 12 mm/s.",
        )

        # Airex condition (PCTL2 / PCTR2 — unstable foam pad)
        test_story += _prop_page(
            prop.airex, "airex",
            title="BALANCE — AIREX PAD (EYES CLOSED, UNSTABLE SURFACE)",
            norm_vel=50,
            lsi_note="Eyes-closed single-leg stance on Airex foam pad. "
                     "Higher COP values are expected — the unstable surface "
                     "challenges proprioceptive and neuromuscular systems. "
                     "Symmetry between limbs (LSI ≥ 85%) is the key target.",
        )

    # ── Triple Hop for Distance (manually entered — no test page) ─────────────
    th = (patient_data or {}).get("triple_hop") or {}
    th_surg = th.get("surg_in")
    th_ns   = th.get("ns_in")
    th_lsi  = th.get("lsi")
    if th_surg is not None or th_ns is not None:
        def _fmt_hop(v):
            if v is None:
                return "—"
            ft  = int(v) // 12
            ins = int(round(v)) % 12
            return f"{ft}' {ins}\""
        _reg("Triple Hop",
             _fmt_hop(th_surg), _fmt_hop(th_ns), th_lsi,
             "Hop", "triple_hop_lsi")
        # Flag yellow-zone (75–89%) — red zone (<75) is caught by _reg()
        if th_lsi is not None and 75 <= th_lsi < 90:
            clinical_notes.append(
                f"Triple Hop: LSI {th_lsi:.1f}% — below 90% target")

    # ── Summary (built after test pages so all metrics are collected) ─────────
    domain_avg = {k: sum(v) / len(v) for k, v in domain_scores.items() if v}

    rtr_metrics_full = {**(rtr_metrics or {}), **_rtr}
    if patient_data.get("months_since_surgery") is not None:
        rtr_metrics_full["_months_since_surgery"] = patient_data["months_since_surgery"]

    rtr_result = compute_rtr_score(rtr_metrics_full)

    summary_story = _exec_summary(
        s=s,
        patient_data=patient_data,
        lsi_summary=lsi_summary,
        domain_scores=domain_avg,
        clinical_notes=clinical_notes,
        rtr_result=rtr_result,
    )

    full_story = summary_story + test_story
    doc.build(full_story)
    return output_path
