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
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
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


# ─── Layout constants ─────────────────────────────────────────────────────────
PAGE    = letter if PAGE_SIZE == "letter" else A4
MARGIN  = 0.5 * inch
W_FULL  = 7.5 * inch          # usable page width (8.5 - 2×0.5)
W_HALF  = (W_FULL - 0.15 * inch) / 2   # half-width for side-by-side figures
H_PAGE  = 10.0 * inch         # usable page height (11 - 2×0.5)

# ─── Colours ──────────────────────────────────────────────────────────────────
C_HEADER = colors.HexColor(COLORS["header_bg"])
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
            surgery_date: str = "", months_post_op: float = None) -> Table:
    # Build the patient info line
    info_parts = [
        f"<b>Patient:</b> {name}",
        f"<b>Surgical Side:</b> {side}",
        f"<b>Date:</b> {date}",
    ]
    if dob and dob != "—":
        info_parts.insert(1, f"<b>DOB:</b> {dob}")

    # Surgery info
    if months_post_op is not None:
        mo = months_post_op
        if mo < 12:
            time_str = f"{mo:.1f} months post-op"
        else:
            yrs = mo / 12
            time_str = f"{yrs:.1f} years post-op"
        if surgery_date:
            sd = _surgery_date_display(surgery_date)
            info_parts.append(f"<b>Surgery:</b> {sd} ({time_str})")
        else:
            info_parts.append(f"<b>Post-op:</b> {time_str}")
    elif surgery_date:
        info_parts.append(f"<b>Surgery:</b> {_surgery_date_display(surgery_date)}")

    content = [
        Paragraph(CLINIC_NAME, s["h1"]),
        Paragraph(CLINIC_SUBTITLE, s["h1sub"]),
        Spacer(1, 3),
        Paragraph(" &nbsp;&nbsp; ".join(info_parts), s["pt"]),
    ]
    if clinician:
        content.append(Paragraph(f"<b>Clinician:</b> {clinician}",
            ParagraphStyle("cl", fontSize=7.5,
                           textColor=colors.HexColor("#AACCEE"),
                           fontName="Helvetica")))
    tbl = Table([[content]], colWidths=[W_FULL])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), C_HEADER),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12),
    ]))
    return tbl


def _sec(text: str, s: dict) -> List:
    return [Paragraph(f"  {text}", s["sec"]), Spacer(1, 3)]


# ─── Compact metrics table ────────────────────────────────────────────────────

def _metrics_table(rows: list, s: dict) -> Table:
    """
    rows: list of (label, surg_val, ns_val, lsi_val)
    Compact version — tight padding, colour-coded rows, no separate status column.
    """
    data = [[
        Paragraph("<b>Metric</b>", s["body"]),
        Paragraph("<b>Surgical</b>", s["body"]),
        Paragraph("<b>Non-Surg</b>", s["body"]),
        Paragraph("<b>LSI</b>", s["body"]),
    ]]
    for row in rows:
        name, sv, nv, lv = row[:4]
        try:
            bhex = _badge_color(lv).hexval()[2:]
        except Exception:
            bhex = "AAAAAA"
        data.append([
            Paragraph(name, s["body"]),
            _fmt(sv),
            _fmt(nv),
            Paragraph(
                f"<font color='#{bhex}'><b>{_fmt(lv, suffix='%')}</b></font>",
                ParagraphStyle("lv", fontSize=8.5, alignment=TA_CENTER)),
        ])

    col_w = [3.0*inch, 1.1*inch, 1.1*inch, 1.0*inch]
    tbl   = Table(data, colWidths=col_w)
    cmds  = [
        ("BACKGROUND",    (0,0), (-1,0),  C_HEADER),
        ("TEXTCOLOR",     (0,0), (-1,0),  C_WHITE),
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
        bg = _row_bg(row[3])
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
        "LSI = Surgical ÷ Non-Surgical × 100",
        s["small"])


# ─── Score influence helper ───────────────────────────────────────────────────

# Human-readable labels for RTR breakdown keys
_INFLUENCE_NAMES = {
    "drop_jump_rsi":    "RSI (reactive strength)",
    "landing_lsi":      "Landing symmetry",
    "rfd_lsi":          "Rate of force dev.",
    "peak_grf_lsi":     "Peak GRF symmetry",
    "sl_jump_lsi":      "Single-leg jump",
    "knee_valgus_surg": "Knee valgus (surgical)",
    "cop_velocity_lsi": "Balance symmetry",
    "cop_vel_surg_abs": "Balance quality (surg)",
    "cop_vel_ns_abs":   "Balance quality (NS)",
    "endurance_lsi":    "Endurance symmetry",
    "fatigue_drift":    "Fatigue drift",
}


def _score_influence_box(rtr_result: dict, s: dict,
                         width: float, tf: float,
                         perf_score: float,
                         months_post_op: float = None) -> list:
    """
    Compact 'Score Drivers' panel showing top strengths and detractors.
    Returns a list of flowables sized to `width` inches.
    """
    breakdown = rtr_result.get("breakdown", {})
    red_factor = rtr_result.get("red_factor", 1.0)
    score      = rtr_result.get("score", 0)

    # Split into strengths (green) and detractors (yellow/red), sort by score
    strengths  = sorted(
        [(k, v) for k, v in breakdown.items() if v["color"] == "green"],
        key=lambda x: -x[1]["score"])[:3]
    detractors = sorted(
        [(k, v) for k, v in breakdown.items() if v["color"] in ("red", "yellow")],
        key=lambda x: x[1]["score"])[:4]

    C_BOX  = colors.HexColor("#f5f5f5")
    C_STR  = colors.HexColor("#1a7a3e")   # dark green text
    C_DET  = colors.HexColor("#b03020")   # dark red text
    C_HEAD = colors.HexColor("#39414a")
    C_ORG  = colors.HexColor("#fc6c0f")
    C_SUB  = colors.HexColor("#666666")

    s_hd = ParagraphStyle("inf_hd", fontSize=7.5, fontName="Helvetica-Bold",
                           textColor=C_WHITE, leading=10)
    s_str = ParagraphStyle("inf_str", fontSize=7, fontName="Helvetica",
                            textColor=C_STR, leading=9.5, spaceAfter=1)
    s_det = ParagraphStyle("inf_det", fontSize=7, fontName="Helvetica",
                            textColor=C_DET, leading=9.5, spaceAfter=1)
    s_eqn = ParagraphStyle("inf_eqn", fontSize=6.5, fontName="Helvetica",
                            textColor=C_SUB, leading=9)
    s_none = ParagraphStyle("inf_none", fontSize=7, fontName="Helvetica",
                             textColor=C_SUB, leading=9)

    col_w  = (width * inch - 2) / 2   # two equal sub-columns inside box

    # Strengths column
    str_items = [Paragraph("<b>STRENGTHS</b>", s_hd)]
    if strengths:
        for k, v in strengths:
            name = _INFLUENCE_NAMES.get(k, k)
            str_items.append(Paragraph(f"▲ {name}", s_str))
    else:
        str_items.append(Paragraph("None recorded", s_none))

    # Detractors column
    det_items = [Paragraph("<b>DETRACTORS</b>", s_hd)]
    if detractors:
        for k, v in detractors:
            name = _INFLUENCE_NAMES.get(k, k)
            det_items.append(Paragraph(f"▼ {name}", s_det))
    else:
        det_items.append(Paragraph("None recorded", s_none))

    inner = Table([[str_items, det_items]], colWidths=[col_w, col_w])
    inner.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, 0), C_STR),
        ("BACKGROUND",    (1, 0), (1, 0), C_DET),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))

    # Score equation line
    eqn_parts = []
    if perf_score != score:
        eqn_parts.append(f"Perf {perf_score:.0f}")
        if red_factor < 1.0:
            eqn_parts.append(f"×{red_factor:.2f} red-flag")
        if tf < 0.99:
            eqn_parts.append(f"×{tf:.2f} time")
        eqn_parts.append(f"= {score:.0f}/100")
    eqn_line = [Paragraph("  →  ".join(eqn_parts), s_eqn)] if eqn_parts else []

    outer = Table([[inner]] + ([[eqn_line]] if eqn_line else []),
                  colWidths=[width * inch])
    outer.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_BOX),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ("LINEABOVE",     (0, 0), (-1, 0),  1.0, C_ORG),
    ]))
    return [outer]


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
    ))
    story.append(Spacer(1, 5))

    # ── Row 1: RTS badge (left) + key metrics table (right) ──────────────────
    score       = rtr_result.get("score", 0)
    perf_score  = rtr_result.get("perf_score", score)
    grade       = rtr_result.get("grade", "—")
    red_count   = rtr_result.get("red_count", 0)
    tf          = rtr_result.get("time_factor", 1.0)
    time_note   = rtr_result.get("time_note")

    # Badge colour — orange accent only for Ready; charcoal variants otherwise.
    # Green/yellow/red are reserved for data traffic-light display.
    gc = (colors.HexColor("#fc6c0f") if grade == "Ready"       else
          colors.HexColor("#39414a") if grade == "Progressing" else
          colors.HexColor("#4a3030") if grade == "Caution"     else
          colors.HexColor("#3a2828"))   # Not Ready

    # Sub-line: show what's constraining the score
    sub_parts = []
    if red_count > 0:
        sub_parts.append(f"{red_count} red flag{'s' if red_count > 1 else ''}")
    if months_post_op is not None and tf < 0.99:
        sub_parts.append(f"{months_post_op:.0f} mo post-op")
    sub_line = " · ".join(sub_parts) if sub_parts else ""

    # Score number — large, centred vertically
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

    badge_text = (f"<b>RTS Score</b><br/>{grade}"
                  + (f"<br/><font size='7'>{sub_line}</font>" if sub_line else ""))

    badge = Table([[
        score_cell,
        Paragraph(badge_text,
                  ParagraphStyle("gr", fontSize=11, textColor=C_WHITE,
                                 fontName="Helvetica-Bold", leading=15,
                                 spaceAfter=0)),
    ]], colWidths=[1.05 * inch, 1.95 * inch])
    badge.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), gc),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
    ]))

    # Pull top 6 LSI metrics for the summary table
    top_lsi = [(label, sv, nv, lv)
               for label, (sv, nv, lv) in list(lsi_summary.items())[:6]]
    summary_tbl = _metrics_table(top_lsi, s) if top_lsi else Spacer(1, 1)

    row1 = Table([[badge, summary_tbl]],
                 colWidths=[3.1 * inch, 4.4 * inch])
    row1.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(row1)
    story.append(Spacer(1, 4))

    # ── Row 2: Radar (left) + LSI bar chart (right) ───────────────────────────
    # Left column: Return to Sport Profile + Score Drivers panel below
    # Right column: LSI bar chart (all tests)
    LEFT_W  = 3.15   # inches
    RIGHT_W = W_FULL / inch - LEFT_W - 0.05   # ~4.30"

    lsi_vals = {label: lv for label, (sv, nv, lv) in lsi_summary.items()
                if lv is not None and lv == lv}

    left_col  = []
    right_col = []

    # Radar on the LEFT
    if len(domain_scores) >= 3:
        radar_fig = fig_lib.rtr_radar(domain_scores, figsize=(LEFT_W, LEFT_W * 0.9))
        left_col.append(_fig_img(radar_fig, width_in=LEFT_W))
        left_col.append(Spacer(1, 3))

    # Score influence panel below radar
    left_col += _score_influence_box(
        rtr_result, s, width=LEFT_W,
        tf=tf, perf_score=perf_score,
        months_post_op=months_post_op)

    # LSI bar chart on the RIGHT
    if lsi_vals:
        n      = len(lsi_vals)
        bar_h  = max(2.5, min(n * 0.30, 5.2))   # tight per-bar height, capped
        bar_fig = fig_lib.lsi_bar_chart(
            lsi_vals,
            title="Limb Symmetry Index — All Tests",
            figsize=(RIGHT_W, bar_h))
        right_col.append(_fig_img(bar_fig, width_in=RIGHT_W))

    row2 = Table([[left_col, right_col]],
                 colWidths=[LEFT_W * inch, RIGHT_W * inch])
    row2.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(row2)
    story.append(Spacer(1, 4))

    # ── Clinical flags ────────────────────────────────────────────────────────
    story += _sec("CLINICAL FLAGS", s)

    if time_note:
        tf_pct = round(tf * 100)
        story.append(Paragraph(
            f"<b>Time post-op:</b> {time_note} [Time readiness: {tf_pct}%]",
            ParagraphStyle("tnote", parent=s["body"],
                           textColor=colors.HexColor("#7B3F00") if tf < 0.75
                           else colors.HexColor("#5A5A00") if tf < 0.92
                           else C_TEXT)))

    if clinical_notes:
        for note in clinical_notes[:6]:
            story.append(Paragraph(f"• {note}", s["body"]))
    elif not time_note:
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

    # ── helper: build a compact one-page test section ─────────────────────────
    def _test_page(title: str, rows: list, interp_lines: list,
                   fig_full=None,  cap_full="",
                   fig_l=None,     cap_l="",
                   fig_r=None,     cap_r="",
                   fig_l2=None,    cap_l2="",
                   fig_r2=None,    cap_r2="") -> list:
        """
        Lay out one test section:
          title bar
          metrics table
          interpretation box
          optional full-width figure
          optional pair of side-by-side figures (row 1)
          optional pair of side-by-side figures (row 2)
        Always ends with PageBreak.
        """
        pg = []
        pg += _sec(title, s)
        pg.append(_metrics_table(rows, s))
        pg.append(Spacer(1, 3))
        pg.append(_interp_box(interp_lines, s))
        pg.append(Spacer(1, 3))

        if fig_full is not None:
            pg += _fig_full(fig_full, cap_full, s)

        if fig_l is not None and fig_r is not None:
            pg += _fig_pair(fig_l, cap_l, fig_r, cap_r, s)
        elif fig_l is not None:
            # Single figure — render at half-width to keep page compact
            img = _fig_img(fig_l, width_in=W_HALF / inch)
            cap = Paragraph(cap_l, s["cap"])
            pg += [img, cap, Spacer(1, 4)]
        elif fig_r is not None:
            img = _fig_img(fig_r, width_in=W_HALF / inch)
            cap = Paragraph(cap_r, s["cap"])
            pg += [img, cap, Spacer(1, 4)]

        if fig_l2 is not None and fig_r2 is not None:
            pg += _fig_pair(fig_l2, cap_l2, fig_r2, cap_r2, s)
        elif fig_l2 is not None:
            img = _fig_img(fig_l2, width_in=W_HALF / inch)
            cap = Paragraph(cap_l2, s["cap"])
            pg += [img, cap, Spacer(1, 4)]

        pg.append(PageBreak())
        return pg

    test_story = []

    # ── DROP JUMP ─────────────────────────────────────────────────────────────
    dj = test_results.get("drop_jump")
    if dj:
        sigs = all_signals.get("drop_jump", {})
        rows = []

        if dj.grf_overall:
            o = dj.grf_overall
            _reg("DJ Peak GRF",   o.surgical.peak_force_N,   o.non_surgical.peak_force_N,   o.lsi_peak,    "Force", "peak_grf_lsi")
            _reg("DJ RFD 100ms",  o.surgical.rfd_Ns,         o.non_surgical.rfd_Ns,         o.lsi_rfd,     "Force", "rfd_lsi")
            rows += [
                ("Peak GRF (N)",       o.surgical.peak_force_N,  o.non_surgical.peak_force_N,  o.lsi_peak),
                ("Impulse (N·s)",      o.surgical.impulse_Ns,    o.non_surgical.impulse_Ns,    o.lsi_impulse),
                ("RFD 100ms (N/s)",    o.surgical.rfd_Ns,        o.non_surgical.rfd_Ns,        o.lsi_rfd),
            ]
        _reg("DJ Load Rate",  dj.loading_rate_surg_ns, dj.loading_rate_ns_ns,  dj.loading_rate_lsi, "Force")
        _reg("DJ Land Sym",   None,                    None,                   dj.landing_lsi_200ms, "Force", "landing_lsi")
        rows += [
            ("Loading Rate 50ms (N/s)", dj.loading_rate_surg_ns,  dj.loading_rate_ns_ns,  dj.loading_rate_lsi),
            ("Landing Sym 200ms (%)",   "—",                      "—",                    dj.landing_lsi_200ms),
            ("Impact Transient (N)",    dj.impact_transient_surg, dj.impact_transient_ns, dj.impact_transient_lsi),
            ("Time-to-Peak (ms)",       dj.time_to_peak_surg_ms,  dj.time_to_peak_ns_ms,  None),
            ("RSI",                     _fmt(dj.rsi, 3),          "—",                    None),
            ("Contact Time (s)",        _fmt(dj.contact_time_s, 3), "—",                  None),
        ]
        if dj.kinematics:
            k = dj.kinematics
            _reg("DJ Knee Flexion", k.surg_knee.peak_flexion_deg, k.non_surg_knee.peak_flexion_deg, k.lsi_peak_flexion, "Kinematics")
            rows += [
                ("Peak Knee Flexion (°)", k.surg_knee.peak_flexion_deg, k.non_surg_knee.peak_flexion_deg, k.lsi_peak_flexion),
                ("Peak Knee Valgus (°)",  k.surg_knee.peak_valgus_deg,  k.non_surg_knee.peak_valgus_deg,  k.lsi_peak_valgus),
            ]
        if dj.rsi == dj.rsi:
            _rtr["drop_jump_rsi"] = dj.rsi

        rsi = dj.rsi if dj.rsi == dj.rsi else None
        ll  = dj.landing_lsi_200ms if dj.landing_lsi_200ms == dj.landing_lsi_200ms else None
        interp = []
        if rsi:
            interp.append(
                f"<b>RSI {rsi:.2f}</b> ({'✓ meets' if rsi >= 1.3 else '⚠ below'} RTR target ≥ 1.30). "
                + ("Reactive capacity is adequate." if rsi >= 1.3
                   else "Stiff-landing pattern likely — progressive plyometric loading indicated."))
        if ll:
            interp.append(
                f"<b>Landing symmetry {ll:.0f}% LSI</b>: "
                + ("Good bilateral loading at contact." if 90 <= ll <= 110
                   else f"Surgical limb loading {abs(ll-100):.0f}% {'more' if ll>100 else 'less'} than non-surgical — {'protective offloading of NS side detected' if ll>100 else 'surgical guarding detected'}.")
            )
        interp.append("<b>Targets:</b> RSI ≥ 1.30 · Landing LSI 90–110% · Load rate LSI ≥ 90% · Knee flexion ≥ 60° · Valgus < 5°.")

        # Figures: GRF full width, flex+valgus as pair
        grf_fig  = fig_lib.force_time_curve(sigs.get("fz_surg"), sigs.get("fz_ns"), rate_grf,
                      "Drop Jump — Bilateral Vertical GRF", bw_n, figsize=(7.5, 2.2)) \
                   if sigs.get("fz_surg") is not None else None
        flex_fig = fig_lib.joint_angle_overlay(sigs["knee_flex_surg"], sigs.get("knee_flex_ns"),
                      rate_kin, "Knee Flexion", "Flexion (°)", figsize=(3.5, 2.2)) \
                   if sigs.get("knee_flex_surg") is not None else None
        valg_fig = fig_lib.joint_angle_overlay(sigs["knee_valgus_surg"], sigs.get("knee_valgus_ns"),
                      rate_kin, "Knee Valgus", "Valgus (°)", figsize=(3.5, 2.2)) \
                   if sigs.get("knee_valgus_surg") is not None else None

        test_story += _test_page("DROP JUMP", rows, interp,
            fig_full=grf_fig,  cap_full="Bilateral GRF (× body weight). Orange = surgical, Blue = non-surgical.",
            fig_l=flex_fig,    cap_l="Knee flexion. Target ≥ 60° at contact.",
            fig_r=valg_fig,    cap_r="Knee valgus. Target < 5°. Elevated valgus = ACL re-injury risk.")

    # ── DROP LANDING ──────────────────────────────────────────────────────────
    dl = test_results.get("drop_landing")
    if dl:
        sigs = all_signals.get("drop_landing", {})
        rows = []
        if dl.grf_overall:
            o = dl.grf_overall
            _reg("DL Peak GRF", o.surgical.peak_force_N, o.non_surgical.peak_force_N, o.lsi_peak, "Force")
            rows += [
                ("Peak GRF (N)",       o.surgical.peak_force_N,  o.non_surgical.peak_force_N,  o.lsi_peak),
                ("Impulse (N·s)",      o.surgical.impulse_Ns,    o.non_surgical.impulse_Ns,    o.lsi_impulse),
            ]
        _reg("DL Load Rate",  dl.loading_rate_surg_ns, dl.loading_rate_ns_ns,  dl.loading_rate_lsi, "Force")
        _reg("DL Impact",     dl.impact_transient_surg, dl.impact_transient_ns, dl.impact_transient_lsi, "Force")
        rows += [
            ("Loading Rate 50ms (N/s)", dl.loading_rate_surg_ns,  dl.loading_rate_ns_ns,  dl.loading_rate_lsi),
            ("Impact Transient (N)",    dl.impact_transient_surg, dl.impact_transient_ns, dl.impact_transient_lsi),
        ]
        if dl.kinematics:
            k = dl.kinematics
            rows += [
                ("Peak Knee Flexion (°)", k.surg_knee.peak_flexion_deg, k.non_surg_knee.peak_flexion_deg, k.lsi_peak_flexion),
                ("Peak Knee Valgus (°)",  k.surg_knee.peak_valgus_deg,  k.non_surg_knee.peak_valgus_deg,  k.lsi_peak_valgus),
            ]

        lr  = dl.loading_rate_lsi  if dl.loading_rate_lsi  == dl.loading_rate_lsi  else None
        imp = dl.impact_transient_lsi if dl.impact_transient_lsi == dl.impact_transient_lsi else None
        interp = [
            "Absorptive landing — assesses shock attenuation without a re-jump. Lower, more gradual force = better control.",
        ]
        if lr:
            interp.append(
                f"<b>Load rate LSI {lr:.0f}%</b>: "
                + ("Symmetric loading strategy." if 90 <= lr <= 110
                   else f"Surgical limb loading rate {abs(lr-100):.0f}% {'higher' if lr>100 else 'lower'} — {'NS side protective offloading' if lr>100 else 'surgical inhibition'}."))
        if imp:
            interp.append(f"<b>Impact transient LSI {imp:.0f}%</b>. Target: < 2× BW peak, LSI 90–110%, no sharp spike.")
        interp.append("<b>Targets:</b> Peak GRF < 3× BW · Load rate LSI 90–110% · Knee flexion ≥ 60° · Valgus < 5°.")

        grf_fig  = fig_lib.force_time_curve(sigs.get("fz_surg"), sigs.get("fz_ns"), rate_grf,
                      "Drop Landing — Bilateral GRF", bw_n, figsize=(7.5, 2.2)) \
                   if sigs.get("fz_surg") is not None else None
        flex_fig = fig_lib.joint_angle_overlay(sigs["knee_flex_surg"], sigs.get("knee_flex_ns"),
                      rate_kin, "Knee Flexion", "Flexion (°)", figsize=(3.5, 2.2)) \
                   if sigs.get("knee_flex_surg") is not None else None
        valg_fig = fig_lib.joint_angle_overlay(sigs["knee_valgus_surg"], sigs.get("knee_valgus_ns"),
                      rate_kin, "Knee Valgus", "Valgus (°)", figsize=(3.5, 2.2)) \
                   if sigs.get("knee_valgus_surg") is not None else None

        test_story += _test_page("DROP LANDING", rows, interp,
            fig_full=grf_fig,  cap_full="Bilateral GRF. Smooth progressive rise = good shock absorption. Sharp spike = stiff landing.",
            fig_l=flex_fig,    cap_l="Knee flexion at landing.",
            fig_r=valg_fig,    cap_r="Knee valgus at landing. > 8° surgical = elevated ACL stress.")

    # ── MAX VERTICAL JUMP ─────────────────────────────────────────────────────
    mvj = test_results.get("max_vertical")
    if mvj:
        sigs = all_signals.get("max_vertical", {})
        rows = []
        if mvj.grf_concentric:
            c = mvj.grf_concentric
            _reg("MVJ Concentric",  c.surgical.peak_force_N, c.non_surgical.peak_force_N, c.lsi_peak,    "Force")
            _reg("MVJ Impulse",     c.surgical.impulse_Ns,   c.non_surgical.impulse_Ns,   c.lsi_impulse, "Force")
            rows += [
                ("Peak Concentric GRF (N)",  c.surgical.peak_force_N, c.non_surgical.peak_force_N, c.lsi_peak),
                ("Propulsion Impulse (N·s)", c.surgical.impulse_Ns,   c.non_surgical.impulse_Ns,   c.lsi_impulse),
            ]
        _reg("MVJ Peak Force",  mvj.peak_force_surg_N, mvj.peak_force_ns_N, mvj.peak_force_lsi, "Force", "peak_grf_lsi")
        rows += [
            ("Peak Force Surg / NS (N)", mvj.peak_force_surg_N, mvj.peak_force_ns_N, mvj.peak_force_lsi),
            ("Propulsion LSI (%)",       "—",                   "—",                 mvj.propulsion_lsi),
            ("Jump Height",              _fmt(mvj.jump_height_cm, 1), "—",           None),
            ("Flight Time (s)",          _fmt(mvj.flight_time_s, 3),  "—",           None),
            ("Unweighting Imp Surg/NS",  _fmt(mvj.unweighting_impulse_surg, 1),
                                         _fmt(mvj.unweighting_impulse_ns,   1), None),
        ]
        if mvj.kinematics:
            k = mvj.kinematics
            rows += [
                ("Peak Knee Flexion (°)", k.surg_knee.peak_flexion_deg, k.non_surg_knee.peak_flexion_deg, k.lsi_peak_flexion),
                ("Peak Knee Valgus (°)",  k.surg_knee.peak_valgus_deg,  k.non_surg_knee.peak_valgus_deg,  k.lsi_peak_valgus),
            ]

        pk  = mvj.peak_force_lsi   if mvj.peak_force_lsi   == mvj.peak_force_lsi   else None
        pro = mvj.propulsion_lsi   if mvj.propulsion_lsi   == mvj.propulsion_lsi   else None
        interp = ["Bilateral CMJ — sensitive to subtle propulsion deficits masked at lower intensities."]
        if pk:
            interp.append(
                f"<b>Peak force LSI {pk:.0f}%</b>: "
                + ("Symmetric bilateral output." if 90 <= pk <= 110
                   else f"Surgical limb generating {abs(pk-100):.0f}% {'more' if pk>100 else 'less'} force — {'possible NS compensation' if pk>100 else 'quad/hip extensor deficit; strength work required'}."))
        if pro:
            interp.append(f"<b>Propulsion LSI {pro:.0f}%</b>. Targets: ≥ 90% propulsion, ≥ 90% peak force, matched unweighting impulse.")

        grf_fig  = fig_lib.force_time_curve(sigs.get("fz_surg"), sigs.get("fz_ns"), rate_grf,
                      "Max Vertical Jump — Bilateral GRF", bw_n, figsize=(7.5, 2.2)) \
                   if sigs.get("fz_surg") is not None else None
        flex_fig = fig_lib.joint_angle_overlay(sigs.get("knee_flex_surg"), sigs.get("knee_flex_ns"),
                      rate_kin, "Knee Flexion During CMJ", "Flexion (°)", figsize=(3.5, 2.2)) \
                   if sigs.get("knee_flex_surg") is not None else None

        test_story += _test_page("MAX VERTICAL JUMP (CMJ)", rows, interp,
            fig_full=grf_fig,  cap_full="Full CMJ cycle: unweighting → eccentric → propulsion → flight. Propulsive peaks should be symmetric.",
            fig_l=flex_fig,    cap_l="Knee flexion depth during CMJ. Asymmetric depth = compensatory strategy.",
            fig_r=None,        cap_r="")

    # ── ENDURANCE SQUAT ───────────────────────────────────────────────────────
    esq = test_results.get("endurance_squat")
    if esq:
        sigs = all_signals.get("endurance_squat", {})
        _reg("Endurance LSI", esq.mean_lsi_peak, None, esq.mean_lsi_peak, "Endurance", "endurance_lsi")
        if esq.fatigue_drift_pct == esq.fatigue_drift_pct:
            _rtr["fatigue_drift"] = esq.fatigue_drift_pct
        rows = [
            ("Mean Peak Force LSI (%)",  esq.mean_lsi_peak,     "—", esq.mean_lsi_peak),
            ("LSI First 10s (%)",        esq.lsi_first_third,   "—", esq.lsi_first_third),
            ("LSI Last 10s (%)",         esq.lsi_last_third,    "—", esq.lsi_last_third),
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

        test_story += _test_page("ENDURANCE SQUAT (30s)", rows, interp,
            fig_l=lsi_fig, cap_l="Per-rep LSI across 30s. Declining trend = neuromuscular fatigue breakdown.",
            fig_r=grf_fig, cap_r="Full 30s GRF trace. Force magnitude and rhythm across the set.")

    # ── SINGLE-LEG VERTICAL JUMP ──────────────────────────────────────────────
    slj = test_results.get("single_leg_jump")
    if slj and slj.surgical and slj.non_surgical:
        sv, nv = slj.surgical, slj.non_surgical
        sigs   = all_signals.get("single_leg_jump", {})
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
            ("Peak Knee Flexion (°)",    sv.peak_knee_flexion_deg, nv.peak_knee_flexion_deg, None),
            ("Peak Knee Valgus (°)",     sv.peak_valgus_deg,       nv.peak_valgus_deg,       None),
        ]
        jh  = slj.lsi_jump_height if slj.lsi_jump_height == slj.lsi_jump_height else None
        pk  = slj.lsi_peak_force  if slj.lsi_peak_force  == slj.lsi_peak_force  else None
        interp = ["Highest-demand test — isolates each limb, removing bilateral compensation. Strongest re-injury risk predictor."]
        if jh:
            interp.append(
                f"<b>Jump height LSI {jh:.0f}%</b>: "
                + ("✓ Meets RTR threshold (≥ 90%)." if jh >= 90
                   else f"{'⚠' if jh >= 75 else '✗'} {100-jh:.0f}% below non-surgical. {'Continued plyometric loading required' if jh >= 75 else 'RTR not recommended — significant power deficit'}."))
        if pk and pk < 90:
            interp.append(f"<b>Peak force LSI {pk:.0f}%</b>: Force deficit on surgical side — prioritise quad/hip extensor strengthening.")
        interp.append("<b>Targets:</b> Jump height LSI ≥ 90% · Peak GRF LSI ≥ 90% · Valgus < 5° on both limbs.")

        grf_fig  = fig_lib.force_time_curve(sigs.get("fz_surg"), sigs.get("fz_ns"), rate_grf,
                      "Single-Leg Jump — Surgical vs Non-Surgical GRF", bw_n, figsize=(7.5, 2.2)) \
                   if sigs.get("fz_surg") is not None or sigs.get("fz_ns") is not None else None
        flex_fig = fig_lib.joint_angle_overlay(sigs.get("knee_flex_surg"), sigs.get("knee_flex_ns"),
                      rate_kin, "Knee Flexion", "Flexion (°)", figsize=(3.5, 2.2)) \
                   if sigs.get("knee_flex_surg") is not None else None
        valg_fig = fig_lib.joint_angle_overlay(sigs.get("knee_valgus_surg"), sigs.get("knee_valgus_ns"),
                      rate_kin, "Knee Valgus", "Valgus (°)", figsize=(3.5, 2.2)) \
                   if sigs.get("knee_valgus_surg") is not None else None

        test_story += _test_page("SINGLE-LEG VERTICAL JUMP", rows, interp,
            fig_full=grf_fig, cap_full="Overlaid GRF for each limb tested separately. Propulsive peak height and shape should match.",
            fig_l=flex_fig,   cap_l="Knee flexion strategy per limb.",
            fig_r=valg_fig,   cap_r="Knee valgus per limb. Elevated surgical valgus under single-leg load = re-injury risk.")

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
                cap_full="COP trajectory — eyes closed. Tighter ellipse = better joint position sense. 95% confidence ellipse shown.")

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

    # ── Summary (built after test pages so all metrics are collected) ─────────
    domain_avg = {k: sum(v)/len(v) for k, v in domain_scores.items() if v}

    # Merge time modifier into metrics dict (scorer reads "_months_since_surgery")
    _metrics_for_score = dict(_rtr if _rtr else (rtr_metrics or {}))
    _metrics_for_score["_months_since_surgery"] = patient_data.get("months_since_surgery")
    rtr_result = compute_rtr_score(_metrics_for_score)

    summary = _exec_summary(s, patient_data, lsi_summary,
                             domain_avg, clinical_notes, rtr_result)

    doc.build(summary + [PageBreak()] + test_story)
    return output_path
