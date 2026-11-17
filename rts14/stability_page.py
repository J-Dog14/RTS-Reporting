"""
rts14/stability_page.py
=======================
The Ankle Stability report's "ANKLE STABILITY" section, inserted right after
the summary page:

  1. Stability criteria scorecard (PASS / PROGRESSING / DEFICIT / FAIL / NOT TESTED)
  2. How the ankle is controlling load — auto-written key findings
  3. Static balance by limb and surface (COP velocity + mediolateral sway)
  4. Single-leg landing: vertical force and ML shear settling (first 1 s)
  5. Single-leg landing: frontal-plane hip / knee / ankle + ankle sagittal (first 500 ms)
  6. Where the patient fell short — failed criteria ranked by severity
     (descriptive only: the report states results, it does not prescribe)

Single-leg landing metrics come from the landing phase of every single-leg
vertical trial (same plate, FP3). If a trial was cropped too soon after
landing, the affected metrics show NOT TESTED rather than a wrong number.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import Spacer, Table, TableStyle, KeepTogether, CondPageBreak

from config import COLORS
from helpers.figures import _style
from rts14.achilles_page import _num, _status, _fmt

GREEN, YELLOW, RED, GREY = COLORS["green"], COLORS["yellow"], COLORS["red"], "#AAAAAA"
SC, NC = COLORS["surgical"], COLORS["non_surg"]


def _inv(s, n):
    """Lower-is-better metric → symmetry % (100 = equal, < 100 = surgical worse)."""
    s, n = _num(s), _num(n)
    return 100.0 * n / s if s and n is not None else None


def _sym(s, n):
    s, n = _num(s), _num(n)
    return 100.0 * s / n if n and s is not None else None


def _band_status(v, lo=90, hi=110, lo2=80, hi2=120):
    v = _num(v)
    if v is None:
        return "NOT TESTED", GREY
    if lo <= v <= hi:
        return "PASS", GREEN
    if lo2 <= v <= hi2:
        return "PROGRESSING", YELLOW
    return "DEFICIT", RED


def _delta_status(s, n, ok=3.0, warn=5.0):
    s, n = _num(s), _num(n)
    if s is None or n is None:
        return "NOT TESTED", GREY
    d = abs(s - n)
    return ("PASS", GREEN) if d <= ok else (("PROGRESSING", YELLOW) if d <= warn else ("DEFICIT", RED))


def stab(all_signals):
    return ((all_signals or {}).get("single_leg_jump") or {}).get("stability") or {}


def _crv_rom(curves, key):
    y = (curves or {}).get(key)
    if y is None or np.all(np.isnan(y)):
        return None
    return float(np.nanmax(y) - np.nanmin(y))


# ─── metrics shared with the profile's score hook ────────────────────────────

def score_metrics(test_results, all_signals):
    """[(label, surg, ns, lsi, domain, rtr_key)] for the LSI chart / radar / score."""
    out = []
    st = stab(all_signals)
    s, n = st.get("s") or {}, st.get("n") or {}
    for key, label, rk in (("land_dpsi", "SL Landing — Dynamic Stability (DPSI)", "land_dpsi_lsi"),
                           ("land_mlsi", "SL Landing — ML Stability", "land_mlsi_lsi"),
                           ("land_tts_ml", "SL Landing — Time to Stabilise (ML)", "land_tts_lsi")):
        v = _inv(s.get(key), n.get(key))
        if v is not None:
            out.append((label, s.get(key), n.get(key), v, "Landing", rk))
    prop = (test_results or {}).get("proprioception")
    for cond, tag, rk in ((getattr(prop, "standard", None), "Firm", "balance_ml_lsi"),
                          (getattr(prop, "airex", None), "Airex", "balance_ml_airex_lsi")):
        if cond is None or cond.surgical is None or cond.non_surgical is None:
            continue
        a, b = cond.surgical.range_ml_mm, cond.non_surgical.range_ml_mm
        v = _inv(a, b)
        if getattr(cond.surgical, "failed", False) and v is not None:
            v = min(v, 50.0)                 # failed stance hold caps the score, same rule as COP velocity
        if v is not None:
            out.append((f"Balance ({tag}) — ML Sway", a, b, v, "Balance", rk))
    return out


def collect(test_results, all_signals, patient_data):
    T, S = test_results or {}, all_signals or {}
    crit = []

    def add(group, name, sv, nv, lsi_v, target, status, note=""):
        crit.append(dict(group=group, name=name, sv=sv, nv=nv, lsi=lsi_v,
                         target=target, status=status[0], color=status[1], note=note))

    # ── Static balance ───────────────────────────────────────────────────────
    prop = T.get("proprioception")
    for cond, label in ((getattr(prop, "standard", None), "eyes closed, firm"),
                        (getattr(prop, "airex", None), "eyes closed, Airex")):
        if cond is None or cond.surgical is None:
            add("Static balance", f"Stance hold — {label}", "—", "—", None, "hold", ("NOT TESTED", GREY),
                "not collected")
            continue
        sf = bool(getattr(cond.surgical, "failed", False))
        nf = bool(getattr(cond.non_surgical, "failed", False)) if cond.non_surgical else False
        add("Static balance", f"Stance hold — {label}", "FAIL" if sf else "hold", "FAIL" if nf else "hold",
            None, "hold, no touchdown", ("FAIL", RED) if sf else ("PASS", GREEN),
            "both limbs failed" if (sf and nf) else ("non-surgical failed" if nf else ""))
        vs = cond.surgical.mean_velocity_mm_s
        vn = cond.non_surgical.mean_velocity_mm_s if cond.non_surgical else None
        lv = cond.lsi_cop_velocity
        add("Static balance", f"COP velocity — {label}", _fmt(vs, 1, " mm/s"), _fmt(vn, 1, " mm/s"),
            lv, "LSI ≥ 90%", _status(lv, 90, 75))
        ms_ = cond.surgical.range_ml_mm
        mn_ = cond.non_surgical.range_ml_mm if cond.non_surgical else None
        ml = _inv(ms_, mn_)
        add("Static balance", f"Side-to-side (ML) sway — {label}", _fmt(ms_, 0, " mm"), _fmt(mn_, 0, " mm"),
            ml, "LSI ≥ 90%", _status(ml, 90, 75))

    # ── Single-leg landing ───────────────────────────────────────────────────
    st = stab(S)
    s, n = st.get("s") or {}, st.get("n") or {}
    win = _num(st.get("window_s"))
    wtxt = f"{win:.1f} s" if win else "—"
    cens = max(_num(s.get("land_tts_censored")) or 0, _num(n.get("land_tts_censored")) or 0)
    for key, name, d, unit, target in (
            ("land_dpsi", f"Dynamic stability index (DPSI, first {wtxt})", 3, "", "LSI ≥ 90%"),
            ("land_mlsi", "Mediolateral stability index", 3, "", "LSI ≥ 90%"),
            ("land_tts_ml", "Time to stabilise — side-to-side", 2, " s", "LSI ≥ 90%"),
            ("land_tts_v", "Time to stabilise — vertical", 2, " s", "LSI ≥ 90%")):
        v = _inv(s.get(key), n.get(key))
        note = ""
        miss = [lab for lab, dd in (("surgical", s), ("non-surgical", n)) if _num(dd.get(key)) is None]
        if v is None and miss:
            parts = []
            for lab, dd in (("surgical", s), ("non-surgical", n)):
                if lab in miss:
                    w_ = _num(dd.get("land_window_s"))
                    parts.append(f"{lab}: " + (f"only {w_:.2f} s after landing" if w_ is not None
                                               else "landing not detected"))
            note = "; ".join(parts)
        elif key.startswith("land_tts") and cens and v is not None:
            note = f"some trials still settling at {wtxt} — values are minimums"
        add("Single-leg landing", name, _fmt(s.get(key), d, unit), _fmt(n.get(key), d, unit), v, target,
            _status(v, 90, 75), note)
    pk = _sym(s.get("land_peak_bw"), n.get("land_peak_bw"))
    _lnote = next((c["note"] for c in crit if c["group"] == "Single-leg landing" and c["note"]
                   and "after landing" in c["note"] or "not detected" in (c["note"] or "")), "")
    add("Single-leg landing", "Landing peak force", _fmt(s.get("land_peak_bw"), 2, "×BW"),
        _fmt(n.get("land_peak_bw"), 2, "×BW"), pk, "LSI 90–110%", _band_status(pk), "" if pk is not None else _lnote)
    add("Single-leg landing", "Ankle frontal excursion (first 200 ms)", _fmt(s.get("land_ankle_fr_exc"), 1, "°"),
        _fmt(n.get("land_ankle_fr_exc"), 1, "°"), None, "Δ ≤ 3°",
        _delta_status(s.get("land_ankle_fr_exc"), n.get("land_ankle_fr_exc")),
        "" if _num(s.get("land_ankle_fr_exc")) is not None and _num(n.get("land_ankle_fr_exc")) is not None else _lnote)

    # ── Bilateral landing / power ────────────────────────────────────────────
    for test, label in (("drop_landing", "Drop landing"), ("drop_jump", "Drop jump")):
        c = (S.get(test) or {}).get("curves") or {}
        rs, rn = _crv_rom(c.get("s"), "ankle_fr"), _crv_rom(c.get("n"), "ankle_fr")
        if rs is not None and rn is not None:
            add("Bilateral landing", f"{label} — ankle frontal range", _fmt(rs, 1, "°"), _fmt(rn, 1, "°"),
                None, "Δ ≤ 3°", _delta_status(rs, rn))
    dl = T.get("drop_landing")
    if dl:
        add("Bilateral landing", "Drop landing — loading rate", _fmt(dl.loading_rate_surg_ns, 0, " N/s"),
            _fmt(dl.loading_rate_ns_ns, 0, " N/s"), dl.loading_rate_lsi, "LSI ≥ 85%",
            _status(dl.loading_rate_lsi, 85, 70))
    slj = T.get("single_leg_jump")
    if slj:
        add("Single-leg power", "SL jump height", _fmt(slj.surgical.jump_height_cm, 1, " in"),
            _fmt(slj.non_surgical.jump_height_cm, 1, " in"), slj.lsi_jump_height, "LSI ≥ 90%",
            _status(slj.lsi_jump_height, 90, 75))
    th = (patient_data or {}).get("triple_hop") or {}
    if th.get("lsi") is not None:
        add("Single-leg power", "Triple hop for distance", "", "", th["lsi"], "LSI ≥ 90%",
            _status(th["lsi"], 90, 75))
    return crit


# ─── findings (descriptive) ──────────────────────────────────────────────────

def findings(test_results, all_signals, crit):
    out = []
    by = {c["name"]: c for c in crit}
    fails = [c for c in crit if c["group"] == "Static balance" and c["status"] == "FAIL"]
    if fails:
        both = [c for c in fails if c["note"] == "both limbs failed"]
        out.append(f"<b>Static balance failed.</b> The surgical limb could not hold eyes-closed single-leg stance "
                   f"in {len(fails)} of 2 conditions"
                   + (" — the non-surgical limb failed as well, so the deficit is bilateral rather than only "
                      "an asymmetry (a common finding with ankle instability)." if both else "."))
    ml = [c for c in crit if c["name"].startswith("Side-to-side (ML) sway") and _num(c["lsi"]) is not None]
    vel = [c for c in crit if c["name"].startswith("COP velocity") and _num(c["lsi"]) is not None]
    if ml and vel:
        mlo, vlo = min(_num(c["lsi"]) for c in ml), min(_num(c["lsi"]) for c in vel)
        if mlo < vlo - 10 and mlo < 90:
            out.append(f"<b>The sway is side-to-side.</b> Mediolateral sway symmetry ({mlo:.0f}%) is lower than "
                       f"overall COP-velocity symmetry ({vlo:.0f}%) — the surgical limb's corrections are mainly in "
                       f"the frontal plane, the plane the lateral ligaments and peroneals control.")
    d = next((c for c in crit if c["name"].startswith("Dynamic stability index")), None)
    if d and _num(d["lsi"]) is not None and _num(d["lsi"]) < 90:
        out.append(f"<b>Landing takes longer to settle.</b> On the single-leg landing the surgical side's dynamic "
                   f"stability index is {d['sv']} vs {d['nv']} (symmetry {_num(d['lsi']):.0f}%) — more residual "
                   f"force fluctuation after touchdown.")
    a = by.get("Ankle frontal excursion (first 200 ms)")
    if a and a["status"] in ("DEFICIT", "PROGRESSING"):
        out.append(f"<b>Frontal-plane ankle motion differs at touchdown.</b> In the first 200 ms the surgical ankle "
                   f"moves {a['sv']} in the frontal plane vs {a['nv']} on the other side.")
    p = by.get("Landing peak force")
    if p and _num(p["lsi"]) is not None and _num(p["lsi"]) < 90:
        out.append(f"<b>Softer surgical landing.</b> Peak single-leg landing force is {_num(p['lsi']):.0f}% of the "
                   f"other side — the surgical limb is landing more cautiously.")
    h = by.get("SL jump height")
    if h and _num(h["lsi"]) is not None and _num(h["lsi"]) >= 90 and (fails or (d and _num(d["lsi"]) and _num(d["lsi"]) < 90)):
        out.append(f"<b>Power is not the limiter.</b> Single-leg jump height is {_num(h['lsi']):.0f}% of the other "
                   f"side, so the deficits above are in control and stability rather than force output.")
    return out


_MEANING = {
    "COP velocity": "more corrective sway on the surgical limb ({pct} symmetry)",
    "ML) sway": "more side-to-side sway on the surgical limb ({pct} symmetry)",
    "DPSI": "the surgical landing settles less well in the first second ({pct} symmetry)",
    "Mediolateral stability": "more side-to-side force fluctuation after landing ({pct} symmetry)",
    "Time to stabilise": "the surgical landing takes longer to settle ({pct} symmetry)",
    "Landing peak force": "peak landing force is {pct} of the other side",
    "frontal": "frontal-plane ankle motion differs between sides",
    "loading rate": "the surgical leg takes up load more slowly on landing — {pct} of the other side",
    "SL jump height": "single-leg jump height is {pct} of the other side",
    "Triple hop": "hop distance is {pct} of the other side",
}


def shortfalls(crit):
    bad = [c for c in crit if c["status"] in ("FAIL", "DEFICIT", "PROGRESSING")]

    def sev(c):
        if c["status"] == "FAIL":
            return -1e6
        v = _num(c["lsi"])
        if v is not None:
            return v if v <= 100 else 200 - v
        return 50.0 if c["status"] == "DEFICIT" else 80.0
    bad.sort(key=sev)
    out = []
    for c in bad:
        v = _num(c["lsi"])
        pct = f"{v:.0f}%" if v is not None else ""
        meaning = next((m for k, m in _MEANING.items() if k in c["name"]), "")
        meaning = meaning.format(pct=pct) if meaning else ""
        if c["status"] == "FAIL":
            who = "both limbs" if c["note"] == "both limbs failed" else "the surgical limb"
            meaning = f"{who} could not hold eyes-closed single-leg stance (foot moved or touched down)"
        if v is not None:
            val = f"symmetry {pct}" if c["group"] in ("Single-leg landing", "Static balance") else f"LSI {pct}"
        else:
            val = f"{c['sv']} vs {c['nv']}" if c["sv"] not in ("", "—", "FAIL", "hold") else (c["sv"] or "")
        lab = {"FAIL": "FAIL", "DEFICIT": "deficit", "PROGRESSING": "below target"}[c["status"]]
        out.append(f"<b>{c['name']}</b> — {val} ({lab}; target {c['target']})" + (f": {meaning}." if meaning else "."))
    for c in [c for c in crit if c["status"] == "NOT TESTED"]:
        why = c["note"] or "not collected in this session"
        out.append(f"<i>Not tested: {c['name'][0].lower() + c['name'][1:]} — {why}, so it is not represented "
                   f"in the score.</i>")
    return out


# ─── figures ─────────────────────────────────────────────────────────────────

def balance_bars(prop, figsize=(7.5, 2.4)):
    conds = [(c, lab) for c, lab in ((getattr(prop, "standard", None), "Firm"),
                                     (getattr(prop, "airex", None), "Airex")) if c is not None]
    if not conds:
        return None
    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor="white")
    for ax, (attr, title, unit) in zip(axes, (("mean_velocity_mm_s", "COP velocity", "mm/s"),
                                              ("range_ml_mm", "Side-to-side (ML) sway range", "mm"))):
        _style(ax, title=f"{title} — eyes closed", ylabel=unit)
        ax.title.set_fontsize(8.5)
        x = np.arange(len(conds))
        for off, tag, col, lab in ((-0.19, "surgical", SC, "Surgical"), (0.19, "non_surgical", NC, "Non-Surgical")):
            vals, failed = [], []
            for c, _ in conds:
                limb = getattr(c, tag, None)
                vals.append(_num(getattr(limb, attr, None)) or 0.0)
                failed.append(bool(getattr(limb, "failed", False)) if limb is not None else False)
            bars = ax.bar(x + off, vals, width=0.36, color=col, label=lab, edgecolor="white")
            for b, v, f in zip(bars, vals, failed):
                if v:
                    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}" + ("\nFAIL" if f else ""),
                            ha="center", va="bottom", fontsize=6.5, color=RED if f else COLORS["text"],
                            fontweight="bold" if f else "normal")
                if f:
                    b.set_hatch("///")
                    b.set_edgecolor("#FFFFFF")
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in conds], fontsize=8)
        top = ax.get_ylim()[1]
        ax.set_ylim(0, top * 1.22)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=6.5, loc="upper left", framealpha=0.6)
    fig.tight_layout(pad=0.7)
    return fig


def _curve_ax(ax, x, cs, csd, cn, cnsd, key):
    for m, sd, col, lab in ((cs, csd, SC, "Surgical"), (cn, cnsd, NC, "Non-Surgical")):
        y = (m or {}).get(key)
        if y is None or np.all(np.isnan(y)):
            continue
        ax.plot(x, y, color=col, linewidth=1.6, label=lab)
        e = (sd or {}).get(key)
        if e is not None and not np.all(np.isnan(e)):
            ax.fill_between(x, y - e, y + e, color=col, alpha=0.16, linewidth=0)


def landing_forces(st, figsize=(7.5, 2.4)):
    cs, csd, cn, cnsd = (st.get("curves_s"), st.get("curves_s_sd"), st.get("curves_n"), st.get("curves_n_sd"))
    if not any((c or {}).get("fz_bw") is not None for c in (cs, cn)):
        return None
    x = np.linspace(0, 1000 * (_num(st.get("window_s")) or 1.0), 101)
    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor="white", sharex=True)
    for ax, key, title, yl, ref in ((axes[0], "fz_bw", "Vertical force", "× body weight", 1.0),
                                    (axes[1], "fml_bw", "Side-to-side (ML) shear", "% BW (vs settled)", 0.0),
                                    (axes[2], "fap_bw", "Fore-aft (AP) shear", "% BW (vs settled)", 0.0)):
        _style(ax, title=title, ylabel=yl, xlabel="ms after touchdown")
        ax.title.set_fontsize(8.5)
        ax.axhline(ref, color="#BBBBBB", linewidth=0.7)
        if key == "fz_bw":
            ax.axhspan(0.95, 1.05, color="#DDDDDD", alpha=0.5, linewidth=0)
        else:
            ax.axhspan(-2.5, 2.5, color="#DDDDDD", alpha=0.5, linewidth=0)
        _curve_ax(ax, x, cs, csd, cn, cnsd, key)
        ax.tick_params(labelsize=6.5)
    axes[0].legend(fontsize=6.5, loc="upper right", framealpha=0.6)
    fig.tight_layout(pad=0.6)
    return fig


def landing_kinematics(st, figsize=(7.5, 2.3)):
    cs, csd, cn, cnsd = (st.get("curves_s"), st.get("curves_s_sd"), st.get("curves_n"), st.get("curves_n_sd"))
    if not any((c or {}).get("ankle_fr") is not None for c in (cs, cn)):
        return None
    x = np.linspace(0, 1000 * min(0.5, _num(st.get("window_s")) or 0.5), 101)
    fig, axes = plt.subplots(1, 4, figsize=figsize, facecolor="white", sharex=True)
    for ax, key, title, yl in ((axes[0], "ankle_fr", "Ankle frontal", "Δ from touchdown (°)"),
                               (axes[1], "knee_fr", "Knee frontal", ""),
                               (axes[2], "hip_fr", "Hip frontal", ""),
                               (axes[3], "ankle_ang", "Ankle sagittal", "+ dorsiflexion (°)")):
        _style(ax, title=title, ylabel=yl, xlabel="ms after touchdown")
        ax.title.set_fontsize(8.5)
        if key == "ankle_fr":
            ax.set_facecolor("#FFF6EF")
            ax.title.set_color(COLORS["accent"])
        ax.axhline(0, color="#CCCCCC", linewidth=0.6)
        ax.axvline(200, color="#999999", linewidth=0.7, linestyle=":")
        _curve_ax(ax, x, cs, csd, cn, cnsd, key)
        ax.tick_params(labelsize=6.5)
    axes[0].legend(fontsize=6, loc="best", framealpha=0.6)
    fig.tight_layout(pad=0.6)
    return fig


# ─── flowables ───────────────────────────────────────────────────────────────

def build(test_results, all_signals, patient_data, P, s, helpers, rate_kin):
    _sec, _fig_img, W_FULL = helpers["_sec"], helpers["_fig_img"], helpers["W_FULL"]
    from helpers.report import Paragraph
    crit = collect(test_results, all_signals, patient_data)
    story = _sec("ANKLE STABILITY — CRITERIA", s)

    scored = [c for c in crit if c["status"] != "NOT TESTED"]
    n_pass = sum(c["status"] == "PASS" for c in scored)
    hdr = ParagraphStyle("st_hdr", parent=s["body"], textColor=colors.white, fontName="Helvetica-Bold", fontSize=7.8)
    cell = ParagraphStyle("st_c", parent=s["body"], fontSize=7.6, leading=9.5)
    grp = ParagraphStyle("st_g", parent=s["body"], fontSize=7.2, textColor=colors.HexColor("#39414a"),
                         fontName="Helvetica-Bold")
    chip = ParagraphStyle("st_chip", parent=s["body"], fontSize=6.8, alignment=TA_CENTER,
                          textColor=colors.white, fontName="Helvetica-Bold", leading=8)
    data = [[Paragraph(h, hdr) for h in ("Criterion", "Surgical", "Non-Surg", "Sym.", "Target", "Status")]]
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(COLORS["header_bg"])),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(COLORS["accent"])),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.6),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor(COLORS["grid"])),
    ]
    last = None
    for c in crit:
        if c["group"] != last:
            data.append([Paragraph(c["group"].upper(), grp), "", "", "", "", ""])
            r = len(data) - 1
            style += [("SPAN", (0, r), (-1, r)), ("BACKGROUND", (0, r), (-1, r), colors.HexColor("#EEF1F4"))]
            last = c["group"]
        lv = _num(c["lsi"])
        name = c["name"] + (f' <font size="6" color="#999999"><i>({c["note"]})</i></font>' if c["note"] else "")
        data.append([Paragraph(name, cell), c["sv"], c["nv"], "—" if lv is None else f"{lv:.0f}%",
                     c["target"], Paragraph(c["status"], chip)])
        style.append(("BACKGROUND", (5, len(data) - 1), (5, len(data) - 1), colors.HexColor(c["color"])))
    tbl = Table(data, colWidths=[2.6 * inch, 1.05 * inch, 1.05 * inch, 0.55 * inch, 1.1 * inch, 1.0 * inch])
    tbl.setStyle(TableStyle(style))
    story.append(Paragraph(
        f"<b>{n_pass} of {len(scored)}</b> ankle-stability criteria met. Symmetry for sway, stability index and "
        f"time-to-stabilise is non-surgical ÷ surgical (lower is better for those), so 100% = equal and below 100% "
        f"= the surgical side is less stable.",
        ParagraphStyle("st_cnt", parent=s["body"], fontSize=8, spaceAfter=3)))
    story.append(tbl)
    story.append(Spacer(1, 6))

    f = findings(test_results, all_signals, crit)
    if f:
        story.append(helpers["_interp_box"](f, s, title="HOW THE ANKLE IS CONTROLLING LOAD"))
        story.append(Spacer(1, 6))

    prop = (test_results or {}).get("proprioception")
    fig = balance_bars(prop) if prop is not None else None
    if fig is not None:
        story.append(CondPageBreak(2.6 * inch))
        story.append(_fig_img(fig, W_FULL / inch))
        story.append(Paragraph("Eyes-closed single-leg stance. Hatched bars = the stance hold failed (foot moved or "
                               "the other foot touched down); the value is from the part of the trial that was held.",
                               s["cap"]))
        story.append(Spacer(1, 6))

    st = stab(all_signals)
    fig = landing_forces(st)
    if fig is not None:
        story.append(CondPageBreak(2.6 * inch))
        story.append(_fig_img(fig, W_FULL / inch))
        _w = _num(st.get("window_s"))
        story.append(Paragraph(f"Single-leg landing (from the single-leg vertical trials), first "
                               f"{(_w or 0):.1f} s after touchdown — the shortest recording across all trials, so "
                               f"both legs are compared over the same window. Mean ± SD of all trials. "
                               "Grey band = stable (vertical ± 5% BW, shear ± 2.5% BW). The longer and bigger the "
                               "excursions outside the band, the harder the limb is working to stabilise.", s["cap"]))
        story.append(Spacer(1, 6))
    fig = landing_kinematics(st)
    if fig is not None:
        story.append(CondPageBreak(2.5 * inch))
        story.append(_fig_img(fig, W_FULL / inch))
        story.append(Paragraph("Joint motion in the first 500 ms of the single-leg landing. Frontal-plane curves "
                               "show change from touchdown; dotted line = 200 ms (window for the ankle frontal "
                               "excursion criterion).", s["cap"]))
        story.append(Spacer(1, 6))

    pri = shortfalls(crit)
    if pri:
        sty = ParagraphStyle("st_pri", parent=s["interp"], fontSize=8, leading=11.5, leftIndent=8)
        rows = [[Paragraph("WHERE THE PATIENT FELL SHORT — MOST SEVERE FIRST", s["interp_hd"])]]
        i = 0
        for p in pri:
            if p.startswith("<i>"):
                rows.append([Paragraph(p, sty)])
            else:
                i += 1
                rows.append([Paragraph(f"{i}.&nbsp; {p}", sty)])
        t = Table(rows, colWidths=[W_FULL])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF6EF")),
            ("LINEABOVE", (0, 0), (-1, 0), 1.5, colors.HexColor(COLORS["accent"])),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(CondPageBreak(1.5 * inch))
        story.append(t)
    return story
