"""
rts14/achilles_page.py
======================
The Achilles report's "CALF & ANKLE" section, inserted right after the
summary page:

  1. Achilles criteria scorecard (PASS / PROGRESSING / DEFICIT / NOT TESTED)
  2. What the calf is doing — auto-written key findings
  3. Push-off work by joint (who does the work: hip, knee or ankle?)
  4. Single-leg jump ankle power + plantarflexor moment, aligned to take-off
  5. Ankle symmetry across tests (moment vs push-off work vs landing absorption)
  6. Where the patient fell short — failed criteria ranked by severity
     (descriptive only: the report states results, it does not prescribe)

Heel raise is optional: if it wasn't collected the row says NOT TESTED and
the score simply renormalises over the remaining criteria.
"""

import numpy as np
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import Spacer, Table, TableStyle, KeepTogether, CondPageBreak

from config import COLORS, LSI_GREEN, LSI_YELLOW
from rts14 import achilles_figs as AF

GREEN, YELLOW, RED, GREY = COLORS["green"], COLORS["yellow"], COLORS["red"], "#AAAAAA"


def _num(v):
    try:
        v = float(v)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def _row(sigs, label):
    for r in (sigs or {}).get("extra_rows") or []:
        if r[0] == label:
            return r
    return None


def _status(value, good, caution, higher_better=True):
    v = _num(value)
    if v is None:
        return "NOT TESTED", GREY
    if higher_better:
        if v >= good:
            return "PASS", GREEN
        if v >= caution:
            return "PROGRESSING", YELLOW
        return "DEFICIT", RED
    if v <= good:
        return "PASS", GREEN
    if v <= caution:
        return "PROGRESSING", YELLOW
    return "DEFICIT", RED


def _fmt(v, d=1, suf=""):
    v = _num(v)
    return "—" if v is None else f"{v:,.{d}f}{suf}"


def collect(test_results, all_signals, patient_data):
    """Pull every Achilles criterion into one list of dicts."""
    S = all_signals or {}
    T = test_results or {}
    crit = []

    def add(group, name, sv, nv, lsi_v, target, status, note=""):
        crit.append(dict(group=group, name=name, sv=sv, nv=nv, lsi=lsi_v,
                         target=target, status=status[0], color=status[1], note=note))

    # ── Calf capacity ─────────────────────────────────────────────────────────
    hr = ((patient_data or {}).get("manual_tests") or {}).get("heel_raise") or {}
    if hr.get("surg") is not None or hr.get("ns") is not None:
        add("Calf capacity", "Single-leg heel raise (reps)", _fmt(hr.get("surg"), 0), _fmt(hr.get("ns"), 0),
            hr.get("lsi"), "LSI ≥ 90%", _status(hr.get("lsi"), 90, 75))
    else:
        add("Calf capacity", "Single-leg heel raise (reps)", "—", "—", None, "LSI ≥ 90%",
            ("NOT TESTED", GREY), "not collected")
    sl = S.get("single_leg_jump", {})
    r = _row(sl, "Ankle +Work (J)")
    if r:
        add("Calf capacity", "SL jump — ankle push-off work", _fmt(r[1], 0, " J"), _fmt(r[2], 0, " J"),
            r[3], "LSI ≥ 90%", _status(r[3], 90, 75))
    am_s, am_n = sl.get("ankle_moment_surg"), sl.get("ankle_moment_ns")
    if _num(am_s) and _num(am_n):
        v = 100 * am_s / am_n
        add("Calf capacity", "SL jump — peak plantarflexor moment", _fmt(am_s, 0, " Nm"),
            _fmt(am_n, 0, " Nm"), v, "LSI ≥ 90%", _status(v, 90, 75))

    # ── Single-leg power ──────────────────────────────────────────────────────
    slj = T.get("single_leg_jump")
    if slj:
        add("Single-leg power", "SL jump height", _fmt(slj.surgical.jump_height_cm, 1, " in"),
            _fmt(slj.non_surgical.jump_height_cm, 1, " in"), slj.lsi_jump_height, "LSI ≥ 90%",
            _status(slj.lsi_jump_height, 90, 75))
    r = _row(sl, "Peak Power (W/kg)")
    if r:
        add("Single-leg power", "SL jump peak power", _fmt(r[1], 1, " W/kg"), _fmt(r[2], 1, " W/kg"),
            r[3], "LSI ≥ 90%", _status(r[3], 90, 75))
    cm = S.get("max_vertical", {})
    r = _row(cm, "Ankle +Work (J)")
    if r:
        add("Single-leg power", "CMJ — ankle push-off work", _fmt(r[1], 0, " J"), _fmt(r[2], 0, " J"),
            r[3], "LSI ≥ 90%", _status(r[3], 90, 75))

    # ── Reactive / rate ───────────────────────────────────────────────────────
    dj = T.get("drop_jump")
    if dj:
        add("Reactive strength", "Drop jump RSI (bilateral)", _fmt(dj.rsi, 2, " m/s"), "", None,
            "≥ 1.30 m/s", _status(dj.rsi, 1.30, 1.0))
        add("Reactive strength", "Drop jump contact time", _fmt(dj.contact_time_s, 2, " s"), "", None,
            "< 0.25 s", _status(dj.contact_time_s, 0.25, 0.35, higher_better=False))
        g = dj.grf_overall
        add("Reactive strength", "Drop jump RFD (100 ms)", _fmt(g.surgical.rfd_Ns, 0, " N/s"),
            _fmt(g.non_surgical.rfd_Ns, 0, " N/s"), g.lsi_rfd, "LSI ≥ 85%", _status(g.lsi_rfd, 85, 70))
        add("Reactive strength", "Drop jump loading rate (50 ms)", _fmt(dj.loading_rate_surg_ns, 0, " N/s"),
            _fmt(dj.loading_rate_ns_ns, 0, " N/s"), dj.loading_rate_lsi, "LSI ≥ 85%",
            _status(dj.loading_rate_lsi, 85, 70))

    # ── Landing ───────────────────────────────────────────────────────────────
    dls = S.get("drop_landing", {})
    r = _row(dls, "Ankle Absorption (J)")
    if r:
        add("Landing control", "Drop landing — ankle absorption", _fmt(r[1], 0, " J"), _fmt(r[2], 0, " J"),
            r[3], "LSI ≥ 90%", _status(r[3], 90, 75))
    dl = T.get("drop_landing")
    if dl:
        add("Landing control", "Drop landing — impact transient", _fmt(dl.impact_transient_surg, 0, " N"),
            _fmt(dl.impact_transient_ns, 0, " N"), dl.impact_transient_lsi, "LSI ≥ 90%",
            _status(dl.impact_transient_lsi, 90, 75))

    # ── Balance ───────────────────────────────────────────────────────────────
    prop = T.get("proprioception")
    for cond, label in ((getattr(prop, "standard", None), "Balance — eyes closed, firm"),
                        (getattr(prop, "airex", None), "Balance — eyes closed, Airex")):
        if cond is None or cond.surgical is None:
            continue
        sf = getattr(cond.surgical, "failed", False)
        nf = getattr(cond.non_surgical, "failed", False) if cond.non_surgical else False
        st = ("FAIL", RED) if sf else _status(cond.lsi_cop_velocity, 90, 75)
        add("Balance", label, "FAIL" if sf else "hold", "FAIL" if nf else "hold",
            cond.lsi_cop_velocity, "hold + LSI ≥ 90%", st,
            "both limbs failed" if (sf and nf) else "")

    # ── Hop (shown, not scored) ───────────────────────────────────────────────
    th = (patient_data or {}).get("triple_hop") or {}
    if th.get("lsi") is not None:
        add("Hop", "Triple hop for distance", "", "", th["lsi"], "LSI ≥ 90%",
            _status(th["lsi"], 90, 75), "hop distance normalises early")
    return crit


def findings(test_results, all_signals, crit):
    """Plain-language key findings, most important first."""
    S = all_signals or {}
    out = []
    sl = S.get("single_leg_jump", {})
    aw = _row(sl, "Ankle +Work (J)")
    hw = _row(sl, "Hip +Work (J)")
    am_s, am_n = _num(sl.get("ankle_moment_surg")), _num(sl.get("ankle_moment_ns"))
    aw_l = _num(aw[3]) if aw else None
    hw_l = _num(hw[3]) if hw else None
    if aw_l is not None and am_s and am_n:
        ml = 100 * am_s / am_n
        if ml >= 90 and aw_l < 80:
            out.append(f"<b>The force is there, the push-off isn't.</b> On the single-leg jump the surgical calf "
                       f"reaches {ml:.0f}% of the other side's peak plantarflexor moment but does only {aw_l:.0f}% "
                       f"of its push-off work. It can produce the moment, but it isn't driving the leg through "
                       f"most of the push-off (see the power curve below) — strength isn't the limiter, using it "
                       f"quickly through range is.")
        elif aw_l < 80:
            out.append(f"<b>Calf push-off deficit.</b> Surgical ankle push-off work on the single-leg jump is "
                       f"{aw_l:.0f}% of the non-surgical side (peak moment {ml:.0f}%).")
    if aw_l is not None and hw_l is not None and hw_l - aw_l >= 15:
        out.append(f"<b>The hip is covering for the calf.</b> Hip push-off work is {hw_l:.0f}% of the other side "
                   f"while the ankle is {aw_l:.0f}% — a hip-dominant strategy. Height and power look closer to "
                   f"symmetric than the calf really is.")
    dj = (test_results or {}).get("drop_jump")
    if dj:
        rfd, lr = _num(dj.grf_overall.lsi_rfd), _num(dj.loading_rate_lsi)
        pk = _num(dj.grf_overall.lsi_peak)
        if rfd is not None and lr is not None and min(rfd, lr) < 80:
            out.append(f"<b>Fast loading is avoided.</b> Drop jump RFD {rfd:.0f}% and loading rate {lr:.0f}% "
                       f"of the non-surgical side" + (f", against peak force {pk:.0f}%" if pk else "")
                       + ". The surgical leg accepts load slowly — the Achilles' job is the opposite.")
        ct = _num(dj.contact_time_s)
        if ct and ct > 0.35:
            out.append(f"<b>Not a reactive jump.</b> Drop jump contact is {ct:.2f} s (target < 0.25 s); "
                       f"RSI {(_num(dj.rsi) or 0):.2f} m/s comes from height, not from a fast stretch-shortening "
                       f"rebound.")
    dls = S.get("drop_landing", {})
    ab = _row(dls, "Ankle Absorption (J)")
    hab = _row(dls, "Hip Absorption (J)")
    if ab and _num(ab[3]) is not None and _num(ab[3]) < 80:
        txt = f"<b>Landing bypasses the ankle.</b> The surgical ankle absorbs {_num(ab[3]):.0f}% of the landing energy the other side does"
        if hab and _num(hab[3]) and _num(hab[3]) > 105:
            txt += f", while the surgical hip absorbs {_num(hab[3]):.0f}%"
        out.append(txt + ".")
    bal = [c for c in crit if c["group"] == "Balance" and c["status"] == "FAIL"]
    if bal:
        both = any(c["note"] == "both limbs failed" for c in bal)
        out.append(f"<b>Balance failed.</b> The surgical limb could not hold eyes-closed single-leg stance in "
                   f"{len(bal)} of 2 conditions" + (" — and the non-surgical limb failed too, so this is a "
                                                    "bilateral deficit, not only an asymmetry." if both else "."))
    th = [c for c in crit if c["group"] == "Hop"]
    if th and _num(th[0]["lsi"]) and _num(th[0]["lsi"]) >= 90 and aw_l is not None and aw_l < 80:
        out.append(f"<b>Hop distance hides it.</b> Triple hop is {_num(th[0]['lsi']):.0f}% — horizontal hopping "
                   f"can be driven by the hip, so it doesn't clear the calf.")
    return out


_MEANING = {
    "ankle push-off work": "the surgical calf contributes {pct} of the push-off work the other calf does",
    "peak plantarflexor moment": "peak calf (plantarflexor) moment is {pct} of the other side",
    "SL jump height": "single-leg jump height is {pct} of the other side",
    "SL jump peak power": "single-leg peak power is {pct} of the other side",
    "RSI": "reactive strength is below the 1.30 m/s target",
    "contact time": "ground contact is too long to be a reactive (stretch-shortening) rebound",
    "RFD": "the surgical leg develops force more slowly — {pct} of the other side",
    "loading rate": "the surgical leg takes up load more slowly on landing — {pct} of the other side",
    "ankle absorption": "the surgical ankle absorbs {pct} of the landing energy the other ankle does",
    "impact transient": "initial landing impact on the surgical leg is {pct} of the other side",
    "Balance": "could not hold eyes-closed single-leg stance",
}


def shortfalls(crit):
    """Failed / deficit criteria, most severe first, each with a one-line meaning."""
    bad = [c for c in crit if c["status"] in ("FAIL", "DEFICIT", "PROGRESSING") and c["group"] != "Hop"]

    def severity(c):
        if c["status"] == "FAIL":
            return -1e6
        v = _num(c["lsi"])
        if v is not None:
            return v
        return 50.0 if c["status"] == "DEFICIT" else 80.0
    bad.sort(key=severity)
    out = []
    for c in bad:
        v = _num(c["lsi"])
        pct = f"{v:.0f}%" if v is not None else ""
        meaning = next((m for k, m in _MEANING.items() if k in c["name"]), "")
        meaning = meaning.format(pct=pct) if meaning else ""
        if c["group"] == "Balance":
            who = "both limbs" if c["note"] == "both limbs failed" else "the surgical limb"
            meaning = f"{who} could not hold eyes-closed single-leg stance (foot moved or touched down)"
        val = (f"LSI {pct}" if v is not None else (c["sv"] or "")).strip()
        lab = {"FAIL": "FAIL", "DEFICIT": "deficit", "PROGRESSING": "below target"}[c["status"]]
        out.append(f"<b>{c['name']}</b> — {val} ({lab}; target {c['target']})"
                   + (f": {meaning}." if meaning else "."))
    nt = [c for c in crit if c["status"] == "NOT TESTED"]
    for c in nt:
        out.append(f"<i>Not tested: {c['name'].split(' (')[0].lower()} — not collected in this session, so it is "
                   f"not represented in the score.</i>")
    return out


# ─── flowables ───────────────────────────────────────────────────────────────

def build(test_results, all_signals, patient_data, P, s, helpers, rate_kin):
    """Return the CALF & ANKLE section flowables."""
    _sec, _interp_box, _fig_img, W_FULL, W_HALF = (helpers["_sec"], helpers["_interp_box"],
                                                   helpers["_fig_img"], helpers["W_FULL"], helpers["W_HALF"])
    from helpers.report import Paragraph   # glyph-safe Paragraph
    crit = collect(test_results, all_signals, patient_data)
    has_hr = any(c["name"].startswith("Single-leg heel raise") and c["status"] != "NOT TESTED" for c in crit)
    story = []
    story += _sec("CALF & ANKLE — ACHILLES CRITERIA", s)

    # ── scorecard ────────────────────────────────────────────────────────────
    scored = [c for c in crit if c["status"] != "NOT TESTED" and c["group"] != "Hop"]
    n_pass = sum(c["status"] == "PASS" for c in scored)
    hdr = ParagraphStyle("ach_hdr", parent=s["body"], textColor=colors.white, fontName="Helvetica-Bold",
                         fontSize=7.8)
    cell = ParagraphStyle("ach_c", parent=s["body"], fontSize=7.6, leading=9.5)
    grp = ParagraphStyle("ach_g", parent=s["body"], fontSize=7.2, textColor=colors.HexColor("#39414a"),
                         fontName="Helvetica-Bold")
    chip = ParagraphStyle("ach_chip", parent=s["body"], fontSize=6.8, alignment=TA_CENTER,
                          textColor=colors.white, fontName="Helvetica-Bold", leading=8)
    data = [[Paragraph(h, hdr) for h in ("Criterion", "Surgical", "Non-Surg", "LSI", "Target", "Status")]]
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(COLORS["header_bg"])),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(COLORS["accent"])),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.6),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor(COLORS["grid"])),
    ]
    last_group = None
    for c in crit:
        if c["group"] != last_group:
            data.append([Paragraph(c["group"].upper(), grp), "", "", "", "", ""])
            r = len(data) - 1
            style += [("SPAN", (0, r), (-1, r)), ("BACKGROUND", (0, r), (-1, r), colors.HexColor("#EEF1F4"))]
            last_group = c["group"]
        lv = _num(c["lsi"])
        name = c["name"] + (f' <font size="6" color="#999999"><i>({c["note"]})</i></font>' if c["note"] else "")
        data.append([Paragraph(name, cell), c["sv"], c["nv"], "—" if lv is None else f"{lv:.0f}%",
                     c["target"], Paragraph(c["status"], chip)])
        r = len(data) - 1
        style.append(("BACKGROUND", (5, r), (5, r), colors.HexColor(c["color"])))
    tbl = Table(data, colWidths=[2.55 * inch, 1.05 * inch, 1.05 * inch, 0.6 * inch, 1.1 * inch, 1.0 * inch])
    tbl.setStyle(TableStyle(style))
    story.append(Paragraph(
        f"<b>{n_pass} of {len(scored)}</b> Achilles criteria met"
        + ("" if has_hr else " · heel raise not collected (score uses the remaining criteria)"),
        ParagraphStyle("ach_cnt", parent=s["body"], fontSize=8.5, spaceAfter=3)))
    story.append(tbl)
    story.append(Spacer(1, 6))

    # ── key findings ─────────────────────────────────────────────────────────
    f = findings(test_results, all_signals, crit)
    if f:
        box = helpers["_interp_box"](f, s, title="WHAT THE CALF IS DOING")
        story.append(box)
        story.append(Spacer(1, 6))

    # ── who does the push-off work ───────────────────────────────────────────
    S = all_signals or {}
    tests = [(lab, S[k]["joint_work"]) for k, lab in (("single_leg_jump", "SL jump"),
                                                     ("max_vertical", "CMJ"), ("drop_jump", "Drop jump"))
             if k in S and S[k].get("joint_work")]
    if tests:
        story.append(CondPageBreak(2.5 * inch))
        story.append(_fig_img(AF.joint_work_stacked(tests, figsize=(7.5, 2.3)), W_FULL / inch))
        story.append(Paragraph("Push-off (positive) joint work per leg. Ankle share in orange — a smaller ankle "
                               "share on the surgical side with a bigger hip share = the hip is doing the calf's job.",
                               s["cap"]))
        story.append(Spacer(1, 6))

    # ── SLJ ankle curves aligned to take-off ─────────────────────────────────
    sl = S.get("single_leg_jump", {})
    if sl.get("ankle_pow_ts_surg") is not None:
        f1 = AF.aligned_curves(sl.get("ankle_pow_ts_surg"), sl.get("ankle_pow_ts_ns"), rate_kin,
                               sl.get("takeoff_k_surg"), sl.get("takeoff_k_ns"),
                               "SL Jump — Ankle Power", "Power (W)", unit=" W")
        f2 = AF.aligned_curves(sl.get("ankle_pf_mom_ts_surg"), sl.get("ankle_pf_mom_ts_ns"), rate_kin,
                               sl.get("takeoff_k_surg"), sl.get("takeoff_k_ns"),
                               "SL Jump — Plantarflexor Moment", "Moment (Nm)", unit=" Nm")
        story.append(CondPageBreak(2.8 * inch))
        story += helpers["_fig_pair"](f1, "Ankle power through push-off. Area above zero = push-off work.",
                                      f2, "Plantarflexor moment. Similar height but less power = slow calf.", s)
        story.append(Spacer(1, 4))

    # ── ankle symmetry across tests ──────────────────────────────────────────
    def lsi_of(test, label):
        r = _row(S.get(test, {}), label)
        return _num(r[3]) if r else None

    def pf(test):
        d = S.get(test, {})
        a, b = _num(d.get("ankle_moment_surg")), _num(d.get("ankle_moment_ns"))
        return 100 * a / b if a and b else None
    lsi_data = {
        "Peak PF moment": {"SL jump": pf("single_leg_jump"), "CMJ": pf("max_vertical"),
                           "Drop jump": pf("drop_jump"), "Landing": pf("drop_landing")},
        "Push-off work": {"SL jump": lsi_of("single_leg_jump", "Ankle +Work (J)"),
                          "CMJ": lsi_of("max_vertical", "Ankle +Work (J)"),
                          "Drop jump": lsi_of("drop_jump", "Ankle +Work (J)")},
        "Landing absorption": {"Landing": lsi_of("drop_landing", "Ankle Absorption (J)"),
                               "Drop jump": lsi_of("drop_jump", "Ankle Absorption (J)")},
    }
    if any(v is not None for d in lsi_data.values() for v in d.values()):
        story.append(CondPageBreak(2.8 * inch))
        story.append(_fig_img(AF.ankle_lsi_by_test(lsi_data, figsize=(7.5, 2.9)), W_FULL / inch))
        story.append(Paragraph("Each line runs from 100% (symmetric) to the surgical ankle's value. Peak moment "
                               "near 100% with push-off work far below it = the calf has strength but not speed.",
                               s["cap"]))
        story.append(Spacer(1, 6))

    # ── where the patient fell short (descriptive) ───────────────────────────
    pri = shortfalls(crit)
    if pri:
        st = ParagraphStyle("ach_pri", parent=s["interp"], fontSize=8, leading=11.5, leftIndent=8)
        rows = [[Paragraph("WHERE THE PATIENT FELL SHORT — MOST SEVERE FIRST", s["interp_hd"])]] + \
               [[Paragraph((f"{i + 1}.&nbsp; " if not p.startswith("<i>") else "") + p, st)]
                for i, p in enumerate(pri)]
        t = Table(rows, colWidths=[W_FULL])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF6EF")),
            ("LINEABOVE", (0, 0), (-1, 0), 1.5, colors.HexColor(COLORS["accent"])),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(KeepTogether([t]))
    return story
