"""
rts14/analysis.py
=================
Per-trial analysis of v14 exports. Each analyse_* function takes ONE trial
and returns a flat dict of metrics (surg/ns already resolved) plus the
windowed signals needed for figures. aggregate() averages the metric dicts
across trials; build objects in pipeline.py then shape them for the report.

Conventions
-----------
• Force in N at rate_f (900 Hz), kinematics/kinetics at rate_k (300 Hz).
  Index conversion: k = round(f * rate_k / rate_f).
• "surg" / "ns" = surgical / non-surgical limb.
• Joint power total = X+Y+Z components (V3D's per-axis contributions).
  Positive work = ∫P>0, negative work (absorption) = ∫P<0 (reported positive).
• Ankle angle is zeroed to the athlete's standing value (+ = dorsiflexion).
"""

import numpy as np

from config import PLATE_SIDE, GRF_CONTACT_THRESHOLD
from helpers.grf import detect_contacts, lsi, impulse, loading_rate, rfd, lowpass

# Force-plate jump kinetics are the Athletic Screen v2 processor (vendored in
# rts14/screen_v2): plate zeroed in flight, velocity integrated backward from
# take-off, power / RPD / QC flags. RTS adds the per-limb split on top.
try:
    from rts14.screen_v2.kinetics import process_trial as _v2_process, TrialError as _V2Error
except Exception:          # scipy missing → fall back to the built-in phases
    _v2_process, _V2Error = None, Exception

G = 9.81
JOINTS = ("Hip", "Knee", "Ankle")


# ─── small helpers ────────────────────────────────────────────────────────────

def plate_for(side: str) -> str:
    for p, s in PLATE_SIDE.items():
        if s and s.upper() == side.upper():
            return p
    return "FP1" if side.upper() == "L" else "FP2"


def other(side: str) -> str:
    return "R" if side.upper() == "L" else "L"


def _k(i_f, tr):
    return int(round(i_f * tr.rate_k / tr.rate_f))


def _seg(arr, a, b):
    if arr is None:
        return None
    a, b = max(0, a), min(len(arr), b)
    return arr[a:b] if b > a else None


def _nanmax(x):
    return float(np.nanmax(x)) if x is not None and len(x) and np.any(~np.isnan(x)) else np.nan


def _nanmin(x):
    return float(np.nanmin(x)) if x is not None and len(x) and np.any(~np.isnan(x)) else np.nan


def _ptp(x):
    return _nanmax(x) - _nanmin(x)


def _first_peak(fz, rate, window_ms=50.0):
    win = max(3, int(window_ms * rate / 1000))
    seg = fz[:win]
    for i in range(1, len(seg) - 1):
        if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1]:
            return float(seg[i])
    return _nanmax(seg)


def _contacts(fz, rate):
    return detect_contacts(np.nan_to_num(fz), threshold=GRF_CONTACT_THRESHOLD,
                           min_frames=int(0.02 * rate))


def _angle(tr, side, joint, comp):
    a = tr.angles.get(f"{side}_{joint}_Angle")
    return None if a is None else a[:, "XYZ".index(comp)]


def _moment(tr, side, joint, comp="X"):
    a = tr.moments.get(f"{side}_{joint}_Moment")
    return None if a is None else a[:, "XYZ".index(comp)]


def _power(tr, side, joint):
    a = tr.powers.get(f"{side}_{joint}_Power")
    return None if a is None else np.nansum(a, axis=1)


def _work(p, rate):
    """(positive J, negative J as a positive number) from a power trace."""
    if p is None or len(p) < 2:
        return np.nan, np.nan
    p = np.nan_to_num(p)
    pos = float(np.trapz(np.clip(p, 0, None), dx=1.0 / rate))
    neg = float(-np.trapz(np.clip(p, None, 0), dx=1.0 / rate))
    return pos, neg


def body_weight_from_force(session):
    """
    Body weight (N) measured from the plates: median vertical force during the
    single-leg balance trials (athlete standing still on FP3), falling back to
    the mean total force over the endurance squat (mean of a cyclic movement
    that starts and ends at rest ≈ body weight).
    """
    vals = []
    for test in ("balance_L_std", "balance_R_std", "balance_L_airex", "balance_R_airex"):
        for t in session.by_test(test):
            f = t.fz("FP3")
            if f is not None and len(f) > 100:
                f = np.nan_to_num(f)
                f = f[f > 0.5 * np.median(f)]
                if len(f):
                    vals.append(float(np.median(f)))
    if vals:
        return float(np.median(vals))
    for t in session.by_test("endurance_squat"):
        a, b = t.fz(plate_for("L")), t.fz(plate_for("R"))
        if a is not None and b is not None:
            n = min(len(a), len(b))
            return float(np.mean(np.nan_to_num(a[:n]) + np.nan_to_num(b[:n])))
    return np.nan


def ankle_neutral(session, side):
    """Standing ankle X angle (first 0.3 s of the endurance trial), else None."""
    for t in session.by_test("endurance_squat"):
        a = _angle(t, side, "Ankle", "X")
        if a is not None and len(a) > 10:
            n = max(5, int(0.3 * t.rate_k))
            return float(np.nanmedian(a[:n]))
    return None


def screen_v2(fz_total, rate, movement, body_n, limbs=None):
    """
    Run the Athletic Screen v2 processor on the summed vertical force.

    limbs : optional {"s": Fz_surg, "n": Fz_ns} — each plate is zeroed with its
            own in-flight median, then limb power = F_limb · v(COM), the usual
            way to split bilateral power between legs.
    Returns (out_dict, arrays, limbs_zeroed) or None.
    """
    if _v2_process is None:
        return None
    try:
        out, arr = _v2_process(np.asarray(fz_total, float), float(rate), movement,
                               session_bw_n=body_n if body_n == body_n else None)
    except _V2Error:
        return None
    except Exception:
        return None
    lz = {}
    if limbs:
        to, ld = out["takeoff_idx"], out["landing_idx"]
        a, b = to + int(0.1 * (ld - to)), ld - int(0.1 * (ld - to))
        for k, f in limbs.items():
            f = np.asarray(f, float)
            off = float(np.median(f[a:b])) if b > a else 0.0
            lz[k] = f - off
    return out, arr, lz


def _limb_power(out, arr, f, rate):
    """Peak / concentric-mean power and concentric work for one limb (W, J)."""
    to, c0, con0 = out["takeoff_idx"], out["contact_start_idx"], out["concentric_start_idx"]
    v = arr["v"]
    p = f[:to] * v[:to]
    seg = p[c0:to]
    con = p[con0:to]
    return {"peak_power": _nanmax(seg),
            "con_mean_power": float(np.mean(con)) if len(con) else np.nan,
            "con_work": float(np.sum(con) / rate) if len(con) else np.nan}


# ─── time-normalised joint curves (for the mechanics grid) ──────────────────

N_PTS = 101

# curve key → (kind, joint, component, sign).  sign makes "+" read naturally:
#   angles  : hip/knee + = flexion, ankle + = dorsiflexion (zeroed to standing)
#   moments : + = extensor (hip, knee) / plantarflexor (ankle), in Nm/kg
#   power   : W/kg, + = generating, − = absorbing
#   frontal : change from the start of the window (sign-free comparison)
CURVE_SPEC = {
    "hip_ang":    ("angle", "Hip", "X", 1),
    "knee_ang":   ("angle", "Knee", "X", 1),
    "ankle_ang":  ("angle", "Ankle", "X", 1),
    "hip_mom":    ("moment", "Hip", "X", -1),
    "knee_mom":   ("moment", "Knee", "X", 1),
    "ankle_mom":  ("moment", "Ankle", "X", -1),
    "hip_pow":    ("power", "Hip", None, 1),
    "knee_pow":   ("power", "Knee", None, 1),
    "ankle_pow":  ("power", "Ankle", None, 1),
    "hip_add":    ("frontal", "Hip", "Y", 1),
    "knee_val":   ("frontal", "Knee", "Y", 1),
    "ankle_fr":   ("frontal", "Ankle", "Y", 1),
}


def _resample(y, n=N_PTS):
    y = np.asarray(y, float)
    if y is None or len(y) < 3 or np.all(np.isnan(y)):
        return np.full(n, np.nan)
    x = np.linspace(0, 1, len(y))
    good = ~np.isnan(y)
    return np.interp(np.linspace(0, 1, n), x[good], y[good])


def norm_curves(tr, side, k0, k1, mass_kg, neutral=None):
    """
    Each joint signal over kinematic frames k0..k1 resampled to 0–100 %.
    Moments / powers divided by body mass (Nm/kg, W/kg).
    """
    out = {}
    m = mass_kg if mass_kg and mass_kg == mass_kg else 1.0
    for key, (kind, joint, comp, sign) in CURVE_SPEC.items():
        if kind in ("angle", "frontal"):
            a = _angle(tr, side, joint, comp)
        elif kind == "moment":
            a = _moment(tr, side, joint, comp)
        else:
            a = _power(tr, side, joint)
        seg = _seg(a, k0, k1)
        if seg is None or len(seg) < 3:
            out[key] = np.full(N_PTS, np.nan)
            continue
        seg = sign * np.asarray(seg, float)
        if kind == "angle" and joint == "Ankle":
            seg = seg - (neutral if neutral is not None else 90.0)
        if kind == "frontal":
            seg = seg - seg[0]
        if kind in ("moment", "power"):
            seg = seg / m
        out[key] = _resample(seg)
    return out


def curve_stats(curves, key):
    """Peak, min, ROM, time-of-peak (% of window) for one normalised curve."""
    y = curves.get(key)
    if y is None or np.all(np.isnan(y)):
        return {}
    return {"peak": float(np.nanmax(y)), "min": float(np.nanmin(y)),
            "rom": float(np.nanmax(y) - np.nanmin(y)), "t_peak": float(np.nanargmax(y))}


# ─── per-limb kinematics / kinetics in a kinematic window ─────────────────────

def limb_mechanics(tr, side, k0, k1, neutral=None, prefix=""):
    """Peak angles, moments and joint work for one limb over frames k0..k1."""
    m = {}
    kx = _seg(_angle(tr, side, "Knee", "X"), k0, k1)
    ky = _seg(_angle(tr, side, "Knee", "Y"), k0, k1)
    kz = _seg(_angle(tr, side, "Knee", "Z"), k0, k1)
    hx = _seg(_angle(tr, side, "Hip", "X"), k0, k1)
    hy = _seg(_angle(tr, side, "Hip", "Y"), k0, k1)
    hz = _seg(_angle(tr, side, "Hip", "Z"), k0, k1)
    ax = _seg(_angle(tr, side, "Ankle", "X"), k0, k1)
    m["knee_flex"]    = _nanmax(kx)
    m["knee_valgus"]  = _nanmax(ky)
    m["tib_ir"]       = _nanmax(kz)
    m["tib_exc"]      = _ptp(kz)
    m["hip_flex"]     = _nanmax(hx)
    m["hip_add"]      = _nanmax(hy)
    m["hip_ir"]       = _nanmax(hz)
    if ax is not None:
        ref = neutral if neutral is not None else 90.0
        m["ankle_df"]  = _nanmax(ax) - ref
        m["ankle_rom"] = _ptp(ax)
    m["knee_ext_mom"] = _nanmax(_seg(_moment(tr, side, "Knee"), k0, k1))
    pf = _nanmin(_seg(_moment(tr, side, "Ankle"), k0, k1))
    m["ankle_pf_mom"] = -pf if pf == pf else np.nan
    for j in JOINTS:
        pos, neg = _work(_seg(_power(tr, side, j), k0, k1), tr.rate_k)
        m[f"{j.lower()}_pos_work"] = pos
        m[f"{j.lower()}_neg_work"] = neg
    tot_pos = np.nansum([m[f"{j.lower()}_pos_work"] for j in JOINTS])
    tot_neg = np.nansum([m[f"{j.lower()}_neg_work"] for j in JOINTS])
    for j in JOINTS:
        m[f"{j.lower()}_pos_share"] = (100 * m[f"{j.lower()}_pos_work"] / tot_pos) if tot_pos > 0 else np.nan
        m[f"{j.lower()}_neg_share"] = (100 * m[f"{j.lower()}_neg_work"] / tot_neg) if tot_neg > 0 else np.nan
    return {prefix + k: v for k, v in m.items()}


def limb_signals(tr, side, k0, k1, neutral=None):
    """Windowed angle traces for figures (report expects these key names)."""
    ax = _seg(_angle(tr, side, "Ankle", "X"), k0, k1)
    am = _seg(_moment(tr, side, "Ankle"), k0, k1)
    return {
        # moment / power traces for the Achilles figures
        "ankle_pf_mom_ts": None if am is None else -am,        # + = plantarflexor
        "ankle_pow_ts":    _seg(_power(tr, side, "Ankle"), k0, k1),
        "knee_pow_ts":     _seg(_power(tr, side, "Knee"), k0, k1),
        "hip_pow_ts":      _seg(_power(tr, side, "Hip"), k0, k1),
        "knee_flex":   _seg(_angle(tr, side, "Knee", "X"), k0, k1),
        "knee_valgus": _seg(_angle(tr, side, "Knee", "Y"), k0, k1),
        "tib_rot":     _seg(_angle(tr, side, "Knee", "Z"), k0, k1),
        "hip_add":     _seg(_angle(tr, side, "Hip", "Y"), k0, k1),
        "hip_ir":      _seg(_angle(tr, side, "Hip", "Z"), k0, k1),
        "ankle_flex":  None if ax is None else ax - (neutral if neutral is not None else 90.0),
    }


def _limb_grf(fz, rate, s, e):
    """Landing / contact GRF metrics for one limb over analog frames s..e."""
    seg = fz[s:e + 1]
    sm = lowpass(seg, rate, 50.0)          # light smoothing for slope/peaks only
    out = {
        "peak_grf":  _nanmax(seg),
        "impulse":   impulse(np.clip(seg, 0, None), rate),
        "rfd":       rfd(sm, rate, 100.0),
        "load_rate": loading_rate(sm, rate, 50.0),
        "impact":    _first_peak(sm, rate, 50.0),
        "t_peak_ms": float(np.nanargmax(seg) / rate * 1000.0) if len(seg) else np.nan,
        "sum200":    float(np.nansum(seg[: int(0.2 * rate)])),
    }
    return out


def _with_lsi(d, keys, out):
    for k in keys:
        out[f"{k}_lsi"] = lsi(d.get(f"s_{k}"), d.get(f"n_{k}"))


_MECH_LSI = ["knee_flex", "knee_ext_mom", "ankle_pf_mom",
             "hip_pos_work", "knee_pos_work", "ankle_pos_work",
             "hip_neg_work", "knee_neg_work", "ankle_neg_work"]


# ─── Drop jump / drop landing ────────────────────────────────────────────────

def analyse_drop(tr, surg, mass_n, neutral, rejump=True):
    ns = other(surg)
    fs_ = tr.fz(plate_for(surg))
    fn_ = tr.fz(plate_for(ns))
    if fs_ is None or fn_ is None:
        return None
    n = min(len(fs_), len(fn_))
    fs_, fn_ = np.nan_to_num(fs_[:n]), np.nan_to_num(fn_[:n])
    tot = fs_ + fn_
    rate = tr.rate_f
    cons = _contacts(tot, rate)
    if not cons:
        return None
    s, e = cons[0]
    d = {"trial": tr.name, "qc": []}
    if rejump:
        v2 = screen_v2(tot, rate, "DJ", mass_n, {"s": fs_, "n": fn_})
        if v2 is not None:
            o, arr, lz = v2
            s, e, land2 = o["contact_start_idx"], o["takeoff_idx"], o["landing_idx"]
            fs_, fn_ = lz["s"], lz["n"]
            d.update(contact_time=o["ct_s"], flight_time=o["flight_time_s"],
                     jump_height_in=o["jh_in"], rsi=o["rsi"],
                     peak_power_wkg=o["pp_w_per_kg"], implied_drop_cm=o.get("implied_drop_cm", np.nan),
                     qc=list(o.get("qc_flags", [])))
            for k, f in (("s", fs_), ("n", fn_)):
                for mk, mv in _limb_power(o, arr, f, rate).items():
                    d[f"{k}_{mk}"] = mv
            _with_lsi(d, ["peak_power", "con_work"], d)
        else:
            if len(cons) < 2:
                return None
            land2 = cons[1][0]
            ft = (land2 - e) / rate
            if ft < 0.1:
                return None
            ct = (e - s) / rate
            jh_m = G * ft ** 2 / 8.0
            d.update(contact_time=ct, flight_time=ft, jump_height_in=jh_m / 0.0254,
                     rsi=jh_m / ct if ct > 0 else np.nan)
        win_end = e
        fig_end = min(n, land2 + int(0.3 * rate))
    else:
        win_end = min(e, s + int(1.0 * rate))
        fig_end = min(n, win_end + int(0.1 * rate))

    gs = _limb_grf(fs_, rate, s, win_end)
    gn = _limb_grf(fn_, rate, s, win_end)
    for k, v in gs.items():
        d[f"s_{k}"] = v
    for k, v in gn.items():
        d[f"n_{k}"] = v
    _with_lsi(d, ["peak_grf", "impulse", "rfd", "load_rate", "impact"], d)
    d["landing_lsi_200ms"] = lsi(gs["sum200"], gn["sum200"])

    k0, k1 = _k(s, tr), _k(win_end, tr) + 1
    d.update(limb_mechanics(tr, surg, k0, k1, neutral.get(surg), "s_"))
    d.update(limb_mechanics(tr, ns,   k0, k1, neutral.get(ns),   "n_"))
    _with_lsi(d, _MECH_LSI, d)
    # time-normalised curves: DJ over the whole contact, DL over the first 500 ms
    ck1 = k1 if rejump else min(k1, k0 + int(0.5 * tr.rate_k))
    mkg = mass_n / G if mass_n == mass_n else None
    d["curves_s"] = norm_curves(tr, surg, k0, ck1, mkg, neutral.get(surg))
    d["curves_n"] = norm_curves(tr, ns, k0, ck1, mkg, neutral.get(ns))
    if rejump and v2 is not None and e > s:
        d["event_pct"] = 100.0 * (o["concentric_start_idx"] - s) / (e - s)

    # figure window
    f0 = max(0, s - int(0.05 * rate))
    kf0, kf1 = _k(f0, tr), _k(fig_end, tr)
    sig = {"fz_surg": fs_[f0:fig_end], "fz_ns": fn_[f0:fig_end],
           "contact_k": _k(s, tr) - kf0, "takeoff_k": _k(win_end, tr) - kf0}
    for side, tag in ((surg, "surg"), (ns, "ns")):
        for key, v in limb_signals(tr, side, kf0, kf1, neutral.get(side)).items():
            sig[f"{key}_{tag}"] = v
        cop = tr.cop.get(plate_for(side))
        sig[f"cop_y_{tag}"] = None if cop is None else cop[f0:fig_end, 1]
    return d, sig


# ─── Countermovement-type jumps (bilateral CMJ, single-leg) ──────────────────

def _cmj_phases(fz, rate, body_n):
    """
    Phase frames for a countermovement-type jump.

    The exported trials are cropped and usually start MID-countermovement
    (no quiet standing), so velocity is integrated BACKWARD from take-off,
    where v_to = g·t_flight/2:
        v(t) = v_to − ∫_t^takeoff (F − BW)/m dt
    bottom = last v=0 crossing before take-off (start of propulsion)
    vmin   = peak downward velocity before bottom (braking starts)
    onset  = first frame moving down faster than 0.05 m/s (or trial start)
    """
    cons = _contacts(fz, rate)
    if len(cons) < 2 or body_n != body_n or body_n <= 0:
        return None
    s, takeoff = cons[0]
    land = cons[1][0]
    flight = (land - takeoff) / rate
    if flight <= 0:
        return None
    bw = body_n
    m = bw / G
    v_to = G * flight / 2.0
    seg = fz[s:takeoff + 1]
    acc = (seg - bw) / m
    # backward cumulative integral: vel[i] = v_to - sum(acc[i:]) / rate
    tail = np.cumsum(acc[::-1])[::-1] / rate
    vel = v_to - tail
    neg = np.where(vel < 0)[0]
    if not len(neg):
        return None
    bottom = s + int(neg[-1]) + 1
    i_vmin = int(np.argmin(vel[: neg[-1] + 1]))
    moving = np.where(vel[: i_vmin + 1] < -0.05)[0]
    onset = s + (int(moving[0]) if len(moving) else 0)
    return {"start": s, "onset": onset, "vmin": s + i_vmin, "bottom": bottom,
            "takeoff": takeoff, "land": land, "bw": bw, "v_to": v_to,
            "v_min": float(vel[i_vmin]), "flight": flight,
            "cropped": bool(abs(vel[0]) > 0.1)}


def analyse_cmj(tr, surg, mass_n, neutral):
    ns = other(surg)
    fs_ = tr.fz(plate_for(surg))
    fn_ = tr.fz(plate_for(ns))
    if fs_ is None or fn_ is None:
        return None
    n = min(len(fs_), len(fn_))
    fs_, fn_ = np.nan_to_num(fs_[:n]), np.nan_to_num(fn_[:n])
    tot = fs_ + fn_
    rate = tr.rate_f
    v2 = screen_v2(tot, rate, "CMJ", mass_n, {"s": fs_, "n": fn_})
    extra = {}
    if v2 is not None:
        o, arr, lz = v2
        fs_, fn_ = lz["s"], lz["n"]
        c0, con0, to = o["contact_start_idx"], o["concentric_start_idx"], o["takeoff_idx"]
        vseg = arr["v"][c0:con0] if con0 > c0 else arr["v"][c0:c0 + 1]
        vmin_i = c0 + int(np.argmin(vseg))
        on = o.get("movement_onset_idx", c0)
        ph = {"onset": on, "vmin": vmin_i, "bottom": con0, "takeoff": to,
              "land": o["landing_idx"], "bw": o["bw_n"], "flight": o["flight_time_s"],
              "v_min": o["vmin_m_s"]}
        extra = {"peak_power_wkg": o["pp_w_per_kg"], "mrsi": o.get("mrsi", np.nan),
                 "cm_depth_cm": o.get("countermovement_depth_m", np.nan) * 100,
                 "peak_rfd_bw": o.get("peak_rfd_bw_per_s", np.nan)}
        for k, f in (("s", fs_), ("n", fn_)):
            for mk, mv in _limb_power(o, arr, f, rate).items():
                extra[f"{k}_{mk}"] = mv
        _with_lsi(extra, ["peak_power", "con_work"], extra)
        extra["qc"] = list(o.get("qc_flags", []))
    else:
        ph = _cmj_phases(tot, rate, mass_n)
    if ph is None or ph["flight"] < 0.1:
        return None
    on, vmin, bot, to = ph["onset"], ph["vmin"], ph["bottom"], ph["takeoff"]
    half = ph["bw"] / 2.0
    jh_m = G * ph["flight"] ** 2 / 8.0
    d = {"trial": tr.name, "flight_time": ph["flight"], "jump_height_in": jh_m / 0.0254,
         "peak_down_vel": -ph["v_min"],
         "contact_time": (to - on) / rate}
    d.update(extra)
    for f, tag in ((fs_, "s"), (fn_, "n")):
        d[f"{tag}_prop_impulse"]  = impulse(f[bot:to + 1] - half, rate)
        d[f"{tag}_brake_impulse"] = impulse(f[vmin:bot + 1] - half, rate)
        d[f"{tag}_unwt_impulse"]  = impulse(np.clip(half - f[on:vmin + 1], 0, None), rate)
        d[f"{tag}_prop_peak"]     = _nanmax(f[bot:to + 1])
        d[f"{tag}_brake_peak"]    = _nanmax(f[vmin:bot + 1])
    _with_lsi(d, ["prop_impulse", "brake_impulse", "prop_peak", "brake_peak"], d)

    k0, k1 = _k(on, tr), _k(to, tr) + 1
    d.update(limb_mechanics(tr, surg, k0, k1, neutral.get(surg), "s_"))
    d.update(limb_mechanics(tr, ns,   k0, k1, neutral.get(ns),   "n_"))
    # propulsion-only joint work
    kb = _k(bot, tr)
    for side, tag in ((surg, "s_"), (ns, "n_")):
        for j in JOINTS:
            pos, _ = _work(_seg(_power(tr, side, j), kb, k1), tr.rate_k)
            d[f"{tag}{j.lower()}_pos_work"] = pos
        tot_pos = np.nansum([d[f"{tag}{j.lower()}_pos_work"] for j in JOINTS])
        for j in JOINTS:
            d[f"{tag}{j.lower()}_pos_share"] = (100 * d[f"{tag}{j.lower()}_pos_work"] / tot_pos
                                                if tot_pos > 0 else np.nan)
    _with_lsi(d, _MECH_LSI, d)
    mkg = ph["bw"] / G
    d["curves_s"] = norm_curves(tr, surg, k0, k1, mkg, neutral.get(surg))
    d["curves_n"] = norm_curves(tr, ns, k0, k1, mkg, neutral.get(ns))
    if to > on:
        d["event_pct"] = 100.0 * (bot - on) / (to - on)

    f0 = max(0, on - int(0.3 * rate))
    f1 = min(n, ph["land"] + int(0.3 * rate))
    kf0, kf1 = _k(f0, tr), _k(f1, tr)
    sig = {"fz_surg": fs_[f0:f1], "fz_ns": fn_[f0:f1],
           "takeoff_k": _k(to, tr) - kf0, "bottom_k": _k(bot, tr) - kf0}
    for side, tag in ((surg, "surg"), (ns, "ns")):
        for key, v in limb_signals(tr, side, kf0, kf1, neutral.get(side)).items():
            sig[f"{key}_{tag}"] = v
        cop = tr.cop.get(plate_for(side))
        # COP only during the propulsive contact (before take-off)
        sig[f"cop_y_{tag}"] = None if cop is None else cop[f0:min(f1, to), 1]
    sig["fz_surg_cop"] = fs_[f0:min(f1, to)]
    sig["fz_ns_cop"]   = fn_[f0:min(f1, to)]
    return d, sig


def analyse_slv(tr, side, mass_n, neutral):
    """One single-leg vertical trial (plate FP3), metrics for that limb only."""
    fz = tr.fz("FP3")
    if fz is None:
        return None
    fz = np.nan_to_num(fz)
    rate = tr.rate_f
    v2 = screen_v2(fz, rate, "SLV", mass_n)
    v2x = {}
    if v2 is not None:
        o, arr, _ = v2
        fz = arr["fz"] if len(arr["fz"]) == len(fz) else fz
        c0, con0, to = o["contact_start_idx"], o["concentric_start_idx"], o["takeoff_idx"]
        ph = {"onset": o.get("movement_onset_idx", c0), "bottom": con0, "takeoff": to,
              "land": o["landing_idx"], "bw": o["bw_n"], "flight": o["flight_time_s"]}
        v2x = {"peak_power_wkg": o["pp_w_per_kg"], "peak_power": o["peak_power_w"],
               "con_work": o["concentric_work_j"], "rpd": o["rpd_max_w_per_s"],
               "qc": list(o.get("qc_flags", []))}
    else:
        ph = _cmj_phases(fz, rate, mass_n)
    if ph is None or ph["flight"] < 0.05:
        return None
    on, bot, to = ph["onset"], ph["bottom"], ph["takeoff"]
    jh_m = G * ph["flight"] ** 2 / 8.0
    sm = lowpass(fz, rate, 50.0)
    dt = max(1, int(0.02 * rate))
    slope = (sm[on + dt:to + 1] - sm[on:to + 1 - dt]) * rate / dt if to - on > dt else np.array([np.nan])
    d = {"trial": tr.name,
         "jump_height_in": jh_m / 0.0254,
         "flight_time": ph["flight"],
         "contact_time": (to - on) / rate,
         "peak_grf": _nanmax(fz[bot:to + 1]),
         "net_impulse": impulse(fz[bot:to + 1] - ph["bw"], rate),
         "peak_rfd": _nanmax(slope)}
    d.update(v2x)
    k0, k1 = _k(on, tr), _k(to, tr) + 1
    d.update(limb_mechanics(tr, side, k0, k1, neutral.get(side)))
    kb = _k(bot, tr)
    for j in JOINTS:
        pos, _ = _work(_seg(_power(tr, side, j), kb, k1), tr.rate_k)
        d[f"{j.lower()}_pos_work"] = pos
    tot_pos = np.nansum([d[f"{j.lower()}_pos_work"] for j in JOINTS])
    for j in JOINTS:
        d[f"{j.lower()}_pos_share"] = 100 * d[f"{j.lower()}_pos_work"] / tot_pos if tot_pos > 0 else np.nan

    d["curves"] = norm_curves(tr, side, k0, k1, ph["bw"] / G, neutral.get(side))
    if to > on:
        d["event_pct"] = 100.0 * (bot - on) / (to - on)

    f0 = max(0, on - int(0.3 * rate))
    f1 = min(len(fz), ph["land"] + int(0.3 * rate))
    kf0, kf1 = _k(f0, tr), _k(f1, tr)
    sig = {"fz": fz[f0:f1], "takeoff_k": _k(to, tr) - kf0, "bottom_k": _k(bot, tr) - kf0}
    sig.update(limb_signals(tr, side, kf0, kf1, neutral.get(side)))
    cop = tr.cop.get("FP3")
    sig["cop_y"] = None if cop is None else cop[f0:min(f1, to), 1]
    sig["fz_cop"] = fz[f0:min(f1, to)]
    return d, sig


# ─── aggregation ─────────────────────────────────────────────────────────────

def aggregate(results, key_metric=None):
    """
    results : list of (metrics_dict, signals_dict) per valid trial
    Returns (mean_dict, sd_dict, representative_signals, n_trials, trial_names).
    LSIs are recomputed from the averaged limb values, not averaged.
    """
    results = [r for r in results if r is not None]
    if not results:
        return None
    keys = sorted({k for d, _ in results for k in d
                   if k not in ("trial", "qc") and not isinstance(d[k], dict)})
    mean, sd = {}, {}
    for k in keys:
        vals = np.array([d.get(k, np.nan) for d, _ in results], dtype=float)
        mean[k] = float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else np.nan
        sd[k] = float(np.nanstd(vals)) if np.sum(~np.isnan(vals)) > 1 else np.nan
    # LSIs from mean limb values
    for k in keys:
        if k.endswith("_lsi"):
            base = k[:-4]
            if f"s_{base}" in mean and f"n_{base}" in mean:
                mean[k] = lsi(mean[f"s_{base}"], mean[f"n_{base}"])
    rep = results[0][1]
    if key_metric and key_metric in mean:
        target = mean[key_metric]
        best = min(results, key=lambda r: abs(r[0].get(key_metric, np.inf) - target)
                   if r[0].get(key_metric) == r[0].get(key_metric) else np.inf)
        rep = best[1]
    names = [d["trial"] for d, _ in results]
    qc = sorted({f"{d['trial']}: {f}" for d, _ in results for f in (d.get("qc") or [])})
    mean["_qc"] = qc
    # curves: mean ± SD across trials, per curve key
    for ck in sorted({k for d, _ in results for k in d if isinstance(d.get(k), dict)}):
        stack = {}
        for d, _ in results:
            for key, arr in (d.get(ck) or {}).items():
                stack.setdefault(key, []).append(arr)
        mean[ck] = {key: np.nanmean(np.vstack(v), axis=0) for key, v in stack.items()}
        mean[ck + "_sd"] = {key: (np.nanstd(np.vstack(v), axis=0) if len(v) > 1 else np.zeros(N_PTS))
                            for key, v in stack.items()}
    return mean, sd, rep, len(results), names
