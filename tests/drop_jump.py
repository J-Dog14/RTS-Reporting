"""
tests/drop_jump.py
===================
Drop Jump — land from a 12-inch box into a max-effort vertical jump.

Metrics computed
----------------
From V3D scalars (Data.txt):        used directly if present
From GRF time-series (Forces.txt):  RSI, contact time, flight time, jump height,
                                    landing LSI (200 ms), loading rate, impact transients
From Joints.txt:                    knee / hip kinematics per limb

Result fields (DropJumpResult)
------------------------------
contact_time_s             : duration from landing contact to takeoff (s)
flight_time_s              : airborne time after push-off (s)
jump_height_cm             : estimated from flight time (cm)
rsi                        : Reactive Strength Index = jump_height_m / contact_time_s
rsi_modified               : flight_time / contact_time (dimensionless)
landing_lsi_200ms          : bilateral loading symmetry, first 200 ms (%)
landing_ai_200ms           : asymmetry index, first 200 ms (%)
loading_rate_surg_ns       : loading rate, surgical limb (N/s)
loading_rate_ns_ns         : loading rate, non-surgical limb (N/s)
loading_rate_lsi           : LSI of loading rate (%)
impact_transient_surg      : first Fz peak within 50 ms, surgical (N)
impact_transient_ns        : first Fz peak within 50 ms, non-surgical (N)
impact_transient_lsi       : LSI of impact transients (%)
time_to_peak_surg_ms       : ms from contact to peak Fz, surgical
time_to_peak_ns_ms         : ms from contact to peak Fz, non-surgical
grf_eccentric/concentric/overall : BilateralResult objects
kinematics                 : BilateralKinematics object
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from helpers.grf import (
    detect_contacts, peak_force, impulse, loading_rate, lsi,
    asymmetry_index, analyse_phased, analyse_bilateral,
)
from helpers.kinematics import analyse_bilateral_kinematics


@dataclass
class DropJumpResult:
    # ── Timing ────────────────────────────────────────────────────────────────
    contact_time_s:          float = np.nan
    flight_time_s:           float = np.nan
    jump_height_cm:          float = np.nan

    # ── Reactive strength ─────────────────────────────────────────────────────
    rsi:                     float = np.nan
    rsi_modified:            float = np.nan

    # ── Landing symmetry ──────────────────────────────────────────────────────
    landing_lsi_200ms:       float = np.nan
    landing_ai_200ms:        float = np.nan

    # ── Loading characteristics ───────────────────────────────────────────────
    loading_rate_surg_ns:    float = np.nan
    loading_rate_ns_ns:      float = np.nan
    loading_rate_lsi:        float = np.nan

    impact_transient_surg:   float = np.nan
    impact_transient_ns:     float = np.nan
    impact_transient_lsi:    float = np.nan

    time_to_peak_surg_ms:    float = np.nan
    time_to_peak_ns_ms:      float = np.nan

    # ── GRF phase analysis ────────────────────────────────────────────────────
    grf_eccentric:           object = None
    grf_concentric:          object = None
    grf_overall:             object = None

    # ── Kinematics ────────────────────────────────────────────────────────────
    kinematics:              object = None

    # ── Raw arrays (for figures) ──────────────────────────────────────────────
    fz_surg:  object = field(default=None, repr=False)
    fz_ns:    object = field(default=None, repr=False)
    rate_f:   float  = np.nan


def analyse(
    fz_surg:    Optional[np.ndarray],
    fz_ns:      Optional[np.ndarray],
    rate_f:     float,
    bw_n:       float,
    surg_side:  str,
    knee_surg:  dict = None,
    knee_ns:    dict = None,
    hip_surg:   dict = None,
    hip_ns:     dict = None,
    rate_k:     float = 200.0,
    scalars:    dict  = None,
) -> DropJumpResult:
    """
    Run the full drop jump analysis.

    fz_surg / fz_ns  : vertical GRF time series for each limb (N), averaged
                       across trials; FP1 or FP2 Z component from Forces.txt
    rate_f           : GRF sample rate (Hz) — typically 1000
    bw_n             : body weight in Newtons
    surg_side        : 'L' or 'R'
    knee_surg/ns     : dicts with keys 'flex_ext', 'valgus', 'tib_rot'
    hip_surg/ns      : dicts with key 'flex_ext'
    rate_k           : kinematic sample rate (Hz) — typically 200
    scalars          : flat dict of V3D pre-computed metrics (may be None/empty)
    """
    r = DropJumpResult()
    if scalars is None:
        scalars = {}

    if fz_surg is None or fz_ns is None:
        return r

    r.fz_surg = fz_surg
    r.fz_ns   = fz_ns
    r.rate_f  = rate_f

    fz_total = fz_surg + fz_ns

    # ── Contact periods ───────────────────────────────────────────────────────
    contacts_total = detect_contacts(fz_total)
    contacts_surg  = detect_contacts(fz_surg)

    # Contact time from surgical limb first contact
    if contacts_surg:
        s, e = contacts_surg[0]
        r.contact_time_s = (e - s) / rate_f

    # Flight time from combined total: between end of first contact & start of second
    if len(contacts_total) >= 2:
        takeoff = contacts_total[0][1]
        land    = contacts_total[1][0]
        r.flight_time_s  = max(0.0, (land - takeoff) / rate_f)
        r.jump_height_cm = (9.81 * r.flight_time_s ** 2 / 8) * 100

    # ── RSI ───────────────────────────────────────────────────────────────────
    if (not np.isnan(r.jump_height_cm) and not np.isnan(r.contact_time_s)
            and r.contact_time_s > 0):
        r.rsi          = (r.jump_height_cm / 100.0) / r.contact_time_s
        r.rsi_modified = r.flight_time_s / r.contact_time_s

    # ── Clip to first contact window (robust for files with pre-contact silence)
    # Use the start of the total signal's first contact as the landing start.
    contact_start = contacts_total[0][0] if contacts_total else 0
    fz_s_land = fz_surg[contact_start:]
    fz_ns_land = fz_ns[contact_start:]

    # ── Landing symmetry — first 200 ms of landing contact ───────────────────
    win_200 = int(0.200 * rate_f)
    s200  = float(np.sum(fz_s_land[:win_200]))
    ns200 = float(np.sum(fz_ns_land[:win_200]))
    r.landing_lsi_200ms = lsi(s200, ns200)
    r.landing_ai_200ms  = asymmetry_index(s200, ns200)

    # ── Loading rate (first 50 ms of landing contact) ─────────────────────────
    r.loading_rate_surg_ns = loading_rate(fz_s_land,  rate_f, window_ms=50.0)
    r.loading_rate_ns_ns   = loading_rate(fz_ns_land, rate_f, window_ms=50.0)
    r.loading_rate_lsi     = lsi(r.loading_rate_surg_ns, r.loading_rate_ns_ns)

    # ── Impact transients (first local peak within 50 ms of landing) ──────────
    r.impact_transient_surg = _first_peak(fz_s_land,  rate_f, 50.0)
    r.impact_transient_ns   = _first_peak(fz_ns_land, rate_f, 50.0)
    r.impact_transient_lsi  = lsi(r.impact_transient_surg, r.impact_transient_ns)

    # ── Time to peak force ────────────────────────────────────────────────────
    _, pk_surg = peak_force(fz_surg)
    _, pk_ns   = peak_force(fz_ns)
    r.time_to_peak_surg_ms = pk_surg / rate_f * 1000.0
    r.time_to_peak_ns_ms   = pk_ns   / rate_f * 1000.0

    # ── GRF phased analysis ───────────────────────────────────────────────────
    phased = analyse_phased(fz_surg, fz_ns, rate_f, bw_n)
    r.grf_eccentric  = phased["eccentric"]
    r.grf_concentric = phased["concentric"]
    r.grf_overall    = phased["overall"]

    # ── Kinematics ────────────────────────────────────────────────────────────
    if knee_surg or knee_ns:
        r.kinematics = analyse_bilateral_kinematics(
            knee_surg or {}, knee_ns or {},
            rate_k, "Drop Jump", hip_surg, hip_ns,
        )

    return r


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _first_peak(fz: np.ndarray, rate: float, window_ms: float) -> float:
    """Return the first local maximum within window_ms (impact transient)."""
    win = max(3, int(window_ms * rate / 1000))
    seg = fz[:win]
    if len(seg) < 3:
        return float(np.nanmax(seg)) if len(seg) > 0 else np.nan
    for i in range(1, len(seg) - 1):
        if seg[i] > seg[i - 1] and seg[i] > seg[i + 1]:
            return float(seg[i])
    return float(np.nanmax(seg))
