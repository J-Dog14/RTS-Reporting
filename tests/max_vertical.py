"""
tests/max_vertical.py
======================
Max-effort bilateral countermovement vertical jump (CMJ).

Metrics computed
----------------
From GRF time-series (Vertical Forces.txt):
    jump_height_cm         : estimated from flight time
    flight_time_s          : airborne duration
    contact_time_s         : ground contact duration
    propulsion_lsi         : LSI of concentric impulse (%)
    unweighting_impulse_surg/ns : impulse below 50% BW per limb (N·s)
    peak_force_surg/ns_N   : peak Fz per limb
    peak_force_lsi         : LSI of peak force (%)
    grf_eccentric / concentric / overall : BilateralResult objects

From Joints.txt:
    kinematics             : BilateralKinematics object

Result fields (MaxVerticalResult)
----------------------------------
See dataclass below.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from helpers.grf import (
    detect_contacts, peak_force, impulse, lsi,
    analyse_phased,
)
from helpers.kinematics import analyse_bilateral_kinematics


@dataclass
class MaxVerticalResult:
    # ── Jump metrics ──────────────────────────────────────────────────────────
    jump_height_cm:             float = np.nan
    flight_time_s:              float = np.nan
    contact_time_s:             float = np.nan

    # ── Symmetry ──────────────────────────────────────────────────────────────
    propulsion_lsi:             float = np.nan
    peak_force_surg_N:          float = np.nan
    peak_force_ns_N:            float = np.nan
    peak_force_lsi:             float = np.nan

    # ── Unweighting impulse (limb-specific countermove depth indicator) ────────
    unweighting_impulse_surg:   float = np.nan
    unweighting_impulse_ns:     float = np.nan

    # ── GRF phase objects ─────────────────────────────────────────────────────
    grf_eccentric:              object = None
    grf_concentric:             object = None
    grf_overall:                object = None

    # ── Kinematics ────────────────────────────────────────────────────────────
    kinematics:                 object = None

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
) -> MaxVerticalResult:
    """
    Run the full max vertical jump analysis.

    fz_surg / fz_ns  : vertical GRF time series per limb (N), averaged trials
    rate_f           : GRF sample rate (Hz)
    bw_n             : body weight in Newtons
    surg_side        : 'L' or 'R'
    knee_surg/ns     : dicts with keys 'flex_ext', 'valgus', 'tib_rot'
    hip_surg/ns      : dicts with key 'flex_ext'
    rate_k           : kinematic sample rate (Hz)
    scalars          : flat dict of V3D pre-computed metrics (may be None/empty)
    """
    r = MaxVerticalResult()
    if scalars is None:
        scalars = {}

    if fz_surg is None or fz_ns is None:
        return r

    r.fz_surg = fz_surg
    r.fz_ns   = fz_ns
    r.rate_f  = rate_f

    fz_total = fz_surg + fz_ns

    # ── Jump height from flight time ──────────────────────────────────────────
    contacts = detect_contacts(fz_total)
    if len(contacts) >= 2:
        takeoff = contacts[0][1]
        land    = contacts[1][0]
        r.flight_time_s  = max(0.0, (land - takeoff) / rate_f)
        r.jump_height_cm = (9.81 * r.flight_time_s ** 2 / 8) * 100

    if contacts:
        s, e = contacts[0]
        r.contact_time_s = (e - s) / rate_f

    # ── Peak force ────────────────────────────────────────────────────────────
    pk_s, _ = peak_force(fz_surg)
    pk_ns, _ = peak_force(fz_ns)
    r.peak_force_surg_N = pk_s
    r.peak_force_ns_N   = pk_ns
    r.peak_force_lsi    = lsi(pk_s, pk_ns)

    # ── GRF phased analysis ───────────────────────────────────────────────────
    phased = analyse_phased(fz_surg, fz_ns, rate_f, bw_n)
    r.grf_eccentric  = phased["eccentric"]
    r.grf_concentric = phased["concentric"]
    r.grf_overall    = phased["overall"]

    # Propulsion LSI from concentric phase impulse
    if r.grf_concentric:
        r.propulsion_lsi = lsi(
            r.grf_concentric.surgical.impulse_Ns,
            r.grf_concentric.non_surgical.impulse_Ns,
        )

    # ── Unweighting impulse ───────────────────────────────────────────────────
    # Area under BW/2 that each limb drops below its share of body weight
    per_limb_bw = bw_n / 2.0
    r.unweighting_impulse_surg = _unwt_impulse(fz_surg, per_limb_bw, rate_f)
    r.unweighting_impulse_ns   = _unwt_impulse(fz_ns,   per_limb_bw, rate_f)

    # ── Kinematics ────────────────────────────────────────────────────────────
    if knee_surg or knee_ns:
        r.kinematics = analyse_bilateral_kinematics(
            knee_surg or {}, knee_ns or {},
            rate_k, "Max Vertical", hip_surg, hip_ns,
        )

    return r


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _unwt_impulse(fz: np.ndarray, threshold_n: float, rate: float) -> float:
    """Integrate the area where Fz falls below threshold_n (unweighting impulse)."""
    below = np.where(fz < threshold_n, threshold_n - fz, 0.0)
    return float(np.trapz(below, dx=1.0 / rate))
