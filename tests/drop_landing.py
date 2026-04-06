"""
tests/drop_landing.py
======================
Drop Landing — land as softly as possible from a 12-inch box.

Focus: bilateral shock-absorption strategy, loading symmetry, and joint kinematics.

Metrics computed
----------------
From GRF time-series (Forces.txt):
    impact_transient_surg/ns     : initial Fz peak within 50 ms (N)
    impact_transient_lsi         : LSI of impact transients (%)
    loading_rate_surg/ns_ns      : average N/s in first 50 ms
    loading_rate_lsi             : LSI of loading rates (%)
    mean_load_surg/ns_bw         : mean Fz normalised to body weight (×BW)
    landing_duration_s           : total contact duration (s)
    peak_force_surg/ns_N         : peak Fz per limb (N)
    peak_force_lsi               : LSI of peak force (%)

From Joints.txt:
    kinematics                   : BilateralKinematics (knee flexion, valgus, etc.)

Result fields (DropLandingResult)
----------------------------------
See dataclass below.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from helpers.grf import (
    detect_contacts, peak_force, loading_rate, lsi,
    asymmetry_index, analyse_phased, analyse_bilateral,
)
from helpers.kinematics import analyse_bilateral_kinematics


@dataclass
class DropLandingResult:
    # ── Impact transient (first 50 ms spike) ─────────────────────────────────
    impact_transient_surg:  float = np.nan
    impact_transient_ns:    float = np.nan
    impact_transient_lsi:   float = np.nan

    # ── Loading rate (first 50 ms) ────────────────────────────────────────────
    loading_rate_surg_ns:   float = np.nan
    loading_rate_ns_ns:     float = np.nan
    loading_rate_lsi:       float = np.nan

    # ── Peak force ────────────────────────────────────────────────────────────
    peak_force_surg_N:      float = np.nan
    peak_force_ns_N:        float = np.nan
    peak_force_lsi:         float = np.nan

    # ── Mean load (normalised) ────────────────────────────────────────────────
    mean_load_surg_bw:      float = np.nan
    mean_load_ns_bw:        float = np.nan

    # ── Timing ────────────────────────────────────────────────────────────────
    landing_duration_s:     float = np.nan

    # ── GRF phase objects ─────────────────────────────────────────────────────
    grf_eccentric:          object = None
    grf_overall:            object = None

    # ── Kinematics ────────────────────────────────────────────────────────────
    kinematics:             object = None

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
) -> DropLandingResult:
    """
    Run the full drop landing analysis.

    fz_surg / fz_ns  : vertical GRF time series per limb (N)
    rate_f           : GRF sample rate (Hz)
    bw_n             : body weight in Newtons
    surg_side        : 'L' or 'R'
    knee_surg/ns     : dicts with keys 'flex_ext', 'valgus', 'tib_rot'
    hip_surg/ns      : dicts with key 'flex_ext'
    rate_k           : kinematic sample rate (Hz)
    scalars          : flat dict of V3D pre-computed metrics (may be None/empty)
    """
    r = DropLandingResult()
    if scalars is None:
        scalars = {}

    if fz_surg is None or fz_ns is None:
        return r

    r.fz_surg = fz_surg
    r.fz_ns   = fz_ns
    r.rate_f  = rate_f

    # Detect contact and clip to first contact onset
    contacts      = detect_contacts(fz_surg + fz_ns)
    contact_start = contacts[0][0] if contacts else 0
    fz_s_contact  = fz_surg[contact_start:]
    fz_ns_contact = fz_ns[contact_start:]

    # ── Impact transient (first local peak within 50 ms of contact onset) ────
    r.impact_transient_surg = _first_peak(fz_s_contact,  rate_f, 50.0)
    r.impact_transient_ns   = _first_peak(fz_ns_contact, rate_f, 50.0)
    r.impact_transient_lsi  = lsi(r.impact_transient_surg, r.impact_transient_ns)

    # ── Loading rate (first 50 ms of contact) ─────────────────────────────────
    r.loading_rate_surg_ns = loading_rate(fz_s_contact,  rate_f, window_ms=50.0)
    r.loading_rate_ns_ns   = loading_rate(fz_ns_contact, rate_f, window_ms=50.0)
    r.loading_rate_lsi     = lsi(r.loading_rate_surg_ns, r.loading_rate_ns_ns)

    # ── Peak force ────────────────────────────────────────────────────────────
    pk_s, _ = peak_force(fz_surg)
    pk_ns, _ = peak_force(fz_ns)
    r.peak_force_surg_N = pk_s
    r.peak_force_ns_N   = pk_ns
    r.peak_force_lsi    = lsi(pk_s, pk_ns)

    # ── Mean load normalised to BW ────────────────────────────────────────────
    if bw_n > 0:
        r.mean_load_surg_bw = float(np.nanmean(fz_surg)) / bw_n
        r.mean_load_ns_bw   = float(np.nanmean(fz_ns))   / bw_n

    # ── Landing duration (total contact) ─────────────────────────────────────
    if contacts:
        s, e = contacts[0]
        r.landing_duration_s = (e - s) / rate_f

    # ── GRF phased analysis ───────────────────────────────────────────────────
    phased = analyse_phased(fz_surg, fz_ns, rate_f, bw_n)
    r.grf_eccentric = phased["eccentric"]
    r.grf_overall   = phased["overall"]

    # ── Kinematics ────────────────────────────────────────────────────────────
    if knee_surg or knee_ns:
        r.kinematics = analyse_bilateral_kinematics(
            knee_surg or {}, knee_ns or {},
            rate_k, "Drop Landing", hip_surg, hip_ns,
        )

    return r


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _first_peak(fz: np.ndarray, rate: float, window_ms: float) -> float:
    """Return the first local maximum within window_ms (impact transient spike)."""
    win = max(3, int(window_ms * rate / 1000))
    seg = fz[:win]
    if len(seg) < 3:
        return float(np.nanmax(seg)) if len(seg) > 0 else np.nan
    for i in range(1, len(seg) - 1):
        if seg[i] > seg[i - 1] and seg[i] > seg[i + 1]:
            return float(seg[i])
    return float(np.nanmax(seg))
