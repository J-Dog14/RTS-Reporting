"""
tests/single_leg_jump.py
=========================
Single-Leg Vertical Jump — tested separately on each limb.

The left and right legs use separate data files (SLVL / SLVR) because only
one force plate (FP3) is used for single-leg tests.  main.py calls this
module twice — once per side — then combines the results here.

Metrics computed
----------------
Jump height (cm):
  Preferred source: JH_IN_L / JH_IN_R from scalar Data.txt files (V3D computed).
  Fallback: flight-time method from FP3 Z-component time series.

From GRF time-series (Single Leg Vertical Left/Right Forces.txt, FP3 Z):
  flight_time_s          : airborne duration (s)
  contact_time_s         : ground contact duration (s)
  peak_force_N           : peak vertical GRF (N)
  peak_force_bw          : peak force normalised to BW (×BW)
  propulsion_impulse_Ns  : concentric phase impulse (N·s)
  rfd_100ms              : Rate of Force Development in first 100 ms (N/s)

From Joints.txt (single-leg side):
  peak_knee_flexion_deg  : peak knee flexion during push-off (°)
  peak_valgus_deg        : peak knee valgus (°)
  peak_tib_ir_deg        : peak tibial internal rotation (°)

LSI values:
  lsi_jump_height        : jump height LSI (%)
  lsi_peak_force         : peak GRF LSI (%)
  lsi_impulse            : propulsion impulse LSI (%)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from helpers.grf import detect_contacts, peak_force, impulse, rfd, lsi
from helpers.kinematics import analyse_knee


@dataclass
class LimbResult:
    """Single-limb metrics for one SL jump trial."""
    side: str   # "Surgical" | "Non-Surgical"

    jump_height_cm:        float = np.nan
    flight_time_s:         float = np.nan
    contact_time_s:        float = np.nan
    peak_force_N:          float = np.nan
    peak_force_bw:         float = np.nan
    propulsion_impulse_Ns: float = np.nan
    rfd_100ms:             float = np.nan

    # Kinematics
    peak_knee_flexion_deg: float = np.nan
    peak_valgus_deg:       float = np.nan
    peak_tib_ir_deg:       float = np.nan

    # Raw GRF (for figures)
    fz:    object = field(default=None, repr=False)
    rate_f: float = np.nan


@dataclass
class SingleLegJumpResult:
    surgical:     LimbResult = None
    non_surgical: LimbResult = None

    lsi_jump_height: float = np.nan
    lsi_peak_force:  float = np.nan
    lsi_impulse:     float = np.nan


def analyse_limb(
    fz:         Optional[np.ndarray],
    rate_f:     float,
    bw_n:       float,
    side:       str,
    jump_height_scalar: float = np.nan,
    knee_data:  dict = None,
    rate_k:     float = 200.0,
) -> LimbResult:
    """
    Analyse one limb's single-leg vertical jump.

    fz                   : FP3 Z-component time series (N), averaged across trials
    rate_f               : GRF sample rate (Hz)
    bw_n                 : body weight in Newtons
    side                 : "Surgical" or "Non-Surgical"
    jump_height_scalar   : V3D-computed jump height (from Data.txt); used in
                           preference to the flight-time estimate if available
    knee_data            : dict with 'flex_ext', 'valgus', 'tib_rot' arrays
    rate_k               : kinematic sample rate (Hz)
    """
    r = LimbResult(side=side)
    if fz is None or len(fz) == 0:
        return r

    r.fz     = fz
    r.rate_f = rate_f

    contacts = detect_contacts(fz)

    # ── Jump height ───────────────────────────────────────────────────────────
    # Prefer scalar from Data.txt; fall back to flight-time calculation
    if not np.isnan(jump_height_scalar):
        r.jump_height_cm = float(jump_height_scalar)

    if len(contacts) >= 2:
        takeoff = contacts[0][1]
        land    = contacts[1][0]
        r.flight_time_s = max(0.0, (land - takeoff) / rate_f)
        if np.isnan(r.jump_height_cm):
            r.jump_height_cm = (9.81 * r.flight_time_s ** 2 / 8) * 100

    # ── Contact time ──────────────────────────────────────────────────────────
    if contacts:
        s, e = contacts[0]
        r.contact_time_s = (e - s) / rate_f
        contact_fz = fz[s:e + 1]

        pk_val, _ = peak_force(contact_fz)
        r.peak_force_N          = pk_val
        r.peak_force_bw         = pk_val / bw_n if bw_n > 0 else np.nan
        r.propulsion_impulse_Ns = impulse(contact_fz, rate_f)
        r.rfd_100ms             = rfd(contact_fz, rate_f, window_ms=100.0)

    # ── Kinematics ────────────────────────────────────────────────────────────
    if knee_data:
        k = analyse_knee(
            knee_data.get("flex_ext"),
            knee_data.get("valgus"),
            knee_data.get("tib_rot"),
            rate_k,
            side,
        )
        r.peak_knee_flexion_deg = k.peak_flexion_deg
        r.peak_valgus_deg       = k.peak_valgus_deg
        r.peak_tib_ir_deg       = k.peak_tibial_ir_deg

    return r


def combine(surg_result: LimbResult, ns_result: LimbResult) -> SingleLegJumpResult:
    """
    Combine two per-limb results into a SingleLegJumpResult with LSI values.

    Typically called by main.py after running analyse_limb() for each side.
    """
    r = SingleLegJumpResult()
    r.surgical     = surg_result
    r.non_surgical = ns_result

    if surg_result and ns_result:
        r.lsi_jump_height = lsi(surg_result.jump_height_cm,       ns_result.jump_height_cm)
        r.lsi_peak_force  = lsi(surg_result.peak_force_N,         ns_result.peak_force_N)
        r.lsi_impulse     = lsi(surg_result.propulsion_impulse_Ns, ns_result.propulsion_impulse_Ns)

    return r
