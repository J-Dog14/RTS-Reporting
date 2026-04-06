"""
helpers/kinematics.py
======================
Joint angle analysis — hip, knee, ankle.

All functions accept 1-D numpy arrays (angle time series in degrees)
and return plain dataclasses.  No hidden coupling to other modules.

Key functions
-------------
analyse_knee(flex_ext, valgus, tib_rot, rate, side)  → KneeMetrics
analyse_hip(flex_ext, ab_add, rot, rate, side)        → HipMetrics
analyse_bilateral_kinematics(surg_data, ns_data, ...)  → BilateralKinematics
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.grf import lsi


# ─── Data containers ──────────────────────────────────────────────────────────

@dataclass
class KneeMetrics:
    side: str   # "Surgical" | "Non-Surgical"

    # Sagittal plane (flexion positive)
    peak_flexion_deg:       float = np.nan
    flexion_at_contact_deg: float = np.nan
    flexion_excursion_deg:  float = np.nan   # contact → peak

    # Frontal plane (valgus positive)
    peak_valgus_deg:        float = np.nan
    mean_valgus_deg:        float = np.nan
    valgus_at_contact_deg:  float = np.nan
    time_to_peak_valgus_s:  float = np.nan

    # Transverse plane (internal rotation positive)
    peak_tibial_ir_deg:     float = np.nan
    tibial_rot_excursion:   float = np.nan


@dataclass
class HipMetrics:
    side: str

    peak_flexion_deg:    float = np.nan
    flexion_at_contact_deg: float = np.nan
    peak_adduction_deg:  float = np.nan   # positive = adduction
    mean_adduction_deg:  float = np.nan
    peak_ir_deg:         float = np.nan


@dataclass
class BilateralKinematics:
    """Bilateral kinematic comparison for one test."""
    test_name:     str
    surg_knee:     KneeMetrics = None
    non_surg_knee: KneeMetrics = None
    surg_hip:      HipMetrics  = None
    non_surg_hip:  HipMetrics  = None

    # LSI for key metrics
    lsi_peak_flexion: float = np.nan
    lsi_peak_valgus:  float = np.nan
    lsi_hip_flexion:  float = np.nan


# ─── Single-side analysis ─────────────────────────────────────────────────────

def analyse_knee(
    flex_ext:       Optional[np.ndarray],   # sagittal: + = flexion
    valgus:         Optional[np.ndarray],   # frontal:  + = valgus
    tib_rot:        Optional[np.ndarray],   # transverse: + = internal rotation
    rate:           float,
    side:           str,
) -> KneeMetrics:
    """Compute all knee metrics from three optional angle time series."""
    m = KneeMetrics(side=side)

    if flex_ext is not None and len(flex_ext) > 0:
        fe = np.nan_to_num(flex_ext)
        m.flexion_at_contact_deg = float(fe[0])
        pk_idx = int(np.argmax(fe))
        m.peak_flexion_deg      = float(fe[pk_idx])
        m.flexion_excursion_deg = m.peak_flexion_deg - m.flexion_at_contact_deg

    if valgus is not None and len(valgus) > 0:
        vv = np.nan_to_num(valgus)
        m.valgus_at_contact_deg = float(vv[0])
        pk_idx = int(np.argmax(vv))
        m.peak_valgus_deg       = float(vv[pk_idx])
        m.mean_valgus_deg       = float(np.mean(vv))
        m.time_to_peak_valgus_s = float(pk_idx / rate)

    if tib_rot is not None and len(tib_rot) > 0:
        tr = np.nan_to_num(tib_rot)
        m.peak_tibial_ir_deg  = float(np.max(tr))
        m.tibial_rot_excursion = float(np.ptp(tr))

    return m


def analyse_hip(
    flex_ext:    Optional[np.ndarray],
    ab_adduction: Optional[np.ndarray],
    int_ext_rot: Optional[np.ndarray],
    rate:        float,
    side:        str,
) -> HipMetrics:
    """Compute hip metrics from three optional angle time series."""
    m = HipMetrics(side=side)

    if flex_ext is not None and len(flex_ext) > 0:
        fe = np.nan_to_num(flex_ext)
        m.flexion_at_contact_deg = float(fe[0])
        m.peak_flexion_deg       = float(np.max(fe))

    if ab_adduction is not None and len(ab_adduction) > 0:
        aa = np.nan_to_num(ab_adduction)
        m.peak_adduction_deg = float(np.max(aa))
        m.mean_adduction_deg = float(np.mean(aa))

    if int_ext_rot is not None and len(int_ext_rot) > 0:
        ir = np.nan_to_num(int_ext_rot)
        m.peak_ir_deg = float(np.max(ir))

    return m


# ─── Bilateral analysis ───────────────────────────────────────────────────────

def analyse_bilateral_kinematics(
    surg_knee_data:     dict,   # {flex_ext, valgus, tib_rot} — all optional arrays
    non_surg_knee_data: dict,
    rate:               float,
    test_name:          str,
    surg_hip_data:      dict = None,
    non_surg_hip_data:  dict = None,
) -> BilateralKinematics:
    """
    Run knee and hip analysis for both limbs and compute LSI values.
    Input dicts use keys: 'flex_ext', 'valgus', 'tib_rot' (knee)
                          'flex_ext', 'ab_adduction', 'int_ext_rot' (hip)
    All values are optional numpy arrays — missing keys are silently skipped.
    """
    def _g(d, k):
        return d.get(k) if d else None

    sk = analyse_knee(_g(surg_knee_data,     "flex_ext"),
                      _g(surg_knee_data,     "valgus"),
                      _g(surg_knee_data,     "tib_rot"),
                      rate, "Surgical")

    nk = analyse_knee(_g(non_surg_knee_data, "flex_ext"),
                      _g(non_surg_knee_data, "valgus"),
                      _g(non_surg_knee_data, "tib_rot"),
                      rate, "Non-Surgical")

    sh = analyse_hip(_g(surg_hip_data,     "flex_ext"),
                     _g(surg_hip_data,     "ab_adduction"),
                     _g(surg_hip_data,     "int_ext_rot"),
                     rate, "Surgical")

    nh = analyse_hip(_g(non_surg_hip_data, "flex_ext"),
                     _g(non_surg_hip_data, "ab_adduction"),
                     _g(non_surg_hip_data, "int_ext_rot"),
                     rate, "Non-Surgical")

    return BilateralKinematics(
        test_name=test_name,
        surg_knee=sk,
        non_surg_knee=nk,
        surg_hip=sh,
        non_surg_hip=nh,
        lsi_peak_flexion = lsi(sk.peak_flexion_deg, nk.peak_flexion_deg),
        lsi_peak_valgus  = lsi(sk.peak_valgus_deg,  nk.peak_valgus_deg),
        lsi_hip_flexion  = lsi(sh.peak_flexion_deg, nh.peak_flexion_deg),
    )


# ─── Utility ─────────────────────────────────────────────────────────────────

def trim_to_contact(angle: np.ndarray, fz: np.ndarray,
                    threshold: float = 20.0) -> np.ndarray:
    """
    Trim an angle signal to the contact window defined by fz > threshold.
    Handles the case where kinematics and GRF are at different sample rates.
    """
    in_contact = np.where(fz > threshold)[0]
    if len(in_contact) == 0:
        return angle

    scale = len(angle) / len(fz)
    k_start = int(in_contact[0]  * scale)
    k_end   = int(in_contact[-1] * scale)
    return angle[k_start:k_end + 1]
