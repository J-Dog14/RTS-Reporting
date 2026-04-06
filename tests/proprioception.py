"""
tests/proprioception.py
========================
Single-leg eyes-closed static balance — proprioception assessment.

Protocol
--------
Two conditions are exported, each with one trial per side:

  PCTL1 / PCTR1  — STANDARD:  eyes closed on flat ground (firm surface)
  PCTL2 / PCTR2  — AIREX BAG: eyes closed on unstable foam pad (harder)

Each condition is analysed independently.  The Airex condition will
naturally produce higher COP values — it is NOT averaged with the standard
condition.  Thresholds differ between conditions accordingly.

COP metrics (per side, per condition)
--------------------------------------
mean_velocity_mm_s    : mean COP speed (mm/s) — most sensitive RTR metric
total_excursion_mm    : total path length of COP (mm)
range_ap_mm           : anterior-posterior COP range (mm)
range_ml_mm           : mediolateral COP range (mm)
ellipse_area_mm2      : 95% confidence ellipse area (mm²)
rms_displacement_mm   : RMS distance from mean COP (mm)

Bilateral comparison (per condition)
--------------------------------------
lsi_cop_velocity  : LSI of mean COP velocity — (non_surg / surg) × 100
lsi_ellipse_area  : LSI of ellipse area

Note: higher COP velocity / area = worse balance (lower_better metric).
LSI = (non_surgical / surgical) × 100 so that 100% means equal stability.
Surgical < 100% means the surgical side has WORSE balance.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from helpers.grf import analyse_cop, COPMetrics


# ─── Result dataclasses ───────────────────────────────────────────────────────

@dataclass
class PropLimbResult:
    """Balance metrics for one stance limb in one condition."""
    side: str   # "Surgical" | "Non-Surgical"

    mean_velocity_mm_s:   float = np.nan
    total_excursion_mm:   float = np.nan
    range_ap_mm:          float = np.nan
    range_ml_mm:          float = np.nan
    ellipse_area_mm2:     float = np.nan
    rms_displacement_mm:  float = np.nan

    # COP arrays (trimmed, for figures)
    cop_x:  object = field(default=None, repr=False)   # ML (mm)
    cop_y:  object = field(default=None, repr=False)   # AP (mm)
    rate:   float  = np.nan


@dataclass
class PropConditionResult:
    """
    Results for one balance condition (standard firm surface or Airex foam).
    Contains separate surgical and non-surgical limb results.
    """
    condition:    str               # "Standard" | "Airex"
    surgical:     PropLimbResult = None
    non_surgical: PropLimbResult = None

    lsi_cop_velocity: float = np.nan   # (non_surg / surg) × 100
    lsi_ellipse_area: float = np.nan


@dataclass
class ProprioceptionResult:
    """
    Full proprioception result containing both balance conditions.

    standard : PCTL1 / PCTR1  — firm flat ground
    airex    : PCTL2 / PCTR2  — Airex foam pad (unstable)

    For backward compatibility the top-level surgical / non_surgical /
    lsi_* attributes mirror the STANDARD condition.
    """
    standard:     PropConditionResult = None   # flat ground
    airex:        PropConditionResult = None   # foam pad

    # Convenience aliases → standard condition
    @property
    def surgical(self) -> Optional[PropLimbResult]:
        return self.standard.surgical if self.standard else None

    @property
    def non_surgical(self) -> Optional[PropLimbResult]:
        return self.standard.non_surgical if self.standard else None

    @property
    def lsi_cop_velocity(self) -> float:
        return self.standard.lsi_cop_velocity if self.standard else np.nan

    @property
    def lsi_ellipse_area(self) -> float:
        return self.standard.lsi_ellipse_area if self.standard else np.nan


# ─── Main analysis function ───────────────────────────────────────────────────

def analyse(
    # Standard condition (PCTL1 / PCTR1 — firm surface)
    cop_x_surg_std:  Optional[np.ndarray],
    cop_y_surg_std:  Optional[np.ndarray],
    cop_x_ns_std:    Optional[np.ndarray],
    cop_y_ns_std:    Optional[np.ndarray],
    # Airex condition (PCTL2 / PCTR2 — foam pad)
    cop_x_surg_airex: Optional[np.ndarray],
    cop_y_surg_airex: Optional[np.ndarray],
    cop_x_ns_airex:   Optional[np.ndarray],
    cop_y_ns_airex:   Optional[np.ndarray],
    # Config
    rate_f:           float,
    surg_side:        str,
    trim_seconds:     float = 2.0,
) -> ProprioceptionResult:
    """
    Analyse COP data for both balance conditions independently.

    Parameters
    ----------
    cop_x_surg_std / cop_y_surg_std   : Standard trial COP arrays, surgical limb
    cop_x_ns_std   / cop_y_ns_std     : Standard trial COP arrays, non-surgical limb
    cop_x_surg_airex / cop_y_surg_airex : Airex trial COP arrays, surgical limb
    cop_x_ns_airex   / cop_y_ns_airex   : Airex trial COP arrays, non-surgical limb
    rate_f        : COFP sample rate (Hz)
    surg_side     : 'L' or 'R'
    trim_seconds  : seconds removed from each end of every trial
    """
    r = ProprioceptionResult()

    # ── Standard condition ────────────────────────────────────────────────────
    std_has_data = any(a is not None for a in
                       [cop_x_surg_std, cop_y_surg_std, cop_x_ns_std, cop_y_ns_std])
    if std_has_data:
        r.standard = _analyse_condition(
            condition="Standard",
            cop_x_surg=cop_x_surg_std, cop_y_surg=cop_y_surg_std,
            cop_x_ns=cop_x_ns_std,     cop_y_ns=cop_y_ns_std,
            rate_f=rate_f, trim_seconds=trim_seconds,
        )

    # ── Airex condition ───────────────────────────────────────────────────────
    airex_has_data = any(a is not None for a in
                         [cop_x_surg_airex, cop_y_surg_airex,
                          cop_x_ns_airex,   cop_y_ns_airex])
    if airex_has_data:
        r.airex = _analyse_condition(
            condition="Airex",
            cop_x_surg=cop_x_surg_airex, cop_y_surg=cop_y_surg_airex,
            cop_x_ns=cop_x_ns_airex,     cop_y_ns=cop_y_ns_airex,
            rate_f=rate_f, trim_seconds=trim_seconds,
        )

    return r


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _analyse_condition(condition: str,
                       cop_x_surg, cop_y_surg,
                       cop_x_ns,   cop_y_ns,
                       rate_f: float,
                       trim_seconds: float) -> PropConditionResult:
    """Analyse both limbs for a single balance condition."""
    c = PropConditionResult(condition=condition)

    c.surgical     = _analyse_limb(cop_x_surg, cop_y_surg, rate_f,
                                   trim_seconds, "Surgical")
    c.non_surgical = _analyse_limb(cop_x_ns,   cop_y_ns,   rate_f,
                                   trim_seconds, "Non-Surgical")

    c.lsi_cop_velocity = _prop_lsi(c.surgical.mean_velocity_mm_s,
                                    c.non_surgical.mean_velocity_mm_s)
    c.lsi_ellipse_area = _prop_lsi(c.surgical.ellipse_area_mm2,
                                    c.non_surgical.ellipse_area_mm2)
    return c


def _analyse_limb(cop_x: Optional[np.ndarray],
                  cop_y: Optional[np.ndarray],
                  rate: float,
                  trim_s: float,
                  label: str) -> PropLimbResult:
    """Trim and compute COP metrics for one limb / one trial."""
    result = PropLimbResult(side=label, rate=rate)

    if cop_x is None or cop_y is None or len(cop_x) < 20:
        return result

    x = _trim(cop_x, rate, trim_s)
    y = _trim(cop_y, rate, trim_s)

    if len(x) < 20:
        return result

    result.cop_x = x
    result.cop_y = y

    metrics: COPMetrics = analyse_cop(x, y, rate, label)

    result.mean_velocity_mm_s  = metrics.mean_velocity_mm_s
    result.total_excursion_mm  = metrics.total_excursion_mm
    result.range_ap_mm         = metrics.range_ap_mm
    result.range_ml_mm         = metrics.range_ml_mm
    result.ellipse_area_mm2    = metrics.ellipse_area_mm2
    result.rms_displacement_mm = metrics.rms_displacement_mm

    return result


def _trim(arr: np.ndarray, rate: float, seconds: float) -> np.ndarray:
    """Remove `seconds` from each end of an array."""
    n = int(seconds * rate)
    if 2 * n >= len(arr):
        return arr
    return arr[n:-n]


def _prop_lsi(surg_val: float, ns_val: float) -> float:
    """
    Proprioception LSI: (non_surgical / surgical) × 100.
    Used for lower-is-better metrics — surgical side worse → LSI < 100.
    """
    if (surg_val is None or ns_val is None or
            surg_val != surg_val or ns_val != ns_val or surg_val == 0):
        return np.nan
    return float(ns_val / surg_val) * 100.0
