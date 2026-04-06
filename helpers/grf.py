"""
helpers/grf.py
===============
Ground Reaction Force analysis functions.

All functions take raw numpy arrays and return plain numbers or arrays —
no hidden dependencies, easy to call from any test module.

Key functions
-------------
detect_contacts(fz)               → list of (start, end) frame tuples
peak_force(fz)                    → (value_N, frame_index)
impulse(fz, rate)                 → float (N·s)
loading_rate(fz, rate)            → float (N/s, first 50 ms)
rfd(fz, rate)                     → float (N/s, rate of force development)
lsi(surg, non_surg)               → float (%, limb symmetry index)
asymmetry_index(surg, non_surg)   → float (%, asymmetry index)
split_phases(fz)                  → (eccentric_array, concentric_array)
analyse_side(fz, rate, bw_n)      → SideMetrics dataclass
analyse_bilateral(...)            → BilateralResult dataclass
analyse_cop(cop_x, cop_y, rate)   → COPMetrics dataclass
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GRF_CONTACT_THRESHOLD, MIN_CONTACT_FRAMES, RFD_WINDOW_MS, LOADING_RATE_WINDOW_MS


# ─── Data containers ──────────────────────────────────────────────────────────

@dataclass
class SideMetrics:
    """GRF metrics for one limb in one phase."""
    side: str                        # "Surgical" | "Non-Surgical"
    peak_force_N:    float = np.nan
    peak_force_bw:   float = np.nan  # normalised to body weight
    impulse_Ns:      float = np.nan
    mean_force_N:    float = np.nan
    duration_s:      float = np.nan
    loading_rate_Ns: float = np.nan  # N/s in first LOADING_RATE_WINDOW_MS
    rfd_Ns:          float = np.nan  # N/s rate of force development
    peak_frame:      int   = 0


@dataclass
class BilateralResult:
    """Bilateral GRF comparison for one test / phase."""
    label: str
    surgical:    SideMetrics = None
    non_surgical: SideMetrics = None
    lsi_peak:    float = np.nan   # (surg / non_surg) × 100
    lsi_impulse: float = np.nan
    lsi_rfd:     float = np.nan
    ai_peak:     float = np.nan   # asymmetry index


@dataclass
class COPMetrics:
    """Centre-of-pressure balance metrics."""
    side: str
    mean_velocity_mm_s:  float = np.nan
    total_excursion_mm:  float = np.nan
    range_ap_mm:         float = np.nan
    range_ml_mm:         float = np.nan
    ellipse_area_mm2:    float = np.nan  # 95% confidence ellipse
    rms_displacement_mm: float = np.nan


# ─── Core calculation functions ───────────────────────────────────────────────

def detect_contacts(fz: np.ndarray,
                    threshold: float = None,
                    min_frames: int = None) -> List[Tuple[int, int]]:
    """
    Find continuous periods where Fz exceeds the threshold.
    Returns list of (start_frame, end_frame) tuples.

    threshold  : minimum force in N to count as contact (default from config)
    min_frames : ignore contacts shorter than this (filters edge noise)
    """
    if threshold is None:
        threshold = GRF_CONTACT_THRESHOLD
    if min_frames is None:
        min_frames = MIN_CONTACT_FRAMES

    in_contact = fz > threshold
    contacts, start = [], None

    for i, c in enumerate(in_contact):
        if c and start is None:
            start = i
        elif not c and start is not None:
            if (i - 1 - start) >= min_frames:
                contacts.append((start, i - 1))
            start = None

    if start is not None and (len(fz) - 1 - start) >= min_frames:
        contacts.append((start, len(fz) - 1))

    return contacts


def peak_force(fz: np.ndarray) -> Tuple[float, int]:
    """Return (peak_Fz_N, frame_index)."""
    idx = int(np.nanargmax(fz))
    return float(fz[idx]), idx


def impulse(fz: np.ndarray, rate: float) -> float:
    """Total GRF impulse using the trapezoid rule (N·s)."""
    return float(np.trapz(fz, dx=1.0 / rate))


def loading_rate(fz: np.ndarray, rate: float, window_ms: float = None) -> float:
    """
    Average loading rate over the first window_ms of the signal.
    Returns N/s.
    """
    if window_ms is None:
        window_ms = LOADING_RATE_WINDOW_MS
    n_frames = max(2, int(window_ms * rate / 1000))
    seg = fz[:n_frames]
    if len(seg) < 2:
        return np.nan
    return float((seg[-1] - seg[0]) / (len(seg) / rate))


def rfd(fz: np.ndarray, rate: float, window_ms: float = None) -> float:
    """
    Rate of Force Development: slope from contact to peak within window_ms.
    Returns N/s.
    """
    if window_ms is None:
        window_ms = RFD_WINDOW_MS
    n_frames = max(2, int(window_ms * rate / 1000))
    seg = fz[:n_frames]
    if len(seg) < 2:
        return np.nan
    pk_idx = int(np.nanargmax(seg))
    delta_t = pk_idx / rate
    if delta_t == 0:
        return np.nan
    return float((seg[pk_idx] - seg[0]) / delta_t)


def lsi(surg_val: float, non_surg_val: float) -> float:
    """
    Limb Symmetry Index (%).
    Formula: (surgical / non-surgical) × 100
    Returns NaN if either value is invalid.
    """
    if (non_surg_val is None or surg_val is None or
            non_surg_val != non_surg_val or surg_val != surg_val or
            non_surg_val == 0):
        return np.nan
    return float(surg_val / non_surg_val) * 100.0


def asymmetry_index(surg_val: float, non_surg_val: float) -> float:
    """
    Asymmetry Index (%).
    Formula: |surg - non_surg| / (surg + non_surg) × 100
    """
    try:
        denom = abs(surg_val) + abs(non_surg_val)
        if denom == 0:
            return np.nan
        return abs(surg_val - non_surg_val) / denom * 100.0
    except (TypeError, ValueError):
        return np.nan


def split_phases(fz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a force-time curve at peak force into:
        eccentric  (contact → peak)    — loading phase
        concentric (peak → toe-off)    — push-off phase
    """
    _, pk_idx = peak_force(fz)
    return fz[:pk_idx + 1], fz[pk_idx:]


# ─── Higher-level analysis ────────────────────────────────────────────────────

def analyse_side(fz: np.ndarray, rate: float, bw_n: float,
                 side_label: str) -> SideMetrics:
    """
    Compute all GRF metrics for one limb segment.

    fz        : 1-D numpy array of vertical GRF (already clipped to contact)
    rate      : sample rate in Hz
    bw_n      : body weight in Newtons (for normalisation)
    side_label: "Surgical" or "Non-Surgical"
    """
    m = SideMetrics(side=side_label)
    if fz is None or len(fz) == 0:
        return m

    fz = np.nan_to_num(fz, nan=0.0)
    pk_val, pk_idx = peak_force(fz)

    m.peak_force_N    = pk_val
    m.peak_force_bw   = pk_val / bw_n if bw_n > 0 else np.nan
    m.peak_frame      = pk_idx
    m.impulse_Ns      = impulse(fz, rate)
    m.mean_force_N    = float(np.mean(fz))
    m.duration_s      = len(fz) / rate
    m.loading_rate_Ns = loading_rate(fz, rate)
    m.rfd_Ns          = rfd(fz, rate)

    return m


def analyse_bilateral(fz_surg: np.ndarray, fz_non_surg: np.ndarray,
                      rate: float, bw_n: float,
                      label: str = "Overall") -> BilateralResult:
    """
    Full bilateral GRF analysis for one trial / phase.
    Returns a BilateralResult with LSI values computed.
    """
    surg_m = analyse_side(fz_surg,     rate, bw_n, "Surgical")
    ns_m   = analyse_side(fz_non_surg, rate, bw_n, "Non-Surgical")

    return BilateralResult(
        label=label,
        surgical=surg_m,
        non_surgical=ns_m,
        lsi_peak    = lsi(surg_m.peak_force_N, ns_m.peak_force_N),
        lsi_impulse = lsi(surg_m.impulse_Ns,   ns_m.impulse_Ns),
        lsi_rfd     = lsi(surg_m.rfd_Ns,       ns_m.rfd_Ns),
        ai_peak     = asymmetry_index(surg_m.peak_force_N, ns_m.peak_force_N),
    )


def analyse_phased(fz_surg: np.ndarray, fz_non_surg: np.ndarray,
                   rate: float, bw_n: float) -> dict:
    """
    Run bilateral analysis for eccentric, concentric, and overall phases.
    Returns: {'eccentric': BilateralResult, 'concentric': BilateralResult, 'overall': BilateralResult}
    """
    ecc_s,  con_s  = split_phases(fz_surg)
    ecc_ns, con_ns = split_phases(fz_non_surg)

    return {
        "eccentric":  analyse_bilateral(ecc_s,   ecc_ns,  rate, bw_n, "Eccentric"),
        "concentric": analyse_bilateral(con_s,   con_ns,  rate, bw_n, "Concentric"),
        "overall":    analyse_bilateral(fz_surg, fz_non_surg, rate, bw_n, "Overall"),
    }


def analyse_cop(cop_x: np.ndarray, cop_y: np.ndarray,
                rate: float, side: str) -> COPMetrics:
    """
    Compute Centre-of-Pressure balance metrics for one limb.

    cop_x : mediolateral COP (mm)
    cop_y : anterior-posterior COP (mm)
    rate  : sample rate in Hz
    side  : "Surgical" | "Non-Surgical"
    """
    m = COPMetrics(side=side)
    if cop_x is None or cop_y is None or len(cop_x) < 10:
        return m

    cop_x = np.nan_to_num(cop_x)
    cop_y = np.nan_to_num(cop_y)
    dt = 1.0 / rate

    dx = np.diff(cop_x)
    dy = np.diff(cop_y)
    path = np.sqrt(dx**2 + dy**2)

    m.total_excursion_mm  = float(np.sum(path))
    m.mean_velocity_mm_s  = float(np.mean(path) / dt) if len(path) > 0 else np.nan
    m.range_ap_mm         = float(np.ptp(cop_y))
    m.range_ml_mm         = float(np.ptp(cop_x))
    # RMS displacement from mean position (translation-invariant)
    cx = cop_x - np.mean(cop_x)
    cy = cop_y - np.mean(cop_y)
    m.rms_displacement_mm = float(np.sqrt(np.mean(cx**2 + cy**2)))

    # 95% confidence ellipse area from covariance eigenvalues
    try:
        cov = np.cov(cop_x - np.mean(cop_x), cop_y - np.mean(cop_y))
        ev  = np.linalg.eigvalsh(cov)
        chi2_95 = 5.991  # chi-squared, 2 DOF, 95%
        m.ellipse_area_mm2 = float(np.pi * chi2_95
                                   * np.sqrt(max(ev[0], 0))
                                   * np.sqrt(max(ev[1], 0)))
    except Exception:
        pass

    return m
