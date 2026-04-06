"""
tests/endurance_squat.py
=========================
30-second full-ROM bilateral squat — continuous repetitions.

Key insight: FATIGUE-INDUCED ASYMMETRY DRIFT.  Does LSI worsen over the 30 s?
Neuromuscular deficits that a single-rep test misses often reveal themselves
here, when protective strategies break down under fatigue.

Metrics computed
----------------
From GRF time-series (Endurance Squat Forces.txt):
    n_cycles             : number of squat reps detected
    mean_lsi_peak        : average per-rep LSI across all reps (%)
    lsi_first_third      : mean LSI, first third of reps (%)
    lsi_last_third       : mean LSI, final third of reps (%)
    fatigue_drift_pct    : lsi_last_third - lsi_first_third  (negative = worsening)
    peak_force_surg/ns_N : mean peak Fz per limb across all reps (N)

From Joints.txt (optional):
    valgus_drift_deg     : mean valgus angle shift, last vs first third (surgical)
    mean_peak_valgus_surg: average peak valgus angle, surgical limb (°)

Arrays for plotting:
    lsi_over_time        : per-rep LSI values (numpy array)
    time_axis            : rep timestamps in seconds (numpy array)
    reps                 : list of SquatRep objects
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from helpers.grf import lsi


@dataclass
class SquatRep:
    """Metrics for one squat repetition."""
    rep_number:       int
    timestamp_s:      float
    peak_force_surg:  float = np.nan
    peak_force_ns:    float = np.nan
    lsi_peak:         float = np.nan
    peak_valgus_surg: float = np.nan
    peak_valgus_ns:   float = np.nan


@dataclass
class EnduranceSquatResult:
    n_cycles:              int   = 0
    reps:                  List[SquatRep] = field(default_factory=list)

    mean_lsi_peak:         float = np.nan
    lsi_first_third:       float = np.nan
    lsi_last_third:        float = np.nan
    fatigue_drift_pct:     float = np.nan   # negative = worsening under fatigue

    peak_force_surg_N:     float = np.nan
    peak_force_ns_N:       float = np.nan

    mean_peak_valgus_surg: float = np.nan
    valgus_drift_deg:      float = np.nan   # positive = worsening

    lsi_over_time:         object = field(default=None, repr=False)
    time_axis:             object = field(default=None, repr=False)

    # ── Raw arrays (for figures) ──────────────────────────────────────────────
    fz_surg:  object = field(default=None, repr=False)
    fz_ns:    object = field(default=None, repr=False)
    rate_f:   float  = np.nan


def analyse(
    fz_surg:          Optional[np.ndarray],
    fz_ns:            Optional[np.ndarray],
    rate_f:           float,
    bw_n:             float,
    surg_side:        str,
    knee_valgus_surg: Optional[np.ndarray] = None,
    knee_valgus_ns:   Optional[np.ndarray] = None,
    rate_k:           float = 200.0,
    scalars:          dict  = None,
) -> EnduranceSquatResult:
    """
    Run the 30-second endurance squat analysis.

    fz_surg / fz_ns      : vertical GRF time series per limb (N), full 30 s
    rate_f               : GRF sample rate (Hz)
    bw_n                 : body weight in Newtons
    surg_side            : 'L' or 'R'
    knee_valgus_surg/ns  : 1-D array of knee valgus angle (°), same duration
    rate_k               : kinematic sample rate (Hz)
    scalars              : flat dict of V3D pre-computed metrics (optional)
    """
    r = EnduranceSquatResult()
    if scalars is None:
        scalars = {}

    if fz_surg is None or fz_ns is None:
        return r

    r.fz_surg = fz_surg
    r.fz_ns   = fz_ns
    r.rate_f  = rate_f

    # ── Combined signal and smoothing ─────────────────────────────────────────
    fz_total = fz_surg + fz_ns

    win = max(1, int(0.05 * rate_f))   # 50 ms smoothing window
    kernel = np.ones(win) / win
    smooth = np.convolve(fz_total, kernel, mode="same")

    # ── Find squat rep peaks (each standing phase = local max in combined Fz) ─
    min_dist = int(0.5 * rate_f)    # at least 0.5 s between reps
    peaks = _find_peaks(smooth, min_dist)

    if len(peaks) < 2:
        return r   # not enough reps detected

    # ── Per-rep analysis ──────────────────────────────────────────────────────
    for i in range(len(peaks) - 1):
        s_f = peaks[i]
        e_f = peaks[i + 1]

        seg_s  = fz_surg[s_f:e_f]
        seg_ns = fz_ns[s_f:e_f]
        if len(seg_s) == 0:
            continue

        pk_s  = float(np.nanmax(seg_s))
        pk_ns = float(np.nanmax(seg_ns))

        rep = SquatRep(
            rep_number=i + 1,
            timestamp_s=s_f / rate_f,
            peak_force_surg=pk_s,
            peak_force_ns=pk_ns,
            lsi_peak=lsi(pk_s, pk_ns),
        )

        # Peak valgus in this rep window (resampled from kinematic rate)
        if knee_valgus_surg is not None and len(knee_valgus_surg) > 0:
            rep.peak_valgus_surg = _peak_in_window(
                knee_valgus_surg, s_f, e_f, rate_f, rate_k)

        if knee_valgus_ns is not None and len(knee_valgus_ns) > 0:
            rep.peak_valgus_ns = _peak_in_window(
                knee_valgus_ns, s_f, e_f, rate_f, rate_k)

        r.reps.append(rep)

    r.n_cycles = len(r.reps)
    if r.n_cycles == 0:
        return r

    # ── Summary statistics ────────────────────────────────────────────────────
    lsi_vals  = np.array([rep.lsi_peak       for rep in r.reps if not np.isnan(rep.lsi_peak)])
    times     = np.array([rep.timestamp_s    for rep in r.reps if not np.isnan(rep.lsi_peak)])
    pk_s_all  = np.array([rep.peak_force_surg for rep in r.reps if not np.isnan(rep.peak_force_surg)])
    pk_ns_all = np.array([rep.peak_force_ns   for rep in r.reps if not np.isnan(rep.peak_force_ns)])

    r.lsi_over_time = lsi_vals
    r.time_axis     = times
    r.mean_lsi_peak = float(np.nanmean(lsi_vals)) if len(lsi_vals) > 0 else np.nan
    r.peak_force_surg_N = float(np.nanmean(pk_s_all)) if len(pk_s_all) > 0 else np.nan
    r.peak_force_ns_N   = float(np.nanmean(pk_ns_all)) if len(pk_ns_all) > 0 else np.nan

    # ── Fatigue drift — compare first and last third of reps ─────────────────
    if r.n_cycles >= 3:
        third = max(1, r.n_cycles // 3)
        r.lsi_first_third   = float(np.nanmean(lsi_vals[:third]))
        r.lsi_last_third    = float(np.nanmean(lsi_vals[-third:]))
        r.fatigue_drift_pct = r.lsi_last_third - r.lsi_first_third

    # ── Valgus fatigue drift ──────────────────────────────────────────────────
    valgus_all = [rep.peak_valgus_surg for rep in r.reps
                  if not np.isnan(rep.peak_valgus_surg)]
    if len(valgus_all) >= 3:
        r.mean_peak_valgus_surg = float(np.nanmean(valgus_all))
        third = max(1, len(valgus_all) // 3)
        early = float(np.nanmean(valgus_all[:third]))
        late  = float(np.nanmean(valgus_all[-third:]))
        r.valgus_drift_deg = late - early

    return r


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _find_peaks(arr: np.ndarray, min_dist_frames: int) -> List[int]:
    """Simple local-maximum peak detector (no scipy dependency)."""
    peaks = []
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_dist_frames:
                peaks.append(i)
    return peaks


def _peak_in_window(arr: np.ndarray, s_f: int, e_f: int,
                    rate_f: float, rate_k: float) -> float:
    """
    Extract the peak value of `arr` (sampled at rate_k) that corresponds
    to the GRF frame window [s_f, e_f] (sampled at rate_f).
    """
    scale = rate_k / rate_f
    ks = int(s_f * scale)
    ke = int(e_f * scale)
    ke = min(ke, len(arr))
    if ke <= ks or ks >= len(arr):
        return np.nan
    seg = arr[ks:ke]
    return float(np.nanmax(seg)) if len(seg) > 0 else np.nan
