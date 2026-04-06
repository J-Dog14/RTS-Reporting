"""
helpers/norms.py
=================
Published normative reference values for ACL return-to-sport assessment,
plus the RTR composite scoring engine.

RTR Scoring Model
-----------------
The score is NOT a simple weighted average of LSI values.  It uses three
evidence-based stages:

  1. Strict 3-zone metric scoring
       Green  (≥ good threshold) → 85–100 points
       Yellow (≥ caution threshold) → 50–85 points
       Red    (< caution threshold) → 0–50 points
     This is significantly more punishing than linear-to-threshold scaling.
     A value of 68.9% when the threshold is 90% scores ≈ 38, not 77.

  2. Red-flag penalty
       Every RED metric multiplies the base score by a penalty factor.
       A single red metric → ×0.88; two reds → ×0.76; three → ×0.65 etc.
       This means you cannot be "Ready" with significant deficits anywhere,
       even if other metrics are excellent.

  3. Time-since-surgery modifier
       A multiplicative factor on the performance score (see time_factor()):
         0–6 months   → factor 0.80 (flat)
         6–10 months  → linear ramp 0.80 → 1.00 (0.05 per month)
         ≥ 10 months  → factor 1.00
       Grade may still be capped separately by time (time_grade_cap()).
       Being far out from surgery does NOT inflate a poor score —
       a bad performer at 18 months still scores poorly.

Grade thresholds (applied to final time-adjusted score):
  ≥ 85  → Ready
  ≥ 68  → Progressing
  ≥ 50  → Caution
  < 50  → Not Ready

References
----------
[1]  Grindem et al. 2016 BJSM         — LSI thresholds & re-injury risk by month
[2]  Hewett et al. 2005 AJSM           — Valgus moment & ACL injury risk
[3]  Paterno et al. 2010 AJSM          — Neuromuscular deficits post-ACLR
[4]  Noyes et al. 1991 AJSM            — Single-leg hop & functional symmetry
[5]  Myer et al. 2006 AJSM             — Predictors of ACL injury
[6]  Walsh et al. 2007 JSR             — COP metrics in ACLR athletes
[7]  Sugimoto et al. 2014 AJSM         — Fatigue and neuromuscular control
[8]  Flanagan & Comyns 2008 NSCA J     — RSI interpretation
[9]  Buckthorpe et al. 2019 BJSM       — Strength & function criteria post-ACLR
[10] Dingenen & Gokeler 2017 Sports Med — RTR testing frameworks
[11] Kyritsis et al. 2016 BJSM         — Time < 9 months → 4–7× re-injury risk
[12] Webster & Feller 2019 OJSM        — 24-month graft maturation evidence
[13] Beischer et al. 2020 OJSM         — 65% don't meet criteria at 9 months
"""

from dataclasses import dataclass
from typing import Optional, Dict
import math


# ─── Norm container ───────────────────────────────────────────────────────────

@dataclass
class Norm:
    name:      str
    units:     str     = ""
    good:      Optional[float] = None    # at/above this = green (higher_better)
    caution:   Optional[float] = None    # at/above this = yellow, below = red
    direction: str     = "higher_better" # "higher_better" | "lower_better"
    citation:  str     = ""
    notes:     str     = ""

    def classify(self, value: float) -> str:
        """Returns 'green', 'yellow', 'red', or 'grey' (if no data)."""
        try:
            if value is None or value != value:
                return "grey"
            v = float(value)
        except (TypeError, ValueError):
            return "grey"

        if self.direction == "higher_better":
            if self.good is not None and v >= self.good:
                return "green"
            if self.caution is not None and v >= self.caution:
                return "yellow"
            return "red"
        else:   # lower_better
            if self.good is not None and v <= self.good:
                return "green"
            if self.caution is not None and v <= self.caution:
                return "yellow"
            return "red"

    def score_0_100(self, value: float) -> float:
        """Legacy linear scoring — kept for backwards compatibility."""
        return self.score_0_100_strict(value)

    def score_0_100_strict(self, value: float) -> float:
        """
        Strict 3-zone scoring:
          Green  (≥ good)                  →  85–100  (small bonus for exceeding)
          Yellow (≥ caution, < good)       →  50–85   (linear)
          Red    (< caution)               →   0–50   (linear — NOT zero for near-miss)

        This is deliberately more punishing than (v/good)*100.
        A value just barely in the red zone scores ~49 (not ~83), which
        has meaningful impact on the composite when combined with red-flag penalties.
        """
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 0.0
        if v != v:
            return 0.0

        if self.direction == "higher_better":
            good    = self.good    if self.good    is not None else 90.0
            caution = self.caution if self.caution is not None else 75.0
            if v >= good:
                # Green: 85 at threshold, up to 100 with 20% excess
                excess_pct = min((v - good) / good, 0.20)
                return min(100.0, 85.0 + (excess_pct / 0.20) * 15.0)
            elif v >= caution:
                # Yellow: linear 50→85
                t = (v - caution) / (good - caution)
                return 50.0 + t * 35.0
            else:
                # Red: linear 0→50  (v=0 → 0, v=caution → 50)
                t = v / caution if caution > 0 else 0.0
                return max(0.0, t * 50.0)

        else:  # lower_better
            good    = self.good    if self.good    is not None else 15.0
            caution = self.caution if self.caution is not None else 30.0
            if v <= good:
                return 100.0
            elif v <= caution:
                # Yellow: 85→50
                t = (v - good) / (caution - good)
                return 85.0 - t * 35.0
            else:
                # Red: 50→0.  Assume 3× the caution value = score of 0.
                red_max = caution * 3.0
                t = min(1.0, (v - caution) / (red_max - caution))
                return max(0.0, 50.0 - t * 50.0)


# ─── Normative database ───────────────────────────────────────────────────────

NORMS: Dict[str, Norm] = {

    # ── General LSI ─────────────────────────────────────────────────────────
    "lsi_general": Norm(
        name="Limb Symmetry Index",
        good=90, caution=75, units="%", direction="higher_better",
        citation="Grindem 2016 BJSM; Noyes 1991 AJSM",
        notes="≥90% is the widely-accepted RTR threshold; <75% = significant deficit",
    ),
    "lsi_rfd": Norm(
        name="LSI — Rate of Force Development",
        good=85, caution=70, units="%", direction="higher_better",
        citation="Buckthorpe 2019 BJSM",
        notes="RFD deficits often persist beyond peak-force deficits post-ACLR",
    ),
    "loading_rate_lsi": Norm(
        name="LSI — Loading Rate",
        good=85, caution=70, units="%", direction="higher_better",
        citation="Buckthorpe 2019 BJSM",
        notes="Initial loading rate is sensitive to protective landing strategy",
    ),

    # ── Drop Jump ────────────────────────────────────────────────────────────
    "drop_jump_rsi": Norm(
        name="Reactive Strength Index",
        good=1.4, caution=1.0, units="m/s", direction="higher_better",
        citation="Flanagan & Comyns 2008 NSCA J",
        notes="Trained athletes typically 1.4–2.0; <1.0 suggests protective landing",
    ),
    "drop_jump_rsi_mod": Norm(
        name="RSI Modified (flight÷contact)",
        good=0.60, caution=0.40, units="ratio", direction="higher_better",
        citation="Flanagan & Comyns 2008 NSCA J",
        notes="RSImod = flight time / contact time; removes height from equation",
    ),
    "drop_jump_contact_time": Norm(
        name="Drop Jump Contact Time",
        good=0.25, caution=0.35, units="s", direction="lower_better",
        citation="Flanagan & Comyns 2008 NSCA J",
        notes=">0.35 s suggests avoidance of rapid loading",
    ),
    "landing_lsi_200ms": Norm(
        name="Landing LSI (first 200 ms)",
        good=90, caution=78, units="%", direction="higher_better",
        citation="Paterno 2010 AJSM",
        notes="Bilateral loading symmetry in the first 200 ms of contact",
    ),

    # ── Kinematics ───────────────────────────────────────────────────────────
    "peak_knee_valgus": Norm(
        name="Peak Knee Valgus",
        good=8, caution=14, units="°", direction="lower_better",
        citation="Hewett 2005 AJSM; Myer 2006 AJSM",
        notes=">14° associated with elevated ACL injury risk",
    ),
    "knee_valgus_lsi": Norm(
        name="LSI — Knee Valgus",
        good=90, caution=75, units="%", direction="higher_better",
        citation="Paterno 2010 AJSM",
        notes="Valgus asymmetry is a sensitive ACL re-injury predictor",
    ),
    "tibial_ir": Norm(
        name="Peak Tibial Internal Rotation",
        good=10, caution=18, units="°", direction="lower_better",
        citation="Hewett 2005 AJSM",
        notes="Combined valgus + tibial IR creates the classic ACL injury mechanism",
    ),
    "peak_knee_flexion_landing": Norm(
        name="Peak Knee Flexion (landing)",
        good=60, caution=45, units="°", direction="higher_better",
        citation="Paterno 2010 AJSM",
        notes="Low flexion indicates a stiff landing strategy with higher joint loads",
    ),

    # ── Single-Leg Jump ──────────────────────────────────────────────────────
    "sl_jump_height_lsi": Norm(
        name="LSI — Single-Leg Jump Height",
        good=90, caution=75, units="%", direction="higher_better",
        citation="Grindem 2016 BJSM",
        notes="Strong predictor of re-injury risk post-ACLR",
    ),

    # ── Proprioception / Balance — LSI ───────────────────────────────────────
    "cop_velocity_lsi": Norm(
        name="LSI — COP Velocity",
        good=90, caution=75, units="%", direction="higher_better",
        citation="Walsh 2007 JSR",
        notes="One of the most sensitive RTR metrics for neuromuscular readiness",
    ),

    # ── Proprioception / Balance — ABSOLUTE values ────────────────────────────
    # These catch the clinically important case where BOTH limbs are poor
    # (LSI looks fine, but absolute performance is well below normative).
    # Normative EC single-leg stance on firm surface: ~10–15 mm/s.
    "cop_velocity_abs": Norm(
        name="COP Mean Velocity (Absolute)",
        good=15, caution=30, units="mm/s", direction="lower_better",
        citation="Walsh 2007 JSR; Paterno 2010 AJSM",
        notes="Healthy EC single-leg stance ≈ 10–15 mm/s. "
              "ACLR athletes typically 20–35 mm/s. "
              ">30 mm/s = yellow; >80 mm/s = well into red territory.",
    ),
    "cop_ellipse_area": Norm(
        name="COP 95% Ellipse Area",
        good=150, caution=300, units="mm²", direction="lower_better",
        citation="Paterno 2010 AJSM",
        notes="Normative EC single-leg stance: ~100–200 mm²",
    ),
    "cop_velocity": Norm(
        name="COP Mean Velocity",
        good=15, caution=25, units="mm/s", direction="lower_better",
        citation="Walsh 2007 JSR; Paterno 2010 AJSM",
        notes="Healthy EC single-leg stance: ~10–15 mm/s; ACLR athletes often 20–35",
    ),

    # ── Endurance Squat ──────────────────────────────────────────────────────
    "endurance_mean_lsi": Norm(
        name="Endurance Squat Mean LSI",
        good=90, caution=78, units="%", direction="higher_better",
        citation="Sugimoto 2014 AJSM",
        notes="Fatigued asymmetry is often >5% worse than fresh state",
    ),
    "endurance_fatigue_drift": Norm(
        name="Fatigue-Induced LSI Drift",
        good=-5, caution=-10, units="%", direction="higher_better",
        citation="Sugimoto 2014 AJSM",
        notes="Negative = worsening under fatigue. >10% drift is clinically significant",
    ),
}


# ─── Helper functions ─────────────────────────────────────────────────────────

def classify(metric_key: str, value: float) -> str:
    norm = NORMS.get(metric_key)
    if norm is None:
        return "grey"
    return norm.classify(value)


def get_norm(metric_key: str) -> Optional[Norm]:
    return NORMS.get(metric_key)


# ─── Time-since-surgery modifier ─────────────────────────────────────────────

# Control points (months, factor) — piecewise linear interpolation.
_TIME_CONTROL_POINTS = [
    (0,   0.80),
    (6,   0.80),
    (7,   0.85),
    (8,   0.90),
    (9,   0.95),
    (10,  1.00),
    (24,  1.00),
]


def time_factor(months: float) -> float:
    """
    Map months post-surgery to a 0.80–1.0 multiplier (see _TIME_CONTROL_POINTS).
    The final RTR score = performance_score × time_factor.

    Unknown months (None) → 1.0. Poor performance late post-op still scores poorly.
    """
    if months is None:
        return 1.0   # unknown — don't penalise; clinician must judge
    if months <= 0:
        return _TIME_CONTROL_POINTS[0][1]
    if months >= 24:
        return 1.0
    for i in range(len(_TIME_CONTROL_POINTS) - 1):
        m0, f0 = _TIME_CONTROL_POINTS[i]
        m1, f1 = _TIME_CONTROL_POINTS[i + 1]
        if m0 <= months <= m1:
            t = (months - m0) / (m1 - m0)
            return round(f0 + t * (f1 - f0), 4)
    return 1.0


def time_grade_cap(months: float) -> Optional[str]:
    """
    Return the maximum allowable grade based on time alone, or None
    if time is not a limiting factor.
    """
    if months is None:
        return None
    if months < 6:
        return "Not Ready"
    if months < 9:
        return "Caution"
    return None   # performance-driven above 9 months


# ─── RTR Composite Score ──────────────────────────────────────────────────────

# Each entry: "internal_label" -> ("norm_key", weight)
# Weights represent clinical importance; they are re-normalised to the
# subset of available metrics at run-time (so missing tests don't dilute).
RTR_WEIGHTS = {
    # ── Bilateral reactive loading ────────────────────────────────────────────
    "drop_jump_rsi":     ("drop_jump_rsi",        0.12),   # overall reactive capacity
    "landing_lsi":       ("landing_lsi_200ms",    0.10),   # bilateral landing symmetry
    "rfd_lsi":           ("lsi_rfd",              0.08),   # rate of force development

    # ── Propulsive strength & symmetry ───────────────────────────────────────
    "peak_grf_lsi":      ("lsi_general",          0.08),   # bilateral peak force sym
    "sl_jump_lsi":       ("sl_jump_height_lsi",   0.14),   # single-leg functional power

    # ── Kinematics ───────────────────────────────────────────────────────────
    "knee_valgus_surg":  ("peak_knee_valgus",     0.10),   # re-injury mechanism marker

    # ── Neuromuscular control / balance ──────────────────────────────────────
    "cop_velocity_lsi":  ("cop_velocity_lsi",     0.10),   # LSI symmetry
    "cop_vel_surg_abs":  ("cop_velocity_abs",     0.08),   # absolute surgical limb
    "cop_vel_ns_abs":    ("cop_velocity_abs",     0.04),   # absolute non-surgical limb

    # ── Endurance / fatigue ───────────────────────────────────────────────────
    "endurance_lsi":     ("endurance_mean_lsi",   0.08),   # mean symmetry under fatigue
    "fatigue_drift":     ("endurance_fatigue_drift", 0.08), # does LSI worsen under load?
}

# Penalty multiplier for each additional red metric.
# Thresholds chosen so 1 red → can't be "Ready"; 3 reds → can't be "Progressing".
_RED_PENALTY = [
    1.00,   # 0 reds
    0.88,   # 1 red
    0.76,   # 2 reds
    0.65,   # 3 reds
    0.55,   # 4 reds
    0.46,   # 5 reds
]


def compute_rtr_score(metric_values: dict) -> dict:
    """
    Compute a 0–100 RTR composite score with time-since-surgery modifier.

    metric_values : dict mapping RTR_WEIGHTS keys to raw float values.
                    Optionally include "_months_since_surgery" (float or None).

    Returns: {
        'score'          : float  — final 0–100 score
        'grade'          : str    — "Ready" | "Progressing" | "Caution" | "Not Ready"
        'breakdown'      : dict   — per-metric details
        'perf_score'     : float  — base performance score (before time modifier)
        'red_count'      : int
        'red_factor'     : float
        'time_factor'    : float
        'months_post_op' : float or None
        'time_note'      : str or None  — clinical warning if < 9 months
    }
    """
    months_post_op = metric_values.get("_months_since_surgery")

    # ── Step 1: Strict per-metric scores ─────────────────────────────────────
    weighted_sum  = 0.0
    total_weight  = 0.0
    breakdown     = {}
    red_count     = 0

    for label, (norm_key, weight) in RTR_WEIGHTS.items():
        value = metric_values.get(label)
        if value is None or (isinstance(value, float) and value != value):
            continue
        norm = NORMS.get(norm_key)
        if norm is None:
            continue

        color        = norm.classify(value)
        metric_score = norm.score_0_100_strict(value)

        if color == "red":
            red_count += 1

        weighted_sum  += metric_score * weight
        total_weight  += weight
        breakdown[label] = {
            "value":  value,
            "score":  round(metric_score, 1),
            "weight": weight,
            "color":  color,
        }

    perf_score = (weighted_sum / total_weight) if total_weight > 0 else 0.0

    # ── Step 2: Red-flag penalty ──────────────────────────────────────────────
    red_factor     = _RED_PENALTY[min(red_count, len(_RED_PENALTY) - 1)]
    penalised      = perf_score * red_factor

    # ── Step 3: Time-since-surgery modifier ───────────────────────────────────
    tf             = time_factor(months_post_op)
    final_score    = penalised * tf

    # ── Step 4: Grade (with hard time cap below 9 months) ────────────────────
    cap = time_grade_cap(months_post_op)

    if final_score >= 83:
        grade = "Ready"
    elif final_score >= 66:
        grade = "Progressing"
    elif final_score >= 50:
        grade = "Caution"
    else:
        grade = "Not Ready"

    # Apply time cap: e.g. at 7 months clinician cannot see "Progressing"
    _grade_order = ["Not Ready", "Caution", "Progressing", "Ready"]
    if cap is not None:
        cap_idx   = _grade_order.index(cap)
        grade_idx = _grade_order.index(grade)
        if grade_idx > cap_idx:
            grade = cap

    # ── Time note for the report ──────────────────────────────────────────────
    time_note = None
    if months_post_op is not None:
        if months_post_op < 6:
            time_note = (f"{months_post_op:.1f} months post-op — physiologically too early. "
                         f"Evidence requires ≥ 9 months minimum (Kyritsis 2016 BJSM).")
        elif months_post_op < 9:
            time_note = (f"{months_post_op:.1f} months post-op — below the 9-month evidence "
                         f"threshold. Risk is 4–7× higher than athletes who wait ≥ 9 months "
                         f"(Kyritsis 2016 BJSM). Grade capped at Caution.")
        elif months_post_op < 12:
            time_note = (f"{months_post_op:.1f} months post-op — within evidence window. "
                         f"RTR possible if all performance criteria met.")
        else:
            time_note = (f"{months_post_op:.1f} months post-op — time no longer "
                         f"the limiting factor; performance criteria are decisive.")

    return {
        "score":          round(final_score, 1),
        "grade":          grade,
        "breakdown":      breakdown,
        "perf_score":     round(perf_score, 1),
        "red_count":      red_count,
        "red_factor":     round(red_factor, 2),
        "time_factor":    round(tf, 3),
        "months_post_op": months_post_op,
        "time_note":      time_note,
    }
