"""
profiles/achilles.py
====================
Achilles tendon repair return-to-sport profile.

Clinical emphasis (vs the ACL profile)
--------------------------------------
• The calf–Achilles unit is the limiting tissue. Its job in sport is fast,
  stiff loading on short ground contacts (sprinting, cutting, re-jumping), so
  RATE metrics (RFD, loading rate, impact transient, RSI / contact time) are
  weighted above peak-force magnitude.
• Calf capacity is tested directly with a manually entered single-leg heel
  raise to failure (Silbernagel et al.). Heel-rise deficits commonly persist
  after hop distance has normalised, so triple-hop is shown but not scored.
• Knee valgus / tibial IR are kept in the tables but are NOT treated as
  injury-mechanism red flags. A consistent surgical-side external rotation
  offset is described as a possible toe-out / forefoot-avoidance strategy.
• Ankle angle / moment rows appear when L/R_Ankle_Angle and L/R_Ankle_Moment
  are present in the V3D export; otherwise the report says they are missing.

References
----------
Silbernagel KG et al. — single-leg heel-rise test / heel-rise work LSI after Achilles rupture
Zellers JA et al. 2016 BJSM — return to play after Achilles rupture (systematic review)
Trofa DP et al. 2017 AJSM — professional athletes' return to play after Achilles repair
"""

from profiles.base import ReportProfile, ManualTest
from helpers.norms import Norm


def _domain(label, default):
    """Radar domains for the Achilles report (calf first, knee items left off)."""
    L = label.lower()
    if "ankle" in L or "heel" in L:
        return "Calf"
    if "knee" in L:
        return None
    if L.startswith("balance"):
        return "Balance"
    if L.startswith("endurance"):
        return "Endurance"
    if L.startswith("triple"):
        return "Hop"
    if L.startswith("dj"):
        return "Reactive"
    if L.startswith("dl"):
        return "Landing"
    if L.startswith("sl") or L.startswith("cmj"):
        return "Power"
    return default


def _time_note(m):
    if m is None:
        return None
    if m < 6:
        return (f"{m:.1f} months post-repair — early; tendon remodelling is ongoing and calf capacity "
                f"is rarely restored this soon.")
    if m < 9:
        return (f"{m:.1f} months post-repair — return to sprinting/cutting sport usually falls around "
                f"9–12 months; clearance should be driven by calf capacity and rate criteria.")
    if m < 12:
        return f"{m:.1f} months post-repair — within the usual return window if calf and rate criteria are met."
    return f"{m:.1f} months post-repair — time no longer the limiting factor; performance criteria are decisive."


PROFILE = ReportProfile(
    key="achilles",
    menu_label="Achilles",
    menu_blurb="Achilles tendon repair — calf capacity, rate of loading, reactive strength",
    report_subtitle="Return to Sport Assessment<br/><font size='10'>Achilles Repair</font>",
    file_tag="Achilles",

    # label -> (norm_key, weight). Renormalised to whatever is available.
    rtr_weights={
        # Calf capacity — the primary criterion
        "heel_raise_lsi":   ("heel_raise_lsi",     0.20),
        "ankle_moment_lsi": ("lsi_general",        0.04),   # peak PF moment (SLJ)
        "ankle_work_lsi":   ("lsi_general",        0.10),   # ankle positive work at SLJ push-off
        # Single-leg power
        "sl_jump_lsi":      ("sl_jump_height_lsi", 0.16),
        # Fast loading / reactive (the Achilles' job)
        "drop_jump_rsi":    ("drop_jump_rsi",      0.10),
        "rfd_lsi":          ("lsi_rfd",            0.10),
        "dj_load_rate_lsi": ("loading_rate_lsi",   0.08),
        "dl_load_rate_lsi": ("loading_rate_lsi",   0.06),
        "landing_lsi":      ("landing_lsi_200ms",  0.06),
        # Magnitude / control / endurance
        "peak_grf_lsi":     ("lsi_general",        0.04),
        "dl_peak_grf_lsi":  ("lsi_general",        0.04),
        "cop_velocity_lsi": ("cop_velocity_lsi",   0.06),
        "cop_velocity_airex_lsi": ("cop_velocity_lsi", 0.04),
        "sl_power_lsi":     ("sl_jump_height_lsi", 0.06),   # SL peak power W/kg
        "endurance_lsi":    ("endurance_mean_lsi", 0.04),
        "fatigue_drift":    ("endurance_fatigue_drift", 0.02),
        # Knee valgus intentionally excluded — not the injury mechanism here.
        # Triple hop shown but not scored — hop distance normalises early.
    },
    extra_norms={
        "heel_raise_lsi": Norm(
            name="LSI — Single-Leg Heel Raise",
            good=90, caution=75, units="%", direction="higher_better",
            citation="Silbernagel et al.",
            notes="Heel-rise deficits often persist after hop symmetry has returned",
        ),
    },
    time_points=[(0, 0.80), (6, 0.80), (8, 0.88), (10, 0.95), (12, 1.00), (24, 1.00)],
    time_note=_time_note,
    influence_names={
        "heel_raise_lsi":   "Heel raise (calf capacity)",
        "ankle_moment_lsi": "Ankle PF moment sym.",
        "ankle_work_lsi":   "Ankle push-off work (SLJ)",
        "cop_velocity_airex_lsi": "Balance symmetry (Airex)",
        "sl_power_lsi":     "SL peak power",
        "dj_load_rate_lsi": "DJ Load rate sym.",
    },
    flag_knee_frontal=False,
    ankle_focus=True,
    # Layout: Calf & Ankle page after the summary, then tests in Achilles priority
    extra_pages=["calf_ankle"],
    section_order=["drop_jump", "single_leg_jump", "max_vertical", "drop_landing",
                   "proprioception", "endurance_squat"],
    domain_map=_domain,
    chart_exclude=["Knee"],          # knee flexion / knee moments stay on the test pages
    manual_tests=[
        ManualTest(
            key="heel_raise",
            title="Single-Leg Heel Raise",
            prompt="Reps to failure, each leg (full height, 1 rep / 2 s)\n"
                   "Press Enter to confirm · Esc to skip",
            label="Heel Raise",
            unit="reps",
            domain="Calf",
            rtr_key="heel_raise_lsi",
            integer=True,
        ),
    ],
    text={
        "summary": {
            "focus": ("Achilles profile — rate-of-loading, reactive and calf-capacity metrics are weighted above "
                      "peak force. Knee valgus / tibial rotation are shown for context, not scored."),
            "ankle_missing": ("Ankle angle / moment signals were not found in the V3D export. Add L/R_Ankle_Angle "
                              "and L/R_Ankle_Moment to the export pipeline for plantarflexor-specific metrics."),
            "heel_missing": "",      # heel raise is optional — the Calf & Ankle page says "not collected"
        },
        "drop_jump": {
            "rsi_low": "Long contact / low reactive strength — the surgical calf–Achilles unit is not yet tolerating "
                       "fast stretch-shortening loads. Progress pogo → short-contact drop jumps.",
            "rsi_slow": "The RSI number clears 1.30, but only with a long contact — this is a slow rebound, "
                        "not a reactive (fast stretch-shortening) jump, so it does not clear the calf.",
            "rate_gap": "Force magnitude is close to symmetric but rate is not — the surgical limb avoids fast "
                        "loading. This is the typical Achilles pattern; train rate, not just strength.",
            "targets": "<b>Targets:</b> RSI ≥ 1.30 m/s · Contact < 0.25 s · RFD & load rate LSI ≥ 85–90% · "
                       "Landing LSI 90–110%.",
            "cap_grf":    "Bilateral GRF (normalised to BW). A slower, lower initial rise on the surgical side = "
                          "reduced fast loading through the calf–Achilles unit.",
            "cap_flex":   "Knee flexion. Deeper knee bend on landing can be used to offload the ankle.",
            "cap_ankle":  "Ankle dorsiflexion. Reduced surgical dorsiflexion under load = calf guarding; excess = "
                          "tendon elongation / reduced stiffness.",
            "cap_valgus": "Knee valgus (context only for Achilles — not scored).",
            "cap_tib":    "Tibial / foot rotation. A consistent surgical-side external-rotation offset can reflect "
                          "toe-out (avoiding push-off through the forefoot) or a static calibration offset — confirm with video.",
            "cap_ankle_mom": "Plantarflexor moment through landing and rebound — how hard the calf–Achilles unit is loaded.",
            "cap_ankle_pow": "Ankle power: below zero = calf absorbing the landing, above zero = calf driving the rebound.",
            "cap_cop":    "AP COP during the first contact (zero-centred). Posterior / heel bias on the surgical side "
                          "= reduced forefoot loading through the Achilles.",
            "description": (
                "The Drop Jump is a reactive bilateral task: the athlete drops from a box and immediately jumps as high "
                "and as quickly as possible. It loads the calf–Achilles unit through a fast stretch-shortening cycle — "
                "the same demand as sprinting and cutting. After Achilles repair the common compensations are a longer "
                "ground contact, a slower and lower initial force rise on the surgical limb (reduced RFD / loading rate), "
                "a heel-biased landing, and shifting work to the knee and hip. The Reactive Strength Index (jump height "
                "÷ contact time) summarises how well the athlete uses elastic energy; contact time itself shows whether "
                "the jump was truly reactive (< 0.25 s) or a slow, guarded rebound."),
        },
        "drop_landing": {
            "intro":   "Absorptive landing — assesses how quickly and symmetrically the surgical limb accepts load.",
            "rate_gap": "Peak force is closer to symmetric than loading rate — the surgical limb is accepting load "
                        "slowly (heel-first / guarded).",
            "targets": "<b>Targets:</b> Load rate LSI ≥ 85–90% · Impact transient LSI 90–110% · Peak GRF LSI ≥ 90%.",
            "cap_grf":    "Bilateral GRF. Lower, later initial peak on the surgical side = slower load acceptance.",
            "cap_flex":   "Knee flexion at landing. Extra knee bend can substitute for ankle absorption.",
            "cap_ankle":  "Ankle dorsiflexion at landing — eccentric calf control.",
            "cap_valgus": "Knee valgus at landing (context only for Achilles — not scored).",
            "cap_tib":    "Tibial / foot rotation. Surgical-side external rotation offset = possible toe-out strategy.",
            "cap_ankle_mom": "Plantarflexor moment during landing — eccentric calf loading.",
            "cap_ankle_pow": "Ankle power during landing: the area below zero is the energy the calf absorbs.",
            "cap_cop":    "AP COP during landing contact (zero-centred). Posterior bias on surgical side = heel-dominant / "
                          "reduced forefoot loading.",
            "description": (
                "The Drop Landing isolates load acceptance: the athlete drops from a box and sticks the landing without "
                "a re-jump. After Achilles repair the surgical limb often lands heel-first and accepts load more slowly, "
                "which shows up as a lower loading rate and impact transient rather than a lower peak force. Symmetric "
                "loading rate means the calf is eccentrically controlling the forefoot-to-heel transition on both sides."),
        },
        "max_vertical": {
            "intro":   "Bilateral CMJ — slow stretch-shortening; usually recovers before fast-SSC tasks after Achilles repair.",
            "pk_low":  "reduced triple-extension push-off; check calf and hip/knee extensor contribution",
            "work_gap": "The surgical ankle contributes less push-off work while the hip makes up the difference "
                        "— the calf is being bypassed in the propulsive phase.",
            "targets": "Targets: ≥ 90% propulsion, ≥ 90% peak force.",
            "cap_grf":    "Full CMJ: unweighting → eccentric → propulsion → flight. Propulsive peaks should be symmetric.",
            "cap_flex":   "Knee flexion depth. Asymmetric countermovement depth = compensatory loading strategy.",
            "cap_ankle":  "Ankle dorsiflexion — the final push-off through plantarflexion drives jump height.",
            "cap_valgus": "Knee valgus (context only for Achilles — not scored).",
            "cap_tib":    "Tibial / foot rotation. Surgical-side external rotation offset = possible toe-out strategy.",
            "cap_ankle_mom": "Plantarflexor moment through the countermovement and push-off.",
            "cap_ankle_pow": "Ankle power: the late positive burst before take-off is the calf's push-off.",
            "cap_cop":    "AP COP during the propulsive contact (zero-centred). Less anterior travel on the surgical side "
                          "= reduced push-off through the forefoot.",
            "description": (
                "The Countermovement Jump is a bilateral power test with a relatively long, slow stretch-shortening "
                "cycle. Because the hip and knee can contribute most of the work, CMJ symmetry often returns before "
                "the calf has recovered — a symmetric CMJ does not clear an Achilles. Propulsive impulse and peak force "
                "LSI show whether the surgical limb is contributing to push-off."),
        },
        "endurance_squat": {
            "intro":     "30s bilateral squat — knee-dominant endurance; does not directly load the calf.",
            "drift_bad": "surgical limb offloads under prolonged effort.",
            "targets":   "<b>Targets:</b> Mean LSI ≥ 90% · Fatigue drift < 5%. Pair with the heel-raise test for calf endurance.",
            "description": (
                "The 30-Second Endurance Squat assesses limb symmetry under sustained bilateral loading. It is knee- and "
                "hip-dominant, so for an Achilles repair it mainly checks for whole-limb offloading under fatigue; calf "
                "endurance is assessed with the single-leg heel raise to failure."),
        },
        "single_leg_jump": {
            "intro":   "Isolates each limb — the surgical calf must produce the final push-off without help.",
            "pk_low":  "Force deficit on surgical side — calf / plantarflexor capacity limiting push-off.",
            "work_gap": "The surgical ankle does far less push-off work than the other side while the hip "
                        "makes up the difference — a hip-dominant strategy that protects the calf.",
            "impulse_gap": "Net impulse is closer to symmetric than jump height — the surgical side is reaching force "
                           "by pushing longer, not faster.",
            "targets": "<b>Targets:</b> Jump height LSI ≥ 90% · Net impulse LSI ≥ 90% · Heel-raise LSI ≥ 90%.",
            "cap_grf":    "Per-limb GRF (separate trials). A later, drawn-out surgical peak = slow push-off.",
            "cap_flex":   "Knee flexion per limb. Reduced surgical depth = shortened, guarded countermovement.",
            "cap_ankle":  "Ankle dorsiflexion per limb — plantarflexion range at push-off drives single-leg height.",
            "cap_valgus": "Knee valgus per limb (context only for Achilles — not scored).",
            "cap_tib":    "Tibial / foot rotation per limb. Surgical-side external rotation offset = possible toe-out.",
            "cap_ankle_mom": "Plantarflexor moment per limb. A similar peak with less power = slow calf.",
            "cap_ankle_pow": "Ankle power per limb — the positive burst before take-off is calf push-off work.",
            "cap_cop":    "AP COP during the take-off contact (zero-centred). Less anterior travel = limited push-off "
                          "through the forefoot.",
            "description": (
                "The Single-Leg Vertical Jump isolates each limb so the uninvolved side cannot compensate. Final push-off "
                "depends on the plantarflexors, so after Achilles repair single-leg height is one of the more sensitive "
                "power measures. Compare jump height with net impulse: if impulse is near-symmetric but height is not, "
                "the athlete is producing force slowly over a longer contact rather than quickly through the calf."),
        },
        "balance": {
            "description": (
                "Eyes-closed single-leg stance removes visual feedback, so balance depends on ankle and foot "
                "proprioception and on the calf's ability to make fast small corrections. Higher COP velocity means more "
                "corrective activity. After Achilles repair, reduced calf strength and altered tendon stiffness commonly "
                "increase sway on the surgical limb. Limb symmetry (LSI ≥ 85–90%) is the primary criterion."),
        },
    },
)
