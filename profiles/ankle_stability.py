"""
profiles/ankle_stability.py
===========================
Ankle stability profile — lateral ankle sprain, chronic ankle instability
(CAI) and lateral ligament repair (Broström / internal brace).

Clinical emphasis (vs ACL / Achilles)
-------------------------------------
• The deficit after a lateral sprain is mostly sensorimotor: postural
  control and how quickly the limb settles after landing, often on BOTH
  sides. Static balance (eyes closed, firm + Airex) and single-leg
  jump-landing stability are therefore weighted highest.
• Single-leg landing stability comes from the landing phase of every
  single-leg vertical trial (same plate): Dynamic Postural Stability Index
  (Wikstrom 2005), mediolateral stability index and time to stabilisation.
  These are lower-is-better, so symmetry = non-surgical ÷ surgical.
• Mediolateral (frontal-plane) sway and ankle frontal motion at touchdown
  are reported because inversion control is the injury mechanism. Ankle
  frontal motion is reported as excursion / range (direction-free) until the
  inversion sign convention is confirmed in the V3D model.
• Power and reactive metrics stay in the score at lower weight — after a
  sprain they usually recover before control does.
• Knee valgus / tibial rotation: context only, not scored.

References
----------
Gribble PA et al. 2016 BJSM — consensus on CAI / the ankle-instability continuum
Wikstrom EA et al. 2005 J Athl Train — Dynamic Postural Stability Index
Ross SE, Guskiewicz KM 2004 Clin J Sport Med — time to stabilisation in functional ankle instability
Arnold BL et al. 2009 J Athl Train — balance deficits in ankle instability (meta-analysis)
Doherty C et al. 2016 AJSM — landing biomechanics and recurrence after first-time lateral sprain
"""

from profiles.base import ReportProfile


def _domain(label, default):
    L = label.lower()
    if "knee" in L:
        return None
    if L.startswith("balance"):
        return "Balance"
    if L.startswith("sl landing") or L.startswith("dl"):
        return "Landing"
    if "ankle" in L:
        return "Ankle"
    if L.startswith("endurance"):
        return "Endurance"
    if L.startswith("triple"):
        return "Hop"
    if L.startswith("dj"):
        return "Reactive"
    if L.startswith("sl") or L.startswith("cmj"):
        return "Power"
    return default


def _extra_reg(test_results, all_signals):
    from rts14.stability_page import score_metrics
    return score_metrics(test_results, all_signals)


def _time_note(m):
    if m is None:
        return None
    if m < 3:
        return (f"{m:.1f} months since injury / repair — early; ligament healing and sensorimotor control "
                f"are usually still recovering.")
    if m < 6:
        return (f"{m:.1f} months since injury / repair — return to cutting sport commonly falls around "
                f"3–6 months when control criteria are met.")
    return f"{m:.1f} months since injury / repair — time no longer the limiting factor; control criteria are decisive."


PROFILE = ReportProfile(
    key="ankle_stability",
    menu_label="Ankle Stability",
    menu_blurb="Lateral sprain / CAI / ligament repair — balance, landing stability, frontal-plane control",
    report_subtitle="Return to Sport Assessment<br/><font size='10'>Ankle Stability</font>",
    file_tag="Ankle Stability",

    rtr_weights={
        # Static postural control — the primary deficit in ankle instability
        "cop_velocity_lsi":       ("cop_velocity_lsi",   0.14),
        "cop_velocity_airex_lsi": ("cop_velocity_lsi",   0.10),
        "balance_ml_lsi":         ("cop_velocity_lsi",   0.06),
        "balance_ml_airex_lsi":   ("cop_velocity_lsi",   0.04),
        # Single-leg jump-landing stability
        "land_dpsi_lsi":          ("lsi_general",        0.14),
        "land_mlsi_lsi":          ("lsi_general",        0.08),
        "land_tts_lsi":           ("lsi_general",        0.06),
        # Single-leg function
        "sl_jump_lsi":            ("sl_jump_height_lsi", 0.10),
        "triple_hop_lsi":         ("triple_hop_lsi",     0.08),
        # Bilateral landing / reactive (lower weight — usually recover first)
        "dl_load_rate_lsi":       ("loading_rate_lsi",   0.04),
        "landing_lsi":            ("landing_lsi_200ms",  0.04),
        "drop_jump_rsi":          ("drop_jump_rsi",      0.04),
        "rfd_lsi":                ("lsi_rfd",            0.03),
        "sl_power_lsi":           ("sl_jump_height_lsi", 0.03),
        "endurance_lsi":          ("endurance_mean_lsi", 0.02),
    },
    time_points=[(0, 0.85), (2, 0.85), (3, 0.92), (4, 0.97), (6, 1.00), (24, 1.00)],
    time_note=_time_note,
    influence_names={
        "cop_velocity_airex_lsi": "Balance symmetry (Airex)",
        "balance_ml_lsi":         "ML sway sym. (firm)",
        "balance_ml_airex_lsi":   "ML sway sym. (Airex)",
        "land_dpsi_lsi":          "SL landing stability (DPSI)",
        "land_mlsi_lsi":          "SL landing ML stability",
        "land_tts_lsi":           "SL landing time to stabilise",
        "sl_power_lsi":           "SL peak power",
        "dl_load_rate_lsi":       "DL Load rate sym.",
        "triple_hop_lsi":         "Triple hop",
    },
    flag_knee_frontal=False,
    ankle_focus=True,
    extra_pages=["ankle_stability"],
    section_order=["proprioception", "single_leg_jump", "drop_landing", "drop_jump",
                   "max_vertical", "endurance_squat"],
    domain_map=_domain,
    chart_exclude=["Knee"],
    extra_reg=_extra_reg,
    text={
        "summary": {
            "focus": ("Ankle-stability profile — static balance and single-leg landing stability are weighted "
                      "highest; power and reactive metrics are included at lower weight."),
            "ankle_missing": ("Ankle angle / moment signals were not found in the V3D export. Add L/R_Ankle_Angle "
                              "and L/R_Ankle_Moment to the export pipeline for ankle-specific metrics."),
        },
        "balance": {
            "description": (
                "Eyes-closed single-leg stance removes vision, so balance depends on ankle and foot proprioception "
                "and on fast small corrections by the peroneals and the muscles around the ankle — the system "
                "most affected after a lateral ankle sprain. Higher COP velocity and a wider side-to-side (ML) sway "
                "range mean more corrective activity. Deficits after a sprain are frequently present on BOTH "
                "limbs, so a failed stance hold or high absolute sway matters even when symmetry looks acceptable. "
                "A trial in which the foot moved or the other foot touched down is recorded as a FAIL."),
        },
        "single_leg_jump": {
            "intro":   "Single-leg jump and landing — push-off power, then how quickly the limb stabilises on landing.",
            "pk_low":  "Force deficit on the surgical side during push-off.",
            "impulse_gap": "Net impulse is closer to symmetric than jump height — the surgical side reaches force "
                           "by pushing longer, not faster.",
            "targets": "<b>Targets:</b> Jump height LSI ≥ 90% · Landing stability (DPSI, ML index, time to "
                       "stabilise) symmetry ≥ 90% · Ankle frontal excursion within 3° of the other side.",
            "cap_grf":    "Per-limb GRF (separate trials), push-off and landing.",
            "cap_flex":   "Knee flexion per limb.",
            "cap_ankle":  "Ankle dorsiflexion per limb.",
            "cap_valgus": "Knee valgus per limb (context only for this report type — not scored).",
            "cap_tib":    "Tibial / foot rotation per limb (context only).",
            "cap_ankle_mom": "Plantarflexor moment per limb.",
            "cap_ankle_pow": "Ankle power per limb — positive burst before take-off is push-off work.",
            "cap_cop":    "AP COP during the take-off contact (zero-centred).",
            "description": (
                "The Single-Leg Vertical Jump isolates each limb. For ankle stability the landing matters as much "
                "as the jump: the athlete lands back on the same leg, and the force plate shows how much the limb "
                "has to correct in the first second after touchdown. The Dynamic Postural Stability Index combines "
                "side-to-side, fore-aft and vertical force fluctuation; time to stabilise is how long those "
                "fluctuations take to settle. Both are higher (worse) after a lateral ankle sprain."),
        },
        "drop_landing": {
            "intro":   "Bilateral absorptive landing — load acceptance and frontal-plane ankle control.",
            "rate_gap": "Peak force is closer to symmetric than loading rate — the surgical limb is accepting load "
                        "more slowly.",
            "targets": "<b>Targets:</b> Load rate LSI ≥ 85–90% · Impact transient LSI 90–110% · Ankle frontal "
                       "range within 3° of the other side.",
            "cap_grf":    "Bilateral GRF. Lower, later initial peak on the surgical side = slower load acceptance.",
            "cap_flex":   "Knee flexion at landing.",
            "cap_ankle":  "Ankle dorsiflexion at landing.",
            "cap_valgus": "Knee valgus at landing (context only — not scored).",
            "cap_tib":    "Tibial / foot rotation at landing (context only).",
            "cap_ankle_mom": "Plantarflexor moment during landing.",
            "cap_ankle_pow": "Ankle power during landing: the area below zero is energy absorbed at the ankle.",
            "cap_cop":    "AP COP during landing contact (zero-centred).",
            "description": (
                "The Drop Landing isolates load acceptance: the athlete drops from a box and sticks the landing. "
                "For ankle stability, the key questions are whether the surgical limb accepts load as quickly as "
                "the other side and whether its frontal-plane ankle motion on contact matches the other side."),
        },
        "drop_jump": {
            "rsi_low": "Long contact / low reactive strength on landing and rebound.",
            "rsi_slow": "The RSI number clears 1.30, but only with a long contact — a slow rebound rather than a "
                        "reactive jump.",
            "rate_gap": "Force magnitude is close to symmetric but rate is not — the surgical limb avoids fast loading.",
            "targets": "<b>Targets:</b> RSI ≥ 1.30 m/s · Contact < 0.25 s · RFD & load rate LSI ≥ 85–90%.",
            "cap_grf":    "Bilateral GRF (normalised to BW).",
            "cap_flex":   "Knee flexion.",
            "cap_ankle":  "Ankle dorsiflexion.",
            "cap_valgus": "Knee valgus (context only — not scored).",
            "cap_tib":    "Tibial / foot rotation (context only).",
            "cap_ankle_mom": "Plantarflexor moment through landing and rebound.",
            "cap_ankle_pow": "Ankle power: below zero = absorbing, above zero = driving the rebound.",
            "cap_cop":    "AP COP during the first contact (zero-centred).",
            "description": (
                "The Drop Jump is a reactive bilateral task: drop from a box and immediately jump as high and as "
                "quickly as possible. After an ankle sprain it shows whether the athlete will load the ankle "
                "quickly and stiffly on contact; a long contact or slower surgical-side force rise suggests the "
                "ankle is being protected."),
        },
        "max_vertical": {
            "intro":   "Bilateral CMJ — power reference; usually symmetric well before control recovers.",
            "pk_low":  "reduced push-off on the surgical side",
            "targets": "Targets: ≥ 90% propulsion, ≥ 90% peak force.",
            "cap_grf":    "Full CMJ: unweighting → eccentric → propulsion → flight.",
            "cap_flex":   "Knee flexion depth.",
            "cap_ankle":  "Ankle dorsiflexion.",
            "cap_valgus": "Knee valgus (context only).",
            "cap_tib":    "Tibial / foot rotation (context only).",
            "cap_ankle_mom": "Plantarflexor moment through the countermovement and push-off.",
            "cap_ankle_pow": "Ankle power: the late positive burst before take-off is push-off.",
            "cap_cop":    "AP COP during the propulsive contact (zero-centred).",
            "description": (
                "The Countermovement Jump is a bilateral power test. After an ankle sprain it is mostly a reference: "
                "a symmetric CMJ shows force output is available, so stability deficits elsewhere are about control."),
        },
        "endurance_squat": {
            "intro":     "30 s bilateral squat — whole-limb offloading under fatigue.",
            "drift_bad": "surgical limb offloads under prolonged effort.",
            "targets":   "<b>Targets:</b> Mean LSI ≥ 90% · Fatigue drift < 5%.",
            "description": (
                "The 30-Second Endurance Squat checks limb symmetry under sustained bilateral loading. For ankle "
                "stability it is a secondary check for whole-limb offloading as fatigue builds."),
        },
    },
)
