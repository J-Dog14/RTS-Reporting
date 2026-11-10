"""
profiles/acl.py
===============
ACL reconstruction / knee return-to-sport profile.
This is the original report wording, moved out of helpers/report.py.
"""

from profiles.base import ReportProfile
from helpers.norms import RTR_WEIGHTS as _ACL_WEIGHTS, _TIME_CONTROL_POINTS as _ACL_TIME


def _time_note(m):
    if m is None:
        return None
    if m < 6:
        return (f"{m:.1f} months post-op — physiologically too early. "
                f"Evidence supports ≥ 9 months before return (Kyritsis 2016 BJSM).")
    if m < 9:
        return (f"{m:.1f} months post-op — below the 9-month evidence threshold; "
                f"re-injury risk is substantially higher before 9 months (Kyritsis 2016 BJSM).")
    if m < 12:
        return f"{m:.1f} months post-op — within evidence window. RTS possible if all performance criteria met."
    return f"{m:.1f} months post-op — time no longer the limiting factor; performance criteria are decisive."


VALGUS_TARGET = "< 8°"   # matches NORMS['peak_knee_valgus'].good

PROFILE = ReportProfile(
    key="acl",
    menu_label="ACL / Knee",
    menu_blurb="ACL reconstruction and other knee surgery — valgus, tibial rotation, quad emphasis",
    report_subtitle="Return to Sport Assessment",
    file_tag="",
    rtr_weights={**_ACL_WEIGHTS,
                 # v14 additions
                 "cop_velocity_airex_lsi": ("cop_velocity_lsi", 0.05),   # Airex balance
                 "sl_power_lsi":           ("sl_jump_height_lsi", 0.06), # SL peak power W/kg
                 },
    time_points=list(_ACL_TIME),
    time_note=_time_note,
    flag_knee_frontal=True,
    ankle_focus=False,
    text={
        "summary": {
            "focus": "",
        },
        "drop_jump": {
            "rsi_low": "Below target — consistent with a stiff or guarded landing.",
            "targets": f"<b>Targets:</b> RSI ≥ 1.30 m/s · Landing LSI 90–110% · Load rate LSI ≥ 90% · "
                       f"Knee flexion ≥ 60° · Valgus {VALGUS_TARGET} · Tibial IR < 10°.",
            "cap_grf":    "Bilateral GRF (normalised to BW). Orange = surgical, Blue = non-surgical.",
            "cap_flex":   "Knee flexion. Target ≥ 60° at contact. Reduced depth = stiff landing strategy.",
            "cap_ankle":  "Ankle dorsiflexion. Restricted dorsiflexion → shifts load to quadriceps / increases ACL stress.",
            "cap_valgus": f"Knee valgus. Target {VALGUS_TARGET}. Elevated valgus = medial collapse / ACL re-injury risk.",
            "cap_tib":    "Tibial internal rotation. IR > 10° at landing = increased ACL rotational load.",
            "cap_cop":    "AP COP displacement during the first contact (zero-centred). Smaller excursion on surgical "
                          "limb = restricted fore-aft weight transfer / guarded landing strategy.",
            "description": (
                "The Drop Jump is a reactive bilateral landing task performed from a standard box height. "
                "The athlete drops off the box, absorbs the impact with both limbs simultaneously, then immediately "
                "jumps for maximum height. It tests the athlete's ability to rapidly store and release elastic energy "
                "(reactive strength) and to absorb high-impact loads symmetrically. Key concerns after ACL reconstruction "
                "include stiff-knee landing (reduced flexion, elevated impact transient), medial knee collapse (valgus), "
                "and side-to-side force asymmetries that indicate ongoing limb offloading. The Reactive Strength Index "
                "(RSI = jump height ÷ contact time) summarises the explosive reactive capacity of the neuromuscular system."),
        },
        "drop_landing": {
            "intro":   "Absorptive landing — assesses shock attenuation without a re-jump. Lower, more gradual force = better control.",
            "targets": f"<b>Targets:</b> Peak GRF < 3× BW · Load rate LSI 90–110% · Impact transient LSI 90–110% · "
                       f"Knee flexion ≥ 60° · Valgus {VALGUS_TARGET} · Tibial IR < 10°.",
            "cap_grf":    "Bilateral GRF. Smooth progressive rise = good shock absorption. Sharp impact spike = stiff landing.",
            "cap_flex":   "Knee flexion at landing. Reduced depth = stiff strategy / quad inhibition.",
            "cap_ankle":  "Ankle dorsiflexion. Restricted ROM limits knee flexion depth and increases ACL load.",
            "cap_valgus": f"Knee valgus at landing. Target {VALGUS_TARGET}; higher surgical valgus = elevated ACL stress.",
            "cap_tib":    "Tibial IR at landing. Combined with valgus = high ACL injury-risk pattern.",
            "cap_cop":    "AP COP during landing contact (zero-centred). Posterior bias on surgical side = heel-dominant / reduced forefoot loading.",
            "description": (
                "The Drop Landing isolates the deceleration phase of a jump: the athlete drops from a box and sticks "
                "the landing bilaterally without a subsequent jump. This removes the propulsive demand and focuses entirely "
                "on the athlete's ability to safely absorb and dissipate ground reaction forces. After ACL reconstruction, "
                "common compensations include a stiff-knee strategy (reduced flexion, sharp impact spike), excessive valgus "
                "collapse, and asymmetric weight acceptance—all of which increase mechanical stress at the reconstructed "
                f"ligament. A smooth, progressive force curve with symmetric knee flexion ≥ 60° and valgus {VALGUS_TARGET} is the "
                "clinical target. Loading rate and impact transient LSI reflect side-to-side shock absorption capacity."),
        },
        "max_vertical": {
            "intro":   "Bilateral CMJ — sensitive to subtle propulsion deficits masked at lower intensities.",
            "pk_low":  "consistent with a quadriceps / hip extensor deficit",
            "knee_work_gap": "The surgical knee contributes less push-off work while the hip makes up the "
                             "difference — a hip-dominant strategy offloading the quadriceps.",
            "targets": "Targets: ≥ 90% propulsion, ≥ 90% peak force, matched unweighting impulse.",
            "cap_grf":    "Full CMJ: unweighting → eccentric load → propulsion → flight. Propulsive peaks should be symmetric.",
            "cap_flex":   "Knee flexion depth. Asymmetric countermovement depth = compensatory loading strategy.",
            "cap_ankle":  "Ankle dorsiflexion. Restricted ROM reduces jump height and shifts load away from the knee.",
            "cap_valgus": "Knee valgus during propulsion. Dynamic valgus under load = ACL stress.",
            "cap_tib":    "Tibial IR during propulsion. Coupled with valgus = high re-injury risk pattern.",
            "cap_cop":    "AP COP during the propulsive contact (zero-centred). Anterior shift during propulsion drives jump "
                          "height — asymmetric or reduced excursion on surgical side = limb offloading.",
            "description": (
                "The Countermovement Jump (CMJ) is the gold-standard bilateral power test. The athlete begins in quiet "
                "stance, rapidly descends into a squat (countermovement), then drives upward for maximum jump height. "
                "The full GRF waveform reveals each phase: unweighting, eccentric loading, propulsion, and flight. "
                "After ACL reconstruction, typical deficits include an asymmetric countermovement depth (surgical limb "
                "offloading during the eccentric phase), reduced propulsive impulse, and side-to-side jump height "
                "asymmetry. Propulsive LSI < 90% is a common criterion for RTS decisions. Kinematic analysis identifies "
                "compensatory strategies—valgus, reduced knee flexion, and tibial rotation—that indicate the surgical "
                "limb is not being loaded with full confidence or mechanical efficiency."),
        },
        "endurance_squat": {
            "intro":     "30s bilateral squat — reveals neuromuscular fatigue-induced asymmetry missed by single-rep testing.",
            "drift_bad": "surgical limb offloads under prolonged effort.",
            "targets":   "<b>Targets:</b> Mean LSI ≥ 90% · Fatigue drift < 5% · No progressive offloading across 30s.",
            "description": (
                "The 30-Second Endurance Squat assesses neuromuscular fatigue and limb symmetry under sustained "
                "bilateral loading. The athlete performs continuous bilateral squats to a standardised depth for a "
                "full 30 seconds. Unlike single-repetition tests, this task reveals how well the neuromuscular system "
                "maintains symmetry as it fatigues. Athletes post-ACL reconstruction frequently show an initial LSI "
                "within acceptable range that deteriorates progressively across the set as the nervous system "
                "disinhibits protective guarding. The COP drift trace exposes this compensation: surgical-limb weight "
                "transfer decreases, traces diverge, and loading shifts toward the uninvolved side. This test is "
                "particularly sensitive to residual quadriceps inhibition that is not apparent during brief, "
                "high-energy tasks."),
        },
        "single_leg_jump": {
            "intro":   "Highest-demand test — isolates each limb, removing bilateral compensation. Strongest re-injury risk predictor.",
            "pk_low":  "Force deficit on the surgical side, consistent with a quadriceps / hip extensor deficit.",
            "knee_work_gap": "The surgical knee does less push-off work while the hip makes up the difference "
                             "— quadriceps avoidance under single-leg load.",
            "targets": f"<b>Targets:</b> Jump height LSI ≥ 90% · Peak GRF LSI ≥ 90% · Valgus {VALGUS_TARGET} · Tibial IR < 10° on surgical limb.",
            "cap_grf":    "Per-limb GRF (separate trials). Propulsive peak height and shape should match.",
            "cap_flex":   "Knee flexion per limb. Reduced surgical depth = power deficit or guarded strategy.",
            "cap_ankle":  "Ankle dorsiflexion per limb. Restricted ROM = reduced propulsive range / increased ACL load.",
            "cap_valgus": "Knee valgus per limb. Elevated surgical valgus under single-leg load = re-injury risk.",
            "cap_tib":    "Tibial IR per limb. Surgical-side IR > 10° = high rotational ACL stress pattern.",
            "cap_cop":    "AP COP during the take-off contact (zero-centred, per limb trial). Reduced anterior excursion on surgical side = guarded takeoff / limited push-off.",
            "description": (
                "The Single-Leg Vertical Jump isolates each limb independently, removing the ability of the uninvolved "
                "leg to compensate. The athlete performs a maximal countermovement jump landing on the same leg. "
                "Jump height LSI (surgical vs. non-surgical) is among the most sensitive power-based RTS criteria—"
                "a threshold of ≥ 90% is widely cited in ACL literature. Single-leg testing also reveals subtle "
                "compensations masked in bilateral tasks: reduced propulsive depth, hip drop (Trendelenburg pattern), "
                "and increased tibial rotation are common markers of residual neuromuscular asymmetry."),
        },
        "balance": {
            "description": (
                "The Balance / Proprioception test assesses joint position sense and neuromuscular control under "
                "sensory challenge. The athlete stands on a single leg with eyes closed, removing visual "
                "compensation and isolating the mechanoreceptor feedback system of the ankle and knee. Center of "
                "Pressure (COP) velocity is the most sensitive metric: a higher velocity indicates greater "
                "corrective activity, meaning the neuromuscular system is working harder to maintain stability. "
                "After ACL reconstruction, deficits in mechanoreceptor density within the graft and capsular "
                "tissue result in elevated COP velocity and a larger sway ellipse on the surgical limb. The Airex "
                "pad condition adds an unstable surface to further stress the proprioceptive system, revealing "
                "deficits that may be masked on firm ground. Symmetry between limbs (LSI ≥ 85–90%) is the "
                "primary RTS criterion for this test."),
        },
    },
)
