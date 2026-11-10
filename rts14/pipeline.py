"""
rts14/pipeline.py
=================
v14 export folder → (test_results, all_signals) in the shapes helpers/report.py
already understands, so the ACL / Achilles report profiles work unchanged.

Every metric is computed per trial (rts14/analysis.py) and averaged across
trials. Figures use the trial closest to the mean on that test's key metric
(never a frame-by-frame average of curves).
"""

from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

from helpers.grf import lowpass, lsi
from rts14.loader import load_session, FILES
from rts14 import analysis as A
import tests.endurance_squat as es_mod
import tests.proprioception as prop_mod


def is_v14_folder(folder) -> bool:
    f = Path(folder)
    return (f / FILES["force"]).exists() and (f / FILES["angles"]).exists()


def _g(m, k):
    v = m.get(k, np.nan)
    return np.nan if v is None else v


def _kin(m):
    def knee(t):
        return NS(peak_flexion_deg=_g(m, f"{t}_knee_flex"), peak_valgus_deg=_g(m, f"{t}_knee_valgus"),
                  peak_tibial_ir_deg=_g(m, f"{t}_tib_ir"), tibial_rot_excursion=_g(m, f"{t}_tib_exc"))

    def hip(t):
        return NS(peak_flexion_deg=_g(m, f"{t}_hip_flex"), peak_adduction_deg=_g(m, f"{t}_hip_add"),
                  peak_ir_deg=_g(m, f"{t}_hip_ir"))
    return NS(surg_knee=knee("s"), non_surg_knee=knee("n"), surg_hip=hip("s"), non_surg_hip=hip("n"),
              lsi_peak_flexion=_g(m, "knee_flex_lsi"))


def _work_rows(m, test_tag, kinds=("pos",), ankle_focus_rtr=None):
    """Extra metric-table rows for joint work + ankle share."""
    rows = []
    label = {"pos": "+Work (J)", "neg": "Absorption (J)"}
    for kind in kinds:
        for j in ("Ankle", "Knee", "Hip"):
            jl = j.lower()
            reg = f"{test_tag} {j} {'+Work' if kind == 'pos' else 'Absorption'}" if j == "Ankle" else None
            rtr = ankle_focus_rtr if (j == "Ankle" and kind == "pos") else None
            rows.append((f"{j} {label[kind]}", _g(m, f"s_{jl}_{kind}_work"), _g(m, f"n_{jl}_{kind}_work"),
                         _g(m, f"{jl}_{kind}_work_lsi"), reg, rtr))
        rows.append((f"Ankle share of {'+work' if kind == 'pos' else 'absorption'} (%)",
                     _g(m, f"s_ankle_{kind}_share"), _g(m, f"n_ankle_{kind}_share"), None, None, None))
    return rows


def _scalars(m, sig):
    sig["knee_moment_surg"]  = _g(m, "s_knee_ext_mom")
    sig["knee_moment_ns"]    = _g(m, "n_knee_ext_mom")
    sig["ankle_moment_surg"] = _g(m, "s_ankle_pf_mom")
    sig["ankle_moment_ns"]   = _g(m, "n_ankle_pf_mom")
    return sig


def build_results(folder: str, surg_side: str, body_n: float = None):
    """
    Returns (test_results, all_signals, info)
      info: {"mass_n", "paths", "trials": {test: [names]}, "rates": (f, k)}
    """
    S = load_session(folder)
    surg = surg_side.upper()
    ns_side = A.other(surg)
    # Body weight: trust the plates. V3D's MASS_N comes from the session weight
    # entry, which can be stale (John Jiles' session carried 845 N while he
    # stands at ~941 N on the plate).
    bw_force = A.body_weight_from_force(S)
    bw = body_n if body_n else (bw_force if bw_force == bw_force else S.mass_n)
    bw_warning = None
    if bw_force == bw_force and S.mass_n == S.mass_n and abs(bw_force - S.mass_n) > 0.03 * bw_force:
        bw_warning = (f"Body weight {bw_force / 9.81 / 0.4536:.0f} lb from the plates vs "
                      f"{S.mass_n / 9.81 / 0.4536:.0f} lb in the Visual3D session — plate value used; "
                      f"fix the session weight (it scales joint moments).")
    neutral = {s: A.ankle_neutral(S, s) for s in ("L", "R")}

    results, signals = {}, {}
    info = {"mass_n": bw, "mass_n_v3d": S.mass_n, "bw_warning": bw_warning,
            "paths": S.paths, "trials": {}, "rates": None}
    if S.trials:
        t0 = next(iter(S.trials.values()))
        info["rates"] = (t0.rate_f, t0.rate_k)

    # ── Drop jump ────────────────────────────────────────────────────────────
    agg = A.aggregate([A.analyse_drop(t, surg, bw, neutral, rejump=True)
                       for t in S.by_test("drop_jump")], key_metric="rsi")
    if agg:
        m, sd, rep, n, names = agg
        info["trials"]["drop_jump"] = names
        results["drop_jump"] = NS(
            grf_overall=NS(
                surgical=NS(peak_force_N=m["s_peak_grf"], impulse_Ns=m["s_impulse"], rfd_Ns=m["s_rfd"]),
                non_surgical=NS(peak_force_N=m["n_peak_grf"], impulse_Ns=m["n_impulse"], rfd_Ns=m["n_rfd"]),
                lsi_peak=m["peak_grf_lsi"], lsi_impulse=m["impulse_lsi"], lsi_rfd=m["rfd_lsi"]),
            loading_rate_surg_ns=m["s_load_rate"], loading_rate_ns_ns=m["n_load_rate"],
            loading_rate_lsi=m["load_rate_lsi"], landing_lsi_200ms=m["landing_lsi_200ms"],
            impact_transient_surg=m["s_impact"], impact_transient_ns=m["n_impact"],
            impact_transient_lsi=m["impact_lsi"],
            time_to_peak_surg_ms=m["s_t_peak_ms"], time_to_peak_ns_ms=m["n_t_peak_ms"],
            jump_height_cm=m["jump_height_in"], contact_time_s=m["contact_time"],
            flight_time_s=m["flight_time"], rsi=m["rsi"], kinematics=_kin(m),
            n_trials=n, sd=sd)
        sig = _scalars(m, dict(rep))
        sig["extra_rows"] = ([("Trials analysed", str(n), "—", None, None, None),
                              ("Peak Power (W/kg)", _g(m, "peak_power_wkg"), None, None, None, None),
                              ("Limb Peak Power (W)", _g(m, "s_peak_power"), _g(m, "n_peak_power"),
                               _g(m, "peak_power_lsi"), "DJ Peak Power", None),
                              ("Limb Push-off Work (J)", _g(m, "s_con_work"), _g(m, "n_con_work"),
                               _g(m, "con_work_lsi"), None, None),
                              ("Implied Drop Height (cm)", _g(m, "implied_drop_cm"), None, None, None, None)]
                             + _work_rows(m, "DJ", ("neg", "pos")))
        sig["qc"] = m.get("_qc", [])
        sig["joint_work"] = {"kind": "pos", **{tag: {j: _g(m, f"{tag}_{j}_pos_work") for j in ("hip", "knee", "ankle")} for tag in ("s", "n")}}
        signals["drop_jump"] = sig

    # ── Drop landing ─────────────────────────────────────────────────────────
    agg = A.aggregate([A.analyse_drop(t, surg, bw, neutral, rejump=False)
                       for t in S.by_test("drop_landing")], key_metric="peak_grf_lsi")
    if agg:
        m, sd, rep, n, names = agg
        info["trials"]["drop_landing"] = names
        results["drop_landing"] = NS(
            grf_overall=NS(
                surgical=NS(peak_force_N=m["s_peak_grf"], impulse_Ns=m["s_impulse"], rfd_Ns=m["s_rfd"]),
                non_surgical=NS(peak_force_N=m["n_peak_grf"], impulse_Ns=m["n_impulse"], rfd_Ns=m["n_rfd"]),
                lsi_peak=m["peak_grf_lsi"], lsi_impulse=m["impulse_lsi"], lsi_rfd=m["rfd_lsi"]),
            peak_force_lsi=m["peak_grf_lsi"],
            loading_rate_surg_ns=m["s_load_rate"], loading_rate_ns_ns=m["n_load_rate"],
            loading_rate_lsi=m["load_rate_lsi"],
            impact_transient_surg=m["s_impact"], impact_transient_ns=m["n_impact"],
            impact_transient_lsi=m["impact_lsi"], kinematics=_kin(m), n_trials=n, sd=sd)
        sig = _scalars(m, dict(rep))
        sig["extra_rows"] = ([("Trials analysed", str(n), "—", None, None, None)]
                             + _work_rows(m, "DL", ("neg",)))
        sig["qc"] = m.get("_qc", [])
        sig["joint_work"] = {"kind": "neg", **{tag: {j: _g(m, f"{tag}_{j}_neg_work") for j in ("hip", "knee", "ankle")} for tag in ("s", "n")}}
        signals["drop_landing"] = sig

    # ── CMJ ──────────────────────────────────────────────────────────────────
    agg = A.aggregate([A.analyse_cmj(t, surg, bw, neutral)
                       for t in S.by_test("max_vertical")], key_metric="jump_height_in")
    if agg:
        m, sd, rep, n, names = agg
        info["trials"]["max_vertical"] = names
        results["max_vertical"] = NS(
            grf_concentric=NS(
                surgical=NS(peak_force_N=m["s_prop_peak"], impulse_Ns=m["s_prop_impulse"]),
                non_surgical=NS(peak_force_N=m["n_prop_peak"], impulse_Ns=m["n_prop_impulse"]),
                lsi_peak=m["prop_peak_lsi"], lsi_impulse=m["prop_impulse_lsi"]),
            peak_force_surg_N=m["s_brake_peak"], peak_force_ns_N=m["n_brake_peak"],
            peak_force_lsi=m["brake_peak_lsi"], propulsion_lsi=m["prop_impulse_lsi"],
            braking_impulse_lsi=m["brake_impulse_lsi"],
            jump_height_cm=m["jump_height_in"], flight_time_s=m["flight_time"],
            contact_time_s=m["contact_time"],
            unweighting_impulse_surg=m["s_unwt_impulse"], unweighting_impulse_ns=m["n_unwt_impulse"],
            kinematics=_kin(m), n_trials=n, sd=sd)
        sig = _scalars(m, dict(rep))
        sig["extra_rows"] = ([("Trials analysed", str(n), "—", None, None, None),
                              ("Braking Impulse (N·s)", _g(m, "s_brake_impulse"), _g(m, "n_brake_impulse"),
                               _g(m, "brake_impulse_lsi"), None, None),
                              ("Peak Power (W/kg)", _g(m, "peak_power_wkg"), None, None, None, None),
                              ("Limb Peak Power (W)", _g(m, "s_peak_power"), _g(m, "n_peak_power"),
                               _g(m, "peak_power_lsi"), "CMJ Peak Power", None),
                              ("Limb Push-off Work (J)", _g(m, "s_con_work"), _g(m, "n_con_work"),
                               _g(m, "con_work_lsi"), None, None),
                              ("mRSI (JH ÷ time to take-off)", _g(m, "mrsi"), None, None, None, None),
                              ("Countermovement Depth (cm)", _g(m, "cm_depth_cm"), None, None, None, None)]
                             + _work_rows(m, "CMJ", ("pos",)))
        sig["qc"] = m.get("_qc", [])
        sig["joint_work"] = {"kind": "pos", **{tag: {j: _g(m, f"{tag}_{j}_pos_work") for j in ("hip", "knee", "ankle")} for tag in ("s", "n")}}
        signals["max_vertical"] = sig

    # ── Endurance squat ──────────────────────────────────────────────────────
    ends = S.by_test("endurance_squat")
    if ends:
        t = ends[0]
        fs_ = t.fz(A.plate_for(surg))
        fn_ = t.fz(A.plate_for(ns_side))
        if fs_ is not None and fn_ is not None:
            n = min(len(fs_), len(fn_))
            fs_ = lowpass(np.nan_to_num(fs_[:n]), t.rate_f, 10.0)
            fn_ = lowpass(np.nan_to_num(fn_[:n]), t.rate_f, 10.0)
            res = es_mod.analyse(
                fz_surg=fs_, fz_ns=fn_, rate_f=t.rate_f, bw_n=bw, surg_side=surg,
                knee_valgus_surg=A._angle(t, surg, "Knee", "Y"),
                knee_valgus_ns=A._angle(t, ns_side, "Knee", "Y"), rate_k=t.rate_k)
            results["endurance_squat"] = res
            info["trials"]["endurance_squat"] = [t.name]
            mech_s = A.limb_mechanics(t, surg, 0, 10 ** 9, neutral.get(surg))
            mech_n = A.limb_mechanics(t, ns_side, 0, 10 ** 9, neutral.get(ns_side))
            cs, cn = t.cop.get(A.plate_for(surg)), t.cop.get(A.plate_for(ns_side))
            signals["endurance_squat"] = {
                "fz_surg": fs_, "fz_ns": fn_,
                "cop_y_surg": None if cs is None else cs[:n, 1],
                "cop_y_ns": None if cn is None else cn[:n, 1],
                "knee_moment_surg": mech_s["knee_ext_mom"], "knee_moment_ns": mech_n["knee_ext_mom"],
                "ankle_moment_surg": mech_s["ankle_pf_mom"], "ankle_moment_ns": mech_n["ankle_pf_mom"],
            }

    # ── Single-leg vertical ──────────────────────────────────────────────────
    per = {}
    for side in ("L", "R"):
        per[side] = A.aggregate([A.analyse_slv(t, side, bw, neutral) for t in S.by_test(f"slv_{side}")],
                                key_metric="jump_height_in")
    if per["L"] and per["R"]:
        ms, sds, reps, ns_n, names_s = per[surg]
        mn, sdn, repn, nn_n, names_n = per[ns_side]
        info["trials"]["single_leg_jump"] = names_s + names_n

        def limb(m):
            return NS(jump_height_cm=m["jump_height_in"], peak_force_N=m["peak_grf"],
                      propulsion_impulse_Ns=m["net_impulse"], rfd_100ms=m["peak_rfd"],
                      flight_time_s=m["flight_time"], contact_time_s=m["contact_time"],
                      peak_knee_flexion_deg=m.get("knee_flex", np.nan),
                      peak_valgus_deg=m.get("knee_valgus", np.nan))
        results["single_leg_jump"] = NS(
            surgical=limb(ms), non_surgical=limb(mn),
            lsi_jump_height=lsi(ms["jump_height_in"], mn["jump_height_in"]),
            lsi_peak_force=lsi(ms["peak_grf"], mn["peak_grf"]),
            lsi_impulse=lsi(ms["net_impulse"], mn["net_impulse"]),
            n_trials=(ns_n, nn_n))
        merged = {f"s_{k}": v for k, v in ms.items()}
        merged.update({f"n_{k}": v for k, v in mn.items()})
        for k in ("hip_pos_work", "knee_pos_work", "ankle_pos_work"):
            merged[f"{k}_lsi"] = lsi(ms.get(k), mn.get(k))
        sig = {"fz_surg": reps["fz"], "fz_ns": repn["fz"],
               "cop_y_surg": reps["cop_y"], "cop_y_ns": repn["cop_y"]}
        for key in ("knee_flex", "knee_valgus", "tib_rot", "hip_add", "hip_ir", "ankle_flex",
                    "ankle_pf_mom_ts", "ankle_pow_ts", "knee_pow_ts", "hip_pow_ts",
                    "takeoff_k", "bottom_k"):
            sig[f"{key}_surg"] = reps.get(key)
            sig[f"{key}_ns"] = repn.get(key)
        _scalars(merged, sig)
        sig["extra_rows"] = ([("Trials analysed", str(ns_n), str(nn_n), None, None, None),
                              ("Peak Power (W/kg)", _g(ms, "peak_power_wkg"), _g(mn, "peak_power_wkg"),
                               lsi(ms.get("peak_power_wkg"), mn.get("peak_power_wkg")), "SL Peak Power",
                               "sl_power_lsi"),
                              ("Push-off Work (J)", _g(ms, "con_work"), _g(mn, "con_work"),
                               lsi(ms.get("con_work"), mn.get("con_work")), None, None),
                              ("Peak RPD (W/s)", _g(ms, "rpd"), _g(mn, "rpd"),
                               lsi(ms.get("rpd"), mn.get("rpd")), None, None)]
                             + _work_rows(merged, "SLJ", ("pos",), ankle_focus_rtr="ankle_work_lsi"))
        sig["qc"] = (ms.get("_qc") or []) + (mn.get("_qc") or [])
        sig["joint_work"] = {"kind": "pos", **{tag: {j: _g(merged, f"{tag}_{j}_pos_work") for j in ("hip", "knee", "ankle")} for tag in ("s", "n")}}
        signals["single_leg_jump"] = sig

    # ── Balance ──────────────────────────────────────────────────────────────
    def _bal(test):
        ts = S.by_test(test)
        if not ts:
            return None, None, None
        t = ts[0]
        cop = t.cop.get("FP3")
        if cop is None:
            return None, None, None
        return cop[:, 0], cop[:, 1], t.fz("FP3")

    xs_s, ys_s, fz_s_s = _bal(f"balance_{surg}_std")
    xs_n, ys_n, fz_s_n = _bal(f"balance_{ns_side}_std")
    xa_s, ya_s, fz_a_s = _bal(f"balance_{surg}_airex")
    xa_n, ya_n, fz_a_n = _bal(f"balance_{ns_side}_airex")
    if any(a is not None for a in (xs_s, xs_n, xa_s, xa_n)):
        from config import PROP_TRIM_SECONDS
        rate_f = info["rates"][0] if info["rates"] else 900.0
        res = prop_mod.analyse(
            cop_x_surg_std=xs_s, cop_y_surg_std=ys_s, cop_x_ns_std=xs_n, cop_y_ns_std=ys_n,
            cop_x_surg_airex=xa_s, cop_y_surg_airex=ya_s, cop_x_ns_airex=xa_n, cop_y_ns_airex=ya_n,
            rate_f=rate_f, surg_side=surg, trim_seconds=PROP_TRIM_SECONDS,
            fz={"surg_std": fz_s_s, "ns_std": fz_s_n, "surg_airex": fz_a_s, "ns_airex": fz_a_n})
        results["proprioception"] = res

        def _ls(c):
            if c is None:
                return {"cop_x_surg": None, "cop_y_surg": None, "cop_x_ns": None, "cop_y_ns": None}
            return {"cop_x_surg": c.surgical.cop_x if c.surgical else None,
                    "cop_y_surg": c.surgical.cop_y if c.surgical else None,
                    "cop_x_ns": c.non_surgical.cop_x if c.non_surgical else None,
                    "cop_y_ns": c.non_surgical.cop_y if c.non_surgical else None}
        signals["proprioception"] = {"standard": _ls(res.standard), "airex": _ls(res.airex)}

    return results, signals, info
