"""
rts14
=====
Reader + analysis for the v14 Visual3D pipeline ("model + export only").

Visual3D exports raw-ish signals for ALL trials into a handful of files:

    Force_Data.txt     FP1-3 force (ORIGINAL, analog rate)
    COP_Data.txt       FP1-3 centre of pressure (ORIGINAL, metres)
    Joint_Data.txt     L/R hip, knee, ankle angles (no NEGATE flags)
    Joint_Moments.txt  L/R hip, knee, ankle moments (Nm)
    Joint_Powers.txt   L/R hip, knee, ankle powers (W)
    COP_Pelvis.txt     Pelvis_CG position (m)        [optional]
    General_Data.txt   MASS_N                         [optional]

Everything else — contact detection, jump height, RSI, RFD, loading rate,
impulse, joint work, balance — is computed here, PER TRIAL, and the metrics
(not the curves) are averaged across trials.

    from rts14.pipeline import is_v14_folder, build_results
"""
