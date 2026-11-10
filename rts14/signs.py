"""
rts14/signs.py
==============
Sign conventions for v14 exports (Visual3D raw — no NEGATE flags).

Verified on a real session (John Jiles, DJ trials):
  • Knee angle X: flexion reads NEGATIVE on both sides      → ×-1 both
  • Hip angle X:  flexion positive on both sides             → keep
  • Y / Z (frontal / transverse) mirror between sides        → ×-1 on LEFT
  • Knee moment X: extensor positive on both                 → keep
  • Ankle moment X: plantarflexor NEGATIVE on both           → keep (report
    uses the magnitude of the negative peak)
These match the v13 NEGATE flags, so v13 and v14 values line up.

After conversion (both sides):
  knee X  + = flexion     knee Y + = valgus     knee Z + = internal rotation
  hip  X  + = flexion     hip  Y + = adduction  hip  Z + = internal rotation
  ankle X raw (≈90° offset; + = toward dorsiflexion) — zeroed in analysis
"""

import numpy as np

# signal-name prefix → per-component multipliers (X, Y, Z)
_TABLE = {
    "R_Knee_Angle":  (-1,  1,  1),
    "L_Knee_Angle":  (-1, -1, -1),
    "R_Hip_Angle":   ( 1,  1,  1),
    "L_Hip_Angle":   ( 1, -1, -1),
    "R_Ankle_Angle": ( 1,  1,  1),
    "L_Ankle_Angle": ( 1, -1, -1),
    "R_Knee_Moment": ( 1,  1,  1),
    "L_Knee_Moment": ( 1, -1, -1),
    "R_Hip_Moment":  ( 1,  1,  1),
    "L_Hip_Moment":  ( 1, -1, -1),
    "R_Ankle_Moment": (1,  1,  1),
    "L_Ankle_Moment": (1, -1, -1),
}


def apply_signs(signal: str, arr: np.ndarray) -> np.ndarray:
    m = _TABLE.get(signal)
    if m is None:
        return arr
    return arr * np.array(m, dtype=float)
