"""
rts14/loader.py
===============
Fast reader for v14 Visual3D ASCII exports.

Each export file has 5 header rows (path, signal, type, folder, component)
then data rows; column 0 is the frame number. Every trial is a column group
identified by its C3D path in row 1, so one file holds the whole session.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from rts14.signs import apply_signs

# Sampling rates. Force frames are exactly 3x kinematic frames; free-fall of
# the pelvis during DJ flight (~280 Hz) and V3D's own flight-time jump height
# both put the cameras at 300 Hz and the plates at 900 Hz.
ANALOG_RATE    = 900.0
KINEMATIC_RATE = 300.0

FILES = {
    "force":   "Force_Data.txt",
    "cop":     "COP_Data.txt",
    "angles":  "Joint_Data.txt",
    "moments": "Joint_Moments.txt",
    "powers":  "Joint_Powers.txt",
    "pelvis":  "COP_Pelvis.txt",
    "general": "General_Data.txt",
}

# Trial-name → test key. Matched on the C3D file name (without extension).
_TEST_PATTERNS = [
    (re.compile(r"^PCT\s*L\s*0*1\b", re.I), "balance_L_std"),
    (re.compile(r"^PCT\s*L\s*0*2\b", re.I), "balance_L_airex"),
    (re.compile(r"^PCT\s*R\s*0*1\b", re.I), "balance_R_std"),
    (re.compile(r"^PCT\s*R\s*0*2\b", re.I), "balance_R_airex"),
    (re.compile(r"^SLV\s*L", re.I),         "slv_L"),
    (re.compile(r"^SLV\s*R", re.I),         "slv_R"),
    (re.compile(r"^DJ", re.I),              "drop_jump"),
    (re.compile(r"^DL", re.I),              "drop_landing"),
    (re.compile(r"^(Vert|CMJ)", re.I),      "max_vertical"),
    (re.compile(r"^End", re.I),             "endurance_squat"),
]


def test_key(trial_name: str) -> Optional[str]:
    for pat, key in _TEST_PATTERNS:
        if pat.search(trial_name.strip()):
            return key
    return None


@dataclass
class Trial:
    name:   str                       # "DJ 1"
    path:   str                       # full C3D path
    test:   Optional[str]             # "drop_jump", "slv_L", ...
    # {signal: (n, 3) array}  e.g. force["FP1"][:, 2] = Fz
    force:   Dict[str, np.ndarray] = field(default_factory=dict)
    cop:     Dict[str, np.ndarray] = field(default_factory=dict)
    angles:  Dict[str, np.ndarray] = field(default_factory=dict)
    moments: Dict[str, np.ndarray] = field(default_factory=dict)
    powers:  Dict[str, np.ndarray] = field(default_factory=dict)
    pelvis:  Optional[np.ndarray] = None
    rate_f:  float = ANALOG_RATE
    rate_k:  float = KINEMATIC_RATE

    def fz(self, plate: str) -> Optional[np.ndarray]:
        a = self.force.get(plate)
        return None if a is None else a[:, 2]


@dataclass
class Session:
    folder:   str
    trials:   Dict[str, Trial] = field(default_factory=dict)
    mass_n:   float = float("nan")
    paths:    list = field(default_factory=list)   # every C3D path seen

    def by_test(self, test: str):
        """Trials for one test, in trial-number order."""
        ts = [t for t in self.trials.values() if t.test == test]
        return sorted(ts, key=lambda t: _trial_number(t.name))


def _trial_number(name: str) -> int:
    m = re.search(r"(\d+)\s*$", name)
    return int(m.group(1)) if m else 0


def _read_export(path: Path) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """
    Parse one export → {trial_name: {signal: {component: 1-D array}}} plus
    the trial's full path under key "__path__".
    Trailing NaN padding (shorter trials) is trimmed per column.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        hdr = [fh.readline().rstrip("\r\n").split("\t") for _ in range(5)]
    paths, sigs, _types, _folders, comps = hdr
    body = pd.read_csv(path, sep="\t", header=None, skiprows=5,
                       engine="c", na_values=["", "NaN", "nan"],
                       dtype=float, on_bad_lines="skip")
    arr = body.to_numpy(dtype=float)

    out: Dict[str, dict] = {}
    for c in range(1, len(paths)):
        if c >= arr.shape[1]:
            break
        p = paths[c].strip()
        if not p:
            continue
        name = os.path.splitext(os.path.basename(p.replace("\\", "/")))[0]
        col = arr[:, c]
        valid = np.where(~np.isnan(col))[0]
        col = col[: valid[-1] + 1] if len(valid) else col[:0]
        t = out.setdefault(name, {"__path__": p})
        t.setdefault(sigs[c].strip(), {})[comps[c].strip()] = col
    return out


def _stack(comp: Dict[str, np.ndarray]) -> np.ndarray:
    """{X,Y,Z} → (n, 3), padding shorter components with NaN."""
    cols = [comp.get(k, np.array([])) for k in ("X", "Y", "Z")]
    n = max(len(c) for c in cols)
    out = np.full((n, 3), np.nan)
    for i, c in enumerate(cols):
        out[: len(c), i] = c
    return out


def load_session(folder: str) -> Session:
    folder = Path(folder)
    s = Session(folder=str(folder))

    for kind, fname in FILES.items():
        p = folder / fname
        if not p.exists():
            continue
        data = _read_export(p)

        if kind == "general":
            vals = [v.get("MASS_N", {}).get("X") for v in data.values()]
            vals = [float(x[0]) for x in vals if x is not None and len(x)]
            if vals:
                s.mass_n = float(np.median(vals))
            continue

        for name, sigs in data.items():
            path = sigs.pop("__path__")
            if path not in s.paths:
                s.paths.append(path)
            tr = s.trials.get(name)
            if tr is None:
                tr = s.trials[name] = Trial(name=name, path=path, test=test_key(name))
            for sig, comp in sigs.items():
                a = _stack(comp)
                if kind == "force":
                    tr.force[sig] = a
                elif kind == "cop":
                    tr.cop[sig] = a * 1000.0          # m → mm
                elif kind == "angles":
                    tr.angles[sig] = apply_signs(sig, a)
                elif kind == "moments":
                    tr.moments[sig] = apply_signs(sig, a)
                elif kind == "powers":
                    tr.powers[sig] = a                # scalar-ish; unaffected by NEGATE
                elif kind == "pelvis":
                    tr.pelvis = a
    return s
