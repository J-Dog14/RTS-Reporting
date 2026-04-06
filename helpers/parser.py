"""
helpers/parser.py
==================
Parses Visual 3D ASCII export files produced by Export_Data_To_Ascii_File.

Two file types are handled automatically:

  Scalar (Data.txt)
  -----------------
  One numeric value per signal per trial.  Trials sit side-by-side as column
  groups.  Used to import V3D-computed metrics (jump height, LSI, etc.) and
  patient demographics (MASS_N, Affected_Leg).

  Time-series (Forces.txt / COFP.txt / Joints.txt / Moments.txt)
  ---------------------------------------------------------------
  Many rows of frame-by-frame data.  Same column grouping as scalar files.
  Used for force-time curves, joint-angle traces, and COP analysis.

File layout (tab-delimited, 1-based row numbers)
------------------------------------------------
  Row 1  : [empty]  <filepath_trial1>  <filepath_trial1>  ...  <filepath_trial2>  ...
  Row 2  : [empty]  <signal_name>      <signal_name>      ...  <signal_name>      ...
  Row 3  : [empty]  <type>             <type>             ...  (METRIC | FORCE_PLATE_FORCES | ...)
  Row 4  : [empty]  PROCESSED          PROCESSED          ...
  Row 5  : ITEM     X                  Y                  Z   ...  (component labels)
  Row 6+ : <frame>  <value>            <value>            ...  (data; col 1 = frame# or empty)

Column 0 is ALWAYS skipped (contains "ITEM" in row 5, frame number in data rows).
Trial boundaries are identified by changes in the filepath string in row 1.

Usage
-----
    from helpers.parser import parse_v3d_file

    # Load scalar metrics
    data = parse_v3d_file("Drop Jump Data.txt")
    jh   = data.scalar("JH_IN_L")           # float, averaged across trials
    mass = data.scalar("MASS_N")             # float
    leg  = data.scalar_str("Affected_Leg")   # "L" or "R"

    # Load time-series forces
    forces = parse_v3d_file("Drop Jump Forces.txt", rate=1000)
    fz_fp1 = forces.timeseries("FP1", "Z")   # np.ndarray, averaged across trials
    fz_fp2 = forces.timeseries("FP2", "Z")

    # Load joint angles
    joints = parse_v3d_file("Drop Jump Joints.txt", rate=200)
    flex   = joints.timeseries("L_Knee_Angle", "X")
    valgus = joints.timeseries("L_Knee_Angle", "Y")
"""

import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ─── Public API ───────────────────────────────────────────────────────────────

class V3DExport:
    """
    Holds all parsed data from one V3D ASCII export file.

    Attributes
    ----------
    filepath  : str   — source file path
    is_scalar : bool  — True if the file is a METRIC (scalar) file
    rate      : float — sample rate in Hz (meaningful for time-series only)
    _store    : nested dict  {signal -> {component -> {trial_path -> value_or_array}}}
    """

    def __init__(self, filepath: str = "", rate: float = 1000.0):
        self.filepath  = filepath
        self.is_scalar = False
        self.rate      = rate
        # {signal_name: {component_label: {trial_path: float or np.ndarray}}}
        self._store: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # ── Scalar access ──────────────────────────────────────────────────────────

    def scalar(self, signal: str, component: str = None) -> float:
        """
        Return a scalar metric value averaged across all trials.
        Returns np.nan if the signal is not found.
        component: if None, use the first available component.
        """
        comp_data = self._get_comp_data(signal, component)
        if comp_data is None:
            return np.nan
        values = [v for v in comp_data.values()
                  if not isinstance(v, np.ndarray)]
        if not values:
            return np.nan
        try:
            return float(np.nanmean([float(v) for v in values]))
        except (TypeError, ValueError):
            return np.nan

    def scalar_str(self, signal: str) -> str:
        """
        Return a string scalar value (e.g. Affected_Leg = 'L').
        Returns "" if not found.
        """
        comp_data = self._get_comp_data(signal, None)
        if comp_data is None:
            return ""
        for v in comp_data.values():
            if isinstance(v, str):
                return v.strip()
            try:
                return str(v).strip()
            except Exception:
                continue
        return ""

    def all_scalars(self) -> Dict[str, float]:
        """
        Return a flat dict {signal_name: averaged_scalar_value} for all signals.
        Multi-component signals get separate entries: signal + "_" + component.
        """
        out = {}
        for sig, comps in self._store.items():
            comp_list = list(comps.keys())
            if len(comp_list) == 1:
                out[sig] = self.scalar(sig, comp_list[0])
            else:
                for comp in comp_list:
                    out[f"{sig}_{comp}"] = self.scalar(sig, comp)
        return out

    # ── Time-series access ─────────────────────────────────────────────────────

    def timeseries(self, signal: str, component: str = None,
                   averaged: bool = True) -> Optional[np.ndarray]:
        """
        Return a 1-D numpy array for a signal component, averaged across trials
        (or the per-trial dict if averaged=False).

        component: 'X', 'Y', 'Z', or None (uses first available).
        Returns None if signal not found.
        """
        comp_data = self._get_comp_data(signal, component)
        if comp_data is None:
            return None

        arrays = [v for v in comp_data.values() if isinstance(v, np.ndarray)]
        if not arrays:
            return None

        if not averaged:
            return {k: v for k, v in comp_data.items() if isinstance(v, np.ndarray)}

        if len(arrays) == 1:
            return arrays[0].copy()

        # Trials can have different effective lengths (shorter trials are
        # padded with NaN to the longest row count in the file).
        # Find the last non-NaN frame in each trial and trim to the shortest.
        eff_lens = []
        for a in arrays:
            valid = np.where(~np.isnan(a))[0]
            eff_lens.append(int(valid[-1]) + 1 if len(valid) > 0 else 0)

        min_eff = min(eff_lens) if eff_lens else 0
        if min_eff == 0:
            return None

        # Average only over the common (fully-valid) frames
        stack = np.vstack([a[:min_eff] for a in arrays])
        return np.nanmean(stack, axis=0)

    def timeseries_per_trial(self, signal: str, component: str = None
                             ) -> List[np.ndarray]:
        """Return list of arrays, one per trial (not averaged)."""
        comp_data = self._get_comp_data(signal, component)
        if comp_data is None:
            return []
        return [v for v in comp_data.values() if isinstance(v, np.ndarray)]

    # ── Biomechanics convenience helpers ──────────────────────────────────────

    def get_knee(self, side: str) -> dict:
        """
        Return knee kinematics for one side.
        Output dict keys: 'flex_ext', 'valgus', 'tib_rot' (all np.ndarray).
        Missing components are omitted.
        """
        side = side.upper()
        prefix = "L" if side == "L" else "R"
        sig = f"{prefix}_Knee_Angle"
        out = {}
        x = self.timeseries(sig, "X")
        if x is not None: out["flex_ext"] = x
        y = self.timeseries(sig, "Y")
        if y is not None: out["valgus"] = y
        z = self.timeseries(sig, "Z")
        if z is not None: out["tib_rot"] = z
        return out

    def get_hip(self, side: str) -> dict:
        """
        Return hip kinematics for one side.
        V3D exports only the Y component (abduction/adduction) for hip.
        Output dict key: 'flex_ext' mapped to the Y component.
        """
        side = side.upper()
        prefix = "L" if side == "L" else "R"
        sig = f"{prefix}_Hip_Angle"
        out = {}
        y = self.timeseries(sig, "Y")
        if y is not None: out["flex_ext"] = y
        return out

    def get_ankle(self, side: str) -> dict:
        """
        Return ankle kinematics.  V3D exports only X (dorsi/plantar).
        """
        side = side.upper()
        prefix = "L" if side == "L" else "R"
        sig = f"{prefix}_Ankle_Angle"
        out = {}
        x = self.timeseries(sig, "X")
        if x is not None: out["flex_ext"] = x
        return out

    def trial_paths(self) -> List[str]:
        """Return all unique C3D trial filepaths embedded in this export (row 1 values)."""
        paths: set = set()
        for comps in self._store.values():
            for trials in comps.values():
                paths.update(trials.keys())
        return sorted(paths)

    def available_signals(self) -> List[str]:
        """Return a list of all parsed signal names."""
        return sorted(self._store.keys())

    def print_signals(self):
        """Print a diagnostic summary of all loaded signals."""
        kind = "SCALAR" if self.is_scalar else f"TIME-SERIES @ {self.rate:.0f} Hz"
        print(f"\nV3DExport [{kind}]: {Path(self.filepath).name}")
        for sig in sorted(self._store):
            comps = self._store[sig]
            for comp, trials in comps.items():
                n_trials = len(trials)
                sample = next(iter(trials.values()))
                if isinstance(sample, np.ndarray):
                    info = f"{len(sample)} frames"
                else:
                    info = f"value = {sample!r}"
                print(f"  {sig:<40s} [{comp}]  {n_trials} trial(s)  {info}")
        print()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_comp_data(self, signal: str,
                       component: Optional[str]) -> Optional[Dict[str, Any]]:
        """Resolve signal + component to the inner trial dict, case-insensitive."""
        sig_data = self._resolve_signal(signal)
        if sig_data is None:
            return None

        if component is None:
            # Return first available component
            return next(iter(sig_data.values())) if sig_data else None

        comp = component.upper()
        # Try exact then case-insensitive
        for k, v in sig_data.items():
            if k.upper() == comp:
                return v
        return None

    def _resolve_signal(self, signal: str) -> Optional[Dict]:
        """Case-insensitive signal name lookup."""
        if signal in self._store:
            return self._store[signal]
        sig_up = signal.upper()
        for k, v in self._store.items():
            if k.upper() == sig_up:
                return v
        return None

    def _add(self, signal: str, component: str, trial_path: str, value: Any):
        """Store a single value (float or ndarray) for a signal/component/trial."""
        if signal not in self._store:
            self._store[signal] = {}
        if component not in self._store[signal]:
            self._store[signal][component] = {}
        self._store[signal][component][trial_path] = value


# ─── Parser ───────────────────────────────────────────────────────────────────

def parse_v3d_file(filepath: str, rate: float = 1000.0,
                   silent: bool = False) -> V3DExport:
    """
    Parse a V3D ASCII export file and return a V3DExport object.

    filepath : path to the .txt file
    rate     : sample rate in Hz — only relevant for time-series files;
               use ANALOG_RATE (1000) for Forces/COFP, KINEMATIC_RATE (200)
               for Joints/Moments.
    """
    export = V3DExport(filepath=str(filepath), rate=rate)

    try:
        rows = _read_rows(filepath)
    except FileNotFoundError:
        return export
    except Exception as e:
        print(f"[parser] Could not read {filepath}: {e}")
        return export

    if len(rows) < 6:
        if not silent:
            print(f"[parser] {Path(filepath).name}: fewer than 6 rows — skipping")
        return export

    # Pad all rows to the same width
    n_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < n_cols:
            r.append("")

    # Header rows (0-indexed, so row 0 = 1st line of file)
    paths_row  = rows[0]   # trial file paths
    names_row  = rows[1]   # signal names
    types_row  = rows[2]   # METRIC / FORCE_PLATE_FORCES / LINK_MODEL_BASED / etc.
    # rows[3]  = PROCESSED (not used)
    item_row   = rows[4]   # ITEM in col 0, then component labels (X Y Z ...)
    data_rows  = rows[5:]  # actual numeric data

    # Is this a scalar (METRIC) file?
    for cell in types_row[1:]:
        if cell.strip().upper() == "METRIC":
            export.is_scalar = True
            break

    # Build per-column metadata (skip column 0 throughout)
    col_paths  = paths_row[1:]
    col_names  = names_row[1:]
    col_comps  = item_row[1:]

    # Replace empty component labels with "X"
    col_comps = [c if c.strip() else "X" for c in col_comps]

    n_data_cols = len(col_paths)

    if export.is_scalar:
        _parse_scalar(export, data_rows, col_paths, col_names, col_comps, n_data_cols)
    else:
        _parse_timeseries(export, data_rows, col_paths, col_names, col_comps, n_data_cols)

    return export


# ─── Internal parsing helpers ─────────────────────────────────────────────────

def _read_rows(filepath: str) -> List[List[str]]:
    """Read file and split every line on tabs.  Strips CR/LF."""
    rows = []
    with open(filepath, "r", errors="replace") as f:
        for line in f:
            cells = line.rstrip("\r\n").split("\t")
            rows.append([c.strip() for c in cells])
    return rows


def _parse_scalar(export: V3DExport,
                  data_rows: List[List[str]],
                  col_paths: List[str],
                  col_names: List[str],
                  col_comps: List[str],
                  n_data_cols: int):
    """
    Parse a scalar (METRIC) file: one data row, multiple trial columns.
    Values may be numeric or string (e.g. Affected_Leg = 'L').
    """
    # Find first non-empty data row
    data_line: Optional[List[str]] = None
    for row in data_rows:
        # The row has col 0 (frame/empty) + data cols
        if len(row) > 1 and any(c.strip() for c in row[1:]):
            data_line = row
            break

    if data_line is None:
        return

    values = data_line[1:]  # skip col 0

    for i in range(min(n_data_cols, len(values))):
        path   = col_paths[i] if i < len(col_paths) else ""
        name   = col_names[i] if i < len(col_names) else ""
        comp   = col_comps[i] if i < len(col_comps) else "X"
        raw    = values[i]

        if not name:
            continue

        # Try numeric conversion; fall back to string
        val: Any
        if raw.strip() in ("", "nan", "NaN", "#N/A"):
            val = np.nan
        else:
            try:
                val = float(raw)
            except ValueError:
                val = raw.strip()   # string scalar (e.g. "L")

        export._add(name, comp, path, val)


def _parse_timeseries(export: V3DExport,
                      data_rows: List[List[str]],
                      col_paths: List[str],
                      col_names: List[str],
                      col_comps: List[str],
                      n_data_cols: int):
    """
    Parse a time-series file.
    Builds a numeric matrix then slices it per (signal, component, trial).
    """
    # Collect all numeric rows; skip blank lines and header remnants
    matrix_rows = []
    for row in data_rows:
        if not any(c for c in row):
            continue
        numeric_cells = row[1:]  # skip col 0 (frame number)
        vals = []
        valid = False
        for c in numeric_cells:
            if c.strip() in ("", "nan", "NaN", "#N/A"):
                vals.append(np.nan)
            else:
                try:
                    vals.append(float(c))
                    valid = True
                except ValueError:
                    vals.append(np.nan)
        if valid:
            matrix_rows.append(vals)

    if not matrix_rows:
        return

    # Build 2-D array: (n_frames, n_data_cols)
    # Pad rows to equal length
    max_width = max(len(r) for r in matrix_rows)
    for r in matrix_rows:
        while len(r) < max_width:
            r.append(np.nan)

    arr = np.array(matrix_rows, dtype=float)

    # Assign each column to its signal/component/trial
    for i in range(min(n_data_cols, arr.shape[1])):
        path = col_paths[i] if i < len(col_paths) else ""
        name = col_names[i] if i < len(col_names) else ""
        comp = col_comps[i] if i < len(col_comps) else "X"

        if not name:
            continue

        export._add(name, comp, path, arr[:, i])


# ─── Convenience loader for a whole test session folder ──────────────────────

def load_test_files(folder: str, file_keys: List[str],
                    rates: Dict[str, float] = None) -> Dict[str, V3DExport]:
    """
    Load multiple export files for one test session.

    folder    : path to the folder containing V3D export files
    file_keys : list of EXPORT_FILES keys to load (from config.py)
    rates     : {file_key: sample_rate_hz} — defaults to 1000 for all

    Returns a dict {file_key: V3DExport}. Missing files produce empty exports.

    Example
    -------
        from config import EXPORT_FILES
        files = load_test_files(session_dir,
                                ["drop_jump_forces", "drop_jump_joints"],
                                rates={"drop_jump_forces": 1000,
                                       "drop_jump_joints": 200})
        fz = files["drop_jump_forces"].timeseries("FP1", "Z")
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import EXPORT_FILES

    if rates is None:
        rates = {}

    folder_path = Path(folder)
    result = {}
    for key in file_keys:
        filename = EXPORT_FILES.get(key, "")
        if not filename:
            print(f"[parser] Unknown file key: {key!r}")
            result[key] = V3DExport()
            continue
        filepath = folder_path / filename
        rate = rates.get(key, 1000.0)
        if filepath.exists():
            result[key] = parse_v3d_file(str(filepath), rate=rate)
        else:
            result[key] = V3DExport(filepath=str(filepath), rate=rate)

    return result
