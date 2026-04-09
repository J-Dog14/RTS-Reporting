"""
main.py
========
Return-to-Sport Reporting Pipeline — single entry point.

HOW TO USE
----------
  1. Set SESSION_FOLDER to the patient's export folder.
  2. Set PATIENT_NAME and SURGICAL_SIDE for this patient.
  3. Run:  python main.py
     OR:   python main.py "D:\\RTS 2.0\\Data Exports\\Smith John 2024-01-15"

What this script does
---------------------
  1. Reads General Data.txt for body weight (MASS_N).
  2. Loads all V3D export files in the session folder.
  3. Runs each test module (drop_jump, drop_landing, max_vertical, etc.).
  4. Generates a PDF report in the session folder.

If any test file is missing, that test is skipped and shown as "No data".

NOTES ON UNITS
--------------
  • Force data (Forces.txt): Newtons.
  • COFP data (COFP.txt): METRES — converted to mm automatically below.
  • Jump height in Data.txt (JH_IN, JH_IN_L, JH_IN_R): V3D proprietary units.
    Set JUMP_HEIGHT_SCALE in config.py if you want mm/cm conversion.
  • Surgical side (SURGICAL_SIDE): edit in config.py or at the top of this file.
    V3D does NOT export this — you must set it manually.
"""

import sys
import os
import re
import datetime
from pathlib import Path

# ─── USER SETTINGS — edit these for each patient session ─────────────────────
SESSION_FOLDER  = r"D:\RTS 2.0\Data Exports"   # Default; override on command line
REPORT_FILENAME = "RTS_Report.pdf"
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np

ROOT = Path(__file__).resolve().parent


# ─── Surgical-side selection dialog ──────────────────────────────────────────

def _parse_c3d_path(path: str) -> dict:
    """
    Extract patient info from one C3D file path.

    Example:  D:\\RTS 2.0\\Data\\Lyons, Temperance_TL\\2025-12-03_\\SLVL 4.c3d
    Returns:
        display_name : 'Temperance Lyons'   (First Last — for display / filename)
        raw_name     : 'Lyons, Temperance'  (Last, First  — for the report header)
        session_date : '12-03-25'           (MM-DD-YY extracted from path)
        save_dir     : 'D:\\RTS 2.0\\Data\\Lyons, Temperance_TL\\2025-12-03_'
    """
    parts = re.split(r"[\\/]", path)
    for i, part in enumerate(parts):
        if "," in part and len(part) > 3:
            # Strip trailing initials: _TL, _AW, _ABC etc.
            raw = re.sub(r"_[A-Z]{1,4}$", "", part).strip()
            if not raw:
                continue
            # "Lyons, Temperance" → "Temperance Lyons"
            if "," in raw:
                last, first = raw.split(",", 1)
                display = f"{first.strip()} {last.strip()}"
            else:
                display = raw

            # Date from next path segment (e.g. "2025-12-03_")
            date_str = None
            if i + 1 < len(parts):
                date_part = parts[i + 1].rstrip("_").rstrip("-")
                try:
                    dt = datetime.datetime.strptime(date_part, "%Y-%m-%d")
                    date_str = dt.strftime("%m-%d-%y")
                except ValueError:
                    pass

            # Patient session folder = parent directory of the C3D file
            try:
                save_dir = str(Path(path).parent)
            except Exception:
                save_dir = None

            return {
                "display_name": display,
                "raw_name":     raw,
                "session_date": date_str,
                "save_dir":     save_dir,
            }
    return {}


def _collect_session_info(folder: Path) -> dict:
    """
    Scan ALL .txt files in the exports folder.  For each file, extract
    patient info from the embedded C3D file paths.

    Uses majority-vote across all files so that stale files from a previous
    patient don't silently override the current session.

    Returns dict with keys: display_name, raw_name, session_date, save_dir.
    """
    from collections import Counter

    name_display_counts: Counter = Counter()
    name_raw_counts:     Counter = Counter()
    date_counts:         Counter = Counter()
    dir_counts:          Counter = Counter()

    # Sort newest-modified first so ties resolve in favour of current session
    txt_files = sorted(folder.glob("*.txt"),
                       key=lambda p: p.stat().st_mtime if p.exists() else 0,
                       reverse=True)

    for txt_file in txt_files:
        try:
            exp = parse_v3d_file(str(txt_file), rate=ANALOG_RATE, silent=True)
            for path in exp.trial_paths():
                info = _parse_c3d_path(path)
                if info.get("display_name"):
                    name_display_counts[info["display_name"]] += 1
                    name_raw_counts[info["raw_name"]] += 1
                if info.get("session_date"):
                    date_counts[info["session_date"]] += 1
                if info.get("save_dir"):
                    dir_counts[info["save_dir"]] += 1
                break  # one path per file is enough
        except Exception:
            continue

    result = {}
    if name_display_counts:
        result["display_name"] = name_display_counts.most_common(1)[0][0]
        result["raw_name"]     = name_raw_counts.most_common(1)[0][0]
        total = sum(name_display_counts.values())
        wins  = name_display_counts.most_common(1)[0][1]
        print(f"[main] Patient name: {result['display_name']!r} "
              f"(matched in {wins}/{total} export files)")
    if date_counts:
        result["session_date"] = date_counts.most_common(1)[0][0]
    if dir_counts:
        result["save_dir"] = dir_counts.most_common(1)[0][0]

    return result


def _ask_patient_name() -> str:
    """Show a simple dialog for the clinician to type the patient name."""
    try:
        import tkinter as tk
        from tkinter import simpledialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        name = simpledialog.askstring(
            "Patient Name",
            "Patient name not found in session files.\nEnter name (e.g. Smith, John):",
            parent=root,
        )
        root.destroy()
        return (name or "Unknown").strip()
    except Exception:
        return "Unknown"


def _ask_surgical_side(patient_name: str, default: str = "") -> str:
    """
    Show a branded GUI dialog so the clinician can select the surgical limb.
    Returns 'L' or 'R'.  Falls back to `default` if tkinter is unavailable
    (e.g. headless server) or the window is closed without a selection.
    """
    try:
        import tkinter as tk

        chosen = [default.upper() if default.upper() in ("L", "R") else "R"]
        confirmed = [False]

        BG     = "#39414a"
        BG_BTN = "#2a3038"
        FG     = "#ffffff"
        FG_SUB = "#c8cdd3"
        ORANGE = "#fc6c0f"

        root = tk.Tk()
        root.title("RTS Reporting — Surgical Side")
        root.resizable(False, False)
        root.configure(bg=BG)

        # Centre on screen
        W, H = 400, 230
        root.update_idletasks()
        sx = (root.winfo_screenwidth()  - W) // 2
        sy = (root.winfo_screenheight() - H) // 2
        root.geometry(f"{W}x{H}+{sx}+{sy}")
        root.lift()
        root.attributes("-topmost", True)

        # ── Header ────────────────────────────────────────────────────────────
        tk.Label(root, text=CLINIC_NAME,
                 bg=BG, fg=FG,
                 font=("Helvetica", 13, "bold")).pack(pady=(18, 2))

        tk.Label(root, text="Return to Sport Assessment",
                 bg=BG, fg=FG_SUB,
                 font=("Helvetica", 9)).pack()

        tk.Label(root, text=f"Patient:  {patient_name}",
                 bg=BG, fg=FG,
                 font=("Helvetica", 10)).pack(pady=(10, 2))

        tk.Label(root,
                 text="Surgical side not found in session data.\n"
                      "Please select the surgically repaired limb:",
                 bg=BG, fg=FG_SUB,
                 font=("Helvetica", 9),
                 justify="center").pack(pady=(4, 14))

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_frame = tk.Frame(root, bg=BG)
        btn_frame.pack()

        btn_kw = dict(width=9, height=2,
                      font=("Helvetica", 12, "bold"),
                      relief="flat", cursor="hand2",
                      activeforeground=FG)

        def _pick(side: str):
            chosen[0]    = side
            confirmed[0] = True
            root.destroy()

        tk.Button(btn_frame, text="◀  LEFT",
                  bg=ORANGE, fg=FG, activebackground="#d45a00",
                  command=lambda: _pick("L"), **btn_kw).pack(side="left", padx=16)

        tk.Button(btn_frame, text="RIGHT  ▶",
                  bg=BG_BTN, fg=FG, activebackground="#1e2428",
                  command=lambda: _pick("R"), **btn_kw).pack(side="left", padx=16)

        root.mainloop()

        # If closed without clicking a button, re-prompt rather than silently default
        if not confirmed[0]:
            print("[main] WARNING: Side dialog closed without selection — re-prompting.")
            return _ask_surgical_side(patient_name, default)

        return chosen[0]

    except Exception as exc:
        print(f"[main] WARNING: Could not show side-selection dialog ({exc}). "
              f"Defaulting to '{default or 'R'}'.")
        return default.upper() if default.upper() in ("L", "R") else "R"


def _normalize_surgery_date_slash(raw: str) -> str:
    """Pad slash-separated parts so e.g. 3/5/25 parses as MM/DD/YY."""
    s = raw.strip()
    if "/" not in s:
        return s
    parts = [p.strip() for p in s.split("/") if p.strip()]
    if len(parts) == 3:
        m, d, y = parts
        m, d = m.zfill(2), d.zfill(2)
        y = y if len(y) == 4 else y.zfill(2)
        return f"{m}/{d}/{y}"
    if len(parts) == 2:
        m, y = parts
        m = m.zfill(2)
        y = y if len(y) == 4 else y.zfill(2)
        return f"{m}/{y}"
    return s


def _parse_surgery_date(raw: str) -> str:
    """
    US order: MM/DD/YY, MM/DD/YYYY (full date → ISO YYYY-MM-DD),
    or MM/YY / MM/YYYY (month-only → MM/YY).
    Two-digit years: %%y uses 00–68 → 2000–2068, 69–99 → 1900–1999.
    Returns "" if invalid.
    """
    raw_in = raw.strip()
    if not raw_in:
        return ""
    try:
        dt = datetime.datetime.strptime(raw_in, "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        pass
    norm = _normalize_surgery_date_slash(raw_in)
    candidates = [norm, raw_in] if norm != raw_in else [raw_in]
    seen = set()
    to_try = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            to_try.append(c)
    for s in to_try:
        for fmt in ("%m/%d/%y", "%m/%d/%Y"):
            try:
                dt = datetime.datetime.strptime(s, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
    for s in to_try:
        for fmt in ("%m/%y", "%m/%Y"):
            try:
                dt = datetime.datetime.strptime(s, fmt)
                return dt.strftime("%m/%y")
            except ValueError:
                continue
    return ""


def _ask_surgery_date(patient_name: str) -> str:
    """
    Show a branded dialog for surgery date: MM/YY (approx.) or MM/DD/YY when known.
    Returns canonical MM/YY or YYYY-MM-DD, or "" if skipped.
    """
    BG      = "#39414a"   # matches new CLINIC header colour
    BG_DARK = "#2a3038"
    FG      = "#ffffff"
    FG_SUB  = "#c8cdd3"
    ORANGE  = "#fc6c0f"

    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()

        win = tk.Toplevel(root)
        win.title("RTS Reporting — Surgery Date")
        win.resizable(False, False)
        win.configure(bg=BG)
        win.lift()
        win.attributes("-topmost", True)
        win.focus_force()

        W, H = 400, 250
        win.update_idletasks()
        sx = (win.winfo_screenwidth()  - W) // 2
        sy = (win.winfo_screenheight() - H) // 2
        win.geometry(f"{W}x{H}+{sx}+{sy}")

        result    = [""]
        confirmed = [False]

        tk.Label(win, text=CLINIC_NAME,
                 bg=BG, fg=FG,
                 font=("Helvetica", 13, "bold")).pack(pady=(16, 2))
        tk.Label(win, text="Return to Sport Assessment",
                 bg=BG, fg=FG_SUB,
                 font=("Helvetica", 9)).pack()
        tk.Label(win, text=f"Patient:  {patient_name}",
                 bg=BG, fg=FG,
                 font=("Helvetica", 10)).pack(pady=(8, 2))
        tk.Label(win,
                 text="Enter surgery date:\n"
                      "MM/YY (approx.)  or  MM/DD/YY if exact day is known\n"
                      "Press Enter to confirm · Esc to skip",
                 bg=BG, fg=FG_SUB,
                 font=("Helvetica", 9), justify="center").pack(pady=(4, 6))

        entry_var = tk.StringVar()
        entry = tk.Entry(win, textvariable=entry_var,
                         font=("Helvetica", 14, "bold"), justify="center",
                         width=14, bd=0, relief="flat",
                         bg=BG_DARK, fg=FG,
                         insertbackground=ORANGE)
        entry.pack(ipady=7)
        entry.focus_set()

        def _submit(event=None):
            parsed = _parse_surgery_date(entry_var.get())
            if not parsed:
                entry.configure(bg="#5a2a2a")   # flash red for bad format
                entry.after(600, lambda: entry.configure(bg=BG_DARK))
                return
            result[0] = parsed
            confirmed[0] = True
            win.destroy()

        def _skip(event=None):
            confirmed[0] = True
            win.destroy()

        entry.bind("<Return>", _submit)
        entry.bind("<Escape>", _skip)

        btn_frame = tk.Frame(win, bg=BG)
        btn_frame.pack(pady=8)
        tk.Button(btn_frame, text="Confirm", command=_submit,
                  bg=ORANGE, fg=FG,
                  font=("Helvetica", 10, "bold"),
                  relief="flat", cursor="hand2", padx=16).pack(side="left", padx=8)
        tk.Button(btn_frame, text="Skip", command=_skip,
                  bg=BG_DARK, fg=FG_SUB,
                  font=("Helvetica", 10),
                  relief="flat", cursor="hand2", padx=16).pack(side="left", padx=8)

        root.wait_window(win)
        root.destroy()
        return result[0]

    except Exception as exc:
        print(f"[main] WARNING: Could not show surgery-date dialog ({exc}). Skipping.")
        return ""


def _parse_triple_hop(raw: str) -> float:
    """
    Parse a triple-hop distance in FEET.INCHES format → total inches.

    Rules:
      "11.11"  → 11 ft 11 in → 143 in   (decimal digits are literal inches)
      "11.1"   → 11 ft  1 in → 133 in
      "11.10"  → 11 ft 10 in → 142 in
      "11"     → 11 ft  0 in → 132 in   (no decimal = 0 inches)
      "143"    → treated as already-inches if > 30 ft (unrealistic feet value)

    Returns float inches, or None if unparseable.
    """
    s = raw.strip().replace("'", "").replace('"', "").replace(",", ".")
    if not s:
        return None
    try:
        if "." in s:
            feet_str, inch_str = s.split(".", 1)
            feet  = int(feet_str)
            # The digits AFTER the decimal are literal inches (not a fraction)
            # "1" → 1 inch, "10" → 10 inches, "01" → 1 inch
            inches = int(inch_str.lstrip("0") or "0")
        else:
            feet   = int(s)
            inches = 0
        total_in = feet * 12 + inches
        if total_in <= 0:
            return None
        return float(total_in)
    except (ValueError, AttributeError):
        return None


def _ask_triple_hop(patient_name: str) -> dict:
    """
    Branded dialog: two entry fields for surgical and non-surgical triple-hop
    distances (in feet.inches format).  Returns dict with keys:
        surg_in, ns_in, lsi    (all floats, or None if skipped)
    """
    BG      = "#39414a"
    BG_DARK = "#2a3038"
    FG      = "#ffffff"
    FG_SUB  = "#c8cdd3"
    ORANGE  = "#fc6c0f"

    result = {"surg_in": None, "ns_in": None, "lsi": None}

    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()

        win = tk.Toplevel(root)
        win.title("RTS Reporting — Triple Hop for Distance")
        win.resizable(False, False)
        win.configure(bg=BG)
        win.lift()
        win.attributes("-topmost", True)
        win.focus_force()

        W, H = 420, 310
        win.update_idletasks()
        sx = (win.winfo_screenwidth()  - W) // 2
        sy = (win.winfo_screenheight() - H) // 2
        win.geometry(f"{W}x{H}+{sx}+{sy}")

        confirmed = [False]

        tk.Label(win, text=CLINIC_NAME,
                 bg=BG, fg=FG, font=("Helvetica", 13, "bold")).pack(pady=(14, 2))
        tk.Label(win, text="Triple Hop for Distance",
                 bg=BG, fg=FG_SUB, font=("Helvetica", 9)).pack()
        tk.Label(win, text=f"Patient:  {patient_name}",
                 bg=BG, fg=FG, font=("Helvetica", 10)).pack(pady=(6, 2))
        tk.Label(win,
                 text="Enter best hop distance: FEET.INCHES\n"
                      "  e.g.  11.11 = 11 ft 11 in  |  11.1 = 11 ft 1 in\n"
                      "Press Enter to confirm · Esc to skip",
                 bg=BG, fg=FG_SUB, font=("Helvetica", 8), justify="center").pack(pady=(4, 8))

        # ── Entry fields ──────────────────────────────────────────────────────
        row_frame = tk.Frame(win, bg=BG)
        row_frame.pack()

        def _entry_col(parent, label_text):
            f = tk.Frame(parent, bg=BG)
            f.pack(side="left", padx=18)
            tk.Label(f, text=label_text, bg=BG, fg=FG_SUB,
                     font=("Helvetica", 9)).pack()
            var = tk.StringVar()
            e = tk.Entry(f, textvariable=var,
                         font=("Helvetica", 14, "bold"), justify="center",
                         width=10, bd=0, relief="flat",
                         bg=BG_DARK, fg=FG, insertbackground=ORANGE)
            e.pack(ipady=6)
            return var, e

        surg_label = "Surgical Limb"
        ns_label   = "Non-Surgical"
        surg_var, surg_entry = _entry_col(row_frame, surg_label)
        ns_var,   ns_entry   = _entry_col(row_frame, ns_label)

        def _submit(event=None):
            sv = _parse_triple_hop(surg_var.get())
            nv = _parse_triple_hop(ns_var.get())
            bad_s = sv is None and surg_var.get().strip()
            bad_n = nv is None and ns_var.get().strip()
            if bad_s:
                surg_entry.configure(bg="#5a2a2a")
                surg_entry.after(600, lambda: surg_entry.configure(bg=BG_DARK))
            if bad_n:
                ns_entry.configure(bg="#5a2a2a")
                ns_entry.after(600, lambda: ns_entry.configure(bg=BG_DARK))
            if bad_s or bad_n:
                return
            result["surg_in"] = sv
            result["ns_in"]   = nv
            if sv and nv and nv > 0:
                result["lsi"] = round(sv / nv * 100.0, 1)
            confirmed[0] = True
            win.destroy()

        def _skip(event=None):
            confirmed[0] = True
            win.destroy()

        surg_entry.bind("<Return>", lambda e: ns_entry.focus_set())
        ns_entry.bind("<Return>", _submit)
        win.bind("<Escape>", _skip)
        surg_entry.focus_set()

        btn_frame = tk.Frame(win, bg=BG)
        btn_frame.pack(pady=12)
        tk.Button(btn_frame, text="Confirm", command=_submit,
                  bg=ORANGE, fg=FG, font=("Helvetica", 10, "bold"),
                  relief="flat", cursor="hand2", padx=16).pack(side="left", padx=8)
        tk.Button(btn_frame, text="Skip", command=_skip,
                  bg=BG_DARK, fg=FG_SUB, font=("Helvetica", 10),
                  relief="flat", cursor="hand2", padx=16).pack(side="left", padx=8)

        root.wait_window(win)
        root.destroy()
        return result

    except Exception as exc:
        print(f"[main] WARNING: Could not show triple-hop dialog ({exc}). Skipping.")
        return result


def _months_since(date_str: str, reference: datetime.date = None) -> float:
    """
    Parse surgery date: YYYY-MM-DD, MM/DD/YY(Y), or MM/YY(Y).
    Full slash dates before month-only so 01/02/25 is Jan 2 not Jan 2025.
    Returns None if empty or unparseable.
    """
    if not date_str:
        return None
    s = date_str.strip()
    norm = _normalize_surgery_date_slash(s)
    candidates = [norm, s] if norm != s else [s]
    seen, to_try = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            to_try.append(c)
    ref = reference or datetime.date.today()
    for cand in to_try:
        for fmt in ("%Y-%m-%d", "%m/%d/%y", "%m/%d/%Y", "%m/%y", "%m/%Y"):
            try:
                surgery = datetime.datetime.strptime(cand, fmt).date()
                delta = ref - surgery
                return max(0.0, delta.days / 30.44)
            except ValueError:
                continue
    return None


sys.path.insert(0, str(ROOT))

from config import (
    EXPORT_FILES, PLATE_SIDE, SCALAR_SIGNAL_NAMES,
    ANALOG_RATE, KINEMATIC_RATE, JUMP_HEIGHT_SCALE,
    PATIENT_NAME, SURGICAL_SIDE, SURGERY_DATE, COFP_TO_MM, CLINIC_NAME,
)
from helpers.parser import parse_v3d_file, V3DExport
from helpers.report import build_report

import tests.drop_jump      as dj_mod
import tests.drop_landing   as dl_mod
import tests.max_vertical   as mv_mod
import tests.endurance_squat as es_mod
import tests.single_leg_jump as sl_mod
import tests.proprioception  as prop_mod


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load(folder: Path, key: str, rate: float, verbose: bool = False) -> V3DExport:
    """
    Load one export file; return empty export if file is missing.

    Tries the exact configured filename first, then falls back to a
    case-insensitive and whitespace-normalised search so that filenames
    like 'PCTL 1 COFP.txt' are found even when config says 'PCTL1 COFP.txt'.
    """
    filename = EXPORT_FILES.get(key, "")
    if not filename:
        return V3DExport()

    # 1. Exact match
    path = folder / filename
    if path.exists():
        exp = parse_v3d_file(str(path), rate=rate)
        if verbose:
            sigs = exp.available_signals()
            if sigs:
                print(f"  Loaded '{filename}' → signals: {sigs}")
            else:
                print(f"  Loaded '{filename}' but no signals were parsed!")
        return exp

    # 2. Case-insensitive + whitespace-normalised fallback
    try:
        target_norm = re.sub(r"\s+", "", filename).lower()
        for f in sorted(folder.iterdir()):
            if not f.is_file() or not f.suffix.lower() == ".txt":
                continue
            cand_norm = re.sub(r"\s+", "", f.name).lower()
            if cand_norm == target_norm:
                print(f"  [load] Fuzzy match: '{filename}' → '{f.name}'")
                exp = parse_v3d_file(str(f), rate=rate)
                if verbose:
                    sigs = exp.available_signals()
                    print(f"         Signals: {sigs}")
                return exp
    except Exception:
        pass

    if verbose:
        print(f"  [load] NOT FOUND: '{filename}' (also searched for fuzzy match)")
    return V3DExport(filepath=str(path), rate=rate)


def _surg_plate(surg_side: str) -> str:
    """Return 'FP1' or 'FP2' for the surgical side."""
    for plate, side in PLATE_SIDE.items():
        if side and side.upper() == surg_side.upper():
            return plate
    return "FP1"


def _ns_plate(surg_side: str) -> str:
    """Return the force plate for the non-surgical side."""
    sp  = _surg_plate(surg_side)
    for plate, side in PLATE_SIDE.items():
        if side is not None and plate != sp:
            return plate
    return "FP2"


def _bilateral_forces(forces: V3DExport, surg_side: str):
    """
    Return (fz_surg, fz_ns) from a bilateral Forces.txt file.
    Uses FP1/FP2 Z-components, mapped by PLATE_SIDE in config.py.
    """
    sp  = _surg_plate(surg_side)
    nsp = _ns_plate(surg_side)
    fz_s  = forces.timeseries(sp,  "Z")
    fz_ns = forces.timeseries(nsp, "Z")
    return fz_s, fz_ns


def _get_knee(joints: V3DExport, side: str) -> dict:
    return joints.get_knee(side) if joints else {}


def _get_hip(joints: V3DExport, side: str) -> dict:
    return joints.get_hip(side) if joints else {}


def _get_ankle(joints: V3DExport, side: str) -> dict:
    return joints.get_ankle(side) if joints else {}


def _scalars(data_export: V3DExport) -> dict:
    """Flat dict of all scalar values from a Data.txt file."""
    return data_export.all_scalars() if data_export else {}


def _ns_side(surg_side: str) -> str:
    return "L" if surg_side.upper() == "R" else "R"


def _cofp_mm(export: V3DExport, plate: str, component: str) -> np.ndarray:
    """
    Get COP time series in mm.
    COFP files export in METRES — multiply by COFP_TO_MM to get mm.
    """
    arr = export.timeseries(plate, component)
    if arr is None:
        return None
    return arr * COFP_TO_MM


def _peak_knee_moment(moments_export: V3DExport, side: str) -> float:
    """
    Extract peak knee extension moment (Nm) for the given side from a Moments.txt export.
    V3D knee moment X component: positive = extension moment (quadriceps demand).
    Returns NaN if data unavailable.
    """
    signal = f"{side.upper()}_Knee_Moment"
    ts = moments_export.timeseries(signal, "X")
    if ts is None or len(ts) == 0:
        return float("nan")
    return float(np.nanmax(ts))


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run(session_folder: str,
        patient_name: str = None,
        surg_side: str = None):
    """
    Run the full RTS pipeline for one patient session.

    session_folder : path containing all V3D export .txt files
    patient_name   : overrides config.PATIENT_NAME if provided
    surg_side      : overrides config.SURGICAL_SIDE if provided ('L' or 'R')
    """
    folder = Path(session_folder)
    if not folder.exists():
        print(f"[main] ERROR: Session folder not found: {folder}")
        sys.exit(1)

    # ── Load general data (body weight) ──────────────────────────────────────
    general = _load(folder, "general_data", rate=ANALOG_RATE)
    bw_n    = general.scalar(SCALAR_SIGNAL_NAMES["patient_mass_n"])
    if np.isnan(bw_n) or bw_n <= 0:
        print("[main] WARNING: MASS_N not found — defaulting to 750 N.")
        bw_n = 750.0

    # ── Collect session info from ALL export files (majority vote) ────────────
    # Scans every .txt file for embedded C3D paths → extracts name, date,
    # and the patient's original data folder (where we save the PDF).
    if patient_name and patient_name.strip():
        # CLI override: treat as "Last, First" → convert to "First Last"
        raw = patient_name.strip()
        if "," in raw:
            last, first = raw.split(",", 1)
            display_name = f"{first.strip()} {last.strip()}"
            raw_name     = raw
        else:
            display_name = raw
            raw_name     = raw
        session_info = {}
    else:
        session_info = _collect_session_info(folder)
        display_name = session_info.get("display_name", "")
        raw_name     = session_info.get("raw_name", display_name)
        if not display_name:
            raw_name     = _ask_patient_name()
            display_name = raw_name

    # ── Surgical side ─────────────────────────────────────────────────────────
    if surg_side and surg_side.upper() in ("L", "R"):
        side = surg_side.upper()
    elif SURGICAL_SIDE and SURGICAL_SIDE.strip().upper() in ("L", "R"):
        side = SURGICAL_SIDE.strip().upper()
    else:
        side = _ask_surgical_side(display_name)

    ns = _ns_side(side)

    # ── Surgery date & months since surgery ───────────────────────────────────
    # Priority: 1) config.SURGERY_DATE  2) dialog prompt  3) None (unknown)
    if SURGERY_DATE and SURGERY_DATE.strip():
        surgery_date_str = SURGERY_DATE.strip()
    else:
        surgery_date_str = _ask_surgery_date(display_name)

    months_post_op = _months_since(surgery_date_str)
    if months_post_op is not None:
        print(f"  Surgery date : {surgery_date_str}  "
              f"({months_post_op:.1f} months post-op)")
    else:
        print("  Surgery date : unknown (skipped — time modifier not applied)")

    # ── Triple Hop for Distance ────────────────────────────────────────────────
    triple_hop = _ask_triple_hop(display_name)
    if triple_hop.get("lsi") is not None:
        print(f"  Triple Hop   : Surg={triple_hop['surg_in']:.1f}\" "
              f"NS={triple_hop['ns_in']:.1f}\"  LSI={triple_hop['lsi']:.1f}%")
    else:
        print("  Triple Hop   : skipped")

    # ── Session date (from embedded C3D paths, fall back to today) ────────────
    # date_str format: MM-DD-YY  e.g. "12-03-25"
    date_str_short = session_info.get("session_date") or \
                     datetime.date.today().strftime("%m-%d-%y")
    # Human-readable date for the report header
    try:
        dt = datetime.datetime.strptime(date_str_short, "%m-%d-%y")
        test_date_long = dt.strftime("%B %d, %Y")
    except ValueError:
        test_date_long = datetime.date.today().strftime("%B %d, %Y")

    # ── Output folder: patient's original data folder if it exists ────────────
    raw_save_dir = session_info.get("save_dir")
    if raw_save_dir and Path(raw_save_dir).exists():
        out_folder = Path(raw_save_dir)
    else:
        out_folder = folder
        if raw_save_dir:
            print(f"[main] NOTE: Patient folder not accessible on this machine "
                  f"({raw_save_dir}) — saving to session folder instead.")

    print(f"\n{'='*62}")
    print(f"  RTS Reporting Pipeline")
    print(f"  Patient  : {display_name}")
    print(f"  Session  : {folder.name}")
    print(f"  Date     : {test_date_long}")
    print(f"  Surg side: {side}")
    print(f"  Body wt  : {bw_n:.1f} N  ({bw_n/9.81:.1f} kg)")
    print(f"  Saving to: {out_folder}")
    print(f"{'='*62}\n")

    patient_data = {
        "name":               display_name,
        "dob":                "—",
        "surgical_side":      side,
        "test_date":          test_date_long,
        "clinician":          "",
        "bw_kg":              bw_n / 9.81,
        "rate_grf":           ANALOG_RATE,
        "rate_kin":           KINEMATIC_RATE,
        "session_dir":        str(folder),
        "surgery_date":       surgery_date_str or "",
        "months_since_surgery": months_post_op,   # float or None
        "triple_hop":         triple_hop,          # {surg_in, ns_in, lsi} or all None
    }

    test_results = {}
    all_signals  = {}   # raw arrays keyed by test name, for figure generation

    # ── Drop Jump ─────────────────────────────────────────────────────────────
    print("[main] -- Drop Jump ------------------------------")
    dj_data   = _load(folder, "drop_jump_data",   rate=ANALOG_RATE)
    dj_forces = _load(folder, "drop_jump_forces",  rate=ANALOG_RATE)
    dj_joints = _load(folder, "drop_jump_joints",  rate=KINEMATIC_RATE)
    sc        = _scalars(dj_data)

    fz_s, fz_ns = _bilateral_forces(dj_forces, side)
    if fz_s is not None and fz_ns is not None:
        knee_surg_dj = _get_knee(dj_joints, side)
        knee_ns_dj   = _get_knee(dj_joints, ns)
        hip_surg_dj  = _get_hip(dj_joints, side)
        hip_ns_dj    = _get_hip(dj_joints, ns)

        res = dj_mod.analyse(
            fz_surg=fz_s, fz_ns=fz_ns,
            rate_f=ANALOG_RATE, bw_n=bw_n, surg_side=side,
            knee_surg=knee_surg_dj,
            knee_ns=knee_ns_dj,
            hip_surg=hip_surg_dj,
            hip_ns=hip_ns_dj,
            rate_k=KINEMATIC_RATE, scalars=sc,
        )
        # Prefer V3D-computed scalars over time-series estimates
        if not np.isnan(sc.get("RSI", np.nan)):
            res.rsi = float(sc["RSI"])
        if not np.isnan(sc.get("JH_IN", np.nan)):
            res.jump_height_cm = float(sc["JH_IN"]) * JUMP_HEIGHT_SCALE
        if not np.isnan(sc.get("CT", np.nan)):
            res.contact_time_s = float(sc["CT"])

        test_results["drop_jump"] = res
        dj_cofp    = _load(folder, "drop_jump_cofp",    rate=ANALOG_RATE)
        dj_moments = _load(folder, "drop_jump_moments", rate=KINEMATIC_RATE)
        ankle_surg_dj  = _get_ankle(dj_joints, side)
        ankle_ns_dj    = _get_ankle(dj_joints, ns)
        all_signals["drop_jump"] = {
            "fz_surg":           fz_s,
            "fz_ns":             fz_ns,
            "knee_flex_surg":    knee_surg_dj.get("flex_ext"),
            "knee_flex_ns":      knee_ns_dj.get("flex_ext"),
            "knee_valgus_surg":  knee_surg_dj.get("valgus"),
            "knee_valgus_ns":    knee_ns_dj.get("valgus"),
            "tib_rot_surg":      knee_surg_dj.get("tib_rot"),
            "tib_rot_ns":        knee_ns_dj.get("tib_rot"),
            "ankle_flex_surg":   ankle_surg_dj.get("flex_ext"),
            "ankle_flex_ns":     ankle_ns_dj.get("flex_ext"),
            "hip_add_surg":      hip_surg_dj.get("ab_adduction"),
            "hip_add_ns":        hip_ns_dj.get("ab_adduction"),
            "hip_ir_surg":       hip_surg_dj.get("int_ext_rot"),
            "hip_ir_ns":         hip_ns_dj.get("int_ext_rot"),
            "cop_y_surg":        _cofp_mm(dj_cofp, _surg_plate(side), "Y"),
            "cop_y_ns":          _cofp_mm(dj_cofp, _ns_plate(side),   "Y"),
            "knee_moment_surg":  _peak_knee_moment(dj_moments, side),
            "knee_moment_ns":    _peak_knee_moment(dj_moments, ns),
        }
        print(f"  RSI           = {res.rsi:.3f}")
        print(f"  Jump height   = {res.jump_height_cm:.2f} (V3D units)")
        print(f"  Contact time  = {res.contact_time_s:.3f} s")
        print(f"  Landing LSI   = {res.landing_lsi_200ms:.1f}%")
        print(f"  Load rate LSI = {res.loading_rate_lsi:.1f}%")
    else:
        print("  [SKIPPED] Forces file missing or empty.")
    print()

    # ── Drop Landing ──────────────────────────────────────────────────────────
    print("[main] -- Drop Landing ---------------------------")
    dl_data   = _load(folder, "drop_landing_data",   rate=ANALOG_RATE)
    dl_forces = _load(folder, "drop_landing_forces",  rate=ANALOG_RATE)
    dl_joints = _load(folder, "drop_landing_joints",  rate=KINEMATIC_RATE)
    sc        = _scalars(dl_data)

    fz_s, fz_ns = _bilateral_forces(dl_forces, side)
    if fz_s is not None and fz_ns is not None:
        knee_surg_dl = _get_knee(dl_joints, side)
        knee_ns_dl   = _get_knee(dl_joints, ns)
        hip_surg_dl  = _get_hip(dl_joints, side)
        hip_ns_dl    = _get_hip(dl_joints, ns)

        res = dl_mod.analyse(
            fz_surg=fz_s, fz_ns=fz_ns,
            rate_f=ANALOG_RATE, bw_n=bw_n, surg_side=side,
            knee_surg=knee_surg_dl,
            knee_ns=knee_ns_dl,
            hip_surg=hip_surg_dl,
            hip_ns=hip_ns_dl,
            rate_k=KINEMATIC_RATE, scalars=sc,
        )
        test_results["drop_landing"] = res
        dl_cofp    = _load(folder, "drop_landing_cofp",    rate=ANALOG_RATE)
        dl_moments = _load(folder, "drop_landing_moments", rate=KINEMATIC_RATE)
        ankle_surg_dl  = _get_ankle(dl_joints, side)
        ankle_ns_dl    = _get_ankle(dl_joints, ns)
        all_signals["drop_landing"] = {
            "fz_surg":           fz_s,
            "fz_ns":             fz_ns,
            "knee_flex_surg":    knee_surg_dl.get("flex_ext"),
            "knee_flex_ns":      knee_ns_dl.get("flex_ext"),
            "knee_valgus_surg":  knee_surg_dl.get("valgus"),
            "knee_valgus_ns":    knee_ns_dl.get("valgus"),
            "tib_rot_surg":      knee_surg_dl.get("tib_rot"),
            "tib_rot_ns":        knee_ns_dl.get("tib_rot"),
            "ankle_flex_surg":   ankle_surg_dl.get("flex_ext"),
            "ankle_flex_ns":     ankle_ns_dl.get("flex_ext"),
            "hip_add_surg":      hip_surg_dl.get("ab_adduction"),
            "hip_add_ns":        hip_ns_dl.get("ab_adduction"),
            "hip_ir_surg":       hip_surg_dl.get("int_ext_rot"),
            "hip_ir_ns":         hip_ns_dl.get("int_ext_rot"),
            "cop_y_surg":        _cofp_mm(dl_cofp, _surg_plate(side), "Y"),
            "cop_y_ns":          _cofp_mm(dl_cofp, _ns_plate(side),   "Y"),
            "knee_moment_surg":  _peak_knee_moment(dl_moments, side),
            "knee_moment_ns":    _peak_knee_moment(dl_moments, ns),
        }
        print(f"  Impact LSI    = {res.impact_transient_lsi:.1f}%")
        print(f"  Load rate LSI = {res.loading_rate_lsi:.1f}%")
        print(f"  Peak force LSI= {res.peak_force_lsi:.1f}%")
    else:
        print("  [SKIPPED] Forces file missing or empty.")
    print()

    # ── Max Vertical Jump ─────────────────────────────────────────────────────
    print("[main] -- Max Vertical Jump ----------------------")
    mv_data   = _load(folder, "vertical_data",   rate=ANALOG_RATE)
    mv_forces = _load(folder, "vertical_forces",  rate=ANALOG_RATE)
    mv_joints = _load(folder, "vertical_joints",  rate=KINEMATIC_RATE)
    sc        = _scalars(mv_data)

    fz_s, fz_ns = _bilateral_forces(mv_forces, side)
    if fz_s is not None and fz_ns is not None:
        knee_surg_mv = _get_knee(mv_joints, side)
        knee_ns_mv   = _get_knee(mv_joints, ns)
        hip_surg_mv  = _get_hip(mv_joints, side)
        hip_ns_mv    = _get_hip(mv_joints, ns)

        res = mv_mod.analyse(
            fz_surg=fz_s, fz_ns=fz_ns,
            rate_f=ANALOG_RATE, bw_n=bw_n, surg_side=side,
            knee_surg=knee_surg_mv,
            knee_ns=knee_ns_mv,
            hip_surg=hip_surg_mv,
            hip_ns=hip_ns_mv,
            rate_k=KINEMATIC_RATE, scalars=sc,
        )
        # Apply V3D scalars
        if not np.isnan(sc.get("JH_IN", np.nan)):
            res.jump_height_cm = float(sc["JH_IN"]) * JUMP_HEIGHT_SCALE

        # Peak force LSI from "Force Left"/"Force Right" scalars
        fl = sc.get("Force Left",  np.nan)
        fr = sc.get("Force Right", np.nan)
        if not (np.isnan(fl) or np.isnan(fr)):
            pk_surg = fr if side == "R" else fl
            pk_ns   = fl if side == "R" else fr
            from helpers.grf import lsi as _lsi
            res.peak_force_surg_N = pk_surg
            res.peak_force_ns_N   = pk_ns
            res.peak_force_lsi    = _lsi(pk_surg, pk_ns)

        test_results["max_vertical"] = res
        mv_cofp    = _load(folder, "vertical_cofp",    rate=ANALOG_RATE)
        mv_moments = _load(folder, "vertical_moments", rate=KINEMATIC_RATE)
        ankle_surg_mv  = _get_ankle(mv_joints, side)
        ankle_ns_mv    = _get_ankle(mv_joints, ns)
        all_signals["max_vertical"] = {
            "fz_surg":           fz_s,
            "fz_ns":             fz_ns,
            "knee_flex_surg":    knee_surg_mv.get("flex_ext"),
            "knee_flex_ns":      knee_ns_mv.get("flex_ext"),
            "knee_valgus_surg":  knee_surg_mv.get("valgus"),
            "knee_valgus_ns":    knee_ns_mv.get("valgus"),
            "tib_rot_surg":      knee_surg_mv.get("tib_rot"),
            "tib_rot_ns":        knee_ns_mv.get("tib_rot"),
            "ankle_flex_surg":   ankle_surg_mv.get("flex_ext"),
            "ankle_flex_ns":     ankle_ns_mv.get("flex_ext"),
            "hip_add_surg":      hip_surg_mv.get("ab_adduction"),
            "hip_add_ns":        hip_ns_mv.get("ab_adduction"),
            "hip_ir_surg":       hip_surg_mv.get("int_ext_rot"),
            "hip_ir_ns":         hip_ns_mv.get("int_ext_rot"),
            "cop_y_surg":        _cofp_mm(mv_cofp, _surg_plate(side), "Y"),
            "cop_y_ns":          _cofp_mm(mv_cofp, _ns_plate(side),   "Y"),
            "knee_moment_surg":  _peak_knee_moment(mv_moments, side),
            "knee_moment_ns":    _peak_knee_moment(mv_moments, ns),
        }
        print(f"  Jump height   = {res.jump_height_cm:.2f} (V3D units)")
        print(f"  Propulsion LSI= {res.propulsion_lsi:.1f}%")
        print(f"  Peak force LSI= {res.peak_force_lsi:.1f}%")
    else:
        print("  [SKIPPED] Forces file missing or empty.")
    print()

    # ── Endurance Squat ───────────────────────────────────────────────────────
    print("[main] -- Endurance Squat ------------------------")
    es_forces = _load(folder, "endurance_forces", rate=ANALOG_RATE, verbose=True)
    es_joints = _load(folder, "endurance_joints", rate=KINEMATIC_RATE, verbose=True)

    fz_s, fz_ns = _bilateral_forces(es_forces, side)
    if fz_s is None or fz_ns is None:
        sp  = _surg_plate(side)
        nsp = _ns_plate(side)
        sigs = es_forces.available_signals()
        if sigs:
            print(f"  [DIAG] Forces file loaded but FP signals not found.")
            print(f"  [DIAG] Looking for '{sp}'/'{nsp}' (Z). File has: {sigs}")
        else:
            print(f"  [DIAG] Forces file not found or empty.")
            print(f"  [DIAG] Expected: '{EXPORT_FILES.get('endurance_forces')}'")
            txt_files = sorted(f.name for f in folder.glob("*ndur*") or folder.glob("*squat*"))
            if not txt_files:
                txt_files = sorted(f.name for f in folder.glob("*.txt") if "ndur" in f.name.lower() or "squat" in f.name.lower())
            if txt_files:
                print(f"  [DIAG] Files with 'endur'/'squat' in name: {txt_files}")
    if fz_s is not None and fz_ns is not None:
        k_surg = _get_knee(es_joints, side)
        k_ns   = _get_knee(es_joints, ns)
        res = es_mod.analyse(
            fz_surg=fz_s, fz_ns=fz_ns,
            rate_f=ANALOG_RATE, bw_n=bw_n, surg_side=side,
            knee_valgus_surg=k_surg.get("valgus"),
            knee_valgus_ns=k_ns.get("valgus"),
            rate_k=KINEMATIC_RATE,
        )
        test_results["endurance_squat"] = res
        es_cofp    = _load(folder, "endurance_cofp",    rate=ANALOG_RATE)
        es_moments = _load(folder, "endurance_moments", rate=KINEMATIC_RATE)
        all_signals["endurance_squat"] = {
            "fz_surg":          fz_s,
            "fz_ns":            fz_ns,
            "cop_y_surg":       _cofp_mm(es_cofp, _surg_plate(side), "Y"),
            "cop_y_ns":         _cofp_mm(es_cofp, _ns_plate(side),   "Y"),
            "knee_moment_surg": _peak_knee_moment(es_moments, side),
            "knee_moment_ns":   _peak_knee_moment(es_moments, ns),
        }
        print(f"  Reps detected = {res.n_cycles}")
        print(f"  Mean LSI      = {res.mean_lsi_peak:.1f}%")
        print(f"  Fatigue drift = {res.fatigue_drift_pct:.1f}%")
    else:
        print("  [SKIPPED] Forces file missing or empty.")
    print()

    # ── Single-Leg Vertical Jump ──────────────────────────────────────────────
    print("[main] -- Single-Leg Vertical Jump ---------------")
    sl_l_data   = _load(folder, "sl_left_data",    rate=ANALOG_RATE)
    sl_l_forces = _load(folder, "sl_left_forces",  rate=ANALOG_RATE)
    sl_l_joints = _load(folder, "sl_left_joints",  rate=KINEMATIC_RATE)
    sl_r_data   = _load(folder, "sl_right_data",   rate=ANALOG_RATE)
    sl_r_forces = _load(folder, "sl_right_forces", rate=ANALOG_RATE)
    sl_r_joints = _load(folder, "sl_right_joints", rate=KINEMATIC_RATE)

    fz_left  = sl_l_forces.timeseries("FP3", "Z")
    fz_right = sl_r_forces.timeseries("FP3", "Z")

    jh_l = sl_l_data.scalar(SCALAR_SIGNAL_NAMES["jump_height_l"])
    jh_r = sl_r_data.scalar(SCALAR_SIGNAL_NAMES["jump_height_r"])
    if not np.isnan(jh_l): jh_l *= JUMP_HEIGHT_SCALE
    if not np.isnan(jh_r): jh_r *= JUMP_HEIGHT_SCALE

    if fz_left is not None or fz_right is not None:
        if side == "L":
            fz_surg_sl, fz_ns_sl = fz_left,  fz_right
            jh_surg,    jh_ns    = jh_l,      jh_r
            knee_surg_sl = _get_knee(sl_l_joints, "L")
            knee_ns_sl   = _get_knee(sl_r_joints, "R")
        else:
            fz_surg_sl, fz_ns_sl = fz_right, fz_left
            jh_surg,    jh_ns    = jh_r,     jh_l
            knee_surg_sl = _get_knee(sl_r_joints, "R")
            knee_ns_sl   = _get_knee(sl_l_joints, "L")

        surg_limb = sl_mod.analyse_limb(
            fz=fz_surg_sl, rate_f=ANALOG_RATE, bw_n=bw_n,
            side="Surgical",     jump_height_scalar=jh_surg,
            knee_data=knee_surg_sl, rate_k=KINEMATIC_RATE,
        )
        ns_limb = sl_mod.analyse_limb(
            fz=fz_ns_sl,   rate_f=ANALOG_RATE, bw_n=bw_n,
            side="Non-Surgical", jump_height_scalar=jh_ns,
            knee_data=knee_ns_sl,   rate_k=KINEMATIC_RATE,
        )
        res = sl_mod.combine(surg_limb, ns_limb)
        test_results["single_leg_jump"] = res
        # Load per-side COFP and moments (side-specific files for single-leg tests)
        sl_surg_cofp_key = "sl_left_cofp"  if side == "L" else "sl_right_cofp"
        sl_ns_cofp_key   = "sl_right_cofp" if side == "L" else "sl_left_cofp"
        sl_surg_mom_key  = "sl_left_moments"  if side == "L" else "sl_right_moments"
        sl_ns_mom_key    = "sl_right_moments" if side == "L" else "sl_left_moments"
        sl_surg_cofp = _load(folder, sl_surg_cofp_key, rate=ANALOG_RATE)
        sl_ns_cofp   = _load(folder, sl_ns_cofp_key,   rate=ANALOG_RATE)
        sl_surg_mom  = _load(folder, sl_surg_mom_key,  rate=KINEMATIC_RATE)
        sl_ns_mom    = _load(folder, sl_ns_mom_key,    rate=KINEMATIC_RATE)
        # Per-side ankle (joints loaded from SLVL/SLVR files)
        sl_surg_joints = sl_l_joints if side == "L" else sl_r_joints
        sl_ns_joints   = sl_r_joints if side == "L" else sl_l_joints
        ankle_surg_sl  = _get_ankle(sl_surg_joints, side)
        ankle_ns_sl    = _get_ankle(sl_ns_joints,   ns)
        hip_surg_sl    = _get_hip(sl_surg_joints, side)
        hip_ns_sl      = _get_hip(sl_ns_joints,   ns)
        all_signals["single_leg_jump"] = {
            "fz_surg":           fz_surg_sl,
            "fz_ns":             fz_ns_sl,
            "knee_flex_surg":    knee_surg_sl.get("flex_ext"),
            "knee_flex_ns":      knee_ns_sl.get("flex_ext"),
            "knee_valgus_surg":  knee_surg_sl.get("valgus"),
            "knee_valgus_ns":    knee_ns_sl.get("valgus"),
            "tib_rot_surg":      knee_surg_sl.get("tib_rot"),
            "tib_rot_ns":        knee_ns_sl.get("tib_rot"),
            "ankle_flex_surg":   ankle_surg_sl.get("flex_ext"),
            "ankle_flex_ns":     ankle_ns_sl.get("flex_ext"),
            "hip_add_surg":      hip_surg_sl.get("ab_adduction"),
            "hip_add_ns":        hip_ns_sl.get("ab_adduction"),
            "hip_ir_surg":       hip_surg_sl.get("int_ext_rot"),
            "hip_ir_ns":         hip_ns_sl.get("int_ext_rot"),
            "cop_y_surg":        _cofp_mm(sl_surg_cofp, "FP3", "Y"),
            "cop_y_ns":          _cofp_mm(sl_ns_cofp,   "FP3", "Y"),
            "knee_moment_surg":  _peak_knee_moment(sl_surg_mom, side),
            "knee_moment_ns":    _peak_knee_moment(sl_ns_mom,   ns),
        }
        print(f"  Surg jump ht = {res.surgical.jump_height_cm:.2f} (V3D units)")
        print(f"  NS   jump ht = {res.non_surgical.jump_height_cm:.2f} (V3D units)")
        print(f"  Jump ht LSI  = {res.lsi_jump_height:.1f}%")
        print(f"  Peak force LSI={res.lsi_peak_force:.1f}%")
    else:
        print("  [SKIPPED] Forces files missing or empty.")
    print()

    # ── Proprioception ────────────────────────────────────────────────────────
    print("[main] -- Proprioception -------------------------")
    #  PCTL1 / PCTR1 = Standard (flat firm ground)
    #  PCTL2 / PCTR2 = Airex bag (unstable foam — harder, higher COP expected)
    pctl1 = _load(folder, "pctl1_cofp", rate=ANALOG_RATE, verbose=True)
    pctl2 = _load(folder, "pctl2_cofp", rate=ANALOG_RATE, verbose=True)
    pctr1 = _load(folder, "pctr1_cofp", rate=ANALOG_RATE, verbose=True)
    pctr2 = _load(folder, "pctr2_cofp", rate=ANALOG_RATE, verbose=True)

    def _cop(exp, comp):
        return _cofp_mm(exp, "FP3", comp)

    # Standard (trial 1 per side)
    if side == "L":
        std_xs, std_ys = _cop(pctl1, "X"), _cop(pctl1, "Y")
        std_xn, std_yn = _cop(pctr1, "X"), _cop(pctr1, "Y")
        airex_xs, airex_ys = _cop(pctl2, "X"), _cop(pctl2, "Y")
        airex_xn, airex_yn = _cop(pctr2, "X"), _cop(pctr2, "Y")
    else:
        std_xs, std_ys = _cop(pctr1, "X"), _cop(pctr1, "Y")
        std_xn, std_yn = _cop(pctl1, "X"), _cop(pctl1, "Y")
        airex_xs, airex_ys = _cop(pctr2, "X"), _cop(pctr2, "Y")
        airex_xn, airex_yn = _cop(pctl2, "X"), _cop(pctl2, "Y")

    from config import PROP_TRIM_SECONDS
    has_prop = any(a is not None for a in [std_xs, std_xn, airex_xs, airex_xn])
    if has_prop:
        try:
            res = prop_mod.analyse(
                cop_x_surg_std=std_xs,   cop_y_surg_std=std_ys,
                cop_x_ns_std=std_xn,     cop_y_ns_std=std_yn,
                cop_x_surg_airex=airex_xs, cop_y_surg_airex=airex_ys,
                cop_x_ns_airex=airex_xn,   cop_y_ns_airex=airex_yn,
                rate_f=ANALOG_RATE,
                surg_side=side,
                trim_seconds=PROP_TRIM_SECONDS,
            )
            test_results["proprioception"] = res

            # Pass both conditions' COP arrays to the report
            def _limb_sigs(cond):
                if cond is None:
                    return {"cop_x_surg": None, "cop_y_surg": None,
                            "cop_x_ns": None,   "cop_y_ns": None}
                s = cond.surgical
                n = cond.non_surgical
                return {
                    "cop_x_surg": s.cop_x if s else None,
                    "cop_y_surg": s.cop_y if s else None,
                    "cop_x_ns":   n.cop_x if n else None,
                    "cop_y_ns":   n.cop_y if n else None,
                }

            all_signals["proprioception"] = {
                "standard": _limb_sigs(res.standard),
                "airex":    _limb_sigs(res.airex),
            }

            if res.standard and res.standard.surgical:
                s = res.standard.surgical
                print(f"  [Std]  Surg COP vel  = {s.mean_velocity_mm_s:.1f} mm/s")
                print(f"  [Std]  Surg ellipse  = {s.ellipse_area_mm2:.0f} mm²")
                print(f"  [Std]  COP vel LSI   = {res.standard.lsi_cop_velocity:.1f}%")
            if res.airex and res.airex.surgical:
                s = res.airex.surgical
                print(f"  [Airex] Surg COP vel = {s.mean_velocity_mm_s:.1f} mm/s")
                print(f"  [Airex] Surg ellipse = {s.ellipse_area_mm2:.0f} mm²")
                print(f"  [Airex] COP vel LSI  = {res.airex.lsi_cop_velocity:.1f}%")
        except Exception as _prop_err:
            import traceback
            print(f"  [ERROR] Proprioception analysis failed: {_prop_err}")
            print(traceback.format_exc())
    else:
        print("  [SKIPPED] COFP files missing or empty.")
        # Diagnostic: show what PCTL/PCTR filenames the code expected
        for _key in ("pctl1_cofp", "pctl2_cofp", "pctr1_cofp", "pctr2_cofp"):
            _fn = EXPORT_FILES.get(_key, "")
            _fp = folder / _fn
            _exists = "✓ FOUND" if _fp.exists() else "✗ MISSING"
            print(f"  [DIAG]  {_key}: '{_fn}' → {_exists}")
        # Show any PCTL/PCTR files that ARE present
        _prop_files = sorted(
            f.name for f in folder.glob("*.txt")
            if any(x in f.name.upper() for x in ("PCTL", "PCTR", "PROP", "BALANCE"))
        )
        if _prop_files:
            print(f"  [DIAG]  PCTL/PCTR files found in folder: {_prop_files}")
        else:
            print(f"  [DIAG]  No PCTL/PCTR files found at all — check config.py EXPORT_FILES")
    print()

    # ── RTR composite score ───────────────────────────────────────────────────
    rtr_metrics = _collect_rtr_metrics(test_results)
    rtr_metrics["_months_since_surgery"] = months_post_op  # passed through to scorer
    if triple_hop.get("lsi") is not None:
        rtr_metrics["triple_hop_lsi"] = triple_hop["lsi"]

    # ── Generate PDF ──────────────────────────────────────────────────────────
    # Filename format: "RTS Report_Temperance Lyons_12-03-25.pdf"
    # (RTS Report first, First Last name, session date MM-DD-YY)
    safe_name   = re.sub(r'[\\/:*?"<>|]', "_", display_name).strip()
    output_path = str(out_folder / f"RTS Report_{safe_name}_{date_str_short}.pdf")
    print(f"[main] Generating PDF → {output_path}")
    build_report(
        output_path=output_path,
        patient_data=patient_data,
        test_results=test_results,
        all_signals=all_signals,
        rtr_metrics=rtr_metrics,
    )
    print(f"\n✓ Done!  Report saved to: {output_path}\n")
    return output_path


# ─── RTR composite metric collector ──────────────────────────────────────────

def _collect_rtr_metrics(results: dict) -> dict:
    """Map test results → RTR_WEIGHTS keys for compute_rtr_score()."""
    m = {}

    # ── Drop Jump ─────────────────────────────────────────────────────────────
    dj = results.get("drop_jump")
    if dj:
        if not _isnan(dj.rsi):
            m["drop_jump_rsi"] = dj.rsi
        if not _isnan(dj.landing_lsi_200ms):
            m["landing_lsi"] = dj.landing_lsi_200ms
        if dj.grf_overall and not _isnan(dj.grf_overall.lsi_rfd):
            m["rfd_lsi"] = dj.grf_overall.lsi_rfd
        if dj.grf_overall and not _isnan(dj.grf_overall.lsi_peak):
            m["peak_grf_lsi"] = dj.grf_overall.lsi_peak
        if dj.kinematics:
            kin = dj.kinematics
            if hasattr(kin, "valgus_surg") and not _isnan(kin.valgus_surg):
                m["knee_valgus_surg"] = abs(kin.valgus_surg)

    # ── Drop Landing ─────────────────────────────────────────────────────────
    dl = results.get("drop_landing")
    if dl:
        if not _isnan(dl.peak_force_lsi):
            m["dl_peak_grf_lsi"] = dl.peak_force_lsi
        if not _isnan(dl.loading_rate_lsi):
            m["dl_load_rate_lsi"] = dl.loading_rate_lsi

    # ── Single-Leg Vertical Jump ──────────────────────────────────────────────
    slj = results.get("single_leg_jump")
    if slj and not _isnan(slj.lsi_jump_height):
        m["sl_jump_lsi"] = slj.lsi_jump_height

    # ── Proprioception / Balance ──────────────────────────────────────────────
    prop = results.get("proprioception")
    if prop and prop.standard and not _isnan(prop.standard.lsi_cop_velocity):
        m["cop_velocity_lsi"] = prop.standard.lsi_cop_velocity

    # ── Endurance Squat ───────────────────────────────────────────────────────
    end = results.get("endurance")
    if end:
        if not _isnan(end.mean_lsi_peak):
            m["endurance_lsi"] = end.mean_lsi_peak
        if not _isnan(end.fatigue_drift_pct):
            m["fatigue_drift"] = end.fatigue_drift_pct

    return m


def _isnan(v) -> bool:
    """Return True if v is None or a float NaN."""
    if v is None:
        return True
    try:
        import math
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


if __name__ == "__main__":
    # CLI: python main.py "D:\path\to\session"
    session = sys.argv[1] if len(sys.argv) > 1 else SESSION_FOLDER
    run(session_folder=session)