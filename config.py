"""
config.py
=========
All user-configurable settings for the RTS Reporting pipeline.
Edit this file to match your lab's hardware, naming conventions, and branding.

Sections
--------
1. Clinic / Branding
2. Hardware Setup
3. Force Plate Assignment
4. Joint Signal Names
5. File Naming Conventions
6. Analysis Thresholds
7. Report Styling
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. CLINIC / BRANDING
# ─────────────────────────────────────────────────────────────────────────────
CLINIC_NAME     = "BYoung Physical Therapy"
CLINIC_SUBTITLE = "Return to Sport Assessment"
CLINIC_LOGO     = None   # Absolute path to a .png logo, e.g. r"C:\Clinic\logo.png"

# ─────────────────────────────────────────────────────────────────────────────
# PATIENT / SESSION (edit before each run if not using command-line args)
# ─────────────────────────────────────────────────────────────────────────────
PATIENT_NAME   = ""               # Leave blank — name is read from session file paths
                                  # Set only as a last resort override, e.g. "Smith, John"
SURGICAL_SIDE  = ""               # "L", "R", or "" to prompt via popup each run
                                  # (V3D does not export this; set it here or leave
                                  #  blank to get a selection dialog every session)
SURGERY_DATE   = ""               # ISO "YYYY-MM-DD", or US "MM/YY", "MM/YYYY",
                                  # "MM/DD/YY", "MM/DD/YYYY", or "" to prompt each run.
                                  # Used for months-since-surgery (RTR time modifier).


# ─────────────────────────────────────────────────────────────────────────────
# 2. HARDWARE SETUP
# ─────────────────────────────────────────────────────────────────────────────

# AMTI force plate analog sample rate (Hz) — used for Forces and COFP files
ANALOG_RATE = 1000

# Kinematic / marker data rate from QTM/V3D (Hz) — used for Joints and Moments
KINEMATIC_RATE = 200


# ─────────────────────────────────────────────────────────────────────────────
# 3. FORCE PLATE ASSIGNMENT
#    FP1/FP2 are used for bilateral tests (Drop Jump, Drop Landing, Max VJ,
#    Endurance Squat). FP3 is used for single-leg tests and proprioception.
#
#    Set the physical side of each plate to match your lab layout.
#    "R" = right side, "L" = left side.
# ─────────────────────────────────────────────────────────────────────────────
PLATE_SIDE = {
    "FP1": "R",   # FP1 is the RIGHT force plate
    "FP2": "L",   # FP2 is the LEFT force plate
    "FP3": None,  # FP3 is side-agnostic (single-leg — side set by test)
}

# Minimum force (N) to be considered "in contact" (raise if noise triggers)
GRF_CONTACT_THRESHOLD = 20.0

# Minimum number of frames to count a contact period (filters edge noise)
MIN_CONTACT_FRAMES = 20


# ─────────────────────────────────────────────────────────────────────────────
# 4. JOINT SIGNAL NAMES
#    These are the EXACT signal names as they appear in the V3D Joints.txt
#    and Moments.txt export files (row 2 of the file, after the path row).
#
#    V3D exports these components per signal:
#      L_Knee_Angle / R_Knee_Angle  → X (flex/ext), Y (valgus), Z (tib rot)
#      L_Hip_Angle  / R_Hip_Angle   → Y only (ab/adduction)
#      L_Ankle_Angle/ R_Ankle_Angle → X only (dorsi/plantarflexion)
# ─────────────────────────────────────────────────────────────────────────────
JOINT_SIGNAL_NAMES = {
    "L_knee":  "L_Knee_Angle",
    "R_knee":  "R_Knee_Angle",
    "L_hip":   "L_Hip_Angle",
    "R_hip":   "R_Hip_Angle",
    "L_ankle": "L_Ankle_Angle",
    "R_ankle": "R_Ankle_Angle",
}

# Scalar metrics exported by V3D in Data.txt files
# Keys are internal labels used in the code; values are V3D signal names
SCALAR_SIGNAL_NAMES = {
    # General / Demographics (from General Data.txt)
    "patient_mass_n":   "MASS_N",           # Body weight in Newtons

    # Bilateral test scalars (Drop Jump / Vertical / Drop Landing Data.txt)
    "rsi":              "RSI",              # Reactive Strength Index (V3D computed)
    "jump_height":      "JH_IN",           # Jump height, bilateral (V3D computed)
    "jump_time":        "JT",              # Flight time in Vertical Data.txt
    "contact_time":     "CT",              # Contact time (V3D computed)
    "force_left":       "Force Left",      # Peak force, left side (N)
    "force_right":      "Force Right",     # Peak force, right side (N)
    "force_asi_ecc":    "Force_ASI_Eccentric",
    "force_asi_con":    "Force_ASI_Concentric",
    "lkx":              "L_KNEE_X",        # Left knee flexion angle (°)
    "rkx":              "R_KNEE_X",        # Right knee flexion angle (°)
    "lhx":              "L_HIP_X",         # Left hip angle (°)
    "rhx":              "R_HIP_X",         # Right hip angle (°)

    # Single-Leg Jump (from Single Leg Vertical Left/Right Data.txt)
    "jump_height_l":    "JH_IN_L",        # Jump height, left leg (V3D units)
    "jump_height_r":    "JH_IN_R",        # Jump height, right leg (V3D units)
    "pelvis_depth_in":  "Pelvis_Depth_in", # Pelvis drop depth (inches)
}

# COFP files export COP in METRES (global plate coordinates).
# Multiply by this factor to convert to millimetres before COP analysis.
COFP_TO_MM = 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. FILE NAMING CONVENTIONS
#    Exact filenames as V3D creates them in the export folder.
#    Keys are internal labels; values are the filename strings.
#    Edit the right-hand side to match your actual exported filenames.
# ─────────────────────────────────────────────────────────────────────────────
EXPORT_FILES = {
    # Patient demographics
    "general_data":          "DJ General Data.txt",

    # Drop Jump
    "drop_jump_data":        "Drop Jump Data.txt",
    "drop_jump_forces":      "Drop Jump Forces.txt",
    "drop_jump_cofp":        "Drop Jump COFP.txt",
    "drop_jump_joints":      "Drop Jump Joints.txt",
    "drop_jump_moments":     "Drop Jump Moments.txt",

    # Drop Landing
    "drop_landing_data":     "Drop Landing Data.txt",
    "drop_landing_forces":   "Drop Landing Forces.txt",
    "drop_landing_cofp":     "Drop Landing COFP.txt",
    "drop_landing_joints":   "Drop Landing Joints.txt",
    "drop_landing_moments":  "Drop Landing Moments.txt",

    # Max Vertical Jump (Bilateral CMJ)
    "vertical_data":         "Vertical Data.txt",
    "vertical_forces":       "Vertical Forces.txt",
    "vertical_cofp":         "Vertical COFP.txt",
    "vertical_joints":       "Vertical Joints.txt",
    "vertical_moments":      "Vertical Moments.txt",

    # 30-Second Endurance Squat
    # NOTE: V3D exports the scalar file with "Squat" in the name but
    #       the time-series files omit it — match actual filenames exactly.
    "endurance_data":        "Endurance Squat Data.txt",
    "endurance_forces":      "Endurance Forces.txt",
    "endurance_cofp":        "Endurance COFP.txt",
    "endurance_joints":      "Endurance Joints.txt",

    # Single-Leg Vertical Jump — Left
    # Note: Data.txt uses full name; Forces/COFP/Joints use abbreviated "SLVL"
    "sl_left_data":          "Single Leg Vertical Left Data.txt",
    "sl_left_forces":        "SLVL Forces.txt",
    "sl_left_cofp":          "SLVL COFP.txt",
    "sl_left_joints":        "SLVL Joints.txt",

    # Single-Leg Vertical Jump — Right
    "sl_right_data":         "Single Leg Vertical Right Data.txt",
    "sl_right_forces":       "SLVR Forces.txt",
    "sl_right_cofp":         "SLVR COFP.txt",
    "sl_right_joints":       "SLVR Joints.txt",

    # Proprioception — Control Left trials
    "pctl1_data":            "PCTL1 Data.txt",
    "pctl1_cofp":            "PCTL1 COFP.txt",
    "pctl1_forces":          "PCTL1 Forces.txt",
    "pctl2_data":            "PCTL2 Data.txt",
    "pctl2_cofp":            "PCTL2 COFP.txt",
    "pctl2_forces":          "PCTL2 Forces.txt",

    # Proprioception — Control Right trials
    "pctr1_data":            "PCTR1 Data.txt",
    "pctr1_cofp":            "PCTR1 COFP.txt",
    "pctr1_forces":          "PCTR1 Forces.txt",
    "pctr2_data":            "PCTR2 Data.txt",
    "pctr2_cofp":            "PCTR2 COFP.txt",
    "pctr2_forces":          "PCTR2 Forces.txt",
}


# ─────────────────────────────────────────────────────────────────────────────
# 6. ANALYSIS THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

# Limb Symmetry Index traffic-light bands (%)
LSI_GREEN  = 90    # ≥ this value = green (pass)
LSI_YELLOW = 75    # ≥ this value = yellow (caution), below = red (fail)

# Rate of Force Development computation window (ms from first contact)
RFD_WINDOW_MS = 100

# Loading Rate computation window (ms from first contact)
LOADING_RATE_WINDOW_MS = 50

# Proprioception: trim this many seconds from each end of balance trial
PROP_TRIM_SECONDS = 2.0

# Jump height unit scaling — set to 1.0 if V3D exports in cm,
# or 100.0 if V3D exports in metres and you want cm in the report
JUMP_HEIGHT_SCALE = 1.0   # adjust if JH_IN_L / JH_IN_R are in metres


# ─────────────────────────────────────────────────────────────────────────────
# 7. REPORT STYLING
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    # ── Data / clinical colors — DO NOT change (green/yellow/red traffic light) ──
    "green":     "#2ecc71",
    "yellow":    "#f39c12",
    "red":       "#e74c3c",
    "surgical":  "#E07B54",   # warm orange — surgical limb traces (unchanged)
    "non_surg":  "#4A90D9",   # cool blue — non-surgical limb traces (unchanged)

    # ── Branding / page design ────────────────────────────────────────────────
    "header_bg": "#39414a",   # charcoal — page headers, section bars
    "header_fg": "#FFFFFF",
    "accent":    "#fc6c0f",   # brand orange — used sparingly (badges, highlights)
    "grid":      "#E8E8E8",
    "text":      "#2c2c2c",
    "lt_grey":   "#F5F5F5",
}

PAGE_SIZE = "letter"   # "letter" or "A4"
