"""
profiles/base.py
================
A ReportProfile bundles everything that changes between injury types:

  • menu label / report subtitle / output filename tag
  • clinical text (interpretations, figure captions, "About this test" boxes)
  • which metrics feed the RTS composite score, and how heavily
  • time-since-surgery curve and time notes
  • extra normative thresholds
  • extra manually-entered clinic tests (e.g. heel-raise reps for Achilles)

The data processing (parsing, GRF / COP / kinematics maths) is shared by all
profiles — only the interpretation layer changes.

Adding a new profile
--------------------
1. Copy profiles/achilles.py → profiles/<new>.py and edit the text / weights.
2. Register it in profiles/__init__.py (PROFILES list).
It will then appear automatically in the report-type picker in main.py.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class ManualTest:
    """A clinic test entered by hand (surgical vs non-surgical value)."""
    key:       str                 # internal key, e.g. "heel_raise"
    title:     str                 # dialog title, e.g. "Single-Leg Heel Raise"
    prompt:    str                 # dialog instructions
    label:     str                 # label in LSI chart / tables
    unit:      str = ""            # e.g. "reps"
    domain:    str = "Hop"         # radar-chart domain
    rtr_key:   Optional[str] = None  # RTR_WEIGHTS label it feeds (or None)
    integer:   bool = False        # parse as int


@dataclass
class ReportProfile:
    key:             str                       # "acl", "achilles", ...
    menu_label:      str                       # shown on the picker button
    menu_blurb:      str                       # one-line description under the button
    report_subtitle: str                       # right side of the page header
    file_tag:        str = ""                  # "" → "RTS Report_..."; "Achilles" → "RTS Achilles Report_..."

    # Scoring ---------------------------------------------------------------
    rtr_weights:     Dict[str, Tuple[str, float]] = field(default_factory=dict)
    extra_norms:     dict = field(default_factory=dict)       # norm_key -> Norm
    time_points:     List[Tuple[float, float]] = field(default_factory=list)
    time_note:       Optional[Callable[[float], Optional[str]]] = None
    influence_names: Dict[str, str] = field(default_factory=dict)  # extra S&D labels

    # Behaviour flags -------------------------------------------------------
    flag_knee_frontal: bool = True    # put knee valgus / tibial rotation deltas in Clinical Flags
    ankle_focus:       bool = False   # show ankle rows prominently + warn if ankle data missing

    # Extra manual-entry tests (beyond Triple Hop, which every profile gets)
    manual_tests:    List[ManualTest] = field(default_factory=list)

    # Layout ------------------------------------------------------------------
    section_order:   Optional[List[str]] = None   # test page order (keys: drop_jump, drop_landing,
                                                  # max_vertical, endurance_squat, single_leg_jump,
                                                  # proprioception); None = default order
    extra_pages:     List[str] = field(default_factory=list)   # e.g. ["calf_ankle"]
    domain_map:      Optional[Callable[[str, str], Optional[str]]] = None  # (LSI label, default domain) → radar domain
    chart_exclude:   List[str] = field(default_factory=list)   # substrings hidden from the summary LSI chart

    # Text ------------------------------------------------------------------
    # text[test_key][field] -> str.  test_key in:
    #   drop_jump, drop_landing, max_vertical, endurance_squat,
    #   single_leg_jump, balance, summary
    text: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def t(self, test: str, fld: str, default: str = "") -> str:
        return self.text.get(test, {}).get(fld, default)
