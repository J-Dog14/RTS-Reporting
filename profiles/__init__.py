"""
profiles
========
Injury-specific report profiles. main.py asks which one to use, then passes
it to build_report(). See profiles/base.py for how to add a new one.
"""

from profiles.base import ReportProfile, ManualTest
from profiles.acl import PROFILE as ACL
from profiles.achilles import PROFILE as ACHILLES

# Order here = order of buttons in the picker
PROFILES = [ACL, ACHILLES]
PROFILES_BY_KEY = {p.key: p for p in PROFILES}
DEFAULT_PROFILE = "acl"


def get_profile(key: str = None) -> ReportProfile:
    """Return the profile for `key` (case-insensitive); default ACL."""
    if not key:
        return PROFILES_BY_KEY[DEFAULT_PROFILE]
    k = key.strip().lower()
    if k not in PROFILES_BY_KEY:
        raise KeyError(f"Unknown report profile '{key}'. "
                       f"Options: {', '.join(PROFILES_BY_KEY)}")
    return PROFILES_BY_KEY[k]
