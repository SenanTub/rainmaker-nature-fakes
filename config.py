"""Settings shared by every script. Edit here, not inside the other scripts."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT
RADAR_DIR = ROOT / "radar_files"      # raw PAHG files land here
RESULTS_DIR = Path(os.environ.get("RM_RESULTS", ROOT / "results"))
RESULTS_DIR.mkdir(exist_ok=True)

# Radar and time window (UTC)
SITE = "PAHG"
START_UTC = "2026-08-23T06:00:00"
END_UTC = "2026-08-23T11:00:00"

# Radar tilts to analyse (degrees). 0.48 is the lowest tilt, closest to
# Rainmaker's "lowest unblocked gate" QPE; 0.88 matches their video outlines.
TILTS_DEG = [0.48, 0.88]
TILT_TOL_DEG = 0.2

# Rainmaker's 27-member Z = a * S^b ensemble (Z in mm^6/m^3, S in mm/h)
A_VALUES = list(range(100, 501, 50))   # 100, 150, ..., 500
B_VALUES = [2.0, 2.1, 2.2]
OFFSET_DB = 5.0                        # their background adjustment
B_REFERENCE = 2.1                      # used for headline ratios

# Box handling
FLANK_GAP_KM = 1.0          # gap between the streak box and each next-door box
INFLATE_KM = [0, 1, 2, 4]   # outline-size test: grow box by this on every side
PRE_SCANS = 2               # scans before first detection used for "before" background
MIN_VALID_GATES = 20        # fewer valid gates than this -> flagged
MIN_FLANK_VALID_FRAC = 0.5  # next-door boxes mostly black (no data) -> flagged

# Fallback drift (km/h, east, north) if no signature has two keyframes.
# Measured from signature D: (+4, -106.75) at 08:59 -> (-2, -123.25) at 10:02.
DRIFT_KMH_FALLBACK = (-5.70, -15.68)

ACRE_FOOT_M3 = 1233.48
