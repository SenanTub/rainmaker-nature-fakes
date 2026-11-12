"""The two cases. Criteria shared by both come from the paper's 71-signature catalogue
(Table S1a: detection 8-33 min, peak area 2-165 km2, lifetime up to 141 min,
motion 1.9-25 m/s) and its attribution protocol (moves within 30 deg of the wind)."""
from datetime import datetime, timezone

U = lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc)

SHARED = dict(min_enh_db=5.0, area_km2=(2.0, 170.0), max_minutes=145.0,
              speed_kmh=(5.0, 110.0), dir_tol_deg=30.0, window_min=(5.0, 60.0), wedge_km=45.0)

CASES = {
    "kenai": dict(radar_dir="radar_files", tilt=0.48, band_km=(95.0, 140.0),
                  flow_toward_deg=225.0,   # report: "northeasterly flow"
                  sites={"AK36": (9.8, -98.1), "AK52": (11.3, -99.5)},    # from Rainmaker's video
                  releases={"A": ("AK36", U("2026-08-23T06:38:18")), "B": ("AK36", U("2026-08-23T07:06:54")),
                            "C": ("AK36", U("2026-08-23T07:36:48")), "D": ("AK52", U("2026-08-23T08:19:46")),
                            "E": ("AK52", U("2026-08-23T08:43:25")), "F": ("AK52", U("2026-08-23T09:07:24")),
                            "G": ("AK52", U("2026-08-23T09:29:52"))}),
    "pdt025": dict(radar_dir="radar_files_PDT025", tilt=0.48, band_km=(22.0, 65.0),
                   flow_toward_deg=215.0,  # paper Fig. 6: plume moved SW, within 15 deg of the sonde wind
                   sites={"PDT1": (-16.1, -22.9)},   # ~28 km from KPDT (Table S1b first-detection range), on the plume line
                   releases={"SS1": ("PDT1", U("2026-01-23T22:16:00")), "SS2": ("PDT1", U("2026-01-23T23:32:00")),
                             "SS3": ("PDT1", U("2026-01-24T00:28:00")), "SS4": ("PDT1", U("2026-01-24T01:36:00"))}),
}
