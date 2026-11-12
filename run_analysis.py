"""Run the whole check on the downloaded radar files.

Run:  python run_analysis.py
Writes results/per_scan.csv, results/water_by_scan.csv, results/summary.csv
and prints the headline table.
"""
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
from boxes import Signatures
from measure import background_mask, gate_stats, water_volumes, mean_dbz

NAME_RE = re.compile(r"(\d{8})_(\d{6})")


def file_time(path):
    m = NAME_RE.search(path.name)
    return datetime.strptime(m[1] + m[2], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def scan_durations_hours(times):
    """Each scan stands for the time from halfway-to-previous to halfway-to-next."""
    t = np.array([x.timestamp() for x in times], dtype=float)
    if len(t) == 1:
        return np.array([7 / 60])
    mids = (t[1:] + t[:-1]) / 2
    edges = np.concatenate([[t[0] - (mids[0] - t[0])], mids, [t[-1] + (t[-1] - mids[-1])]])
    return np.diff(edges) / 3600.0


def analyse_sweep(sweep, tilt, sigs, dt_h, stats_rows, water_rows):
    for sig in sigs.names:
        box = sigs.box_at(sig, sweep.time)
        if box is None:
            continue
        # other streaks and exclusion zones: never background, never part of this streak
        others = sigs.keep_out_mask(sweep.x_km, sweep.y_km, sweep.time, sig)
        stats, _ = gate_stats(sweep, box, sigs.flank_boxes(box), others)
        key = dict(signature=sig, tilt=tilt, time_utc=sweep.time.isoformat(timespec="seconds"))
        stats_rows.append({**key, "tilt_actual": sweep.tilt_deg,
                           "x0": box.x0, "x1": box.x1, "y0": box.y0, "y1": box.y1, **stats})
        x, y = sweep.x_km, sweep.y_km
        m_bg = background_mask(x, y, sigs.flank_boxes(box), others)
        # second definition of "next door": a ring 1-3 km around the outline
        m_ring = box.inflate(3).mask(x, y) & ~box.inflate(1).mask(x, y) & ~others
        stats_rows[-1]["dbz_ring"] = mean_dbz(sweep.dbz, m_ring, sweep.area_m2)
        stats_rows[-1]["contrast_ring_db"] = stats_rows[-1]["dbz_in"] - stats_rows[-1]["dbz_ring"]
        ring_rows = water_volumes(sweep, box, m_ring, dt_h, exclude=others)
        ring = {(r["a"], r["b"]): r["linear_m3"] for r in ring_rows}
        ringres = {(r["a"], r["b"]): r["residbg_m3"] for r in ring_rows}
        for k in config.INFLATE_KM:
            # same background estimate for every outline size, so only the outline changes
            for row in water_volumes(sweep, box.inflate(k), m_bg, dt_h, exclude=others):
                row["ring_m3"] = ring[(row["a"], row["b"])] if k == 0 else np.nan
                row["ringres_m3"] = ringres[(row["a"], row["b"])] if k == 0 else np.nan
                water_rows.append({**key, "inflate_km": k, **row})


def totals(water, by):
    cols = [c for c in ["theirs_m3", "linear_m3", "none_m3", "ring_m3", "resid5_m3", "residbg_m3", "ringres_m3"] if c in water]
    t = water.groupby(by)[cols].sum(min_count=1).reset_index()
    t["theirs_over_linear"] = t["theirs_m3"] / t["linear_m3"]
    if "ring_m3" in t:
        t["theirs_over_ring"] = t["theirs_m3"] / t["ring_m3"]
    t["none_over_theirs"] = t["none_m3"] / t["theirs_m3"]
    for c in cols:
        t[c.replace("_m3", "_af")] = t[c] / config.ACRE_FOOT_M3
    return t


def summarise(stats, water):
    flags = stats.set_index(["signature", "tilt", "time_utc"])["flag"]
    water = water.join(flags, on=["signature", "tilt", "time_utc"])
    out = []
    for subset, w in (("unflagged", water[~water["flag"]]), ("all_scans", water)):
        per_sig = totals(w, ["signature", "tilt", "inflate_km", "a", "b"])
        overall = totals(w, ["tilt", "inflate_km", "a", "b"]).assign(signature="ALL")
        both = pd.concat([per_sig, overall], ignore_index=True).assign(scans=subset)
        out.append(both)
    summary = pd.concat(out, ignore_index=True)

    # Snowfall-weighted contrast over each streak's life (unflagged scans)
    ref = water[(water.inflate_km == 0) & (water.b == config.B_REFERENCE) & (water.a == 300)]
    w = ref.set_index(["signature", "tilt", "time_utc"])["none_m3"]
    s = stats.join(w, on=["signature", "tilt", "time_utc"])
    s = s[~s["flag"] & s["contrast_db"].notna()]
    life = s.groupby(["signature", "tilt"]).apply(
        lambda g: pd.Series({
            "contrast_db_weighted": np.average(g.contrast_db, weights=g.none_m3.clip(lower=1e-12)),
            "contrast_db_median": g.contrast_db.median(),
            "contrast_db_min": g.contrast_db.min(),
            "contrast_db_max": g.contrast_db.max(),
            "contrast_pre_db_median": g.contrast_pre_db.median() if "contrast_pre_db" in g else np.nan,
            "contrast_ring_db_median": g.contrast_ring_db.median() if "contrast_ring_db" in g else np.nan,
            "n_scans_used": len(g),
        }), include_groups=False).reset_index()
    return summary, life


def main():
    from radar_io import list_radar_files, read_sweeps

    use_outlines = (config.DATA_DIR / "outlines.npz").exists() and "--boxes" not in sys.argv
    if use_outlines:
        from outlines import OutlineSignatures
        sigs = OutlineSignatures()
        print("Using Rainmaker's own outlines from the video (data/outlines.npz)")
    else:
        sigs = Signatures()
        print("Using rectangle boxes from data/keyframes.csv")
    files = list_radar_files()
    times = [file_time(p) for p in files]
    durations = scan_durations_hours(times)
    print(f"{len(files)} radar files, drift used {sigs.drift_kmh[0]:+.1f} km/h east, "
          f"{sigs.drift_kmh[1]:+.1f} km/h north")

    # Which files are the "before" scans for each signature?
    pre_files = {}
    for sig in sigs.names:
        first = sigs.windows[sig][0]
        before = [p for p, t in zip(files, times) if t < first]
        pre_files[sig] = set() if use_outlines else set(before[-config.PRE_SCANS:])

    stats_rows, water_rows, pre_rows = [], [], []
    for frame, (path, dt_h) in enumerate(zip(files, durations), start=1):
        print("reading", path.name)
        sweeps = read_sweeps(path)
        ref_time = sweeps[0.88].time if 0.88 in sweeps else None
        for tilt, sweep in sweeps.items():
            if use_outlines:
                hours = (sweep.time - ref_time).total_seconds() / 3600 if ref_time else 0.0
                sigs.set_scan(frame, hours)
            analyse_sweep(sweep, tilt, sigs, dt_h, stats_rows, water_rows)
            for sig in sigs.names:
                if path in pre_files[sig]:
                    box = sigs.box_at(sig, sigs.windows[sig][0], ignore_window=True)
                    m = box.mask(sweep.x_km, sweep.y_km)
                    pre_rows.append(dict(signature=sig, tilt=tilt,
                                         z_lin=10 ** (mean_dbz(sweep.dbz, m, sweep.area_m2) / 10)))

    stats = pd.DataFrame(stats_rows)
    water = pd.DataFrame(water_rows)
    if stats.empty:
        raise SystemExit("No scans fell inside any signature window. Check data/windows.csv and the file times.")

    pre = pd.DataFrame(pre_rows)
    if not pre.empty:
        pre_db = pre.groupby(["signature", "tilt"])["z_lin"].mean().pipe(lambda z: 10 * np.log10(z))
        stats = stats.join(pre_db.rename("dbz_pre"), on=["signature", "tilt"])
        stats["contrast_pre_db"] = stats["dbz_in"] - stats["dbz_pre"]

    summary, life = summarise(stats, water)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    stats.to_csv(config.RESULTS_DIR / "per_scan.csv", index=False)
    water.to_csv(config.RESULTS_DIR / "water_by_scan.csv", index=False)
    summary.to_csv(config.RESULTS_DIR / "summary.csv", index=False)
    life.to_csv(config.RESULTS_DIR / "contrast_by_signature.csv", index=False)

    headline = summary[(summary.scans == "unflagged") & (summary.inflate_km == 0)
                       & (summary.b == config.B_REFERENCE) & (summary.a == 300)]
    headline = headline.merge(life, on=["signature", "tilt"], how="left")
    cols = ["signature", "tilt", "n_scans_used", "contrast_db_weighted", "contrast_ring_db_median",
            "theirs_over_linear", "theirs_over_ring", "theirs_af", "linear_af"]
    print("\nHeadline (b = 2.1, unflagged scans, boxes not grown).")
    print("theirs_over_linear > 1 means the 5 dB method counts MORE new water than streak-minus-next-door.\n")
    print(headline[cols].round(2).to_string(index=False))
    print(f"\nAll files written to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
