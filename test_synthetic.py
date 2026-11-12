"""Check the whole pipeline on FAKE radar data where the right answer is known.
Run this first; every check should print PASS. Nothing here is a real result.

Run:  python test_synthetic.py
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
from boxes import Signatures
from measure import bias_ratio, break_even_db, snow_rate, to_linear
from radar_io import Sweep
from run_analysis import analyse_sweep, scan_durations_hours, summarise

OUT = config.ROOT / "results_synthetic"
CONTRAST = {"A": 5, "B": 6, "C": 7, "D": 8, "E": 10, "F": 12, "G": 15}  # made-up dB values
BACKGROUND_DBZ = 10.0
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def polar_grid():
    r = np.arange(0.125, 150, 0.25)                 # km, 250 m gates
    az = np.arange(0.25, 360, 0.5)                  # degrees, 0.5 deg rays
    R, AZ = np.meshgrid(r, az)
    x, y = R * np.sin(np.deg2rad(AZ)), R * np.cos(np.deg2rad(AZ))
    area = (R * 1000) * np.deg2rad(0.5) * 250.0
    return x, y, area


def fake_sweep(t, sigs, grid, rng, noise_db=0.0, streak_fill=1.0, holes=False):
    x, y, area = grid
    dbz = np.full(x.shape, BACKGROUND_DBZ) + (rng.normal(0, noise_db, x.shape) if noise_db else 0)
    for sig in sigs.names:
        b = sigs.box_at(sig, t)
        if b is None:
            continue
        # streak covers the central `streak_fill` share of the box width
        pad = b.width * (1 - streak_fill) / 2
        inner = b.inflate(0).__class__(b.x0 + pad, b.x1 - pad, b.y0, b.y1)
        dbz[inner.mask(x, y)] += CONTRAST[sig]
    mask = np.zeros(x.shape, dtype=bool)
    if holes:
        mask |= (x > 12) & (x < 16) & (y < -125)    # a black patch, like the video
    return Sweep(time=t, tilt_deg=0.48, dbz=np.ma.array(dbz, mask=mask), x_km=x, y_km=y,
                 area_m2=area, beam_msl_m=np.full(x.shape, 2000.0))


def scan_times():
    t0 = datetime(2026, 8, 23, 6, 0, 0, tzinfo=timezone.utc)
    return [t0 + timedelta(minutes=7 * i) for i in range(0, 43)]


def run_fake(sigs, **kw):
    grid, rng = polar_grid(), np.random.default_rng(1)
    times = scan_times()
    stats_rows, water_rows = [], []
    for t, dt in zip(times, scan_durations_hours(times)):
        analyse_sweep(fake_sweep(t, sigs, grid, rng, **kw), 0.48, sigs, dt, stats_rows, water_rows)
    stats, water = pd.DataFrame(stats_rows), pd.DataFrame(water_rows)
    summary, life = summarise(stats, water)
    return stats, water, summary, life


def main():
    sigs = Signatures()

    # 1. Box positions: midpoint of D's two keyframes is the average of both
    t_mid = datetime(2026, 8, 23, 9, 30, 34, tzinfo=timezone.utc)
    b = sigs.box_at("D", t_mid)
    check("box interpolation (D at midpoint)", abs(b.centre[0] - 1.0) < 0.05 and abs(b.centre[1] + 115.0) < 0.05,
          f"centre {b.centre[0]:+.2f}, {b.centre[1]:+.2f} km")
    vx, vy = sigs.drift_kmh
    check("drift estimated from D", abs(vx + 5.7) < 0.2 and abs(vy + 15.7) < 0.2, f"{vx:+.1f}, {vy:+.1f} km/h")

    # 2. Drift model reproduces the report's range limits: nearest edge of the
    #    box at first detection, farthest corner at last detection
    def near(b):
        return np.hypot(np.clip(0, b.x0, b.x1), np.clip(0, b.y0, b.y1))

    def far(b):
        return max(np.hypot(px, py) for px in (b.x0, b.x1) for py in (b.y0, b.y1))

    worst_last, worst_first = 0, 0
    for sig in sigs.names:
        first, last = sigs.windows[sig]
        r_first, r_last = near(sigs.box_at(sig, first)), far(sigs.box_at(sig, last))
        lo, hi = sigs.report_ranges[sig]
        worst_last = max(worst_last, abs(r_last - hi))
        worst_first = max(worst_first, lo - r_first)
        print(f"      {sig}: predicted {r_first:5.1f}-{r_last:5.1f} km   report {lo:5.1f}-{hi:5.1f} km")
    check("far edge at last detection matches report", worst_last < 3, f"worst gap {worst_last:.1f} km")
    print(f"      NOTE: at first detection boxes reach up to {worst_first:.1f} km closer than the report.")
    print("      Young streaks are smaller than the keyframe boxes; add first-appearance keyframes from the video.")

    # 3. Analytic facts
    check("5 dB keeps ~58% (b=2.1)", abs(10 ** (-0.5 / 2.1) - 0.578) < 0.002)
    check("break-even 7.2-8.6 dB", abs(break_even_db(2.0) - 7.2) < 0.1 and abs(break_even_db(2.2) - 8.6) < 0.1)
    s_bg = snow_rate(to_linear(10.0), 300, 2.1)
    s_5 = snow_rate(to_linear(15.0 - 5.0), 300, 2.1)
    check("at exactly 5 dB contrast, method returns the background", abs(s_5 - s_bg) < 1e-12)

    # 4. Clean fake data: streak fills the box, no noise
    stats, water, summary, life = run_fake(sigs)
    contrast_err = (stats.groupby("signature").contrast_db.median() - pd.Series(CONTRAST)).abs().max()
    check("measured contrast = planted contrast", contrast_err < 1e-6, f"max error {contrast_err:.2e} dB")
    sel = summary[(summary.scans == "all_scans") & (summary.inflate_km == 0) & (summary.signature != "ALL")]
    errs = []
    for r in sel.itertuples():
        errs.append(abs(r.theirs_over_linear / bias_ratio(CONTRAST[r.signature], r.b) - 1))
    check("ratio matches the formula for every (a, b)", max(errs) < 1e-6, f"max error {max(errs):.1e}")
    a_spread = sel.groupby(["signature", "b"]).theirs_over_linear.agg(np.ptp).max()
    check("ratio does not depend on a", a_spread < 1e-9)

    # 5. Streak only fills half the box: 5 dB method grows with box size, linear does not
    stats, water, summary, life = run_fake(sigs, streak_fill=0.5)
    tot = summary[(summary.scans == "all_scans") & (summary.signature == "ALL") & (summary.b == 2.1) & (summary.a == 300)]
    tot = tot.set_index("inflate_km")
    grow_theirs = tot.loc[4, "theirs_m3"] / tot.loc[0, "theirs_m3"]
    grow_linear = tot.loc[4, "linear_m3"] / tot.loc[0, "linear_m3"]
    check("outline test: linear stays flat", abs(grow_linear - 1) < 0.02, f"x{grow_linear:.3f}")
    check("outline test: 5 dB method grows", grow_theirs > 1.5, f"x{grow_theirs:.2f}")

    # 6. Noisy data with black patches: contrast still recovered on average
    stats, water, summary, life = run_fake(sigs, noise_db=2.0, holes=True)
    med = stats.groupby("signature").contrast_db.median()
    err = (med - pd.Series(CONTRAST)).abs().max()
    check("noisy data: contrast within 0.5 dB", err < 0.5, f"max error {err:.2f} dB")

    # 7. Figures render (fake data)
    OUT.mkdir(exist_ok=True)
    stats.to_csv(OUT / "per_scan.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    life.to_csv(OUT / "contrast_by_signature.csv", index=False)
    import make_figures
    from quicklook import plot_sweep
    make_figures.fig_bias_curve(OUT, OUT / "fig1_bias_curve_FAKE.png")
    make_figures.fig_outline_size(OUT, OUT / "fig2_outline_size_FAKE.png")
    make_figures.fig_contrast_life(OUT, OUT / "fig3_contrast_over_life_FAKE.png")
    t = datetime(2026, 8, 23, 9, 1, tzinfo=timezone.utc)
    plot_sweep(fake_sweep(t, sigs, polar_grid(), np.random.default_rng(2), noise_db=2, holes=True),
               sigs, OUT / "quicklook_FAKE.png")
    check("figures written", all((OUT / f).exists() for f in
                                 ["fig1_bias_curve_FAKE.png", "fig2_outline_size_FAKE.png",
                                  "fig3_contrast_over_life_FAKE.png", "quicklook_FAKE.png"]))

    print(f"\n{sum(results)}/{len(results)} checks passed")


if __name__ == "__main__":
    main()
