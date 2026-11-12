"""How often does nature fake a seeding signature? One script for any case.
Run:  python nature_fakes.py kenai   |   python nature_fakes.py pdt025
Stage 1 (cached) grids every scan; stage 2 tracks, classifies, and measures base rate,
corridor matches and the chance of coincidence (Monte Carlo, one feature per release)."""
import math, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd

import config, lvl2
import lookalikes as L
from cases import CASES, SHARED


def wedge(x, y, site_xy, flow, km):
    dx, dy = x - site_xy[0], y - site_xy[1]
    brg = np.degrees(np.arctan2(dx, dy)) % 360
    return (np.hypot(dx, dy) <= km) & (np.abs((brg - flow + 180) % 360 - 180) <= SHARED["dir_tol_deg"])


def stage1(name, case):
    out = config.RESULTS_DIR / f"nf_objects_{name}.pkl"
    if out.exists():
        return pickle.load(open(out, "rb"))
    scans = []
    for f in sorted(Path(config.ROOT / case["radar_dir"]).glob("*_V06")):
        cut = L.pick_cut(lvl2.read_cuts(str(f)), case["tilt"])
        if cut is None:
            continue
        g = L.to_grid(cut)
        bg = L.background(g)
        scans.append(dict(file=f.name, time=cut["time"], objs=L.objects(g, bg, SHARED["min_enh_db"]),
                          echo_frac=float(((g >= 0) & L.BAND).mean() / L.BAND.mean())))
    pickle.dump(scans, open(out, "wb"))
    return scans


def classify(tr, times, min_scans, require_growth=True):
    fails = []
    n = len(tr)
    dur = (times[tr[-1][0]] - times[tr[0][0]]).total_seconds() / 60
    if n < min_scans: fails.append("too short")
    if dur > SHARED["max_minutes"]: fails.append("too long")
    if not all(SHARED["area_km2"][0] <= o["area"] <= SHARED["area_km2"][1] for _, o in tr): fails.append("size")
    if n >= 2:
        st = [o["area"] * 10 ** (o["enh_db"] / 10) for _, o in tr]
        kp = int(np.argmax(st))
        if require_growth and kp == 0: fails.append("no growth")
        if n >= 3 and kp == n - 1: fails.append("no decay")
        dt = dur / 60
        vx, vy = (tr[-1][1]["x"] - tr[0][1]["x"]) / dt, (tr[-1][1]["y"] - tr[0][1]["y"]) / dt
        ang = abs((np.degrees(np.arctan2(vx, vy)) - L.FLOW_TOWARD_DEG + 180) % 360 - 180)
        if ang > SHARED["dir_tol_deg"] or not (SHARED["speed_kmh"][0] <= math.hypot(vx, vy) <= SHARED["speed_kmh"][1]):
            fails.append("wrong drift")
    return not fails, fails


def chance_one_to_one(rate, corr_km2, rel_min, window, k_obs, n=200_000, seed=3):
    rng = np.random.default_rng(seed)
    t0, t1 = rel_min.min() + window[0], rel_min.max() + window[1]
    mu = rate * corr_km2 / 1000 * (t1 - t0) / 60
    counts = rng.poisson(mu, n)
    res = np.zeros(n, int)
    rs = np.sort(rel_min)
    for i in np.nonzero(counts)[0]:
        ts = np.sort(rng.uniform(t0, t1, counts[i]))
        used = np.zeros(len(ts), bool)
        for r in rs:
            idx = np.nonzero(~used & (ts >= r + window[0]) & (ts <= r + window[1]))[0]
            if len(idx):
                used[idx[0]] = True; res[i] += 1
    return mu, (res >= k_obs).mean(), np.bincount(res, minlength=len(rel_min) + 1) / n


def stage2(name, case, scans, min_scans=3, local_km=None, require_growth=True):
    times = [s["time"] for s in scans]
    tracks = L.track([s["objs"] for s in scans], times)
    looks = [tr for tr in tracks if classify(tr, times, min_scans, require_growth)[0]]
    win = SHARED["window_min"]
    corr = np.zeros(L.GX.shape, bool)
    for xy in case["sites"].values():
        corr |= wedge(L.GX, L.GY, xy, case["flow_toward_deg"], SHARED["wedge_km"])
    corr &= L.BAND
    ctrl = L.BAND & ~corr
    if local_km:
        near = np.zeros(L.GX.shape, bool)
        for xy in case["sites"].values():
            near |= np.hypot(L.GX - xy[0], L.GY - xy[1]) <= local_km
        ctrl &= near
    cell = lambda o: (int(np.argmin(np.abs(L._c[::-1] - o["y"]))), int(np.argmin(np.abs(L._c - o["x"]))))
    hours = (times[-min_scans] - times[0]).total_seconds() / 3600
    n_ctrl = sum(ctrl[cell(tr[0][1])] for tr in looks)
    exposure = ctrl.sum() * L.GRID_KM ** 2 / 1000 * hours          # thousand km2 x hours
    rate = n_ctrl / exposure
    rate_hi = (3.0 if n_ctrl == 0 else n_ctrl + 1.96 * math.sqrt(n_ctrl) + 1.0) / exposure   # ~95% upper bound
    # one-to-one matching of lookalike births in the corridor to releases
    rel = sorted(case["releases"].items(), key=lambda kv: kv[1][1])
    births = sorted([tr for tr in looks if corr[cell(tr[0][1])]], key=lambda tr: times[tr[0][0]])
    used, matched = set(), {}
    for sig, (site, t) in rel:
        for j, tr in enumerate(births):
            m = (times[tr[0][0]] - t).total_seconds() / 60
            o = tr[0][1]
            if j not in used and win[0] <= m <= win[1] and wedge(np.array(o["x"]), np.array(o["y"]),
                                                                  case["sites"][site], case["flow_toward_deg"],
                                                                  SHARED["wedge_km"]):
                used.add(j); matched[sig] = (round(m), round(o["x"], 1), round(o["y"], 1), len(tr)); break
    t_ref = rel[0][1][1]
    rel_min = np.array([(t - t_ref).total_seconds() / 60 for _, (_, t) in rel])
    corr_km2 = corr.sum() * L.GRID_KM ** 2
    mu, p_obs, dist = chance_one_to_one(rate, corr_km2, rel_min, win, max(len(matched), 1))
    _, p_obs_hi, _ = chance_one_to_one(rate_hi, corr_km2, rel_min, win, max(len(matched), 1))
    # births in the corridor before any seeding (a temporal control at the same place)
    pre = [tr for tr in births if times[tr[0][0]] < t_ref]
    pre_hours = max((t_ref - times[0]).total_seconds() / 3600, 0)
    return dict(case=name, growth=require_growth, min_scans=min_scans, local_km=local_km, scans=len(scans), hours=round(hours, 2),
                lookalikes=len(looks), control_lookalikes=int(n_ctrl), rate_per_1000km2_h=rate,
                corridor_km2=corr_km2, p_per_release=1 - math.exp(-rate * corr_km2 / 1000 * (win[1] - win[0]) / 60),
                rate_upper95=rate_hi, releases=len(rel), matched=len(matched), p_matched_by_chance=p_obs,
                p_matched_upper95=p_obs_hi,
                corridor_births_before_seeding=len(pre), hours_before_seeding=round(pre_hours, 2),
                matches=matched), looks, tracks


if __name__ == "__main__":
    name = sys.argv[1]
    case = CASES[name]
    L.configure(case)
    scans = stage1(name, case)
    rows = []
    for growth in (True, False):
        for ms in (3, 4, 5):
            for local in (None, 60):
                r, *_ = stage2(name, case, scans, ms, local, growth)
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / f"nf_summary_{name}.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["growth", "min_scans", "local_km", "hours", "control_lookalikes", "rate_per_1000km2_h", "rate_upper95",
            "p_per_release", "matched", "releases", "p_matched_by_chance", "p_matched_upper95", "corridor_births_before_seeding"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    for r in rows:
        if r["local_km"] is None:
            print("growth" if r["growth"] else "no-growth", r["min_scans"], "scans -> matches:", r["matches"])
