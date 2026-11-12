"""How often does nature fake a seeding signature?

A blind, objective version of Rainmaker's attribution criteria (Satrio et al.
2026, Atmosphere 17:1093, Sec. 2.4), run over a whole radar domain.

ALL thresholds below are fixed in advance from the literature, NOT tuned on
the seeded signatures:
  - enhancement >= 5 dB over the local background (Zaremba et al.)
  - echo >= 0 dBZ (the paper measures signature area above 0 dBZ)
  - object area 2 to 80 km2 (paper: median peak area 30.8 km2; UAS signatures are small)
  - seen in >= 3 consecutive scans (~20 min) and lasting <= 120 min
    (paper: median lifetime 51 min)
  - grows then fades: peak brightness or area after the first scan
  - drifts within 30 deg of the wind at seeding level (paper: 58 of 64 signatures
    did). Wind direction taken from the report's synoptic description
    ("northeasterly flow" -> toward 225 deg), not from the signatures.
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import lvl2

# ---- fixed parameters ----------------------------------------------------
GRID_KM = 0.5
HALF_KM = 140.0
RANGE_BAND_KM = (95.0, 140.0)       # same distances from the radar as the Kenai signatures
BG_BLOCK_KM = 10.0
RHO_MIN = 0.70
MIN_ENH_DB, MIN_DBZ = 5.0, 0.0
AREA_KM2 = (2.0, 80.0)
MIN_SCANS, MAX_MINUTES = 3, 120
DIR_TOL_DEG = 30.0

N = int(2 * HALF_KM / GRID_KM)
_c = -HALF_KM + GRID_KM * (np.arange(N) + 0.5)
GX, GY = np.meshgrid(_c, _c[::-1])          # row 0 = north
GR = np.hypot(GX, GY)
GAZ = np.degrees(np.arctan2(GX, GY)) % 360
BAND = (GR >= RANGE_BAND_KM[0]) & (GR <= RANGE_BAND_KM[1])


def configure(case):
    """Set the per-case range band and plume direction; everything else is shared."""
    global RANGE_BAND_KM, BAND, FLOW_TOWARD_DEG
    RANGE_BAND_KM = case["band_km"]
    BAND = (GR >= RANGE_BAND_KM[0]) & (GR <= RANGE_BAND_KM[1])
    FLOW_TOWARD_DEG = case["flow_toward_deg"]


def pick_cut(cuts, tilt):
    cands = [c for c in cuts if abs(c["fixed_angle"] - tilt) <= 0.2]
    return max(cands, key=lambda c: np.isfinite(c["dbz"]).sum()) if cands else None


def to_grid(cut):
    """Nearest-gate dBZ on the Cartesian grid (NaN = no echo)."""
    az = cut["az"] % 360
    order = np.argsort(az)
    az_s = az[order]
    # nearest ray (with wrap-around)
    i = np.searchsorted(az_s, GAZ)
    i0, i1 = (i - 1) % len(az_s), i % len(az_s)
    d0 = np.abs((GAZ - az_s[i0] + 180) % 360 - 180)
    d1 = np.abs((GAZ - az_s[i1] + 180) % 360 - 180)
    ray = order[np.where(d0 <= d1, i0, i1)]
    # ground range of each gate along this tilt
    x, y, _ = lvl2.gate_xyz(cut["range_m"], np.array([0.0]), np.array([np.median(cut["elev"])]))
    s_km = y[0] / 1000.0
    gate = np.clip(np.searchsorted(s_km, GR), 0, len(s_km) - 1)
    g = cut["dbz"][ray, gate]
    if "rho" in cut and np.isfinite(cut["rho"]).any():
        rho = cut["rho"][ray, gate]
        g[~(rho >= RHO_MIN)] = np.nan          # paper's QC: keep gates with rho_hv >= 0.70
    g[GR > s_km[-1]] = np.nan
    return g


def background(g):
    """Median dBZ in 10 km blocks (no-echo counted as -15 dBZ), bilinear-smoothed."""
    b = int(BG_BLOCK_KM / GRID_KM)
    h = np.where(np.isfinite(g), g, -15.0)
    blk = np.median(h.reshape(N // b, b, N // b, b).transpose(0, 2, 1, 3).reshape(N // b, N // b, -1), axis=2)
    # bilinear upsample
    cb = (np.arange(N // b) + 0.5) * b - 0.5
    idx = np.arange(N)
    r = np.interp(idx, cb, np.arange(N // b))
    r0 = np.clip(np.floor(r).astype(int), 0, N // b - 2)
    fr = np.clip(r - r0, 0, 1)
    tmp = blk[r0] * (1 - fr)[:, None] + blk[r0 + 1] * fr[:, None]
    out = tmp[:, r0] * (1 - fr)[None, :] + tmp[:, r0 + 1] * fr[None, :]
    return out


def components(mask):
    lab = np.zeros(mask.shape, int)
    comps = []
    for r0, c0 in zip(*np.nonzero(mask)):
        if lab[r0, c0]:
            continue
        lab[r0, c0] = len(comps) + 1
        q, pts = deque([(r0, c0)]), []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= rr < N and 0 <= cc < N and mask[rr, cc] and not lab[rr, cc]:
                    lab[rr, cc] = len(comps) + 1
                    q.append((rr, cc))
        comps.append(np.array(pts))
    return comps


def objects(g, bg, min_enh=MIN_ENH_DB):
    enh = g - bg
    m = np.isfinite(g) & (g >= MIN_DBZ) & (enh >= min_enh) & BAND
    out = []
    for pts in components(m):
        area = len(pts) * GRID_KM ** 2
        if not (AREA_KM2[0] <= area <= AREA_KM2[1]):
            continue
        r, c = pts[:, 0], pts[:, 1]
        zl = 10 ** (g[r, c] / 10)
        out.append(dict(x=float(np.average(GX[r, c], weights=zl)), y=float(np.average(GY[r, c], weights=zl)),
                        area=area, enh_db=float(np.median(enh[r, c])), peak_dbz=float(g[r, c].max()),
                        pix=pts))
    return out


def field_motion(g1, g2, dt_h):
    """Mean echo motion (km/h) between two scans by phase correlation."""
    a = np.where(np.isfinite(g1) & (GR < 140), np.clip(g1, 0, None), 0.0)
    b = np.where(np.isfinite(g2) & (GR < 140), np.clip(g2, 0, None), 0.0)
    F = np.fft.fft2(a - a.mean()) * np.conj(np.fft.fft2(b - b.mean()))
    R = np.fft.ifft2(F / (np.abs(F) + 1e-9)).real
    r, c = np.unravel_index(np.argmax(R), R.shape)
    dr = r if r < N // 2 else r - N
    dc = c if c < N // 2 else c - N
    # b is a shifted by (-dr, -dc) cells; rows increase southward
    vx = -dc * GRID_KM / dt_h
    vy = dr * GRID_KM / dt_h
    return vx, vy


def track(scan_objs, times, v_field):
    """Link objects scan to scan by advecting with the field motion."""
    tracks, open_tr = [], []
    for k, objs in enumerate(scan_objs):
        used, nxt = set(), []
        for tr in open_tr:
            last = tr[-1]
            dt = (times[k] - times[last[0]]).total_seconds() / 3600
            px, py = last[1]["x"] + v_field[0] * dt, last[1]["y"] + v_field[1] * dt
            best, bd = None, MAX_LINK_KM
            for j, o in enumerate(objs):
                d = np.hypot(o["x"] - px, o["y"] - py)
                if j not in used and d < bd:
                    best, bd = j, d
            if best is None:
                tracks.append(tr)
            else:
                used.add(best)
                tr.append((k, objs[best]))
                nxt.append(tr)
        for j, o in enumerate(objs):
            if j not in used:
                nxt.append([(k, o)])
        open_tr = nxt
    return tracks + open_tr


FLOW_TOWARD_DEG = 225.0      # report: "northeasterly flow" -> drifts toward the south-west
MIN_KMH, MAX_KMH = 5.0, 60.0  # moving (not terrain-locked), physically plausible


def classify(tr, times, min_scans=MIN_SCANS):
    """Objective version of the paper's criteria. Returns (passes, list of failed tests)."""
    fails = []
    n = len(tr)
    dur = (times[tr[-1][0]] - times[tr[0][0]]).total_seconds() / 60
    if n < min_scans:
        fails.append("too short")
    if dur > MAX_MINUTES:
        fails.append("too long")
    if n >= 2:
        strength = [o["area"] * 10 ** (o["enh_db"] / 10) for _, o in tr]
        k_peak = int(np.argmax(strength))
        if k_peak == 0:
            fails.append("no growth")
        if n >= 3 and k_peak == n - 1:
            fails.append("no decay")
        dt = (times[tr[-1][0]] - times[tr[0][0]]).total_seconds() / 3600
        vx = (tr[-1][1]["x"] - tr[0][1]["x"]) / dt
        vy = (tr[-1][1]["y"] - tr[0][1]["y"]) / dt
        sp = np.hypot(vx, vy)
        ang = abs((np.degrees(np.arctan2(vx, vy)) - FLOW_TOWARD_DEG + 180) % 360 - 180)
        if ang > DIR_TOL_DEG or not (MIN_KMH <= sp <= MAX_KMH):
            fails.append("not drifting with flow")
    return (len(fails) == 0), fails


def track(scan_objs, times, max_link_km=3.0):
    """Link each object to the nearest object in the next scan (similar size, within max_link_km
    after allowing for drift of up to MAX_KMH). Greedy, one-to-one."""
    tracks, open_tr = [], []
    for k, objs in enumerate(scan_objs):
        used, nxt = set(), []
        for tr in open_tr:
            kk, last = tr[-1]
            if kk != k - 1:
                tracks.append(tr); continue
            dt = (times[k] - times[kk]).total_seconds() / 3600
            best, bd = None, max_link_km + MAX_KMH * dt * 0 + 0.0
            reach = max_link_km
            for j, o in enumerate(objs):
                if j in used:
                    continue
                d = np.hypot(o["x"] - last["x"], o["y"] - last["y"])
                ratio = o["area"] / last["area"]
                if d < reach and 1 / 3 <= ratio <= 3 and (best is None or d < bd):
                    best, bd = j, d
            if best is None:
                tracks.append(tr)
            else:
                used.add(best)
                tr.append((k, objs[best]))
                nxt.append(tr)
        for j, o in enumerate(objs):
            if j not in used:
                nxt.append([(k, o)])
        open_tr = nxt
    return tracks + open_tr
