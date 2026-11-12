"""Minimal NEXRAD Level II (V06, message 31) reader: reflectivity only.
Used automatically when Py-ART is not installed. Standard library + numpy."""
import bz2
import struct
from datetime import datetime, timedelta, timezone

import numpy as np

EARTH_R_EFF_M = 6371000.0 * 4.0 / 3.0   # 4/3 Earth model, as Py-ART uses


def _messages(path):
    raw = open(path, "rb").read()
    pos = 24                                   # volume header
    while pos + 4 <= len(raw):
        size = abs(struct.unpack(">i", raw[pos:pos + 4])[0])
        pos += 4
        if size == 0:
            break
        block = bz2.decompress(raw[pos:pos + size])
        pos += size
        off = 0
        while off + 28 <= len(block):
            size_hw, _, mtype = struct.unpack(">HBB", block[off + 12:off + 16])
            if size_hw == 0:
                off += 2432
                continue
            if mtype == 31:
                yield block[off + 28: off + 12 + 2 * size_hw]
                off += 12 + 2 * size_hw
            else:
                off += 2432


def _radials(path):
    """Yield one dict per radial with azimuth, elevation, cut number, time,
    reflectivity (dBZ, NaN where below threshold), first gate and spacing."""
    for m in _messages(path):
        (icao, ms, mjd, az_num, az, comp, _sp, rad_len, az_res, rad_stat, elev_num, cut, elev,
         blank, idx_mode, nblocks) = struct.unpack(">4sIHHfBBHBBBBfBBH", m[:32])
        ptrs = struct.unpack(f">{nblocks}I", m[32:32 + 4 * nblocks])
        ref, alt, rho = None, None, None
        for p in ptrs:
            btype, name = m[p:p + 1], m[p + 1:p + 4]
            if name == b"VOL":
                site_h, feed_h = struct.unpack(">hH", m[p + 16:p + 20])
                alt = float(site_h + feed_h)
            elif btype == b"D" and name in (b"REF", b"RHO"):
                (ngates, first, spacing, _tover, _snr, _flags, wsize, scale, offset) = struct.unpack(
                    ">HHHHhBBff", m[p + 8:p + 28])
                dtype = ">u1" if wsize == 8 else ">u2"
                vals = np.frombuffer(m[p + 28:p + 28 + ngates * wsize // 8], dtype=dtype).astype(float)
                out = (vals - offset) / scale
                out[vals < 2] = np.nan                 # 0 = below threshold, 1 = range folded
                if name == b"REF":
                    ref = (out, first, spacing)
                else:
                    rho = out
        if ref is None:
            continue
        t = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(days=mjd - 1, milliseconds=ms)
        yield dict(az=az, elev=elev, cut=elev_num, time=t, dbz=ref[0], first_m=ref[1],
                   spacing_m=ref[2], alt=alt, rho=rho)


def read_cuts(path):
    """Group radials into elevation cuts. Returns list of dicts."""
    cuts, alt = {}, None
    for r in _radials(path):
        cuts.setdefault(r["cut"], []).append(r)
        alt = r["alt"] if r["alt"] is not None else alt
    out = []
    for num, rads in sorted(cuts.items()):
        ng = max(len(r["dbz"]) for r in rads)
        dbz = np.full((len(rads), ng), np.nan)
        rho = np.full((len(rads), ng), np.nan)
        for i, r in enumerate(rads):
            dbz[i, :len(r["dbz"])] = r["dbz"]
            if r["rho"] is not None:
                n = min(ng, len(r["rho"]))
                rho[i, :n] = r["rho"][:n]
        first, spacing = rads[0]["first_m"], rads[0]["spacing_m"]
        out.append(dict(
            cut=num,
            fixed_angle=float(np.median([r["elev"] for r in rads])),
            elev=np.array([r["elev"] for r in rads]),
            az=np.array([r["az"] for r in rads]),
            time=sorted(r["time"] for r in rads)[len(rads) // 2],
            range_m=first + spacing * np.arange(ng, dtype=float),
            dbz=dbz,
            rho=rho,
            alt=alt or 0.0,
        ))
    return out


def gate_xyz(range_m, az_deg, elev_deg):
    """Same beam-height geometry as Py-ART's antenna_to_cartesian (metres)."""
    r = range_m[None, :]
    th = np.deg2rad(elev_deg)[:, None]
    az = np.deg2rad(az_deg)[:, None]
    R = EARTH_R_EFF_M
    z = np.sqrt(r ** 2 + R ** 2 + 2.0 * r * R * np.sin(th)) - R
    s = R * np.arcsin(r * np.cos(th) / (R + z))
    return s * np.sin(az), s * np.cos(az), z
