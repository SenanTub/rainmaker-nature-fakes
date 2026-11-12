"""Read one NEXRAD Level II file with Py-ART and return the sweeps we need
as plain numpy arrays, so the rest of the code never touches Py-ART."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

import config


@dataclass
class Sweep:
    time: datetime          # UTC, middle of the sweep
    tilt_deg: float         # actual elevation angle
    dbz: np.ma.MaskedArray  # reflectivity (dBZ), masked where no echo / no data
    x_km: np.ndarray        # east of radar
    y_km: np.ndarray        # north of radar
    area_m2: np.ndarray     # ground area of each gate
    beam_msl_m: np.ndarray  # beam centre height above sea level


def _sweep_time(radar, sl):
    units = radar.time["units"]                      # "seconds since 2026-08-23T07:55:56Z"
    base = units.split("since", 1)[1].strip().replace("Z", "")
    base = datetime.fromisoformat(base).replace(tzinfo=timezone.utc)
    return base + timedelta(seconds=float(np.median(radar.time["data"][sl])))


def read_sweeps(path, tilts=config.TILTS_DEG):
    try:
        import pyart
    except ImportError:
        return _read_sweeps_builtin(path, tilts)

    radar = pyart.io.read_nexrad_archive(str(path))
    angles = radar.fixed_angle["data"]
    refl = radar.fields["reflectivity"]["data"]
    alt = float(radar.altitude["data"][0])
    out = {}

    for target in tilts:
        # NEXRAD repeats low tilts (split cuts): keep the one with most reflectivity gates
        candidates = [i for i, a in enumerate(angles) if abs(a - target) <= config.TILT_TOL_DEG]
        if not candidates:
            continue
        best = max(candidates, key=lambda i: np.ma.count(refl[radar.get_slice(i)]))
        sl = radar.get_slice(best)

        x, y, z = radar.get_gate_x_y_z(best)
        r = radar.range["data"]
        az = radar.get_azimuth(best)
        d_az = np.abs(np.diff(az))
        d_az = np.minimum(d_az, 360 - d_az)
        d_theta = np.deg2rad(np.median(d_az))
        d_r = float(np.median(np.diff(r)))
        area = np.broadcast_to(r[None, :] * d_theta * d_r, x.shape).copy()

        out[target] = Sweep(
            time=_sweep_time(radar, sl),
            tilt_deg=float(angles[best]),
            dbz=np.ma.masked_invalid(refl[sl]),
            x_km=x / 1000.0,
            y_km=y / 1000.0,
            area_m2=area,
            beam_msl_m=z + alt,
        )
    return out


def list_radar_files():
    files = [p for p in sorted(config.RADAR_DIR.glob("*")) if p.is_file() and not p.name.endswith("_MDM")]
    if not files:
        raise SystemExit(f"No radar files in {config.RADAR_DIR}. Run:  python fetch_data.py")
    return files


def _read_sweeps_builtin(path, tilts):
    """Fallback when Py-ART isn't installed: uses lvl2.py (same geometry)."""
    import lvl2
    cuts = lvl2.read_cuts(path)
    out = {}
    for target in tilts:
        cands = [c for c in cuts if abs(c["fixed_angle"] - target) <= config.TILT_TOL_DEG]
        if not cands:
            continue
        c = max(cands, key=lambda c: np.isfinite(c["dbz"]).sum())
        x, y, z = lvl2.gate_xyz(c["range_m"], c["az"], c["elev"])
        d_az = np.abs(np.diff(c["az"]))
        d_az = np.minimum(d_az, 360 - d_az)
        d_theta = np.deg2rad(np.median(d_az))
        d_r = float(np.median(np.diff(c["range_m"])))
        area = np.broadcast_to(c["range_m"][None, :] * d_theta * d_r, x.shape).copy()
        out[target] = Sweep(time=c["time"], tilt_deg=c["fixed_angle"],
                            dbz=np.ma.masked_invalid(c["dbz"]), x_km=x / 1000.0, y_km=y / 1000.0,
                            area_m2=area, beam_msl_m=z + c["alt"])
    return out
