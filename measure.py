"""The maths of the check, on plain numpy arrays (no Py-ART here, so it can be
tested on fake data).

Units: Z in mm^6/m^3 (linear), dBZ = 10 log10 Z, snowfall S in mm/h of liquid
water from Z = a * S^b, so S = (Z / a)^(1/b).
"""
import numpy as np

import config


def to_linear(dbz):
    return 10.0 ** (np.asarray(dbz, dtype=float) / 10.0)


def snow_rate(z_lin, a, b):
    return (z_lin / a) ** (1.0 / b)


def _valid(dbz, mask):
    return mask & ~np.ma.getmaskarray(dbz)


def mean_dbz(dbz, mask, area):
    """Area-weighted mean in linear units, returned in dBZ. Ignores gates with
    no data. Averaging in dB would give a different (wrong) answer."""
    v = _valid(dbz, mask)
    if not v.any():
        return np.nan
    z = to_linear(np.ma.getdata(dbz)[v])
    return 10.0 * np.log10(np.average(z, weights=area[v]))


def background_mask(x, y, flank_boxes, keep_out):
    m = np.zeros(x.shape, dtype=bool)
    for fb in flank_boxes:
        m |= fb.mask(x, y)
    return m & ~keep_out


def gate_stats(sweep, box, flank_boxes, keep_out):
    """Brightness inside the streak box vs next door, for one scan."""
    x, y, dbz, area = sweep.x_km, sweep.y_km, sweep.dbz, sweep.area_m2
    m_in = box.mask(x, y)
    m_bg = background_mask(x, y, flank_boxes, keep_out)

    n_in, n_bg = int(m_in.sum()), int(m_bg.sum())
    v_in, v_bg = _valid(dbz, m_in), _valid(dbz, m_bg)
    dbz_in = mean_dbz(dbz, m_in, area)
    dbz_bg = mean_dbz(dbz, m_bg, area)
    stats = dict(
        n_gates_in=n_in,
        valid_frac_in=v_in.sum() / n_in if n_in else np.nan,
        n_gates_bg=n_bg,
        valid_frac_bg=v_bg.sum() / n_bg if n_bg else np.nan,
        dbz_in=dbz_in,
        dbz_bg=dbz_bg,
        contrast_db=dbz_in - dbz_bg,
        beam_msl_m=float(np.median(sweep.beam_msl_m[m_in])) if n_in else np.nan,
    )
    stats["flag"] = (
        v_in.sum() < config.MIN_VALID_GATES
        or not (stats["valid_frac_bg"] >= config.MIN_FLANK_VALID_FRAC)
    )
    return stats, m_bg


def water_volumes(sweep, box, m_bg, dt_hours, exclude=None, a_values=config.A_VALUES,
                  b_values=config.B_VALUES, offset_db=config.OFFSET_DB):
    """New water (m^3) from one scan, three ways, for every (a, b):
      theirs : keep Z - 5 dB inside the box (Rainmaker's method)
      linear : snowfall inside minus mean snowfall next door, gate by gate
      none   : all snowfall inside the box, no adjustment
    Only gates with an echo count; empty gates add nothing. `exclude` removes
    gates belonging to other streaks."""
    x, y, dbz, area = sweep.x_km, sweep.y_km, sweep.dbz, sweep.area_m2
    v_in = _valid(dbz, box.mask(x, y))
    if exclude is not None:
        v_in &= ~exclude
    v_bg = _valid(dbz, m_bg)
    z_in = to_linear(np.ma.getdata(dbz)[v_in])
    z_bg = to_linear(np.ma.getdata(dbz)[v_bg])
    a_in, a_bg = area[v_in], area[v_bg]
    scale = dt_hours * 1e-3  # mm/h * m^2 * h -> m^3

    rows = []
    for b in b_values:
        for a in a_values:
            s_in = snow_rate(z_in, a, b)
            s_off = snow_rate(z_in * 10 ** (-offset_db / 10), a, b)
            s_bg = np.average(snow_rate(z_bg, a, b), weights=a_bg) if z_bg.size else np.nan
            # Rainmaker's published method (Satrio et al. 2026, Atmosphere 17:1093): subtract a
            # background reflectivity in LINEAR units, negatives set to zero
            z_bg5 = 10 ** (offset_db / 10)
            s_res5 = snow_rate(np.clip(z_in - z_bg5, 0, None), a, b)
            z_bgm = np.average(z_bg, weights=a_bg) if z_bg.size else np.nan
            s_resm = snow_rate(np.clip(z_in - z_bgm, 0, None), a, b)
            rows.append(dict(
                a=a, b=b,
                resid5_m3=float(np.sum(s_res5 * a_in) * scale),
                residbg_m3=float(np.sum(s_resm * a_in) * scale),
                theirs_m3=float(np.sum(s_off * a_in) * scale),
                linear_m3=float(np.sum((s_in - s_bg) * a_in) * scale),
                none_m3=float(np.sum(s_in * a_in) * scale),
            ))
    return rows


# ---- the analytic curve (no data needed) -----------------------------------

def kept_fraction(b, offset_db=config.OFFSET_DB):
    """Share of snowfall Rainmaker's method keeps: 10^(-offset/(10 b))."""
    return 10 ** (-offset_db / (10 * b))


def true_new_fraction(contrast_db, b):
    """Share of the streak's snowfall that is genuinely new, if the streak is
    contrast_db brighter than next door: 1 - 10^(-contrast/(10 b))."""
    return 1 - 10 ** (-np.asarray(contrast_db, dtype=float) / (10 * b))


def bias_ratio(contrast_db, b, offset_db=config.OFFSET_DB):
    """Rainmaker estimate / true new water. >1 means they over-count."""
    return kept_fraction(b, offset_db) / true_new_fraction(contrast_db, b)


def break_even_db(b, offset_db=config.OFFSET_DB):
    return -10 * b * np.log10(1 - kept_fraction(b, offset_db))
