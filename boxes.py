"""Where is each streak box at a given time, and where are its next-door boxes?

Positions come from data/keyframes.csv (read off Rainmaker's video).
Between two keyframes the box edges move in a straight line. Outside the
keyframes the box keeps its size and drifts with the wind.
"""
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config


@dataclass(frozen=True)
class Box:
    x0: float
    x1: float
    y0: float
    y1: float

    @property
    def width(self):
        return self.x1 - self.x0

    @property
    def centre(self):
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def inflate(self, km):
        return Box(self.x0 - km, self.x1 + km, self.y0 - km, self.y1 + km)

    def shift(self, dx=0.0, dy=0.0):
        return Box(self.x0 + dx, self.x1 + dx, self.y0 + dy, self.y1 + dy)

    def mask(self, x, y):
        return (x >= self.x0) & (x <= self.x1) & (y >= self.y0) & (y <= self.y1)


def _utc(s):
    t = datetime.fromisoformat(str(s))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _hours(t1, t0):
    return (t1 - t0).total_seconds() / 3600.0


class Signatures:
    def __init__(self, keyframes_csv=None, windows_csv=None, exclusions_csv=None):
        kf = pd.read_csv(keyframes_csv or config.DATA_DIR / "keyframes.csv")
        win = pd.read_csv(windows_csv or config.DATA_DIR / "windows.csv")
        exc_path = exclusions_csv or config.DATA_DIR / "exclusions.csv"

        self.keyframes = {}
        for sig, g in kf.groupby("signature"):
            rows = sorted(
                (_utc(r.time_utc), Box(r.x_min_km, r.x_max_km, r.y_min_km, r.y_max_km))
                for r in g.itertuples()
            )
            self.keyframes[sig] = rows

        self.windows = {
            r.signature: (_utc(r.first_detected_utc), _utc(r.last_detected_utc))
            for r in win.itertuples()
        }
        self.report_ranges = {
            r.signature: (r.report_range_min_km, r.report_range_max_km) for r in win.itertuples()
        }
        self.names = sorted(self.windows)

        self.exclusions = []
        try:
            for r in pd.read_csv(exc_path).itertuples():
                self.exclusions.append(
                    (_utc(r.start_utc), _utc(r.end_utc), Box(r.x_min_km, r.x_max_km, r.y_min_km, r.y_max_km))
                )
        except FileNotFoundError:
            pass

        self.drift_kmh = self._estimate_drift()

    def _estimate_drift(self):
        """Average velocity of box centres across every pair of keyframes."""
        vels = []
        for rows in self.keyframes.values():
            for (t0, b0), (t1, b1) in zip(rows, rows[1:]):
                dt = _hours(t1, t0)
                if dt > 0:
                    (x0, y0), (x1, y1) = b0.centre, b1.centre
                    vels.append(((x1 - x0) / dt, (y1 - y0) / dt))
        return tuple(np.mean(vels, axis=0)) if vels else config.DRIFT_KMH_FALLBACK

    def active(self, sig, t):
        first, last = self.windows[sig]
        return first <= t <= last

    def box_at(self, sig, t, ignore_window=False):
        if not ignore_window and not self.active(sig, t):
            return None
        rows = self.keyframes.get(sig)
        if not rows:
            return None
        vx, vy = self.drift_kmh
        if t <= rows[0][0]:
            t0, b = rows[0]
            dt = _hours(t, t0)
            return b.shift(vx * dt, vy * dt)
        if t >= rows[-1][0]:
            t0, b = rows[-1]
            dt = _hours(t, t0)
            return b.shift(vx * dt, vy * dt)
        for (t0, b0), (t1, b1) in zip(rows, rows[1:]):
            if t0 <= t <= t1:
                f = _hours(t, t0) / _hours(t1, t0)
                lerp = lambda a, c: a + f * (c - a)
                return Box(lerp(b0.x0, b1.x0), lerp(b0.x1, b1.x1), lerp(b0.y0, b1.y0), lerp(b0.y1, b1.y1))

    def flank_boxes(self, box, inflate_km=0.0):
        """Next-door boxes: same size as the streak box, same distance from the
        radar (same y, since the streaks sit almost due south), one each side,
        starting FLANK_GAP_KM beyond the edge of the streak box grown by
        inflate_km (so they never overlap it)."""
        offset = box.width + inflate_km + config.FLANK_GAP_KM
        return box.shift(dx=-offset), box.shift(dx=+offset)

    def keep_out_mask(self, x, y, t, skip_sig, inflate_km=1.0):
        """Gates belonging to other active streaks (grown by inflate_km) or to
        exclusion zones. Used so background boxes never sample another streak,
        and so a grown streak box never swallows its neighbour."""
        m = np.zeros(x.shape, dtype=bool)
        for sig in self.names:
            if sig == skip_sig:
                continue
            b = self.box_at(sig, t)
            if b is not None:
                m |= b.inflate(inflate_km).mask(x, y)
        for t0, t1, b in self.exclusions:
            if t0 <= t <= t1:
                m |= b.mask(x, y)
        return m
