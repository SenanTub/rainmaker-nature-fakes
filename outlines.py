"""Use Rainmaker's own outline shapes (from extract_outlines.py) instead of
rectangles. Drop-in replacement for boxes.Signatures in run_analysis.py.

Frame n of their video = the n-th radar file. Their outlines are drawn on the
0.88 deg scan; for other tilts the shape is shifted with the wind drift for the
minute or so between the two scans."""
import numpy as np

import config
from extract_outlines import X0_PX, X_PX_PER_KM, Y0_PX, Y_PX_PER_KM, _dilate

PX_PER_KM = (X_PX_PER_KM + Y_PX_PER_KM) / 2


class Region:
    """A pixel mask in the video's coordinates, usable like boxes.Box."""

    def __init__(self, m, r0, c0, dx=0.0, dy=0.0, cache=None):
        self.m, self.r0, self.c0, self.dx, self.dy = m, r0, c0, dx, dy
        self._cache = cache if cache is not None else {}
        cols = np.nonzero(m.any(0))[0]
        self.width = (cols.max() - cols.min() + 1) / X_PX_PER_KM

    def _bbox(self):
        rows, cols = np.nonzero(self.m.any(1))[0], np.nonzero(self.m.any(0))[0]
        x0 = (cols.min() + self.c0 - X0_PX) / X_PX_PER_KM + self.dx
        x1 = (cols.max() + self.c0 - X0_PX) / X_PX_PER_KM + self.dx
        y1 = -95.0 - (rows.min() + self.r0 - Y0_PX) / Y_PX_PER_KM + self.dy
        y0 = -95.0 - (rows.max() + self.r0 - Y0_PX) / Y_PX_PER_KM + self.dy
        return x0, x1, y0, y1

    x0 = property(lambda self: self._bbox()[0])
    x1 = property(lambda self: self._bbox()[1])
    y0 = property(lambda self: self._bbox()[2])
    y1 = property(lambda self: self._bbox()[3])

    def shift(self, dx=0.0, dy=0.0):
        return Region(self.m, self.r0, self.c0, self.dx + dx, self.dy + dy, self._cache)

    def inflate(self, km):
        if km <= 0:
            return self
        if km not in self._cache:
            n = int(round(km * PX_PER_KM))
            self._cache[km] = (_dilate(np.pad(self.m, n), n), self.r0 - n, self.c0 - n)
        m, r0, c0 = self._cache[km]
        return Region(m, r0, c0, self.dx, self.dy, {})

    def mask(self, x, y):
        col = np.rint(X0_PX + (x - self.dx) * X_PX_PER_KM).astype(int) - self.c0
        row = np.rint(Y0_PX + (-95.0 - (y - self.dy)) * Y_PX_PER_KM).astype(int) - self.r0
        ok = (row >= 0) & (row < self.m.shape[0]) & (col >= 0) & (col < self.m.shape[1])
        out = np.zeros(x.shape, bool)
        out[ok] = self.m[row[ok], col[ok]]
        return out


class OutlineSignatures:
    def __init__(self, path=None):
        d = np.load(path or config.DATA_DIR / "outlines.npz")
        shape = tuple(d["shape"])
        self.regions = {}
        for k in d.files:
            if k == "shape":
                continue
            frame, sig = int(k[:2]), k[3:]
            full = np.unpackbits(d[k])[: shape[0] * shape[1]].reshape(shape).astype(bool)
            rr, cc = np.nonzero(full)
            r0, c0 = rr.min(), cc.min()
            self.regions[(frame, sig)] = Region(full[r0:rr.max() + 1, c0:cc.max() + 1], r0, c0)
        self.names = sorted({s for _, s in self.regions})
        from boxes import Signatures
        legacy = Signatures()
        self.drift_kmh, self.exclusions, self.windows = legacy.drift_kmh, legacy.exclusions, legacy.windows
        self.frame, self.shift_h = None, 0.0

    def set_scan(self, frame, hours_from_frame):
        """Call before each sweep: which video frame, and how far in time
        (hours) this sweep is from the 0.88 deg sweep the outline was drawn on."""
        self.frame, self.shift_h = frame, hours_from_frame

    def box_at(self, sig, t=None, ignore_window=False):
        r = self.regions.get((self.frame, sig))
        if r is None:
            return None
        return r.shift(self.drift_kmh[0] * self.shift_h, self.drift_kmh[1] * self.shift_h)

    def flank_boxes(self, region, inflate_km=0.0):
        off = region.width + inflate_km + config.FLANK_GAP_KM
        return region.shift(dx=-off), region.shift(dx=+off)

    def keep_out_mask(self, x, y, t, skip_sig, inflate_km=1.0):
        m = np.zeros(x.shape, bool)
        for sig in self.names:
            if sig != skip_sig:
                r = self.box_at(sig)
                if r is not None:
                    m |= r.inflate(inflate_km).mask(x, y)
        for t0, t1, b in self.exclusions:
            if t0 <= t <= t1:
                m |= b.mask(x, y)
        return m
