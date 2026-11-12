"""Read Rainmaker's own signature outlines out of their radar video (Vid.mp4).

Each outline is drawn in a fixed colour. For every frame we find that colour,
take the area it encloses, and convert pixels to km using the plot's tick marks.
Frame n is the n-th radar file (checked against the times printed on frames
15, 18, 27, 36 and 40, which match the 0.88 deg scan times to the second).

Run:  ffmpeg -i Vid.mp4 -vsync 0 frames/f%02d.png   (once)
      python extract_outlines.py
Writes data/outlines.npz (exact outline shapes) and data/outline_boxes.csv.
"""
from collections import deque
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

import config

FRAMES = sorted((config.ROOT / "frames").glob("f*.png"))
COLOURS = {  # measured from the video
    "A": (86, 219, 252), "B": (252, 178, 66), "C": (252, 103, 180), "D": (186, 138, 252),
    "E": (252, 236, 86), "F": (103, 230, 196), "G": (251, 124, 112),
}
TOL = 60
# pixel <-> km calibration from the tick marks (x: -20..20 km, y: -95..-135 km)
X0_PX, X_PX_PER_KM = 579.0, 19.825
Y0_PX, Y_PX_PER_KM = 102.0, 19.86            # row of y = -95 km
PLOT = (102, 937, 123, 1037)                 # rows, cols of the plot area


def px_to_km(col, row):
    return (col - X0_PX) / X_PX_PER_KM, -95.0 - (row - Y0_PX) / Y_PX_PER_KM


def km_to_px(x, y):
    return X0_PX + x * X_PX_PER_KM, Y0_PX + (-95.0 - y) * Y_PX_PER_KM


def _dilate(m, n=1):
    for _ in range(n):
        d = m.copy()
        d[1:] |= m[:-1]; d[:-1] |= m[1:]; d[:, 1:] |= m[:, :-1]; d[:, :-1] |= m[:, 1:]
        m = d
    return m


def _components(mask):
    lab = np.zeros(mask.shape, int)
    comps, n = [], 0
    for r0, c0 in zip(*np.nonzero(mask)):
        if lab[r0, c0]:
            continue
        n += 1
        q, pts = deque([(r0, c0)]), []
        lab[r0, c0] = n
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and not lab[rr, cc]:
                    lab[rr, cc] = n
                    q.append((rr, cc))
        comps.append(np.array(pts))
    return comps


def outline_interior(img, colour):
    """Boolean mask (full frame) of the area enclosed by this colour's outline,
    or None if that outline isn't in the frame."""
    r0, r1, c0, c1 = PLOT
    sub = img[r0:r1, c0:c1].astype(int)
    m = np.abs(sub - np.array(colour)).sum(-1) < TOL
    if m.sum() < 40:
        return None
    m = _dilate(m, 1)
    rows, cols = np.nonzero(m)
    a, b, c, d = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    wall = np.pad(m[a:b, c:d], 1)
    outside = np.zeros_like(wall)
    outside[0, :] = outside[-1, :] = outside[:, 0] = outside[:, -1] = True
    outside &= ~wall
    while True:   # flood fill from the crop border
        grown = _dilate(outside, 1) & ~wall
        if grown.sum() == outside.sum():
            break
        outside = grown
    holes = ~wall & ~outside
    comps = [p for p in _components(holes) if len(p) > 300]
    if not comps:
        return None
    best = max(comps, key=len)
    inner = np.zeros(img.shape[:2], bool)
    inner[best[:, 0] - 1 + a + r0, best[:, 1] - 1 + c + c0] = True
    return _dilate(inner, 2)        # out to the middle of the drawn line


def main():
    if not FRAMES:
        raise SystemExit("No frames found. Run:  ffmpeg -i Vid.mp4 -vsync 0 frames/f%02d.png")
    masks, rows = {}, []
    for n, f in enumerate(FRAMES, start=1):
        img = (mpimg.imread(f)[:, :, :3] * 255).astype(np.int16)
        for sig, col in COLOURS.items():
            inner = outline_interior(img, col)
            if inner is None:
                continue
            masks[f"{n:02d}_{sig}"] = np.packbits(inner)
            rr, cc = np.nonzero(inner)
            x0, y1 = px_to_km(cc.min(), rr.min())
            x1, y0 = px_to_km(cc.max(), rr.max())
            rows.append(f"{n},{sig},{x0:.2f},{x1:.2f},{y0:.2f},{y1:.2f},{inner.sum() / (X_PX_PER_KM * Y_PX_PER_KM):.1f}")
        print("frame", n, "done")
    np.savez_compressed(config.DATA_DIR / "outlines.npz", shape=np.array(img.shape[:2]), **masks)
    (config.DATA_DIR / "outline_boxes.csv").write_text(
        "frame,signature,x_min_km,x_max_km,y_min_km,y_max_km,area_km2\n" + "\n".join(rows) + "\n")
    print(len(rows), "outlines saved")


if __name__ == "__main__":
    main()
