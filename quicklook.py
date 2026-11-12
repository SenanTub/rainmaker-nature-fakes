"""Draw every scan with the streak boxes and next-door boxes on top, so you
can check by eye that the boxes sit on the streaks in Rainmaker's video.

Run:  python quicklook.py            (writes results/quicklook/*.png)
Solid white = streak box, dashed = next door, grey hatched = exclusion zone.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

import config
from boxes import Signatures

XLIM, YLIM = (-25, 25), (-140, -92)


def plot_sweep(sweep, sigs, out_path):
    try:
        import pyart  # noqa: F401  (registers the NWS radar colour map)
        cmap = "NWSRef" if "NWSRef" in matplotlib.colormaps else "pyart_NWSRef"
    except ImportError:
        cmap = "turbo"
    x, y, dbz = sweep.x_km, sweep.y_km, sweep.dbz
    keep = (x >= XLIM[0] - 5) & (x <= XLIM[1] + 5) & (y >= YLIM[0] - 5) & (y <= YLIM[1] + 5)
    rows = np.where(keep.any(axis=1))[0]
    cols = np.where(keep.any(axis=0))[0]
    cols = np.arange(cols.min(), cols.max() + 1)
    # sort rays by azimuth so pcolormesh gets a tidy, monotonic grid
    c_mid = cols[len(cols) // 2]
    az = np.degrees(np.arctan2(x[rows, c_mid], y[rows, c_mid])) % 360
    rows = rows[np.argsort(az)]
    xs, ys, zs = x[np.ix_(rows, cols)], y[np.ix_(rows, cols)], dbz[np.ix_(rows, cols)]
    # explicit cell edges in range/azimuth, then back to x/y
    r_c = np.hypot(xs[len(rows) // 2], ys[len(rows) // 2])
    az_c = np.unwrap(np.arctan2(xs[:, -1], ys[:, -1]))

    def edges(c):
        mid = (c[1:] + c[:-1]) / 2
        return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])

    R_e, AZ_e = np.meshgrid(edges(r_c), edges(az_c))
    X_e, Y_e = R_e * np.sin(AZ_e), R_e * np.cos(AZ_e)

    fig, ax = plt.subplots(figsize=(7.5, 7))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    m = ax.pcolormesh(X_e, Y_e, zs, cmap=cmap, vmin=-20, vmax=70, shading="flat")
    cb = fig.colorbar(m, ax=ax, label="Reflectivity (dBZ)")
    cb.ax.yaxis.label.set_color("white")
    cb.ax.tick_params(colors="white")

    for sig in sigs.names:
        b = sigs.box_at(sig, sweep.time)
        if b is None:
            continue
        ax.add_patch(Rectangle((b.x0, b.y0), b.width, b.y1 - b.y0, fill=False, ec="white", lw=1.6))
        ax.text(b.x1 + 0.3, b.y1, sig, color="white", fontsize=10, fontweight="bold", va="top")
        for fb in sigs.flank_boxes(b):
            ax.add_patch(Rectangle((fb.x0, fb.y0), fb.width, fb.y1 - fb.y0, fill=False,
                                   ec="white", lw=1, ls="--"))
    for t0, t1, b in sigs.exclusions:
        if t0 <= sweep.time <= t1:
            ax.add_patch(Rectangle((b.x0, b.y0), b.width, b.y1 - b.y0, fill=False,
                                   ec="#898781", lw=1, hatch="///"))

    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.set_xlabel("East–west distance from PAHG (km)", color="white")
    ax.set_ylabel("North–south distance from PAHG (km)", color="white")
    ax.tick_params(colors="white")
    ax.set_title(f"PAHG {sweep.tilt_deg:.2f}°  {sweep.time:%Y-%m-%d %H:%M:%S} UTC", color="white", loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor="black")
    plt.close(fig)


def main():
    from radar_io import list_radar_files, read_sweeps
    sigs = Signatures()
    out = config.RESULTS_DIR / "quicklook"
    out.mkdir(parents=True, exist_ok=True)
    for path in list_radar_files():
        sweeps = read_sweeps(path, tilts=[0.88])
        if 0.88 in sweeps:
            s = sweeps[0.88]
            plot_sweep(s, sigs, out / f"{s.time:%H%M%S}.png")
            print("drew", s.time.strftime("%H:%M:%S"))
    print("Check these against Rainmaker's video:", out)


if __name__ == "__main__":
    main()
