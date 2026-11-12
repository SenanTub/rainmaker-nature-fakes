"""Make the write-up figures from results/*.csv.

Run:  python make_figures.py              (after run_analysis.py)
      python make_figures.py --curve-only (just the maths curve, no data needed)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import style
from measure import bias_ratio, break_even_db

style.apply()


def fig_bias_curve(results_dir, out_path, with_points=True, tilt=0.48):
    d = np.linspace(2.5, 20, 400)
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.axvspan(5, 10, color=style.NEUTRAL_BAND, zorder=0)
    ax.text(7.5, 2.62, "Zaremba et al.: a seeding streak must be\n~5–10 dB brighter to be visible",
            ha="center", va="top", fontsize=8.5, color=style.INK_2)
    ax.axhline(1.0, color=style.AXIS, lw=1.2, zorder=1)
    ax.text(19.8, 1.03, "estimate = true new water", ha="right", va="bottom", fontsize=8.5, color=style.MUTED)

    lo, hi = bias_ratio(d, 2.2), bias_ratio(d, 2.0)
    ax.fill_between(d, np.minimum(lo, hi), np.maximum(lo, hi), color=style.BLUE_LIGHT, lw=0, zorder=2,
                    label="b = 2.0 to 2.2")
    ax.plot(d, bias_ratio(d, 2.1), color=style.THEIRS, zorder=3, label="5 dB method, b = 2.1")
    be = break_even_db(2.1)
    ax.plot([be], [1.0], "o", ms=6, color=style.THEIRS, mec=style.SURFACE, mew=2, zorder=4)
    ax.annotate(f"break-even ≈ {be:.1f} dB", (be, 1.0), xytext=(be - 4.2, 0.6),
                fontsize=8.5, color=style.INK_2, arrowprops=dict(arrowstyle="-", color=style.MUTED, lw=0.8))

    if with_points:
        summ = pd.read_csv(Path(results_dir) / "summary.csv")
        life = pd.read_csv(Path(results_dir) / "contrast_by_signature.csv")
        sel = summ[(summ.scans == "unflagged") & (summ.inflate_km == 0) & (summ.b == config.B_REFERENCE)
                   & (summ.a == 300) & (summ.tilt == tilt) & (summ.signature != "ALL")]
        pts = sel.merge(life[life.tilt == tilt], on=["signature", "tilt"])
        ax.plot(pts.contrast_db_weighted, pts.theirs_over_linear, "o", ms=8, color=style.LINEAR,
                mec=style.SURFACE, mew=2, zorder=5, label="measured streaks (A–G)")
        for i, r in enumerate(pts.sort_values("contrast_db_weighted").itertuples()):
            ax.annotate(r.signature, (r.contrast_db_weighted, r.theirs_over_linear),
                        xytext=(6, 4) if i % 2 == 0 else (6, -13),
                        textcoords="offset points", fontsize=9, color=style.INK)

    ax.set_xlim(2.5, 20)
    ax.set_ylim(0.5, 2.7)
    ax.set_xlabel("How much brighter the streak is than next door (dB)")
    ax.set_ylabel("Rainmaker estimate ÷ true new water")
    ax.set_title("Is the 5 dB adjustment too high or too low? It depends on the contrast")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_outline_size(results_dir, out_path, tilt=0.48):
    summ = pd.read_csv(Path(results_dir) / "summary.csv")
    s = summ[(summ.scans == "unflagged") & (summ.signature == "ALL") & (summ.tilt == tilt)
             & (summ.b == config.B_REFERENCE) & (summ.a == 300)].sort_values("inflate_km")
    base = s.loc[s.inflate_km == 0]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for col, colour, label in (("theirs_m3", style.THEIRS, "5 dB method"),
                               ("linear_m3", style.LINEAR, "streak minus next door")):
        y = s[col].values / float(base[col].iloc[0])
        ax.plot(s.inflate_km, y, "-o", color=colour, ms=7, mec=style.SURFACE, mew=2, label=label)
        ax.annotate(f"{label}\n×{y[-1]:.2f}", (s.inflate_km.iloc[-1], y[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=style.INK)
    ax.axhline(1.0, color=style.AXIS, lw=1.2)
    ax.set_xticks(config.INFLATE_KM)
    ax.set_xlim(-0.3, max(config.INFLATE_KM) + 2.2)
    ax.set_xlabel("Outline grown on every side (km)")
    ax.set_ylabel("New water vs original outline")
    ax.set_title("Bigger outline: which method keeps adding water?")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_contrast_life(results_dir, out_path, tilt=0.48):
    st = pd.read_csv(Path(results_dir) / "per_scan.csv", parse_dates=["time_utc"])
    win = pd.read_csv(config.DATA_DIR / "windows.csv", parse_dates=["first_detected_utc"])
    st = st[(st.tilt == tilt) & ~st.flag].merge(win[["signature", "first_detected_utc"]], on="signature")
    first = st.first_detected_utc.dt.tz_localize("UTC") if st.first_detected_utc.dt.tz is None else st.first_detected_utc
    st["minutes"] = (st.time_utc - first).dt.total_seconds() / 60
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axhspan(5, 10, color=style.NEUTRAL_BAND, zorder=0)
    for i, (sig, g) in enumerate(sorted(st.groupby("signature"))):
        g = g.sort_values("minutes")
        ax.plot(g.minutes, g.contrast_db, "-o", ms=4, color=style.SERIES[i % 8], label=sig)
    ax.set_xlabel("Minutes since first detection (report times)")
    ax.set_ylabel("Streak minus next door (dB)")
    ax.set_title("How bright each streak is over its life")
    ax.legend(ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(config.RESULTS_DIR))
    p.add_argument("--curve-only", action="store_true")
    p.add_argument("--tilt", type=float, default=0.48)
    args = p.parse_args()
    out = Path(args.results)
    out.mkdir(exist_ok=True)
    if args.curve_only:
        fig_bias_curve(out, out / "fig1_bias_curve.png", with_points=False)
    else:
        fig_bias_curve(out, out / "fig1_bias_curve.png", tilt=args.tilt)
        fig_outline_size(out, out / "fig2_outline_size.png", tilt=args.tilt)
        fig_contrast_life(out, out / "fig3_contrast_over_life.png", tilt=args.tilt)
    print("Figures written to", out)


if __name__ == "__main__":
    main()
