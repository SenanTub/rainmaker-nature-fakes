"""Comparison figures for Kenai vs Pendleton (PDT-025)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config, style
style.apply()
OUT = config.RESULTS_DIR
LABEL = {"kenai": "Kenai, 23 Aug 2026 · 7 releases inside natural rain, 100–140 km from the radar",
         "pdt025": "Pendleton PDT-025, 23 Jan 2026 · 4 releases in clear air, 25–60 km from the radar"}
CASES = ["pdt025", "kenai"]
d = pd.concat([pd.read_csv(OUT / f"nf_summary_{c}.csv") for c in CASES])
d["per_corridor_h"] = d.rate_per_1000km2_h * d.corridor_km2 / 1000
d["per_corridor_h_hi"] = d.rate_upper95 * d.corridor_km2 / 1000
main = (d.min_scans == 4) & (d.growth) & (d.local_km.isna())

# --- Figure N1: how often nature fakes it, per corridor-hour
fig, ax = plt.subplots(figsize=(8.5, 3.8))
for i, c in enumerate(CASES):
    g = d[d.case == c]
    m = g[main.loc[g.index] if False else ((g.min_scans == 4) & g.growth & g.local_km.isna())].iloc[0]
    lo, hi = g.per_corridor_h.min(), max(g.per_corridor_h.max(), g.per_corridor_h_hi.max())
    ax.plot([lo, hi], [i, i], color=style.MUTED, lw=2, solid_capstyle="round")
    ax.plot(m.per_corridor_h, i, "o", ms=10, color=style.SERIES[i], mec=style.SURFACE, mew=2, zorder=3)
    if m.per_corridor_h == 0:
        txt = f"none found in 12.8 h\n(95% upper bound: once every {1 / m.per_corridor_h_hi:.0f} h)"
    else:
        txt = f"about once every {1 / m.per_corridor_h:.1f} h\n(across rule settings: every {1 / hi:.1f}–{1 / max(lo, 1e-9):.1f} h)"
    ax.text(hi + 0.04, i, txt, va="center", fontsize=8.5, color=style.INK_2)
    ax.text(0, i + 0.32, LABEL[c], fontsize=9.5, color=style.INK, fontweight="bold")
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlim(0, 3.2)
ax.set_ylim(-0.5, 1.75)
ax.set_xlabel("Natural seeding lookalikes per hour inside one downwind corridor")
ax.set_title("How often does nature fake a seeding signature?")
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(OUT / "figN1_how_often.png", dpi=200); plt.close(fig)

# --- Figure N2: chance the run of matches is coincidence, across rule settings
fig, ax = plt.subplots(figsize=(8.5, 4.6))
floor = 1e-6
for i, c in enumerate(CASES):
    g = d[d.case == c].copy()
    y = i + np.linspace(-0.25, 0.25, len(g))
    p = g.p_matched_by_chance.clip(lower=floor)
    hi = g.p_matched_upper95.clip(lower=floor)
    ax.hlines(y, p, hi, color=style.MUTED, lw=1)
    ax.plot(p, y, "o", ms=6, color=style.SERIES[i], mec=style.SURFACE, mew=1)
    for yy, (_, r) in zip(y, g.iterrows()):
        if r.min_scans == 4 and r.growth and pd.isna(r.local_km):
            ax.plot(max(r.p_matched_by_chance, floor), yy, "o", ms=11, color=style.SERIES[i], mec=style.INK, mew=1.2, zorder=4)
ax.set_xscale("log"); ax.set_xlim(floor * 0.5, 1)
ax.axvline(0.05, color=style.AXIS, ls="--", lw=1)
ax.text(0.055, 1.55, "5 percent", fontsize=8, color=style.MUTED)
for i, c in enumerate(CASES):
    ax.text(floor * 0.55, i + 0.36, LABEL[c].split(" · ")[0], fontsize=9.5, color=style.INK, fontweight="bold")
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_ylim(-0.75, 1.75)
ax.set_xlabel("Chance that natural lookalikes alone explain the matched releases (one per release)")
ax.set_title("Could natural lookalikes explain the matches?")
ax.text(floor * 0.55, -0.68, "Each dot: one rule setting. Line: to the 95% upper bound. Ringed: main setting (4 scans, growth, whole ring).\n"
        "Pendleton dots at 1e-6: no chance matches in 200,000 simulated days.", fontsize=7.5, color=style.MUTED)
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(OUT / "figN2_chance.png", dpi=200); plt.close(fig)
print(d[main][["case", "control_lookalikes", "per_corridor_h", "per_corridor_h_hi", "p_per_release",
               "matched", "releases", "p_matched_by_chance", "p_matched_upper95"]].to_string(index=False))


# --- Figure N3: Kenai map under the main rules
import pickle
import lookalikes as L
from cases import CASES as CS, SHARED
from nature_fakes import stage2
from outlines import OutlineSignatures

case = CS["kenai"]; L.configure(case)
scans = pickle.load(open(OUT / "nf_objects_kenai.pkl", "rb"))
_, looks, _ = stage2("kenai", case, scans, 4, None, True)
sigs = OutlineSignatures()
times = [s["time"] for s in scans]
frame_of = {s["time"]: i + 1 for i, s in enumerate(scans)}


def sig_of(tr):
    best = (None, 0)
    for sg in "ABCDEFG":
        hits = 0
        for k, o in tr:
            reg = sigs.regions.get((k + 1, sg))
            if reg is not None:
                px = o["pix"]
                hits += reg.mask(L.GX[px[:, 0], px[:, 1]], L.GY[px[:, 0], px[:, 1]]).mean() > 0.3
        if hits > best[1]:
            best = (sg, hits)
    return best[0] if best[1] >= 2 else None


fig, ax = plt.subplots(figsize=(8.2, 6.8))
th = np.linspace(0, 2 * np.pi, 400)
for r in case["band_km"]:
    ax.plot(r * np.sin(th), r * np.cos(th), color=style.AXIS, lw=1)
for site, (sx, sy) in case["sites"].items():
    a = np.radians(np.linspace(case["flow_toward_deg"] - 30, case["flow_toward_deg"] + 30, 50))
    ax.fill(np.r_[sx, sx + SHARED["wedge_km"] * np.sin(a), sx], np.r_[sy, sy + SHARED["wedge_km"] * np.cos(a), sy],
            color=style.BLUE_LIGHT, alpha=0.6, lw=0, zorder=1)
    ax.plot(sx, sy, "^", color=style.INK, ms=7, zorder=5)
n_sig = n_nat = 0
for tr in looks:
    xy = np.array([[o["x"], o["y"]] for _, o in tr])
    s = sig_of(tr)
    col, lw, z = (style.LINEAR, 2.4, 4) if s else (style.MUTED, 1.4, 3)
    n_sig += bool(s); n_nat += not s
    ax.plot(xy[:, 0], xy[:, 1], "-", color=col, lw=lw, zorder=z)
    ax.plot(xy[0, 0], xy[0, 1], "o", color=col, ms=4.5, mec=style.SURFACE, zorder=z)
ax.annotate("seeding sites and\ndownwind corridors", (11, -99), xytext=(-80, -45), fontsize=8.5, color=style.INK_2,
            arrowprops=dict(arrowstyle="-", color=style.MUTED, lw=0.8))
ax.plot([], [], "-o", color=style.LINEAR, label=f"overlaps a Rainmaker signature ({n_sig})")
ax.plot([], [], "-o", color=style.MUTED, label=f"natural lookalike ({n_nat})")
ax.set_xlim(-145, 145); ax.set_ylim(-145, 145); ax.set_aspect("equal")
ax.set_xlabel("East–west distance from PAHG (km)"); ax.set_ylabel("North–south distance from PAHG (km)")
ax.set_title("Everything that passes the tests at Kenai, 23 Aug 2026")
ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "figN3_kenai_map.png", dpi=200); plt.close(fig)
print("map:", n_sig, "signature tracks,", n_nat, "natural")
