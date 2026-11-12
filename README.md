# How often does nature fake a seeding signature?

An independent, blind check of Rainmaker's radar attributions using public NEXRAD data.

Rainmaker's 2026 paper ([Satrio et al., *Atmosphere* 17, 1093](https://www.mdpi.com/2073-4433/17/11/1093)) attributes radar "seeding signatures" to its drone releases and argues a natural coincidence is very unlikely, but does not put a number on it. This repo turns the paper's attribution criteria into an objective detector, with every threshold fixed in advance from the paper's 71-signature catalogue, and runs it over every radar scan for two cases. It counts how often the weather produces a matching feature on its own, then asks how likely the matched releases are by chance.

**Full write-up:** [`Tubridy_Rainmaker_nature_fakes.pdf`](Tubridy_Rainmaker_nature_fakes.pdf)

![How often nature fakes a seeding signature](figures/figN1_how_often.png)

## Results

| | Pendleton PDT-025 (clear air) | Kenai, 23 Aug 2026 (inside natural rain) |
|---|---|---|
| Radar, distance band | KPDT, 22–65 km | PAHG, 95–140 km |
| Hours analysed | 12.8 (7.3 before seeding) | 4.9 |
| Natural lookalikes outside the corridors | 0 | 42 |
| Natural lookalikes per downwind corridor | none (at most once per 62 h, 95% bound) | about once per 3.3 h (0.5–4.5 h across settings) |
| Releases matched | 4 of 4 without the growth test, 2 of 4 with it | 6 of 7 |
| Chance the matches are coincidence | under 0.001% for 4 of 4 | 0.03% (0.005–21% across settings) |

In clear air the attribution is effectively certain. Inside natural rain a single faint signature is weak evidence on its own, and the strength of an attribution depends on the day's weather. The write-up suggests reporting a chance-of-coincidence figure per case from the base rate measured that day.

## Detector rules (all from the paper)

| Test | Rule |
|---|---|
| Brighter than surroundings | ≥ 5 dB above the 10 km block median, and ≥ 0 dBZ |
| Clutter filter | correlation coefficient ≥ 0.70 |
| Size | 2–170 km² in every scan |
| Persistence | ≥ 4 scans (3 and 5 tested), ≤ 145 min |
| Lifecycle | grows then fades (also tested without) |
| Drift | within 30° of the plume direction, 5–110 km/h |
| Matching | starts 5–60 min after a release, inside a 45 km, ±30° downwind wedge; one feature per release |

## Running it

```bash
conda create -n rain -c conda-forge python=3.11 arm_pyart boto3 numpy pandas matplotlib
conda activate rain

# 1. Download the public radar data (NOAA NEXRAD Level II on AWS, no account needed)
python fetch_data.py                                   # Kenai: PAHG, 23 Aug 2026 06:00–11:00 UTC -> radar_files/
python fetch_data.py --site KPDT --start 2026-01-23T14:50 --end 2026-01-24T04:05 --dest radar_files_PDT025

# 2. Run the detector for each case (first run grids every scan and caches it)
python nature_fakes.py kenai
python nature_fakes.py pdt025

# 3. Figures
python nature_fakes_figures.py
```

Outputs land in `results/`. The `nf_summary_*.csv` files there are the numbers in the write-up, one row per rule setting.

## Supporting check: the 5 dBZ background step

`run_analysis.py`, `measure.py` and `make_figures.py` check the background subtraction in Rainmaker's Kenai report, using the exact signature outlines traced from Rainmaker's public radar animation (`extract_outlines.py`, results in `data/outlines.npz`). The fixed 5 dBZ subtraction gives 3–22% more water than subtracting the natural precipitation measured beside each signature (ratio 1.03–1.22 depending on tilt and comparison area), implying roughly 47–56 acre-feet against the reported 57.6. `test_synthetic.py` checks this pipeline on synthetic data.

## Files

| File | What it does |
|---|---|
| `lvl2.py` | Reads NEXRAD Level II files directly (reflectivity and correlation coefficient) |
| `lookalikes.py` | Gridding, local background, object finding and tracking |
| `cases.py` | The two cases: release times, sites, plume directions, shared criteria |
| `nature_fakes.py` | Base rates, corridor matching and the chance of coincidence |
| `nature_fakes_figures.py` | Figures for the write-up |
| `fetch_data.py` | Downloads radar files from NOAA's public bucket |
| `data/` | Release windows and signature outlines read from Rainmaker's published report and video |

## Limitations

- Two cases only, one tilt, no vertical or satellite evidence.
- Plume directions and release sites are read from the paper's figures and report text, so they are approximate.
- The rules are simpler than the paper's human protocol, so the numbers describe these objective rules, not Rainmaker's own judgement.

## Sources

- Satrio, M. et al. (2026). Radar Observed Cloud Seeding Signatures from UAS Based Glaciogenic Seeding. *Atmosphere* 17, 1093, and its supplementary tables.
- Rainmaker Research Team. Validation Report: Drone Based, Glaciogenic Cloud Seeding in the Kenai Peninsula.
- NOAA NEXRAD Level II data, KPDT and PAHG, via the NOAA Open Data Dissemination program on AWS.

Independent work by Senan Tubridy. Not affiliated with Rainmaker.
