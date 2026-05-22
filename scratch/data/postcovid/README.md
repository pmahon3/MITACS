# Post-COVID competition data (not version-controlled)

The CSV files in this directory are **gitignored** — they are third-party
data whose redistribution is governed by IEEE DataPort terms. This README
is the reproduction recipe; the probe scripts that consume the data
(`scratch/postcovid_estimator_probe.py`,
`scratch/postcovid_competition_probe.py`) **are** tracked.

## Source

IEEE DataPort *Post-COVID Electricity Load Forecasting* competition
(IEEE document 9739082, <https://ieeexplore.ieee.org/document/9739082>).
The load series and dated batch files were obtained from the public
companion repository:

- GitLab: `JosephdeVilmarest/state-space-post-covid-forecasting`, branch
  `main`, folder `data_raw/` (LGPL-3.0; the IEEE DataPort dataset itself
  carries the competition's own terms).

## Files and how they were assembled

- **`Actuals_Interim.csv`** — downloaded verbatim from `data_raw/`;
  hourly load 2020-11-06 .. 2021-01-15.
- **`Actuals_full_postcovid.csv`** — a contiguous hourly load series
  2020-11-06 .. 2021-02-16, stitched from `Actuals_Interim.csv` plus the
  dated daily batch files `Actuals_<date>_8AM.csv` and `Actuals_Feb 16.csv`
  in `data_raw/`. Each dated `*_8AM.csv` batch covers
  `[day-1 08:00, day 07:00]`; they tile with no gap. Stitching:
  concatenate, `drop_duplicates` on `Time`, sort, `asfreq("h")`. The
  result has 2472 rows and zero missing values. Only the `Time` and
  `Load (kW)` columns are retained.

Access date: 2026-05-21.

## Scope

These probes are **exploratory / inspection-only** (each script carries a
`PROVENANCE-GRADE: INSPECTION-ONLY` banner). They are a curiosity check of
the estimator core on an independent dataset and are **not** part of the
Ontario pipeline, the registered experiment, or the paper.
