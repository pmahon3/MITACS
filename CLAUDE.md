# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase for nonlinear dynamics / empirical-dynamic-modelling analysis of Ontario
electricity demand (IESO). It detrends raw demand into a stationary z-score series, finds an
embedding dimension per day-type, fits local-linear (weighted-least-squares) one-step forecasts
across a θ-grid, and explores the residuals / local linear operators in Dash dashboards.

The current layout was established by commit `629948e "changed course"` on the **`operator`**
branch (the default branch is `main`; on `main` the layout differs — it predates this restructure).

## Critical environment fact

`edynamics.modelling_tools` (used by every `processing/` script: `Lag`, `Embedding`,
`WeightedLeastSquares`, `Exponential`, `Minkowski`, `dimensionality`) is **not** declared
anywhere in this repo — there is no `requirements.txt`, `setup.py`, or `pyproject.toml`.
It is an **editable install** pointing at a sibling project:

```
/Users/pmahon/Research/Dynamics/Takens_Whitney/EmpiricalDynamics/src
```

(see `.venv/lib/python3.10/site-packages/__editable__.edynamics-0.3.14.pth`). If imports of
`edynamics` fail, the fix is in that sibling repo / the venv, not here. Despite
`.venv/pyvenv.cfg` claiming Python 3.12, the live `site-packages` is `python3.10/` — treat the
venv as Python 3.10.

Other key deps already in `.venv`: `torch`, `ray`, `dash`, `plotly`, `pandas`, `bs4`, `pyyaml`.

## Configuration (read this before touching paths)

All pipeline paths and parameters live in **`config/pipeline.yaml`**, loaded once via
`from config import load_config`. `config/` resolves every path relative to `PROJECT_ROOT`
(the repo root, derived from `config/__init__.py`'s location) — **nothing depends on the
current working directory**. The loader rejects leading-slash / `..` paths, so the old
root-anchored-path bug class fails loudly instead of silently reading `/`.

This **replaces** the old `.env` + `data/__init__.py:convert_to_absolute_path` mechanism
(which had a `preprocessing` vs `pre_processing` typo and resolved relative to the wrong
directory). `data.convert_to_absolute_path` is dead — do not reintroduce it. `cfg.embedding_dim(daytype)`
is the single home for the `pd.read_csv(...).idxmax()` embedding-dimension idiom.

## Pipeline (run order and the artifact contract)

Stages 2–4 have been refactored to a `main(cfg)` function plus a thin `if __name__` CLI
(`python -m <module>`); they are config-driven and headless-safe (`matplotlib.use("Agg")`).
Stages 5–6 are **not yet refactored** (see "Status" below). Stages communicate through files;
the **filename convention is the contract** (`save_all`/`load_all` discover artifacts by glob).

1. **Acquire** — `data/scraping.py` scrapes IESO `PUB_Demand*.csv`; `data/ontario/pickling.py` → `.pkl`.
2. **Detrend** — `pre_processing/year_wise_standardization.py` → `stationary_candidate.csv`
   (hourly `zscore` = demand de-seasonalized by month+hour-of-day) plus diagnostic plots.
3. **Day-type tagging** — `processing/clustering/process.py` adds `daytype ∈ {weekday, saturday,
   sunday}` using the configurable **07:00 day-anchor offset** (`data.day_anchor_hours`).
4. **Embedding dimension** — `processing/dimensions/process.py` → `params/results_<daytype>.csv`
   (uses `ray`; `ray.init()` is guarded so re-runs don't double-init).
5. **Fit / residuals** — `processing/{innovations, locality/global, locality/pointwise}/process.py`
   write the canonical artifacts: `residuals_<tag>.pt`, `coeffs_<tag>.pt` `(N,d,d)`,
   `times_<tag>.npy` `(N,)` int64 ns, `thetas_<tag>.npy`. `<tag>` = day-type.
6. **Explore** — Dash apps in `post_processing/` read a run dir via `load_all(run_dir)`:
   `python post_processing/innovations/dashboard.py --run-dir <output_dir> [--port 8050]`.

## edynamics WLS API: critical, non-obvious

The loaded `WeightedLeastSquares` (sibling editable install) **requires
`LocalGLSelector.fit(embedding, anchors)` to be called before `.project(...)`** — it sets
`anchor_times`/`best_theta_vals`/`best_sigma_vals`, else `project()` raises `RuntimeError`.
Every legacy `process.py` calls `.project()` directly and would fail against the installed
library. Use `LocalGLSelector` (dual θ/σ grids) — see `processing/innovations/validation/synthetic.py`
for the correct call chain.

**Drift is correct; the library's diffusion is not.** `RoseResult.coefficients` (drift `C`,
`x_next = x@C`, so `C ≈ Aᵀ` for column-convention VAR `x_{t+1}=Ax_t`) is reliable.
`RoseResult.covariances` is the covariance of the *weighted* residual `Wy−WX@C` ≈ `w²·Q`, which
collapses to ~1e-9 under wide kernels — **it does not estimate the diffusion**. Diffusion must be
recomputed from **raw** residuals `Y−X@C` with only the residual kernel applied (see
`raw_residual_diffusion` in the validation module — this is the validated estimator core).

## Validation gate

`processing/innovations/validation/synthetic.py` recovers known drift `A` and diffusion `Q`
from a VAR(1) ground-truth process through the real call chain.
`python -m processing.innovations.validation.synthetic` must print `RESULT: PASS` before any
Ontario output — or any pre-existing `.pt` artifact — is trusted (existing artifacts may be from
a now-shadowed older library version). It is what caught the `Σ ≈ w²·Q` bug above.

## Status (what is done vs. in progress, as of 2026-05-17)

- ✅ `config/` package; stages 2–4 config-driven (clustering verified end-to-end on real data).
- ✅ VAR(1) validation gate built and passing (drift 4.6%, diffusion 2.6%, eig 0.9%).
- ✅ `innovations` stage rebuilt as the local Gaussian semigroup (`Pi_Delta`) estimator
  (`estimator.py` + `spectral.py` + `interface.py` + config-driven `process.py`); verified
  end-to-end on real Ontario data. Operator naming defers to the `Resolvent_Framework`
  programme (no novelty claims; lag-selection stopping rule flagged `[DEFER-RF]`).
- ⚠️ `processing/locality/{pointwise,global}/process.py` still **crash** on the WLS API change
  (direct `.project()` without `LocalGLSelector.fit`). These are the *old* θ-sweep / manual
  θ-selection workflow; their fate (refactor as a `LocalGLSelector` diagnostic vs. delete as
  superseded by the auto-(θ*,σ*) estimator) is deferred until after the Ontario end-to-end run.
- ⚠️ Cost: `LocalGLSelector.fit` is a triple loop (anchor × θ × σ). At full Ontario weekday
  scale (~44k library times, `sample_frac=0.8`, 25×25 grid) that is ~22M `lstsq` calls. Use
  small grids / low `sample_frac` for first runs; the runner logs anchor/grid sizes.
- ✅ `run_pipeline.py`: config-driven orchestration (stage registry, skip-if-exists,
  `--from/--to/--only/--daytype/--profile/--dry-run/--force`). Run profiles live in
  `pipeline.yaml` (`fast` = cheap validation pass, `full` = production); `--profile`
  overrides. **Default is `fast`** — switch to `full` for production-scale runs.
- ✅ End-to-end verified on real Ontario data (`fast` profile, all 3 day-types, ~82s):
  100% finite drift, all diffusion SPD, interpretable spectra (weekday/saturday d=2
  single dominant mode; sunday d=7, r_hat=3). The expensive `full` run has not been
  executed yet.
- `scratch/` and `processing/innovations/scratch.py` are exploratory; not part of the pipeline.

## Data files

CSV and `.pkl` demand files under `data/` are tracked (large; LFS was removed in `db8ff5b`).
`.gitignore` excludes Python bytecode/caches (`__pycache__/`, `*.pyc`, `.pytest_cache/`) and a
couple of legacy `src/main/resources/...` climate paths no longer in this layout.
