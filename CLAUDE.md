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

## Estimator design (current, post-rebuild)

The `innovations` stage estimates, per anchor, a local linear drift `C_j` and a
diffusion `Σ_j` defining the Gaussian Markov kernel `x' | x ~ N(x C_j, Σ_j)` on the
delay-embedding state space — an estimator of the programme's conditional-regularity
kernel `κ_Q` (see memory `mitacs-theory-correspondence` for the precise mapping and
its qualifiers; operator naming defers to `Resolvent_Framework`, no novelty claims).

Key facts (each was hard-won this session; do not "simplify" without reading the memory):

- **Bandwidth selection is true-LOO cross-validation**, not the edynamics
  `LocalGLSelector`. The GL criterion is degenerate for the normalized Gaussian kernel
  (no interior optimum — score monotone in bandwidth); the `Resolvent_Framework`
  programme has no bandwidth convention to defer to. `_theta_loo_cv` in `estimator.py`
  is the validated rule. (memory `mitacs-theta-rail-pinning`)
- **Diffusion is the plain μ-centred residual covariance** (`Y−X@C_j`), no residual
  kernel. The edynamics kernel-weighted covariance is `≈ w²·Q` and collapses; the
  plain covariance is the unbiased local second moment.
- **`Σ_j` is rank-1 by construction** of a single-variable delay embedding (coords
  2..d of the one-step image are deterministic shifts of the input). The only
  stochastic content is the scalar coordinate-0 innovation. `r_hat`/"multi-mode" is
  therefore incoherent for this embedding and is not reported. (memory
  `mitacs-rank1-structural`)
- **`day_anchor_hour`**: transitions whose one-step target lands on the day-anchor
  hour cross the day-type rollover seam and must be excluded — without this, `Σ_j` is
  catastrophically ill-conditioned. `process.py` passes `cfg.data.day_anchor_hours`.

## Validation gates (both must PASS — `python -m processing.innovations.validation.synthetic`)

Both gates call the **production** estimator directly (not reimplementations) — that
is why they are trustworthy:

1. **Recovery gate**: VAR(1) with known `(A, Q)` → `build_local_gaussian_semigroup`
   must recover drift/diffusion within tolerance.
2. **Non-Gaussianity gate**: Gaussian-innovation VAR(1) must read ≈Gaussian;
   Student-t-innovation VAR(1) must be clearly flagged heavy-tailed — validating
   `innovation_diagnostics` against ground truth.

`processing/innovations/validation/rebaseline.py` is the authoritative
production-path factual record (`mitacs-rebaseline-facts`).

## Discipline rule (the durable lesson of this session)

**A diagnostic that reimplements production logic is a hypothesis, not a finding,
until confirmed through the production code path.** This session, ≥3 confident Ontario
claims (sunday multi-mode; "mild"/"10×" non-Gaussianity; a 4-decimal "match") were all
reimplemented-diagnostic artifacts that inverted when finally run through production
functions. The gates never misled because they call production directly. Any Ontario
diagnostic must call production functions or its output is labelled provisional and
not committed as fact.

## Result provenance (Tier-1, implemented 2026-05-18)

Every claim-grade / method-grade result artifact MUST be written through
`experiment.provenance.make_result()` — never hand-written. It stamps a header with
two distinct, independently-verifiable hashes: `inputs_fingerprint`
(git SHA + frozen_spec hash + lib versions + RNG seeds + declared inputs =
"rerun these, get this") and `body_sha256` (tamper-evidence; `load_verified_result()`
recomputes it on read). Policy is **strict refuse-on-dirty** (same bar as `freeze.py`;
a recorded git SHA that can't reproduce the artifact is the hole this closes) and a
mandatory `Grade` banner (CLAIM-GRADE / METHOD/DESIGN / INSPECTION-ONLY). Predictor-
derived CLAIM results pass `frozen_spec_required=True` so a missing/tampered spec
raises rather than stamping `null`. Shared primitives live in `experiment/_prov_core.py`
(ONE definition; `freeze.py` consumes it — do not copy-paste git/hash helpers). The
emitters are `experiment/backtest.py --emit-result` (C1), `processing/innovations/
validation/synthetic.py --emit-result` (C4), `experiment/emit_theory_correspondence.py`
(C3), `experiment/emit_ieso_infeasibility.py` (C2). The requirements analysis driving
this is `experiment/results/PROVENANCE_REQUIREMENTS.md` (memory
`mitacs-provenance-requirements`); `freeze.py`/the registered experiment is the C5
gold-standard template the rest was raised to. Inspection-only scratch carries a
greppable `PROVENANCE-GRADE: INSPECTION-ONLY` docstring banner — never cite it as a
result.

## Status (as of 2026-05-17)

- ✅ Config-driven pipeline (`config/`, `run_pipeline.py` with stage registry,
  skip-if-exists, `--from/--to/--only/--daytype/--profile/--dry-run/--force`; `fast`/
  `full` profiles in `pipeline.yaml`). Stages 2–6 config-driven.
- ✅ Estimator rebuilt and **both validation gates pass** (recovery + non-Gaussianity).
- ✅ **Authoritative empirical facts** established via production-path re-baseline
  (`mitacs-rebaseline-facts` — trust this file for Ontario facts; all earlier Ontario
  claims are superseded):
  - **The intra-day conditioning set is required**: full-process `Σ_j` is numerically
    unusable (cond up to ~1e20, sunday singular) due to the day-anchor seam; intra-day
    gives cond ~6–1072. The estimand is effectively forced to intra-day.
  - **The scalar one-step innovation is strongly non-Gaussian** (gate-validated
    excess kurtosis ≈24–33, tail ratio ≈2.4–3.8; clean VAR(1) reference reads ≈0/≈1).
    `Σ_j` is a Gaussian second-moment *proxy* of a heavy-tailed law — a mandatory
    writeup caveat.
- 🗄️ `processing/locality/` (old θ-sweep, broken on the WLS API) removed; archived at
  `archive/legacy_locality_sweep.zip` (gitignored; recoverable from git history).
- ⏳ Not yet started: IESO published-forecast comparison + web dashboard; the
  publication writeup. A future production-scale (`full` profile) run is a separate
  question — the re-baseline is the current trustworthy factual record.
- `scratch/` and `processing/innovations/scratch.py` are exploratory; not pipeline.
- `processing/innovations/validation/bandwidth_comparison.py` is a superseded
  investigation record (its conclusion is implemented); kept runnable for trace.

## Data files

CSV and `.pkl` demand files under `data/` are tracked (large; LFS was removed in `db8ff5b`).
`.gitignore` excludes Python bytecode/caches (`__pycache__/`, `*.pyc`, `.pytest_cache/`) and a
couple of legacy `src/main/resources/...` climate paths no longer in this layout.
