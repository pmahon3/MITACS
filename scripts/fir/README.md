# Fir submission: W=2y climatology re-score

Re-runs predictor 2 (mean-iteration, global-OLS θ=0) over the full
500-day post-cutoff backtest with `(μ_{m,h}, σ_{m,h})` refit on the
last 2 years of pre-cutoff data, rather than the full 2003–2024
window baked into the registered spec.

Selected by `scratch/climatology_window.py` (held-out NLL + AIC +
residual diurnal-z amplitude all argmin at W=2y).

**PROVENANCE-GRADE: INSPECTION-ONLY.** This deviates from the
registered spec; the resulting pickle is a *candidate v2 predictor*
output, not a registered claim. Landing it as a registered v2 spec
would require a fresh `experiment.freeze` registration.

## Files

- `rescore_climatology.py` — the job script. Single-CPU; reads the
  spec for embedding dims + cutoff, refits the climatology on the
  chosen window, and runs predictor 2's mean-iteration over every
  post-cutoff delivery day. Outputs a pickle in the same schema as
  `scratch/data/smc_smap_samples/cell_B_meaniter_global.pkl`.

- `submit_rescore.sh` — SLURM submission for Fir (DRA conventions).
  15-min wall-clock, 1 CPU, 4 GB. Edit the `--account=` line to your
  RAPI before submitting.

## Local smoke test (run this first)

```bash
.venv/bin/python -m scripts.fir.rescore_climatology \
    --window-years 2 \
    --out scratch/data/rescore_smoke/cell_B_W2y.pkl \
    --max-days 10
```

Expected: ~3 s wall-clock (post-optimization), MAE around 607 MW
over the 10 days, prints headline numbers and writes the pickle.
(The first-10-days 2025-01-01..2025-01-10 sample is winter —
magnitudes there don't match the full-500-day average and shouldn't
be over-read.)

## Submitting on Fir

1. Ensure the repo is checked out and the `.venv` exists with the
   project's pip set:
   ```bash
   cd ~/Research/Dynamics/MITACS
   module load python/3.10
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip wheel
   python -m pip install edynamics==0.4.0 \
                          pandas scipy matplotlib seaborn pyyaml \
                          bs4 dash plotly
   ```
   `edynamics==0.4.0` is on PyPI; the install transitively pulls
   `numpy`, `scipy`, `tqdm`, `ray`, and `torch`.  No need to clone
   the EmpiricalDynamics sibling repo or do an editable install.

2. Verify the data cache is present (`data/ontario/*.pkl`); the
   `load_actuals` call reads from there. If absent, scrape via
   `python -m data.scraping`.

3. Edit `submit_rescore.sh`:
   - Set `--account=def-<your-RAPI>` (currently `def-CHANGE_ME`).

4. Submit:
   ```bash
   sbatch scripts/fir/submit_rescore.sh
   ```

5. Wait for completion. Outputs:
   - `scratch/data/rescore_W2y/cell_B_meaniter_global_W2y.pkl`
   - `logs/rescore_W2y_<jobid>.out`
   - `logs/rescore_W2y_<jobid>.err`

## After the run completes

Compare to the existing predictor-2 pickle:

```python
import pandas as pd
b_orig = pd.read_pickle('scratch/data/smc_smap_samples/cell_B_meaniter_global.pkl')
b_w2   = pd.read_pickle('scratch/data/rescore_W2y/cell_B_meaniter_global_W2y.pkl')
for label, df in [('orig (full-history)', b_orig), ('W=2y', b_w2)]:
    err = (df['our_forecast_mw'] - df['actual_mw']).abs()
    print(f'{label:>22}  MAE={err.mean():7.1f} MW   n={len(df)}')
```

Then re-run `scratch/bias_diagnostics.py` pointing at the new pickle
to produce the signed-error-by-hour comparison.
