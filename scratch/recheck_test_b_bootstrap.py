"""#41-STANDARD RE-CHECK of the honing memo's TEST A and TEST B.

WHY THIS EXISTS
---------------
The honing memo (mitacs-honing-methodology) recorded "STEP-3 RESULT: BOTH
HYPOTHESES CONFIRMED", incl. TEST B = "optimal d varies 5->8 across hours;
current per-day-type dims FAR BELOW => systematic under-embedding".

#41 then resolved (scratch/diagnose_theta_FINAL_summary.py): every prior
"interior optimum" claim in this project was an artifact of reading an
ARGMIN POSITION WITHOUT A VARIANCE CHECK. TEST B's machinery is the SAME
pattern: scratch/diagnose_embedding_adequacy.py:_elbow() returns the first
d whose SINGLE point-estimate rho is within 0.001 of the SINGLE max rho.
No SE, no replication, no test that d=5-8 is statistically separated from
the current d. Per the #41 discipline, TEST B is SUSPECT until re-derived
under a variance/replication check. This script is that re-derivation.

TWO DEFECTS in the original TEST B (both fixed here):
  D1 (the one #41 names): argmin/elbow with no variance check.
  D2 (advisor-caught): the original rho is IN-SAMPLE fit
      (C = lstsq(X,Y); corrcoef(X@C, Y) on the SAME X,Y). In-sample rho
      rises monotonically with d to numerical saturation -> the "5->8
      elbow" is the same shape-of-objective degeneracy #41 found for
      prediction-error-vs-bandwidth. The honest metric for a FORECASTING
      choice is OUT-OF-SAMPLE rho. This script uses held-out-fold OOS rho.

PRE-REGISTERED DECISION RULE (fixed BEFORE any numbers were seen; this is
the #41 discipline -- the rule is committed, not fit post-hoc):
  * Statistic, per hour h: Delta_rho(h) = rho_OOS(d_best,h) - rho_OOS(d_cur,h)
    where d_cur = per-day-type d IN FORCE at hour h (weekday=2/sat=4/sun=3),
    d_best = the apparent best d on the point estimate (the thing the
    original elbow would have chosen). Bootstrap (X,Y) pairs with
    replacement WITHIN the hour-of-day stratum (obs 24h apart => iid
    bootstrap defensible, no block scheme), B=1000 reps, 95% CI.
  * TEST B REPLICATES iff for >= 12 of 24 hours the 95% CI LOWER BOUND on
    Delta_rho(h) > 0.01 (absolute rho). Otherwise TEST B is RETRACTED:
    "d wants 5-8" was an argmin/in-sample artifact, NOT a finding; do not
    change d on its basis.
  * TEST A (re-run for the same standard): bootstrap corr(dz/dt, z_err),
    B=1000. Replicates iff 95% CI excludes 0.

SCOPE: this script ONLY re-checks A and B under the #41 standard. It does
NOT re-open #30's option choice. A retracted TEST B does not by itself
trigger a build proposal -- that is reported separately.

PROVENANCE-GRADE: INSPECTION-ONLY -- exploratory dev-set re-derivation;
MUST NOT be cited as a result artifact. Its OUTPUT is a methodology
verdict (replicates / retracted), recorded to the honing memo.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from edynamics.modelling_tools import Embedding, Lag

from experiment._actuals import load_actuals, zscore_params, zscore_transform
from experiment.predict import _daytype

CACHE = Path("/tmp/bench2122_fc.pkl")
CUTOFF = pd.Timestamp("2021-10-18 23:00:00")
ANCHOR_H = 7
DMAX = 8
B = 1000
RNG = np.random.default_rng(20260518)

# per-day-type d IN FORCE (the frozen-style choice TEST B claimed is too low)
D_CURRENT = {"weekday": 2, "saturday": 4, "sunday": 3}

# pre-registered thresholds
SEP_MARGIN = 0.01      # Delta_rho lower-bound must exceed this
N_HOURS_REQ = 12       # ... for at least this many of 24 hours


def _emb_xy(df: pd.DataFrame, idx: pd.DatetimeIndex, d: int):
    """Embedding block -> (X, Y) one-step pairs, exactly as TEST B builds
    them (same Embedding/Lag/library_times path, production code)."""
    fl = idx[d:-1]
    if len(fl) < 120:
        return None
    emb = Embedding(
        data=df,
        observers=[Lag(variable_name="zscore", tau=-i) for i in range(d)],
        library_times=fl,
    )
    emb.compile()
    b = emb.block
    X, Y = b.iloc[:-1].values, b.iloc[1:].values
    if len(X) < 60:
        return None
    return X, Y


def _rho_oos(X: np.ndarray, Y: np.ndarray, n_fold: int = 5) -> float:
    """Out-of-sample coord-0 rho via K-fold (fit on train, score on held
    out). Replaces the original IN-SAMPLE corrcoef(X@C, Y) (defect D2)."""
    n = len(X)
    order = np.arange(n)
    folds = np.array_split(order, n_fold)
    preds = np.full(n, np.nan)
    for k in range(n_fold):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(n_fold) if j != k])
        if len(tr) < X.shape[1] + 5:
            return np.nan
        C = np.linalg.lstsq(X[tr], Y[tr], rcond=None)[0]
        preds[te] = (X[te] @ C)[:, 0]
    m = ~np.isnan(preds)
    if m.sum() < 30 or np.std(preds[m]) < 1e-12:
        return np.nan
    return float(np.corrcoef(preds[m], Y[m, 0])[0, 1])


def _boot_delta(X_b, Y_b, X_c, Y_c) -> np.ndarray:
    """Bootstrap Delta_rho = rho_OOS(best) - rho_OOS(cur). Resample row
    indices with replacement; same draw applied to both d's pair-sets is
    not possible (different lengths from different d trim) so resample
    each at its own n -- the two are near-independent samples of the same
    hour stratum, conservative for a difference CI."""
    out = np.empty(B)
    nb, nc = len(X_b), len(X_c)
    for i in range(B):
        ib = RNG.integers(0, nb, nb)
        ic = RNG.integers(0, nc, nc)
        rb = _rho_oos(X_b[ib], Y_b[ib])
        rc = _rho_oos(X_c[ic], Y_c[ic])
        out[i] = rb - rc
    return out


# ---- TEST B re-derived under the #41 standard ---------------------------
def recheck_test_b(df: pd.DataFrame, dvals: pd.DatetimeIndex) -> bool:
    print("=" * 72)
    print("TEST B RE-CHECK -- per-hour d, OOS rho, bootstrap Delta vs d_cur")
    print(f"  pre-registered: REPLICATES iff >={N_HOURS_REQ}/24 hours have")
    print(f"  95% CI lower bound on Delta_rho > {SEP_MARGIN:+.3f}")
    print(f"  d_current (per-day-type, in force): {D_CURRENT}")
    print("-" * 72)
    print("  hod  dtype     d_cur  d_best  rho_cur  rho_best  "
          "Delta  [95% CI]        sep?")
    n_sep = 0
    rows = []
    for h in range(24):
        idx = dvals[dvals.hour == h]
        if len(idx) < 150:
            continue
        # day-type that hour h belongs to (anchor-offset rollover aware):
        # use the modal day-type of the hour's timestamps.
        dts = pd.Series([_daytype(t, ANCHOR_H) for t in idx])
        dtype = dts.mode().iloc[0]
        d_cur = D_CURRENT[dtype]

        rho_by_d = {}
        xy_by_d = {}
        for d in range(1, DMAX + 1):
            xy = _emb_xy(df, idx, d)
            if xy is None:
                continue
            xy_by_d[d] = xy
            rho_by_d[d] = _rho_oos(*xy)
        rho_by_d = {d: r for d, r in rho_by_d.items() if not np.isnan(r)}
        if d_cur not in rho_by_d or len(rho_by_d) < 2:
            continue
        d_best = max(rho_by_d, key=rho_by_d.get)  # what the elbow would pick
        if d_best == d_cur:
            lo, hi, dm = 0.0, 0.0, 0.0
            sep = False
        else:
            boot = _boot_delta(*xy_by_d[d_best], *xy_by_d[d_cur])
            boot = boot[~np.isnan(boot)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            dm = float(np.mean(boot))
            sep = lo > SEP_MARGIN
        n_sep += int(sep)
        rows.append((h, dtype, d_cur, d_best, rho_by_d[d_cur],
                     rho_by_d[d_best], dm, lo, hi, sep))
        print(f"  {h:>3}  {dtype:<9} {d_cur:>5}  {d_best:>6}  "
              f"{rho_by_d[d_cur]:>7.4f}  {rho_by_d[d_best]:>8.4f}  "
              f"{dm:>+6.3f}  [{lo:+.3f},{hi:+.3f}]  {'YES' if sep else 'no'}")
    print("-" * 72)
    replicates = n_sep >= N_HOURS_REQ
    print(f"  hours with statistically separated d_best>d_cur: "
          f"{n_sep}/{len(rows)} (threshold {N_HOURS_REQ})")
    print(f"  ==> TEST B {'REPLICATES' if replicates else 'RETRACTED'} "
          f"under the #41 standard")
    if not replicates:
        print("  Interpretation: 'd wants 5-8' was an argmin/in-sample "
              "artifact.\n  Do NOT change d on its basis. (TEST A is the "
              "surviving leg.)")
    return replicates


# ---- TEST A re-derived under the #41 standard ---------------------------
def recheck_test_a() -> bool:
    fc = pickle.loads(CACHE.read_bytes()).copy()
    fc = fc.dropna(subset=["actual_mw"]).sort_index()
    fc["z_actual"] = (fc["actual_mw"] - fc["mu_mh"]) / fc["sigma_mh"]
    fc["z_err"] = fc["z_pred"] - fc["z_actual"]
    g = fc.assign(dz_dt=fc["z_actual"].diff()).dropna(subset=["dz_dt"])
    x, y = g["dz_dt"].to_numpy(), g["z_err"].to_numpy()
    r = float(np.corrcoef(x, y)[0, 1])
    n = len(x)
    boot = np.empty(B)
    for i in range(B):
        s = RNG.integers(0, n, n)
        boot[i] = np.corrcoef(x[s], y[s])[0, 1]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    excl0 = (lo > 0) or (hi < 0)
    print("=" * 72)
    print("TEST A RE-CHECK -- corr(dz/dt, z_err), bootstrap B=1000")
    print(f"  point r = {r:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   n={n}")
    print(f"  ==> TEST A {'REPLICATES' if excl0 else 'RETRACTED'} "
          f"(CI {'excludes' if excl0 else 'includes'} 0)")
    return excl0


def main() -> None:
    act = load_actuals(cutoff=None)
    zp = zscore_params(CUTOFF)
    z = zscore_transform(act, zp)
    df = z.to_frame("zscore").asfreq("h")
    dvals = df.index[df.index <= CUTOFF]

    a = recheck_test_a()
    print()
    b = recheck_test_b(df, dvals)
    print("=" * 72)
    print(f"SUMMARY: TEST A {'REPLICATES' if a else 'RETRACTED'} | "
          f"TEST B {'REPLICATES' if b else 'RETRACTED'}")
    print("Scope note: this re-check does NOT re-open #30's option choice. "
          "If B is\nretracted, the embedding lever rests on TEST A (option "
          "2, phase state).")


if __name__ == "__main__":
    main()
