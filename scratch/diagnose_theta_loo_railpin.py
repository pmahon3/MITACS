"""EXPLORATORY: is the LOO-CV theta degenerate on real Ontario data?

Surfaced 2026-05-18 from the operator dashboard: per-anchor C_j is
constant to ~4 decimals across ~1000 days (weekday), theta saved as a
single value per day-type. Bug ruled out (C_j at separated anchors are
distinct objects, just nearly equal). This script is the discriminating
diagnostic the advisor required BEFORE concluding "rule failed":
the LOO-CV SCORE CURVE vs theta at a weekday anchor, on the SAME grid
the production estimator uses.

FINDING (one weekday anchor, sub=150, d=2, grid=geomspace(ds.min,
ds.max,18)):
  score is monotone-decreasing then FLAT to ~3 decimals for theta
  >~0.1; argmin lands on the GRID MAXIMUM (idx 17/17); improvement
  from theta~0.1 to theta=grid_max is ~0.4% across a ~50x bandwidth
  change. No sharp interior minimum.

INTERPRETATION: this is the RAIL-PINNING branch (degeneracy CLASS of
GL, expressed as "flat past a threshold" not strict monotone), NOT
"global fit genuinely clearly best". The conditional mean is so
near-linear in this embedding that LOO error carries almost no signal
about theta, so the argmin pins to whatever the grid's upper edge is
(per-anchor ds.max()). Consequence: the weighted regression has
effectively collapsed to OLS (kernel ~constant 1 over the library at
the selected theta), regardless of branch -- an integrity question for
the kappa_Q-estimator-with-locality framing, NOT just a tuning detail.

DECISIVE UPDATE (diagnostic 2, run via the PRODUCTION estimator
_theta_loo_cv on the VAR(1) recovery gate's own data): LOO-CV picks
theta AT the per-anchor grid maximum for 100% of gate anchors
(median theta / median grid_max ratio = 1.00). i.e. the theta-railing
is NOT Ontario-specific -- it is the rule's behaviour EVERYWHERE,
including on the synthetic system the provenance-locked recovery gate
"validates" it on. The gate still PASSES because a globally-linear OLS
fit trivially recovers a globally-linear VAR(1); the gate never tested
that locality was operative. Consequence: the recovery gate does not
validate "local" estimation; "true-LOO-CV is the validated rule"
([[mitacs-theta-rail-pinning]]) does not survive this -- LOO-CV rails
to grid-max on synthetic AND Ontario data. Weekday: flat-within-noise
above a floor (no signal). Sunday: strict-monotone-decreasing (GL-class
no-interior-optimum). Both -> kernel ~constant >=0.99 -> weighted LS
collapses to global OLS. This is an integrity contradiction with a
provenance-locked assumption; surfaced to advisor, fix NOT pre-decided;
#30 honing BLOCKED until resolved (phase-blindness attribution
presupposes localization that is not happening).

PROVENANCE-GRADE: INSPECTION-ONLY -- exploratory dev diagnostic; the
finding is production-path-confirmed (diagnostic 2 calls the production
_theta_loo_cv) but the RESOLUTION is undecided. Hypothesis-forming +
integrity flag, not a fix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from edynamics.modelling_tools import Embedding, Lag

from experiment._actuals import load_actuals, zscore_params, zscore_transform
from experiment.predict import _daytype
from processing.innovations.estimator import _gauss_w

CUT = pd.Timestamp("2024-12-31T23:00:00")


def loo_curve(daytype: str, d: int, anchor_row: int = 1000, sub: int = 150):
    act = load_actuals(cutoff=None)
    z = zscore_transform(act, zscore_params(CUT))
    df = z.to_frame("zscore").asfreq("h")
    dv = df.index[df.index <= CUT]
    idx = dv[[_daytype(t, 7) == daytype for t in dv]]
    fl = idx[d:-1]
    emb = Embedding(
        data=df,
        observers=[Lag(variable_name="zscore", tau=-i) for i in range(d)],
        library_times=fl,
    )
    emb.compile()
    b = emb.block
    X, Y = b.iloc[:-1].values, b.iloc[1:].values
    q = X[anchor_row].copy()
    dists = np.linalg.norm(X - q, axis=1)
    rng = np.random.default_rng(0)
    sid = rng.choice(len(dists), sub, replace=False)
    Xs, Ys, ds = X[sid], Y[sid], dists[sid]
    m = len(sid)
    grid = np.geomspace(max(ds.min(), 1e-3), ds.max(), 18)
    rows = []
    for th in grid:
        w = _gauss_w(ds, th, d)
        tot = 0.0
        for i in range(m):
            keep = np.arange(m) != i
            wi = w[keep]
            Ci = np.linalg.lstsq(
                wi[:, None] * Xs[keep], wi[:, None] * Ys[keep], rcond=None
            )[0]
            tot += np.sum((Ys[i] - Xs[i] @ Ci) ** 2)
        rows.append((th, tot / m, int((w > 0.5 * w.max()).sum())))
    return grid, rows


def main() -> None:
    for dt, d in (("weekday", 2), ("sunday", 7)):
        grid, rows = loo_curve(dt, d)
        sc = np.array([r[1] for r in rows])
        bi = sc.argmin()
        print(f"=== {dt} (d={d}) grid [{grid[0]:.3f},{grid[-1]:.3f}] ===")
        print(f"{'theta':>9} {'LOO':>11} {'eff_N':>7}")
        for th, s, e in rows:
            print(f"{th:9.3f} {s:11.5f} {e:7d}")
        print(
            f"-> argmin idx {bi}/17 theta={grid[bi]:.3f}; "
            f"score span {sc.max():.5f}->{sc.min():.5f} "
            f"({100 * (sc.max() - sc.min()) / sc.max():.2f}% total); "
            f"flat-tail = rail-pinning if argmin at edge\n"
        )


if __name__ == "__main__":
    main()
