"""EXPLORATORY: Reading 1 vs Reading 2 for the z-transform.
(Framework-level discriminator; dev 2021-22; scratch/, NOT recorded.
See memory mitacs-zscore-framework-coupling.)

Q: is the z-transform a benign invertible COORDINATE CHANGE (Reading 1 --
M2/phase/tails are genuine dynamics merely relabelled) or a leaky
implicit SEASONAL MODEL (Reading 2 -- those structures are partly
artifacts of THAT seasonal fit)?

Method: re-run the same multi-step day-ahead forecast under 4
climatology specs of increasing seasonal resolution (all estimated
STRICTLY <= cutoff -> dev/test wall preserved), invert every forecast
back to DEMAND space (so error structure is comparable across specs --
z-space is not), and measure, on the 2021-22 dev window:
  * phase-asymmetry  : corr( d(actual demand)/dt , MW residual )
  * heavy-tailedness  : excess kurtosis of the MW residual
INVARIANT across specs  => Reading 1 (z exonerated; proceed to embedding
                           / phase honing).
SYSTEMATICALLY MOVES    => Reading 2 (the seasonal coupling IS the lever;
                           embedding honing inside z is deck chairs).
PARTIAL                 => partial leak; itself a finding -> resolve
                           further (user-agreed).

Specs (crude -> fine -> none):
  global   : one (mu,sigma) scalar pair
  hour     : per hour-of-day (24)
  monthhour: per (month,hour) -- the CURRENT production spec (288)
  identity : no transform (operate on raw demand) -- framework-aligned
             "no seasonal surrogate" extreme
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from edynamics.modelling_tools import Embedding, Lag

from experiment._actuals import load_actuals
from experiment.predict import _daytype
from processing.innovations.estimator import _local_fit_at

CUTOFF = pd.Timestamp("2021-10-18 23:00:00")
WIN_START = pd.Timestamp("2021-10-19")
WIN_END = pd.Timestamp("2022-02-27")
ANCHOR_H = 7
DIM = 3  # FIXED across specs on purpose: isolate the climatology effect,
#          not confound it with per-spec dim re-selection.


def _clim(act: pd.Series, kind: str):
    """(mu, sigma) lookup keyed per spec, estimated strictly <= cutoff."""
    a = act[act.index <= CUTOFF]
    if kind == "global":
        m, s = a.mean(), a.std()
        key = lambda ts: 0
        mu = {0: m}
        sd = {0: s}
    elif kind == "hour":
        g = a.groupby(a.index.hour)
        mu, sd = g.mean().to_dict(), g.std().to_dict()
        key = lambda ts: ts.hour
    elif kind == "monthhour":
        g = a.groupby([a.index.month, a.index.hour])
        mu, sd = g.mean().to_dict(), g.std().to_dict()
        key = lambda ts: (ts.month, ts.hour)
    elif kind == "identity":
        return None  # no transform
    else:
        raise ValueError(kind)
    return key, mu, sd


def _z(series: pd.Series, clim):
    if clim is None:
        return series.copy()
    key, mu, sd = clim
    ks = [key(ts) for ts in series.index]
    m = np.array([mu[k] for k in ks])
    s = np.array([sd[k] for k in ks])
    return pd.Series((series.values - m) / s, index=series.index)


def _inv(zval: float, ts: pd.Timestamp, clim) -> float:
    if clim is None:
        return zval
    key, mu, sd = clim
    k = key(ts)
    return zval * sd[k] + mu[k]


def _forecast(act: pd.Series, clim) -> pd.DataFrame:
    z = _z(act, clim)
    df = z.to_frame("v").asfreq("h")
    lags = [Lag(variable_name="v", tau=-i) for i in range(DIM)]
    fl = df.index[df.index <= CUTOFF][DIM:-1]
    emb = Embedding(data=df, observers=lags, library_times=fl)
    emb.compile()
    b = emb.block
    X, Y = b.iloc[:-1].values, b.iloc[1:].values

    out = []
    for D in pd.date_range(WIN_START, WIN_END, freq="D"):
        tg = [D + pd.Timedelta(hours=h) for h in range(24)]
        ia = tg[0] - pd.Timedelta(hours=1)
        need = [ia - pd.Timedelta(hours=i) for i in range(DIM)]
        if not all(n in z.index for n in need):
            continue
        zh = {ts: float(z.loc[ts]) for ts in z.index
              if ia - pd.Timedelta(hours=DIM) <= ts <= ia}
        for t in tg:
            prev = t - pd.Timedelta(hours=1)
            lt = [prev - pd.Timedelta(hours=i) for i in range(DIM)]
            try:
                xq = np.array([zh[x] for x in lt], dtype=float)
            except KeyError:
                break
            C, _S, _m, _th, _ = _local_fit_at(X, Y, xq, DIM)
            zn = float(xq @ (C[:, 0] if C.ndim == 2 else C))
            zh[t] = zn
            out.append({"dt": t, "fc_mw": _inv(zn, t, clim)})
    fc = pd.DataFrame(out).set_index("dt")
    fc["actual_mw"] = fc.index.map(act)
    return fc.dropna(subset=["actual_mw"])


def main() -> None:
    act = load_actuals(cutoff=None)
    print(f"{'spec':10} {'n':>5} {'MAPE%':>7} {'phase_corr':>11} "
          f"{'resid_kurt':>11}  (DEMAND-space; invariant=>R1, moves=>R2)")
    res = {}
    for kind in ("global", "hour", "monthhour", "identity"):
        clim = _clim(act, kind)
        fc = _forecast(act, clim)
        r = fc["fc_mw"] - fc["actual_mw"]                 # MW residual
        dd = fc["actual_mw"].diff()                        # demand dz/dt
        m = r.notna() & dd.notna()
        pc = np.corrcoef(dd[m], r[m])[0, 1]
        ek = float(stats.kurtosis(r.dropna(), fisher=True))
        mape = float((r.abs() / fc["actual_mw"]).mean() * 100)
        res[kind] = (len(fc), mape, pc, ek)
        print(f"{kind:10} {len(fc):>5} {mape:>7.2f} {pc:>11.3f} {ek:>11.2f}")

    pcs = np.array([v[2] for v in res.values()])
    eks = np.array([v[3] for v in res.values()])
    print(f"\nphase_corr spread across specs: "
          f"{pcs.min():.3f}..{pcs.max():.3f}  (range {np.ptp(pcs):.3f})")
    print(f"resid_kurt spread across specs: "
          f"{eks.min():.2f}..{eks.max():.2f}  (range {np.ptp(eks):.2f})")
    print("\nREAD: small spread (phase_corr & kurt ~constant across the "
          "4 specs incl. identity) => READING 1 (z is a benign coordinate "
          "change; M2/phase/tails are genuine dynamics) => proceed to "
          "embedding/phase honing. Large/systematic spread => READING 2 "
          "(the seasonal surrogate leaks; the coupling is the lever, not "
          "the embedding). Partial => resolve further.")


if __name__ == "__main__":
    main()
