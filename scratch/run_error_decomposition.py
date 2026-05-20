"""Stage 2: regenerate the 2021-22 dev benchmark WITH the propagated
predictive law N(z_hat_k, Sigma_k), then run the M1 & M2 error-
decomposition diagnostic under the pre-registered decision rule.

This is the executable form of writeup/error_decomposition_diagnostic.tex.
It decides the honing fork: Deficiency A (drift/location mis-centring,
the TEST-A leg) vs Deficiency B (diffusion/shape -- Gaussian proxy of a
heavy-tailed innovation), or NEITHER (-> document both, build nothing).

Discipline honoured:
  * production path only -- reuses benchmark_2021_22_ieso's exact
    building blocks + estimator._local_fit_at; no reimplementation;
  * the {Pi_t} variance composition was Stage-1 gate-validated
    (scratch/multistep_variance_propagation.py: PASS, machine precision,
    incl. the production rank-1 shift-Jacobian shape) BEFORE this runs;
  * shift-aware state Jacobian + rank-1 per-step innovation (the true
    iterated-forecast semantics; see mitacs-rank1-structural);
  * phase coordinate = d/dt of ACTUAL z (TEST A's exact definition --
    NOT redefined);
  * day-level bootstrap (resample whole 24h forecast days), B=1000;
  * M1 and M2 must AGREE in direction or the result is INCONCLUSIVE
    (not adjudicated post-hoc);
  * pre-registered rule from the .tex applied verbatim, thresholds
    fixed before any number is seen.

Caveat (stated, per the .tex scope section): linearisation-around-the-
realised-path variance propagation is the standard EKF-style scheme;
for a linear local model it is exact within the iteration, the only
unpropagated nonlinearity being C_i's own dependence on the local state
(locality) -- which is exactly what the production point forecast also
does. Dev-set measurement; any change it motivates is dev-only + a new
frozen spec before forward scoring.

PROVENANCE-GRADE: INSPECTION-ONLY -- exploratory dev-set diagnostic;
its OUTPUT is a methodology verdict, not a recorded result artifact.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

from edynamics.modelling_tools import Embedding, Lag

from experiment._actuals import load_actuals, zscore_params, zscore_transform
from experiment.predict import _daytype
from processing.innovations.estimator import _local_fit_at

from scratch.benchmark_2021_22_ieso import (
    ANCHOR_H,
    CUTOFF,
    WIN_END,
    WIN_START,
    _refrozen_dims,
)
from scratch.multistep_variance_propagation import (
    augmented_state_transition,
    propagate_predictive_cov,
    state_transition_from_local_fit,
    validate as stage1_validate,
)

NEW_CACHE = Path("/tmp/bench2122_fc_propagated.pkl")
B = 1000
RNG = np.random.default_rng(20260519)

# ---- pre-registered thresholds (FIXED before any number is seen) -------
REL_LOSS_MIN = 0.01      # a lever must clear 1% relative dev-loss reduction
# dominance also requires the two counterfactual CIs to be non-overlapping.


# ===================================================================== #
#  Forecast regeneration WITH propagated predictive covariance           #
# ===================================================================== #
def _propagate_scoped(J_seq, S_seq, scope: str) -> float:
    """k-step predictive variance s2_k under one dimension-change SCOPE.

    The composition P_{i+1}=J^T P J + Sigma is only exactly defined when
    J keeps one dimension throughout. When the embedding dim changes
    between steps (day-type rollover), each scope resolves it differently
    -- all but ``clean`` are UNVALIDATED approximations (no closed-form
    ground truth) and are EXPLORATORY robustness probes only:

      clean    : caller already restricted to single-dim days; plain
                 composition (validated, but only the ~58% single-dim
                 subpopulation).
      augmented: VALIDATED all-days path. regenerate() supplies
                 d_max-augmented (J,Sigma) (uniform d_max, gate-checked
                 bit-identical to clean on single-dim days and to the
                 VAR(1) closed form), so the plain recursion is exactly
                 defined across dim changes -- no boundary branch
                 needed. This is clean's computation EXTENDED to the
                 42% rollover days, not a different method.
      reset    : on a dim change, P resets to 0 and re-accumulates
                 (KNOWN signed bias: understates s2_k after any seam --
                 discards accumulated forecast uncertainty).
      project : on a dim change, map P across via truncation (dim drop)
                / zero-pad (dim rise) of the leading coords (an
                ASSUMPTION about cross-dim state covariance; no ground
                truth -> unvalidated).
      h1      : ignore propagation entirely; s2_k = first step's
                Sigma[0,0] (the exact one-step law -- no boundary issue
                at all, but not multi-step).
    """
    if scope == "h1":
        return float(S_seq[0][0, 0])
    if scope == "augmented":
        # regenerate() already supplied uniform d_max (J,Sigma) via the
        # gate-validated augmented_state_transition; the plain recursion
        # is exactly defined throughout -- no dim-change branch.
        P = propagate_predictive_cov(np.stack(J_seq), np.stack(S_seq))
        return float(P[-1, 0, 0])
    P = None
    for Ji, Si in zip(J_seq, S_seq):
        d = Ji.shape[0]
        if P is None or P.shape[0] != d:
            if P is None or scope == "reset":
                P = np.zeros((d, d))
            elif scope == "project":
                # truncate (dim drop) or zero-pad (dim rise) leading block
                old = P.shape[0]
                Q = np.zeros((d, d))
                k = min(old, d)
                Q[:k, :k] = P[:k, :k]
                P = Q
            else:  # clean: caller guarantees no dim change reaches here
                raise ValueError(
                    f"dim change ({P.shape[0]}->{d}) under scope=clean; "
                    f"caller must restrict to single-dim days"
                )
        P = Ji.T @ P @ Ji + Si
        P = 0.5 * (P + P.T)
    return float(P[0, 0])


def regenerate(scope: str = "clean") -> pd.DataFrame:
    """Re-run the benchmark forecast loop (same construction as
    benchmark_2021_22_ieso.compute) but capture per-step (C, Sigma),
    build the shift-aware Jacobian + rank-1 innovation, and propagate the
    predictive covariance through the 24-step iteration. Adds column
    ``s2_k`` = Sigma_k[0,0] (the k-step scalar predictive variance).

    ``scope`` selects the dimension-change handler (see
    ``_propagate_scoped``). ``clean`` restricts to single-dim delivery
    days (validated but ~58% of days); ``augmented`` is the VALIDATED
    all-days path (d_max-augmented, gate-checked); reset/project/h1 are
    EXPLORATORY unvalidated probes.
    """
    act = load_actuals(cutoff=None)
    zp = zscore_params(CUTOFF)
    z_full = zscore_transform(act, zp)
    dims = _refrozen_dims(z_full)
    D_MAX = max(dims.values())          # augmented fixed state size (=4)
    df = z_full.to_frame("zscore").asfreq("h")
    lib_cache: dict[int, tuple] = {}

    days = pd.date_range(WIN_START, WIN_END, freq="D")
    rows = []
    n_skipped_mixed = 0
    for D in days:
        targets = [D + pd.Timedelta(hours=h) for h in range(24)]
        issue_anchor = targets[0] - pd.Timedelta(hours=1)
        dmax = max(dims.values())
        need = [issue_anchor - pd.Timedelta(hours=i) for i in range(dmax)]
        if not all(n in z_full.index for n in need):
            continue
        # scope=clean (the validated, trustworthy path): restrict to
        # single-embedding-dim delivery days so J^T P J is exactly
        # defined throughout. Other scopes are EXPLORATORY robustness
        # probes that DO include the dim-change days, handling the
        # boundary via _propagate_scoped (unvalidated approximations).
        day_dims = {int(dims[_daytype(t, ANCHOR_H)]) for t in targets}
        if scope == "clean" and len(day_dims) > 1:
            n_skipped_mixed += 1
            continue
        zhist = {
            ts: float(z_full.loc[ts])
            for ts in z_full.index
            if issue_anchor - pd.Timedelta(hours=dmax) <= ts <= issue_anchor
        }
        # per-step Jacobian / innovation accumulators for THIS day
        J_seq, S_seq, step_rows = [], [], []
        for t in targets:
            dt = _daytype(t, ANCHOR_H)
            d = int(dims[dt])
            if d not in lib_cache:
                lags = [Lag(variable_name="zscore", tau=-i) for i in range(d)]
                fl = df.index[df.index <= CUTOFF][d:-1]
                emb = Embedding(data=df, observers=lags, library_times=fl)
                emb.compile()
                b = emb.block
                lib_cache[d] = (b.iloc[:-1].values, b.iloc[1:].values)
            X, Y = lib_cache[d]
            prev = t - pd.Timedelta(hours=1)
            lag_t = [prev - pd.Timedelta(hours=i) for i in range(d)]
            try:
                xq = np.array([zhist[lt] for lt in lag_t], dtype=float)
            except KeyError:
                break
            C, S, _mu, _th, _ = _local_fit_at(X, Y, xq, d)
            z_next = float(xq @ (C[:, 0] if C.ndim == 2 else C))
            zhist[t] = z_next

            if scope == "augmented":
                # gate-validated fixed-d_max builder: uniform J,Sigma so
                # the recursion is exactly defined across dim changes
                J, Sig = augmented_state_transition(
                    C, float(S[0, 0]), d, D_MAX
                )
            else:
                J, Sig = state_transition_from_local_fit(
                    C, float(S[0, 0]), d
                )
            J_seq.append(J)
            S_seq.append(Sig)
            # propagate covariance through the steps taken SO FAR,
            # resolving any dim-change boundary per the active scope
            s2_k = _propagate_scoped(J_seq, S_seq, scope)

            key = (t.month, t.hour)
            mu = float(zp["mu_mh"].loc[key])
            sd = float(zp["sigma_mh"].loc[key])
            step_rows.append({
                "dt": t, "delivery_date": D,
                "z_pred": z_next, "s2_k": s2_k,
                "mu_mh": mu, "sigma_mh": sd,
                "horizon_h": int((t - targets[0]) / pd.Timedelta(hours=1)) + 1,
            })
        rows.extend(step_rows)

    fc = pd.DataFrame(rows).set_index("dt").sort_index()
    fc["actual_mw"] = fc.index.map(act)
    fc = fc.dropna(subset=["actual_mw"])
    fc["z_actual"] = (fc["actual_mw"] - fc["mu_mh"]) / fc["sigma_mh"]
    fc["z_err"] = fc["z_pred"] - fc["z_actual"]
    # phase coordinate: d/dt of ACTUAL z (TEST A's exact definition)
    fc["dz_dt"] = fc["z_actual"].diff()
    out = fc.dropna(subset=["dz_dt", "s2_k"])
    n_kept_days = out["delivery_date"].nunique()
    if scope == "augmented":
        out.attrs["scope"] = (
            f"[augmented -- VALIDATED, ALL-DAYS PRIMARY] all "
            f"{n_kept_days} delivery days incl. day-type-rollover; "
            f"d_max-augmented propagation, gate-checked bit-identical "
            f"to clean on single-dim days and to the VAR(1) closed "
            f"form. This is clean's computation EXTENDED to the ~42% "
            f"rollover days via a validated method -- it RESOLVES the "
            f"load-bearing exclusion, it is not an approximation probe."
        )
    elif scope == "clean":
        out.attrs["scope"] = (
            f"[clean -- VALIDATED, single-dim subpopulation] "
            f"single-embedding-dim delivery days only: {n_kept_days} "
            f"kept, {n_skipped_mixed} excluded as day-type-rollover. "
            f"Corroborates 'augmented' on its 58% subpopulation; the "
            f"all-days verdict is 'augmented'."
        )
    else:
        out.attrs["scope"] = (
            f"[{scope} -- EXPLORATORY, UNVALIDATED dim-change handler, "
            f"NOT a production verdict] all {n_kept_days} delivery days "
            f"incl. rollover; boundary resolved by '{scope}' "
            f"(reset=known understatement bias; project=unvalidated "
            f"cross-dim assumption; h1=one-step only). Superseded by "
            f"the VALIDATED 'augmented' scope; kept as trace only."
        )
    return out


# ===================================================================== #
#  M1 -- bias^2 / variance decomposition of squared error                #
# ===================================================================== #
def run_m1(fc: pd.DataFrame) -> dict:
    """E[e^2 | stratum] = bias^2 (-> A) + dispersion (-> B). Stratify by
    phase quintile and hour-of-day. Report the bias^2 FRACTION of total
    MSE with a day-bootstrap CI."""
    e = fc["z_err"].to_numpy()
    days = fc["delivery_date"].to_numpy()
    uniq_days = np.unique(days)
    ph_q = pd.qcut(fc["dz_dt"], 5, labels=False, duplicates="drop").to_numpy()
    hod = fc.index.hour.to_numpy()

    def _bias2_frac(mask_idx: np.ndarray) -> float:
        ee, pq, hh = e[mask_idx], ph_q[mask_idx], hod[mask_idx]
        mse = float(np.mean(ee**2))
        if mse <= 0:
            return np.nan
        # systematic (bias^2) component = variance EXPLAINED by the
        # phase x hour strata means = MSE - within-stratum residual MS
        df_ = pd.DataFrame({"e": ee, "g": pq * 100 + hh})
        grp = df_.groupby("g")["e"]
        within = float(((df_["e"] - grp.transform("mean")) ** 2).mean())
        return (mse - within) / mse

    point = _bias2_frac(np.arange(len(e)))
    boot = np.empty(B)
    for b in range(B):
        samp = RNG.choice(uniq_days, len(uniq_days), replace=True)
        idx = np.concatenate([np.where(days == dd)[0] for dd in samp])
        boot[b] = _bias2_frac(idx)
    boot = boot[~np.isnan(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"bias2_frac": point, "ci": (float(lo), float(hi)),
            "interpretation": "high bias2_frac -> A (location); "
                              "low -> dispersion-dominated -> B (shape)"}


# ===================================================================== #
#  M2 -- counterfactual proper-score bounds (Gaussian log-loss + CRPS)    #
# ===================================================================== #
def _gauss_logloss(y, m, s2):
    s2 = np.maximum(s2, 1e-9)
    return 0.5 * (np.log(2 * np.pi * s2) + (y - m) ** 2 / s2)


def _gauss_crps(y, m, s2):
    s = np.sqrt(np.maximum(s2, 1e-9))
    z = (y - m) / s
    return s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def _t_logloss(y, m, s2, nu):
    # scale so Var = s2  (t variance = scale^2 * nu/(nu-2))
    scale = np.sqrt(np.maximum(s2, 1e-9) * (nu - 2) / nu)
    return -student_t.logpdf((y - m) / scale, df=nu) + np.log(scale)


def run_m2(fc: pd.DataFrame) -> dict:
    """Three predictive laws, all from production outputs:
      (a) baseline N(z_pred, s2_k)
      (b) A-counterfactual: subtract dev phase-conditional bias from mean
      (c) B-counterfactual: Student-t (dev-fit nu) matched to s2_k
    Score = mean Gaussian log-loss (CRPS reported as corroboration).
    Lever = baseline_loss - counterfactual_loss (positive = improvement),
    day-bootstrapped."""
    y = fc["z_actual"].to_numpy()
    m = fc["z_pred"].to_numpy()
    s2 = fc["s2_k"].to_numpy()
    days = fc["delivery_date"].to_numpy()
    uniq_days = np.unique(days)
    ph_q = pd.qcut(fc["dz_dt"], 5, labels=False, duplicates="drop").to_numpy()

    # (b) A-counterfactual mean: remove the phase-quintile mean bias
    bias_by_q = pd.Series(m - y).groupby(ph_q).transform("mean").to_numpy()
    m_A = m - bias_by_q

    # (c) B-counterfactual nu: method-of-moments from standardised resid
    z_std = (y - m) / np.sqrt(np.maximum(s2, 1e-9))
    k = float(pd.Series(z_std).kurtosis())  # excess kurtosis
    nu = max(4.5, 6.0 / max(k, 1e-6) + 4.0)  # t excess kurt = 6/(nu-4)

    def _losses(idx):
        base = _gauss_logloss(y[idx], m[idx], s2[idx]).mean()
        a = _gauss_logloss(y[idx], m_A[idx], s2[idx]).mean()
        bb = _t_logloss(y[idx], m[idx], s2[idx], nu).mean()
        return base, a, bb

    base, a, bb = _losses(np.arange(len(y)))
    dA = np.empty(B)
    dB = np.empty(B)
    for b in range(B):
        samp = RNG.choice(uniq_days, len(uniq_days), replace=True)
        idx = np.concatenate([np.where(days == dd)[0] for dd in samp])
        bs, as_, bs_b = _losses(idx)
        dA[b] = (bs - as_) / abs(bs)   # relative loss reduction
        dB[b] = (bs - bs_b) / abs(bs)
    ciA = np.percentile(dA, [2.5, 97.5])
    ciB = np.percentile(dB, [2.5, 97.5])
    return {
        "nu_fit": float(nu), "excess_kurt_std_resid": k,
        "baseline_logloss": float(base),
        "dA_rel": float(np.mean(dA)), "ciA": (float(ciA[0]), float(ciA[1])),
        "dB_rel": float(np.mean(dB)), "ciB": (float(ciB[0]), float(ciB[1])),
    }


# ===================================================================== #
#  Pre-registered decision rule (verbatim from the .tex §5.2)            #
# ===================================================================== #
def verdict(m1: dict, m2: dict) -> str:
    loA, hiA = m2["ciA"]
    loB, hiB = m2["ciB"]
    A_clears = loA > REL_LOSS_MIN
    B_clears = loB > REL_LOSS_MIN
    A_gt_B = loA > hiB           # non-overlapping, A larger
    B_gt_A = loB > hiA           # non-overlapping, B larger
    # M1 corroboration: bias2_frac CI lower bound; high -> A, low -> B
    m1_lo = m1["ci"][0]
    m1_points_A = m1_lo > 0.5
    m1_points_B = m1["ci"][1] < 0.5

    if A_clears and A_gt_B:
        direction = "A"
        m1_agrees = m1_points_A
    elif B_clears and B_gt_A:
        direction = "B"
        m1_agrees = m1_points_B
    else:
        return ("NEITHER dominates (CIs overlap or neither clears the "
                "1% relative-loss bar). PRE-REGISTERED OUTCOME: document "
                "BOTH Deficiency A and B as limitations of the univariate "
                "local-Gaussian kappa_Q estimator; build nothing; proceed "
                "to the registered forward experiment unchanged.")
    if not m1_agrees:
        return (f"INCONCLUSIVE: M2 points to {direction} but M1 "
                f"(bias^2 fraction CI {m1['ci']}) does not agree in "
                f"direction. Per the pre-registered rule the disagreement "
                f"IS the finding; no build decision is licensed.")
    nxt = ("phase-aware conditional mean (drift/location)"
           if direction == "A" else
           "drop local-Gaussian -> heavy-tailed innovation law "
           "(diffusion/shape)")
    return (f"{direction} DOMINATES (M1 and M2 agree). PRE-REGISTERED "
            f"OUTCOME: the higher-leverage honing target is {nxt}; the "
            f"other deficiency remains a documented caveat. Any build is "
            f"dev-only + new frozen spec before forward scoring.")


# augmented = VALIDATED all-days PRIMARY; clean = validated single-dim
# corroboration; reset/project/h1 = superseded exploratory trace.
SCOPES = ("augmented", "clean", "reset", "project", "h1")


def _cache_path(scope: str) -> Path:
    return Path(f"/tmp/bench2122_fc_propagated_{scope}.pkl")


def _short_verdict(v: str) -> str:
    if v.startswith("NEITHER"):
        return "NEITHER dominates"
    if v.startswith("INCONCLUSIVE"):
        return "INCONCLUSIVE (M1/M2 disagree)"
    return v.split(" DOMINATES")[0] + " DOMINATES"


def main(force_regen: bool = False) -> None:
    print("Stage-1 gate re-check (must PASS before Stage 2 trusts itself)")
    g = stage1_validate()
    print(f"  Stage-1 ALL_PASSED = {g['ALL_PASSED']}")
    if not g["ALL_PASSED"]:
        print("  ABORT: Stage-1 gate failed; the propagation is not "
              "trustworthy. Stage 2 not run.")
        return

    summary = []
    for scope in SCOPES:
        cache = _cache_path(scope)
        if cache.exists() and not force_regen:
            fc = pickle.loads(cache.read_bytes())
            print(f"\n[{scope}] loaded cache ({len(fc)} rows) {cache}")
        else:
            print(f"\n[{scope}] regenerating (slow forecast loop)...")
            fc = regenerate(scope=scope)
            cache.write_bytes(pickle.dumps(fc))
            print(f"  wrote {len(fc)} rows -> {cache}")

        print(f"SCOPE: {fc.attrs.get('scope', '(missing)')}")
        m1 = run_m1(fc)
        m2 = run_m2(fc)
        v = verdict(m1, m2)
        print("-" * 68)
        print(f"  M1 bias^2 frac = {m1['bias2_frac']:.3f} "
              f"CI[{m1['ci'][0]:.3f},{m1['ci'][1]:.3f}]")
        print(f"  M2 A rel red = {m2['dA_rel']:+.4f} "
              f"CI[{m2['ciA'][0]:+.4f},{m2['ciA'][1]:+.4f}] | "
              f"B rel red = {m2['dB_rel']:+.4f} "
              f"CI[{m2['ciB'][0]:+.4f},{m2['ciB'][1]:+.4f}] "
              f"(nu={m2['nu_fit']:.2f})")
        print(f"  VERDICT: {_short_verdict(v)}")
        summary.append((scope, m1, m2, _short_verdict(v)))

    print("\n" + "=" * 78)
    print("A-vs-B DIAGNOSTIC -- VALIDATED all-days verdict + corroboration")
    print("=" * 78)
    print(f"{'scope':<10}{'bias2':>7}{'A rel':>9}{'B rel':>9}"
          f"{'nu':>6}  verdict")
    role = {
        "augmented": "  <- PRIMARY (validated, all days)",
        "clean": "  <- corroboration (validated, single-dim 58%)",
    }
    for scope, m1, m2, sv in summary:
        tag = role.get(scope, "  (superseded exploratory probe)")
        print(f"{scope:<10}{m1['bias2_frac']:>7.3f}{m2['dA_rel']:>+9.4f}"
              f"{m2['dB_rel']:>+9.4f}{m2['nu_fit']:>6.2f}  {sv}{tag}")
    print("-" * 78)
    vmap = {s: sv for s, _, _, sv in summary}
    aug_v, clean_v = vmap["augmented"], vmap["clean"]
    print(f"PRIMARY VERDICT (augmented, all days, validated): {aug_v}")
    if aug_v == clean_v:
        print(f"  CORROBORATED: clean (single-dim 58%) independently "
              f"gives the same verdict ('{clean_v}'). The load-bearing "
              f"exclusion is now RESOLVED, not merely caveated -- the "
              f"all-days result is validated and agrees with the "
              f"subpopulation result.")
    else:
        print(f"  DIVERGES from clean ('{clean_v}'): including the ~42% "
              f"rollover days via the validated augmentation CHANGES the "
              f"verdict. The all-days 'augmented' result is the trusted "
              f"one (validated, complete); 'clean' was the single-dim "
              f"sub-answer. This is a substantive finding -- if it points "
              f"to A or B dominating, a model fix becomes evidence-"
              f"licensed (then: new frozen spec before forward scoring).")
    print("NOTE: 'augmented' and 'clean' are both validated (gate-"
          "checked bit-identical on single-dim days); 'augmented' "
          "additionally covers the rollover days. reset/project/h1 are "
          "SUPERSEDED unvalidated probes, retained as trace only. "
          "INSPECTION-ONLY; not stamped through make_result.")
    print("=" * 78)


if __name__ == "__main__":
    import sys
    main(force_regen="--force" in sys.argv)
