"""Second discriminating check: sample-kurt-matched single-law null.

The first kurt-matched check (patra_sen_kurt_matched_null.py) matched
POPULATION kurt to Ontario's sample kurt of ~22. But at n=60000 and
very-heavy-tailed t(ν), the sample kurt is systematically less than
population kurt (the fourth moment doesn't exist for ν≤4 and converges
slowly for ν only slightly above 4).

The right comparison is to match SAMPLE kurt: find ν such that AR(1)+t(ν)
through the pipeline produces sample kurt ≈ 22 on at least some seeds.
Then read α̂_L at that ν across multiple seeds.

If at the sample-kurt-matched ν, α̂_L stays ≤ 0.30 across seeds:
Ontario excess remains; mixture-structured claim survives.

Key finding from running this:
  At ν=4.01 (population kurt = 600), sample kurt across 5 seeds
  ranges [6.2, 22.0] — wildly variable due to absent moments.
  But α̂_L^95% across the same seeds: stable at 0.218-0.262.
  Patra-Sen's CDF-based functional is robust where moment statistics
  are not.

Ontario α̂_L^95% (0.49-0.74) remains 2-3× any single-law value
obtained here, even on the single seed where t(4.01) sample kurt
coincidentally matched Ontario's 22.

PROVENANCE-GRADE: INSPECTION-ONLY (discriminating check for the
mixture-structured claim).
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from scipy.stats import norm

from scratch.patra_sen import patra_sen
from scratch.patra_sen_kurt_matched_null import _build_emb, _pool, _sim_ar1_t


def main():
    t0 = time.time()
    n_sim = 24000

    print("=" * 80)
    print("Sample-kurt-matched single-law null")
    print("=" * 80)
    print("Question: at the ν where AR(1)+t(ν) gives SAMPLE kurt ≈ 22 "
          "(Ontario's value),")
    print("what α̂_L^95% does Patra-Sen give? Single seed → multi-seed stability.")
    print()

    # Step 1: single-seed sweep across very-heavy-tail ν
    print("Step 1: single-seed sweep at increasing tail heaviness")
    print(f"{'nu':>6} {'pop_kurt':>10} {'sample_kurt':>12} "
          f"{'alpha_hat':>10} {'L95':>8}")
    sweep = []
    for nu in [4.27, 4.10, 4.05, 4.02, 4.01]:
        z = _sim_ar1_t(n_sim, df_t=nu, a=0.95, seed=int(1000 * nu))
        rng = np.random.default_rng(7)
        emb, libt = _build_emb(z)
        pooled = _pool(emb, libt, rng)
        kurt = float(pd.Series(pooled).kurtosis())
        res = patra_sen(pooled, F_b=lambda x: norm.cdf(x),
                        gamma_grid_size=600)
        pop_kurt = 6 / (nu - 4) if nu > 4 else float("inf")
        print(f"{nu:>6.2f} {pop_kurt:>10.1f} {kurt:>12.2f} "
              f"{res.alpha_hat:>10.4f} {res.alpha_L_95:>8.4f}")
        sweep.append((nu, pop_kurt, kurt, res.alpha_hat, res.alpha_L_95))

    # Step 2: at the sample-kurt-closest ν, check seed stability
    best = min(sweep, key=lambda r: abs(r[2] - 22.0))
    nu_best = best[0]
    print()
    print(f"Step 2: seed stability at ν = {nu_best} "
          f"(sample-kurt-closest to Ontario's 22)")
    print(f"  {'seed':>6} {'sample_kurt':>12} {'alpha_hat':>10} {'L95':>8}")
    ks, ls = [], []
    for seed_base in [100, 200, 300, 400, 500]:
        z = _sim_ar1_t(n_sim, df_t=nu_best, a=0.95, seed=seed_base)
        rng = np.random.default_rng(7)
        emb, libt = _build_emb(z)
        pooled = _pool(emb, libt, rng)
        kurt = float(pd.Series(pooled).kurtosis())
        res = patra_sen(pooled, F_b=lambda x: norm.cdf(x),
                        gamma_grid_size=600)
        print(f"  {seed_base:>6} {kurt:>12.2f} "
              f"{res.alpha_hat:>10.4f} {res.alpha_L_95:>8.4f}")
        ks.append(kurt)
        ls.append(res.alpha_L_95)

    print()
    print(f"  sample kurt across seeds: [{min(ks):.1f}, {max(ks):.1f}] "
          f"(wide; high-kurt sample moments are unstable)")
    print(f"  α̂_L^95% across seeds:    [{min(ls):.3f}, {max(ls):.3f}] "
          f"(narrow; Patra-Sen is CDF-based, not moment-based)")

    print()
    print("=" * 80)
    print("Verdict:")
    print("=" * 80)
    max_l95 = max(ls)
    if max_l95 >= 0.45:
        print(f"  Max α̂_L^95% across seeds = {max_l95:.3f}: Ontario CONSISTENT")
        print(f"  with single very-heavy-tailed AR(1). RETRACT mixture-structured.")
    elif max_l95 < 0.30:
        print(f"  Max α̂_L^95% across seeds = {max_l95:.3f}: Ontario "
              f"(L95 = 0.49-0.74)")
        print(f"  remains 2-3× above any single-law value. "
              f"MIXTURE-STRUCTURED SURVIVES.")
    else:
        print(f"  Max α̂_L^95% = {max_l95:.3f}: borderline; needs further "
              "investigation")


if __name__ == "__main__":
    main()
