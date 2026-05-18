"""EXPLORATORY: WHY is there a systematic diurnal bias? (honing step-2
mechanism test; dev 2021-22; scratch/, not recorded.)

The ~51%-of-MAE diurnal bias (under-fc overnight, over-fc evening) has
three candidate mechanisms, which are DISTINGUISHABLE:

  M1  sigma_mh amplification: z-space error is ~zero-mean & ~flat by
      hour, but sigma_mh varies strongly by hour, so MW bias =
      z_err * sigma_mh inherits the diurnal shape. NOT a model bias --
      an artifact of the z-score round-trip x multi-step z-drift.
      Predicts: z-space signed error ~0 & flat; sigma_mh strongly
      hour-varying; MW-bias shape ~ tracks sigma_mh shape.

  M2  genuine model bias: the kappa_Q local-linear map mis-centres the
      conditional mean at certain daily phases. Predicts: z-space signed
      error ITSELF has the diurnal shape (present even before
      re-scaling).

  M3  climatology offset: mu_mh/sigma_mh round-trip injects a static
      per-hour offset. Predicts: bias present even at HORIZON 1 (before
      iteration drift accumulates).

Decisive views: signed error in Z-SPACE by hour-of-day; the same by
forecast HORIZON; sigma_mh's hour profile vs the MW-bias hour profile.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("/tmp/bench2122_fc.pkl")


def _frame() -> pd.DataFrame:
    if CACHE.exists():
        f = pickle.loads(CACHE.read_bytes())
        if "z_pred" in f.columns:
            return f
    from scratch.benchmark_2021_22_ieso import compute

    fc, _ = compute()
    CACHE.write_bytes(pickle.dumps(fc))
    return fc


def main() -> None:
    fc = _frame().copy().dropna(subset=["actual_mw"]).sort_index()
    # invert the SAME round-trip to get the z-space actual
    fc["z_actual"] = (fc["actual_mw"] - fc["mu_mh"]) / fc["sigma_mh"]
    fc["z_err"] = fc["z_pred"] - fc["z_actual"]          # z-space signed
    fc["mw_err"] = fc["ours_mw"] - fc["actual_mw"]       # MW signed
    fc["hod"] = fc.index.hour

    print(f"n={len(fc)}")
    print(f"overall: z-space signed mean = {fc.z_err.mean():+.4f} "
          f"(z units; ~0 => no genuine centre bias)  | "
          f"MW signed mean = {fc.mw_err.mean():+.0f}")

    print("\n-- signed error by hour-of-day: Z-SPACE vs MW, with "
          "sigma_mh profile --")
    g = fc.groupby("hod").agg(
        z_bias=("z_err", "mean"),
        mw_bias=("mw_err", "mean"),
        sigma_mh=("sigma_mh", "mean"),
    ).round(3)
    print(g.to_string())

    # M1 test: does MW-bias shape track sigma_mh (not z-bias)?
    c_mw_sig = np.corrcoef(g["mw_bias"], g["sigma_mh"])[0, 1]
    c_mw_z = np.corrcoef(g["mw_bias"], g["z_bias"])[0, 1]
    print(f"\nM1  corr(MW-bias_hod, sigma_mh_hod) = {c_mw_sig:+.2f}  "
          f"| corr(MW-bias_hod, z-bias_hod) = {c_mw_z:+.2f}")
    print("    M1 (sigma amplification) if MW-bias tracks sigma_mh while "
          "z-bias is small/flat.")
    print(f"    z-bias spread: std across hours = {g['z_bias'].std():.4f} "
          f"(small => not a genuine z-space centre bias => favours M1)")

    # M2 test: is the z-space bias itself diurnally structured & sizable?
    z_amp = g["z_bias"].abs().mean()
    print(f"\nM2  mean |z-bias| across hours = {z_amp:.4f} z-units "
          f"(sizable & shaped => genuine model bias M2)")

    # M3 test: bias at horizon 1 (pre-drift) vs late horizon
    h = fc.groupby("horizon_h").agg(
        z_bias=("z_err", "mean"), mw_bias=("mw_err", "mean"),
        n=("z_err", "size")
    ).round(3)
    print("\nM3  signed error by forecast horizon (h1 = pre-iteration; "
          "growth with h => multi-step drift, not static climatology):")
    print(h.to_string())
    h1 = h.loc[1, "mw_bias"] if 1 in h.index else float("nan")
    hlast = h["mw_bias"].iloc[-1]
    print(f"    h1 MW-bias = {h1:+.0f} ; last-h MW-bias = {hlast:+.0f}  "
          f"(|grows| => drift/M1-M2 ; |flat & nonzero @ h1| => M3 "
          f"climatology offset)")

    print("\n  VERDICT GUIDE:")
    print("   z-bias flat&~0 + MW-bias~sigma_mh + grows with h -> M1 "
          "(round-trip x z-drift artifact): fix the MECHANISM "
          "(de-bias in z-space / damp multi-step z-drift), NOT a MW "
          "calibration table.")
    print("   z-bias itself shaped&sizable -> M2 genuine: a correction "
          "(or operator change) is legitimate.")
    print("   nonzero bias already at h1, ~flat in h -> M3 climatology: "
          "fix mu_mh/sigma_mh estimation, not the operator.")


if __name__ == "__main__":
    main()
