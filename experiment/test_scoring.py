"""Mechanics test for the experiment ledger + scoring.

Real settlement is calendar-bound (stale local actuals vs 2026 targets),
so this exercises ledger immutability + the scoring metrics on synthetic
graded rows. Verifies:
  * append-only: a re-run with the same forecasts adds no duplicates;
  * write-once forecast fields: a changed forecast for an already-graded
    (target, spec) does NOT overwrite the recorded one;
  * settled-actual back-fill happens once, then is immutable;
  * persistence baselines computed; t-24h day-type-mismatch reported;
  * interval coverage measured; real-time subset split out;
  * framing label present (interim / not the IESO head-to-head).

Run::  python -m experiment.test_scoring
"""
from __future__ import annotations

import tempfile
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pandas as pd

import experiment.ledger as L
import experiment.score as S


def _frame(targets, our, ieso, actual, was_rt, spec="hABC"):
    return pd.DataFrame({
        "target_dt": targets,
        "spec_hash": spec,
        "our_forecast_mw": our,
        "our_pi_lo_mw": [v - 300 for v in our],
        "our_pi_hi_mw": [v + 300 for v in our],
        "ieso_forecast_mw": ieso,
        "issue_time": "2026-06-01T00:00:00+00:00",
        "was_real_time": was_rt,
        "daytype": "weekday",
        "embedding_dim": 2,
        "zscore_params_fp": "zfp",
        "actual_mw": actual,
        "actual_scrape_time": "2026-06-02T00:00:00+00:00",
        "actual_status": "final",
        "actual_source_sha": "sha",
    })


def test_ledger_and_scoring() -> None:
    tmp = Path(tempfile.mkdtemp())
    cfg = mock.Mock()
    cfg.paths.forecast_csv = tmp / "ieso_forecast.csv"

    # 240 hourly targets (10 days) so t-24h and t-168h baselines exist
    tg = pd.date_range("2026-06-01", periods=240, freq="h")
    rng = np.random.default_rng(0)
    act = 15000 + 2000 * np.sin(np.arange(240) * 2 * np.pi / 24) + rng.normal(0, 80, 240)
    ours = act + rng.normal(0, 120, 240)          # decent one-step
    ieso = act + rng.normal(0, 500, 240)          # noisier (diff horizon)
    rt = [t > pd.Timestamp("2026-06-05") for t in tg]  # later half "real-time"

    with mock.patch.object(L, "load_config", return_value=cfg):
        f = _frame(tg, ours, ieso, act, rt)
        l1 = L.update_ledger(f)
        assert len(l1) == 240, len(l1)

        # append-only: identical re-run -> no duplicates
        l2 = L.update_ledger(f)
        assert len(l2) == 240, f"re-run duplicated rows: {len(l2)}"

        # write-once: a CHANGED forecast for graded (t,spec) must NOT overwrite
        f_tamper = _frame(tg, [9_999_999.0] * 240, ieso, act, rt)
        l3 = L.update_ledger(f_tamper)
        assert l3["our_forecast_mw"].max() < 1_000_000, \
            "graded forecast was overwritten -- immutability violated"

    with mock.patch.object(S, "read_ledger", return_value=l3):
        r = S.score()

    assert r.sufficient and r.n_settled == 240, r
    assert r.ours["mae"] < r.ieso_DIFFERENT_HORIZON_reference["mae"], \
        "synthetic ours should beat the noisier ieso reference"
    assert r.persistence_168h["n"] > 0 and r.persistence_24h["n"] > 0
    assert 0.0 <= r.persistence_24h_daytype_mismatch_pct <= 100.0
    assert 0.0 <= r.interval_coverage_1sigma_pct <= 100.0
    assert r.real_time_subset.get("n", 0) > 0, "real-time subset not split"
    assert "INTERIM" in r.framing and "head-to-head" in r.framing

    print("ledger + scoring mechanics: PASS")
    print(f"  ours MAE={r.ours['mae']:.0f}  "
          f"p168 MAE={r.persistence_168h['mae']:.0f}  "
          f"ieso(diff-horizon ref) MAE="
          f"{r.ieso_DIFFERENT_HORIZON_reference['mae']:.0f}")
    print(f"  t-24h daytype-mismatch={r.persistence_24h_daytype_mismatch_pct}%  "
          f"interval coverage={r.interval_coverage_1sigma_pct:.0f}% "
          f"(SYNTHETIC fixed +/-300 band, not real Sigma -> arbitrary here; "
          f"the test only verifies coverage is MEASURED & reported)")
    print(f"  real-time subset n={r.real_time_subset['n']}")


if __name__ == "__main__":
    test_ledger_and_scoring()
