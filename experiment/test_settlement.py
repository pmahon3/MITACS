"""Integrity test for the actuals-settlement write-once / revision rule.

This is the experiment's load-bearing scientific property: a graded
actual must NEVER silently change (IESO restates recent demand). The
rule: write-once per target hour; a differing later scrape APPENDS a
revision row (original preserved); an unchanged re-scrape stamps
confirmation and may promote prelim -> final.

Run::

    python -m experiment.test_settlement
"""
from __future__ import annotations

import tempfile
import unittest.mock as mock
from pathlib import Path

import pandas as pd

import experiment.collect as C

T = pd.Timestamp("2026-03-01 05:00:00")  # one experiment target hour
TARGETS = pd.DatetimeIndex([T])


def _prov(sha: str, created: str) -> dict:
    return {
        "PUB_Demand_2026.csv": {
            "ieso_created_at": created,
            "source_sha": sha,
            "scrape_time": pd.Timestamp.utcnow().isoformat(),
            "source_url": "test",
        }
    }


def _run(tmp: Path, actual_value: float, prov: dict) -> pd.DataFrame:
    """Invoke settle_actuals with a controlled single-hour actual and a
    ledger redirected into ``tmp``."""
    fake_cfg = mock.Mock()
    fake_cfg.paths.forecast_csv = tmp / "ieso_forecast.csv"
    with mock.patch.object(
        C, "load_actuals", side_effect=lambda cutoff=None: pd.Series({T: actual_value})
    ), mock.patch.object(C, "load_config", return_value=fake_cfg):
        return C.settle_actuals(prov, TARGETS)


def test_write_once_and_revision() -> None:
    tmp = Path(tempfile.mkdtemp())

    # 1) first scrape -> one prelim row
    l1 = _run(tmp, 14000.0, _prov("sha_A", "2026-03-02 07:30"))
    assert len(l1) == 1, l1
    assert l1.iloc[0]["status"] == "prelim"
    assert l1.iloc[0]["actual_mw"] == 14000.0

    # 2) same value re-scraped -> NO new row; confirmed + promoted final
    l2 = _run(tmp, 14000.0, _prov("sha_A", "2026-03-02 07:30"))
    assert len(l2) == 1, f"unchanged re-scrape must not append: {l2}"
    assert l2.iloc[0]["status"] == "final"
    assert l2.iloc[0]["confirmed_unchanged_on"] not in ("", None)

    # 3) IESO restatement -> APPEND revision; original row preserved
    l3 = _run(tmp, 14250.0, _prov("sha_B", "2026-03-05 07:30"))
    assert len(l3) == 2, f"restatement must append, not overwrite: {l3}"
    hour_rows = l3[l3["target_hour"] == T].sort_values("scrape_time")
    assert list(hour_rows["actual_mw"]) == [14000.0, 14250.0], hour_rows
    assert (hour_rows["actual_mw"] == 14000.0).any(), "original value lost!"
    assert hour_rows.iloc[-1]["status"] == "revised"

    # gradable = latest value
    grad = C.gradable_actuals(l3)
    assert grad.loc[T] == 14250.0, grad

    print("settlement integrity: PASS")
    print("  prelim -> final on unchanged re-scrape; restatement appended")
    print("  (original value preserved; gradable = latest)")


if __name__ == "__main__":
    test_write_once_and_revision()
