"""Live registered prediction experiment.

A frozen, pre-registered out-of-sample protocol: the model is specified
entirely on demand actuals up to a fixed cutoff (2024-12-31), IESO
publishes day-ahead forecasts on a known cadence, actuals settle later,
and our forecast vs IESO's forecast vs actual accrue into an immutable
ledger over calendar time. The freeze (hash-stamped spec) is what makes
this a *registered* prediction rather than a retrospective fit -- the
model cannot be silently changed after forecasts are issued.

Components:
  freeze.py   -- hash-stamped immutable predictor spec (pre-registration)
  predict.py  -- frozen forecast generator (refuses on hash mismatch)
  collect.py  -- scheduled scrape + actuals settlement
  ledger.py   -- append-only experiment record
  score.py    -- skill / calibration over the accrued settled window
"""
