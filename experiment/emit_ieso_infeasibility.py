"""Promote the IESO retrospective head-to-head INFEASIBILITY finding
(C2) to a provenanced, hashed, citable artifact.

This is likely the project's most novel methodology point: a directly-
verified negative result that no PUBLIC IESO archive is both
Ontario-demand-basis AND day-ahead-issued for historical dates. It is
**CLAIM-GRADE**. Beyond the header, the IESO external-data provenance
is stamped explicitly (access date + the exact probed scope) because
IESO restructures its public site — 404s encountered during this
project are direct evidence — so a bare prose date is not enough; the
access date and enumeration scope are recorded as structured inputs.

``frozen_spec_required=False``: the finding is about IESO's public
archive, not a particular frozen predictor run.

Run::

    python -m experiment.emit_ieso_infeasibility
"""
from __future__ import annotations

from config import PROJECT_ROOT

from .provenance import Grade, make_result

# Date the IESO public archive was enumerated/probed for this finding.
# Stamped as a structured input (not only prose) so a reviewer can see
# exactly when the negative result was established — IESO restructures.
IESO_ACCESS_DATE = "2026-05-18"
IESO_ARCHIVE_ROOT = "https://reports-public.ieso.ca/public/"

BODY = """IESO DAY-AHEAD HEAD-TO-HEAD: RETROSPECTIVE INFEASIBILITY (2026-05-18)
=====================================================================

Goal: compare our day-ahead Ontario-demand forecast vs IESO's published
day-ahead Ontario-demand forecast on historical (settled) delivery days.

FINDING: not achievable from IESO PUBLIC archives. Two independent,
directly-verified blockers:

1. Ontario-demand-basis product (Adequacy Report, Adequacy3
   `ForecastOntDemand`): for delivery days old enough to have settled
   actuals (the gradable Feb-mid-May 2026 window), the public archive
   retains ONLY the end-of-delivery-day version. Verified e.g. delivery
   2026-03-15: only PUB_Adequacy3_20260315.xml and _v120 survive, both
   CreatedAt 2026-03-15T23:53 (end of D). Probe: 0 of 18 gradable
   delivery days had any version with CreatedAt <= D-1 23:59. Wrong
   horizon (sees ~all of D); no recoverable day-ahead version.
   (A few very recent deliveries still retain early versions only
   because intraday revisions had not yet been purged - not the
   gradable window.)

2. True day-ahead-issued products (DATotals, PredispTotals): issued
   ~D-1 (DATotals CreatedAt ~D-1 12:30; Predisp ~D-1 05:10) BUT publish
   only market totals, NOT Ontario demand. "Total Load" runs a
   consistent +2485 MW (median +2499, std 767) above Ontario Demand -
   a definitional offset (losses + dispatchable), not forecast error.
   No Ontario-demand field exists in either
   (MarketQuantity = Total Energy/Loss/Load/Dispatchable/10S/10N/30R).

SCOPE OF REVIEW: all 133 directories at reports-public.ieso.ca/public/
were enumerated and every plausible demand/load-forecast candidate
directly inspected. Non-obvious candidates checked & ruled out:
  - HourlyLFDA      : loss-factor / LFDC $/MWh rate, not a load forecast
  - OntarioElectEnergy: stale energy summary (CreatedAt 2025-05-01)
  - DAHourlyZonal   : day-ahead but ZONAL PRICE (LMP), not demand
  - PredispHourlyZonal: zonal, predispatch -- price/zonal, not Ont demand
Demand-bearing day-ahead products (DATotals, PredispTotals) are
market-Total-Load basis only (see blocker 2).

No PUBLIC IESO archive is BOTH Ontario-demand-basis AND day-ahead-issued
for historical dates. (Verified by full directory enumeration, not a
sampled subset. A direct IESO data request / privileged feed might
surface it; out of scope here -- writeup must say "public archives",
not "any source".)

DISCARDED (product/quantity mismatches, NOT recorded as results):
  - vs Adequacy3 end-of-D final : ours 5.60% / "IESO" 1.88% MAPE
  - vs DATotals Total Load      : ours 5.57% / "IESO" 15.83% MAPE
Both artifacts of comparing against the wrong IESO quantity/horizon.

VALID PATH: the forward registered experiment (experiment/, Tasks
24-27) captures Adequacy3 `ForecastOntDemand` PROSPECTIVELY at D-1
(before purge-down), enabling a fair head-to-head as calendar time
accrues. This is why the prospective infrastructure was built; it is
the methodologically correct comparison, not a workaround.

UNAFFECTED: the model's own validated result stands independently --
Step-4 historical backtest, 500 out-of-sample post-cutoff days,
leakage directly verified (see the provenanced C1 backtest artifact
under experiment/results/, which carries its own header + body hash).
"""


def main() -> None:
    out = (
        PROJECT_ROOT / "experiment" / "results"
        / "ieso_headtohead_infeasibility.txt"
    )
    hdr = make_result(
        path=out,
        grade=Grade.CLAIM,
        title="IESO retrospective day-ahead head-to-head: infeasible "
              "from public archives (directly-verified negative result)",
        body=BODY,
        inputs={
            "ieso_archive_root": IESO_ARCHIVE_ROOT,
            "ieso_access_date": IESO_ACCESS_DATE,
            "ieso_enumeration_scope": "all 133 public directories "
            "enumerated; every demand/load-forecast candidate directly "
            "inspected (not a sampled subset)",
            "ieso_restructures_caveat": "IESO restructures its public "
            "site; this negative result is as of the access date above "
            "and may not hold for a re-probe at a later date",
        },
        seeds={},
        frozen_spec_required=False,
    )
    print(f"wrote provenanced C2 artifact -> {out}")
    print(f"  grade              = {hdr['grade']}")
    print(f"  ieso_access_date   = {IESO_ACCESS_DATE}")
    print(f"  inputs_fingerprint = {hdr['inputs_fingerprint'][:16]}...")
    print(f"  body_sha256        = {hdr['body_sha256'][:16]}...")


if __name__ == "__main__":
    main()
