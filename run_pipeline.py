#!/usr/bin/env python3
"""Top-level pipeline orchestration.

Runs the stages in order, driven entirely by ``config/pipeline.yaml``, with
skip-if-exists idempotency and partial-run selection.

Examples::

    python run_pipeline.py --dry-run
    python run_pipeline.py --from detrend --to dimensions
    python run_pipeline.py --only innovations --daytype saturday
    python run_pipeline.py --force --only clustering

Stage order:
    scrape -> pickle -> detrend -> clustering -> dimensions -> innovations

The legacy ``locality/{pointwise,global}`` sweep stages are intentionally
NOT registered: they crash on the current edynamics WLS API and their fate
is deferred until after the first full Ontario run.
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from config import PipelineConfig, load_config

logger = logging.getLogger("pipeline")


# ──────────────────────────────────────────────────────────────────────────
# Stage adapters
# ──────────────────────────────────────────────────────────────────────────
def _stage_scrape(cfg: PipelineConfig, _dt: str | None) -> None:
    from data.scraping import download_historical_reports

    download_historical_reports(str(cfg.paths.historical_csvs))


def _stage_pickle(cfg: PipelineConfig, _dt: str | None) -> None:
    from data.ontario.pickling import convert_csvs_to_pickles

    convert_csvs_to_pickles(
        input_folder=str(cfg.paths.historical_csvs),
        output_folder=str(cfg.paths.pickled_dir),
    )


def _stage_detrend(cfg: PipelineConfig, _dt: str | None) -> None:
    from pre_processing.year_wise_standardization import main as detrend_main

    detrend_main(cfg)


def _stage_clustering(cfg: PipelineConfig, _dt: str | None) -> None:
    from processing.clustering.process import main as clustering_main

    clustering_main(cfg)


def _stage_dimensions(cfg: PipelineConfig, _dt: str | None) -> None:
    from processing.dimensions.process import main as dimensions_main

    dimensions_main(cfg)


def _stage_innovations(cfg: PipelineConfig, dt: str | None) -> None:
    from processing.innovations.process import main as innovations_main

    innovations_main(cfg, dt)


# ──────────────────────────────────────────────────────────────────────────
# Output predicates (for skip-if-exists)
# ──────────────────────────────────────────────────────────────────────────
def _out_scrape(cfg: PipelineConfig, _dt: str | None) -> list[Path]:
    return [cfg.paths.historical_csvs]


def _out_pickle(cfg: PipelineConfig, _dt: str | None) -> list[Path]:
    return [cfg.paths.pickled_dir]


def _out_detrend(cfg: PipelineConfig, _dt: str | None) -> list[Path]:
    return [cfg.paths.stationary_candidate, cfg.paths.annual_params]


def _out_clustering(cfg: PipelineConfig, _dt: str | None) -> list[Path]:
    return [cfg.paths.clustered_csv]


def _out_dimensions(cfg: PipelineConfig, _dt: str | None) -> list[Path]:
    return [cfg.dimensions_csv(dt) for dt in cfg.data.daytypes]


def _out_innovations(cfg: PipelineConfig, dt: str | None) -> list[Path]:
    dts = (dt,) if dt else cfg.data.daytypes
    return [cfg.paths.operator_output_dir / f"coeffs_{d}.pt" for d in dts]


@dataclass
class Stage:
    name: str
    fn: Callable[[PipelineConfig, str | None], None]
    outputs: Callable[[PipelineConfig, str | None], list[Path]]
    per_daytype: bool = False


REGISTRY: list[Stage] = [
    Stage("scrape", _stage_scrape, _out_scrape),
    Stage("pickle", _stage_pickle, _out_pickle),
    Stage("detrend", _stage_detrend, _out_detrend),
    Stage("clustering", _stage_clustering, _out_clustering),
    Stage("dimensions", _stage_dimensions, _out_dimensions),
    Stage("innovations", _stage_innovations, _out_innovations, per_daytype=True),
]
STAGE_NAMES = [s.name for s in REGISTRY]


def _exists(paths: list[Path]) -> bool:
    return bool(paths) and all(p.exists() for p in paths)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="frm", choices=STAGE_NAMES)
    ap.add_argument("--to", dest="to", choices=STAGE_NAMES)
    ap.add_argument("--only", nargs="+", choices=STAGE_NAMES)
    ap.add_argument("--daytype", help="restrict per-daytype stages to one day type")
    ap.add_argument("--force", action="store_true", help="ignore skip-if-exists")
    ap.add_argument("--dry-run", action="store_true", help="print plan only")
    ap.add_argument(
        "--profile",
        choices=["fast", "full"],
        help="override the config run profile (fast = cheap validation pass)",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(profile=args.profile)
    logger.info("Profile: %s", cfg.profile)

    if args.only:
        selected = [s for s in REGISTRY if s.name in args.only]
    else:
        i = STAGE_NAMES.index(args.frm) if args.frm else 0
        j = STAGE_NAMES.index(args.to) + 1 if args.to else len(REGISTRY)
        selected = REGISTRY[i:j]

    logger.info(
        "Plan: %s%s",
        " -> ".join(s.name for s in selected),
        f"  (daytype={args.daytype})" if args.daytype else "",
    )
    if args.daytype:
        # surface the LocalGLSelector cost knobs up front
        logger.info(
            "Sweep grids: theta=%d, sigma=%d, sample_frac=%.2f "
            "(LocalGLSelector cost ~ anchors x theta x sigma lstsq calls)",
            cfg.theta.theta_count,
            cfg.theta.sigma_count,
            cfg.sampling.sample_frac,
        )

    for stage in selected:
        dt = args.daytype if stage.per_daytype else None
        outs = stage.outputs(cfg, dt)
        if not args.force and _exists(outs):
            logger.info("SKIP  %-12s (outputs exist)", stage.name)
            continue
        if args.dry_run:
            logger.info("RUN   %-12s -> %s", stage.name, [str(p) for p in outs])
            continue
        logger.info("RUN   %-12s ...", stage.name)
        t0 = time.time()
        stage.fn(cfg, dt)
        logger.info("DONE  %-12s (%.1fs)", stage.name, time.time() - t0)


if __name__ == "__main__":
    main()
