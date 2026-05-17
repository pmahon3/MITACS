"""IESO published day-ahead demand forecast scraper + parser.

Source: the IESO **Adequacy Report** (Adequacy3 schema) at
``https://reports-public.ieso.ca/public/Adequacy3/``. This is the modern
replacement for the old ``OntarioZonalDemand`` report the project once
scraped (that path now 404s; recovered from git history 378cccf for
reference only). Each ``PUB_Adequacy3_YYYYMMDD.xml`` is the final
day-ahead adequacy report for delivery date YYYYMMDD; the ``_vNNN``
variants are intraday revisions and are skipped (we want the settled
forecast, mirroring the old scraper's intent).

The Ontario demand forecast lives at (verified against a real file):

    <DeliveryDate>YYYY-MM-DD</DeliveryDate>
    ...
    <OntarioDemand><ForecastOntDemand>
       <Demand><DeliveryHour>1..24</DeliveryHour><EnergyMW>...</EnergyMW></Demand>
       ...
    </ForecastOntDemand></OntarioDemand>

Parsed with a real XML parser (``xml.etree``), not the old fragile
string-splitting. Output: a tidy CSV ``datetime, forecast_mw`` (hour h ->
timestamp DeliveryDate + (h-1) hours, matching the actuals convention in
``pre_processing``: IESO hour 1 == 00:00).

Run::

    python -m data.forecast_scraping            # scrape + parse, config window
    python -m data.forecast_scraping --check    # parse-validate one file only
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

from config import PipelineConfig, load_config

BASE_URL = "https://reports-public.ieso.ca/public/Adequacy3/"


def _final_filenames() -> list[str]:
    """All final ``PUB_Adequacy3_YYYYMMDD.xml`` names currently in the
    index (skips ``_vNNN`` intraday revisions).

    No date filter: IESO retains only a rolling ~4-month window and there
    is no historical forecast archive (verified) -- so the project's
    actuals year-range does not apply here. This is a FORWARD-ARCHIVING
    scraper: each run grabs whatever is currently published and merges it
    into the accumulating CSV; the comparison window grows over calendar
    time.
    """
    import re

    resp = requests.get(BASE_URL, timeout=60)
    resp.raise_for_status()
    pat = re.compile(r"PUB_Adequacy3_\d{8}\.xml")
    return sorted(set(pat.findall(resp.text)))


def _download(name: str, dest_dir: Path) -> Path | None:
    dest = dest_dir / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest  # idempotent
    try:
        r = requests.get(BASE_URL + name, timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return dest
    except requests.RequestException as e:
        print(f"  ! failed {name}: {e}", file=sys.stderr)
        return None


def parse_forecast_xml(path: Path) -> pd.DataFrame:
    """Parse one Adequacy3 file -> DataFrame[datetime, forecast_mw].

    Raises ``ValueError`` if the expected nodes are absent (so a schema
    change fails loudly rather than silently yielding empty data).
    """
    root = ET.parse(path).getroot()

    # Tags are namespaced in the real files; match on local-name.
    def localname(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    def find_first(node, name):
        for el in node.iter():
            if localname(el.tag) == name:
                return el
        return None

    dd = find_first(root, "DeliveryDate")
    fod = find_first(root, "ForecastOntDemand")
    if dd is None or fod is None or dd.text is None:
        raise ValueError(
            f"{path.name}: missing DeliveryDate/ForecastOntDemand "
            f"(schema change or non-forecast file?)"
        )
    base = pd.Timestamp(dd.text.strip())

    rows = []
    for dem in fod.iter():
        if localname(dem.tag) != "Demand":
            continue
        hr = mw = None
        for ch in dem:
            ln = localname(ch.tag)
            if ln == "DeliveryHour":
                hr = int(ch.text)
            elif ln == "EnergyMW":
                mw = float(ch.text)
        if hr is not None and mw is not None:
            rows.append((base + pd.Timedelta(hours=hr - 1), mw))

    if not rows:
        raise ValueError(f"{path.name}: ForecastOntDemand had no Demand rows")
    return pd.DataFrame(rows, columns=["datetime", "forecast_mw"])


def main(cfg: PipelineConfig, check_only: bool = False) -> None:
    xml_dir = cfg.paths.forecast_xml_dir
    xml_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.forecast_csv.parent.mkdir(parents=True, exist_ok=True)

    if check_only:
        names = _final_filenames()
        if not names:
            raise SystemExit("no Adequacy3 files in the public index")
        p = _download(names[-1], xml_dir)
        df = parse_forecast_xml(p)
        print(f"parse-validated {p.name}: {len(df)} rows")
        print(df.head(3).to_string(index=False))
        print(df.tail(2).to_string(index=False))
        return

    names = _final_filenames()
    print(
        f"{len(names)} final Adequacy3 files currently published "
        f"(rolling window; forward-archiving)"
    )
    frames = []
    for i, name in enumerate(names, 1):
        p = _download(name, xml_dir)
        if p is None:
            continue
        try:
            frames.append(parse_forecast_xml(p))
        except ValueError as e:
            print(f"  ! skip {name}: {e}", file=sys.stderr)
        if i % 200 == 0:
            print(f"  ...{i}/{len(names)}")

    if not frames:
        raise SystemExit("no forecast rows parsed")
    new = pd.concat(frames, ignore_index=True)

    # Forward accumulation: merge with any previously-archived CSV so the
    # comparison window grows monotonically across runs. On duplicate
    # datetimes keep the latest scrape (last write wins).
    csv = cfg.paths.forecast_csv
    if csv.exists():
        prior = pd.read_csv(csv, parse_dates=["datetime"])
        new = pd.concat([prior, new], ignore_index=True)
    out = (
        new.drop_duplicates("datetime", keep="last")
        .sort_values("datetime")
        .set_index("datetime")
    )
    out.to_csv(csv)
    print(
        f"archived {len(out)} total forecast rows "
        f"({out.index.min()} .. {out.index.max()}) -> {csv}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check", action="store_true",
        help="download+parse ONE latest file to validate the schema, no full scrape",
    )
    args = ap.parse_args()
    main(load_config(), check_only=args.check)
