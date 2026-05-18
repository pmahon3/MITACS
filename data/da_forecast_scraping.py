"""IESO Day-Ahead Totals (DATotals) scraper -- the TRUE day-ahead forecast.

This is the correct product for the day-ahead head-to-head. The earlier
``forecast_scraping`` (Adequacy3) is a 35-day reliability OUTLOOK whose
public archive keeps only the end-of-delivery-day final -- comparing our
strict D-1 forecast to that was a horizon mismatch, not a model result
(see memory ``mitacs-ieso-product-correction``).

DATotals: ``PUB_DATotals_YYYYMMDD.xml`` has ``<CreatedAt>`` ~D-1 midday
(verified, e.g. 2026-05-17T12:31 for delivery 2026-05-18) and a single
``<DeliveryDate> = D`` -- a genuine day-ahead forecast. We keep the
final ``PUB_DATotals_<D>.xml`` (the settled day-ahead issue) and skip
``_vNNN`` intraday-of-D-1 revisions. ``CreatedAt`` is recorded so the
issue-time horizon is auditable (our predictor issues D-1 23:00, ~10h
later than IESO's ~D-1 12:30 -- a small, honestly-stated edge to us).

Demand path (verified):
  DocBody>DeliveryDate ; Energies>HourlyEnergy>DeliveryHour ;
  MQ[MarketQuantity == "Total Load"]>EnergyMW   (Ontario demand fcst MW)
IESO hour h -> timestamp DeliveryDate + (h-1)h (matches actuals convention).

Run::  python -m data.da_forecast_scraping            # scrape+parse
       python -m data.da_forecast_scraping --check     # 1-file schema check
"""
from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

from config import PipelineConfig, load_config

BASE_URL = "https://reports-public.ieso.ca/public/DATotals/"
_FINAL_RE = re.compile(r"PUB_DATotals_\d{8}\.xml")  # excludes _vNNN and the bare PUB_DATotals.xml


def _final_filenames() -> list[str]:
    r = requests.get(BASE_URL, timeout=60)
    r.raise_for_status()
    return sorted(set(_FINAL_RE.findall(r.text)))


def _download(name: str, dest_dir: Path) -> Path | None:
    dest = dest_dir / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    try:
        r = requests.get(BASE_URL + name, timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return dest
    except requests.RequestException as e:
        print(f"  ! failed {name}: {e}", file=sys.stderr)
        return None


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_da_xml(path: Path) -> pd.DataFrame:
    """One DATotals file -> DataFrame[datetime, da_forecast_mw, ieso_created_at].

    Raises ValueError if expected nodes are absent (schema change fails
    loudly, never silently empty).
    """
    root = ET.parse(path).getroot()

    def find_first(node, name):
        for el in node.iter():
            if _localname(el.tag) == name:
                return el
        return None

    dd = find_first(root, "DeliveryDate")
    created = find_first(root, "CreatedAt")
    if dd is None or dd.text is None:
        raise ValueError(f"{path.name}: no DeliveryDate")
    base = pd.Timestamp(dd.text.strip())
    issue = created.text.strip() if created is not None and created.text else ""

    rows = []
    for he in root.iter():
        if _localname(he.tag) != "HourlyEnergy":
            continue
        hr = None
        load = None
        for ch in he.iter():
            ln = _localname(ch.tag)
            if ln == "DeliveryHour":
                hr = int(ch.text)
        # find the MQ block whose MarketQuantity == "Total Load"
        for mq in he.iter():
            if _localname(mq.tag) != "MQ":
                continue
            label = val = None
            for c in mq:
                cl = _localname(c.tag)
                if cl == "MarketQuantity":
                    label = (c.text or "").strip()
                elif cl == "EnergyMW":
                    val = c.text
            if label == "Total Load" and val is not None:
                load = float(val)
                break
        if hr is not None and load is not None:
            rows.append((base + pd.Timedelta(hours=hr - 1), load))

    if not rows:
        raise ValueError(
            f"{path.name}: no HourlyEnergy/'Total Load' rows "
            f"(schema change?)"
        )
    df = pd.DataFrame(rows, columns=["datetime", "da_forecast_mw"])
    df["ieso_created_at"] = issue
    return df


def main(cfg: PipelineConfig, check_only: bool = False) -> None:
    out_csv = cfg.paths.forecast_csv.parent / "ieso_da_forecast.csv"
    xml_dir = cfg.paths.forecast_xml_dir.parent / "da_xml"
    xml_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if check_only:
        names = _final_filenames()
        if not names:
            raise SystemExit("no DATotals files in the public index")
        p = _download(names[-1], xml_dir)
        df = parse_da_xml(p)
        print(f"parse-validated {p.name}: {len(df)} rows; "
              f"CreatedAt={df['ieso_created_at'].iloc[0]} "
              f"(D-1 issue => true day-ahead)")
        print(df.head(3).to_string(index=False))
        return

    names = _final_filenames()
    print(f"{len(names)} final DATotals files (rolling archive; "
          f"forward-accumulating)")
    frames = []
    for i, name in enumerate(names, 1):
        p = _download(name, xml_dir)
        if p is None:
            continue
        try:
            frames.append(parse_da_xml(p))
        except ValueError as e:
            print(f"  ! skip {name}: {e}", file=sys.stderr)
        if i % 50 == 0:
            print(f"  ...{i}/{len(names)}")

    if not frames:
        raise SystemExit("no DA forecast rows parsed")
    new = pd.concat(frames, ignore_index=True)
    if out_csv.exists():
        prior = pd.read_csv(out_csv, parse_dates=["datetime"])
        new = pd.concat([prior, new], ignore_index=True)
    out = (
        new.drop_duplicates("datetime", keep="last")
        .sort_values("datetime")
        .set_index("datetime")
    )
    out.to_csv(out_csv)
    print(f"archived {len(out)} DA-forecast rows "
          f"({out.index.min()} .. {out.index.max()}) -> {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="download+parse ONE latest file to validate schema")
    args = ap.parse_args()
    main(load_config(), check_only=args.check)
