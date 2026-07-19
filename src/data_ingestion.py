"""Fetch India CPI data (monthly index) and derive YoY / MoM inflation.

Primary source: FRED series INDCPIALLMINMEI (OECD Main Economic Indicators).
Fallback: DBnomics mirror of the IMF CPI database (series M.IN.PCPI_IX).
Both are free and require no API key, keeping the pipeline fully reproducible.
"""
from __future__ import annotations

import io
import pathlib

import pandas as pd
import requests
import yaml

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
DBNOMICS_URL = "https://api.db.nomics.world/v22/series/{sid}?observations=1"


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def fetch_fred(series_id: str) -> pd.DataFrame:
    """Download a series from FRED as a two-column frame (date, cpi_index)."""
    resp = requests.get(FRED_URL.format(sid=series_id), timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = ["date", "cpi_index"]
    df["date"] = pd.to_datetime(df["date"])
    df["cpi_index"] = pd.to_numeric(df["cpi_index"], errors="coerce")
    return df.dropna()


def fetch_dbnomics(series_id: str) -> pd.DataFrame:
    """Fallback fetch from DBnomics (IMF CPI, index 2010=100)."""
    resp = requests.get(DBNOMICS_URL.format(sid=series_id), timeout=60)
    resp.raise_for_status()
    doc = resp.json()["series"]["docs"][0]
    df = pd.DataFrame({"date": doc["period"], "cpi_index": doc["value"]})
    df["date"] = pd.to_datetime(df["date"])
    df["cpi_index"] = pd.to_numeric(df["cpi_index"], errors="coerce")
    return df.dropna()


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Basic sanity checks: monotone dates, no long gaps, positive index."""
    df = df.sort_values("date").reset_index(drop=True)
    assert (df["cpi_index"] > 0).all(), "CPI index must be positive"
    gaps = df["date"].diff().dt.days.dropna()
    if (gaps > 62).any():
        n = int((gaps > 62).sum())
        print(f"warning: {n} gap(s) longer than 2 months in the series")
    return df


def add_inflation(df: pd.DataFrame) -> pd.DataFrame:
    """Derive YoY and MoM inflation from the index level."""
    df = df.copy()
    df["inflation_yoy"] = df["cpi_index"].pct_change(12) * 100
    df["inflation_mom"] = df["cpi_index"].pct_change(1) * 100
    return df


def main() -> None:
    cfg = load_config()
    try:
        df = fetch_fred(cfg["data"]["fred_series_id"])
        source = "FRED"
    except Exception as exc:  # network / schema failure -> fallback
        print(f"FRED fetch failed ({exc}); falling back to DBnomics")
        df = fetch_dbnomics(cfg["data"]["dbnomics_series"])
        source = "DBnomics"

    df = validate(df)
    raw_path = pathlib.Path(cfg["data"]["raw_path"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)

    df = df[df["date"] >= cfg["data"]["start_date"]]
    df = add_inflation(df).dropna(subset=["inflation_yoy"])
    out = pathlib.Path(cfg["data"]["processed_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"source: {source}")
    print(f"rows: {len(df)}, span: {df['date'].min():%Y-%m} to {df['date'].max():%Y-%m}")
    print(f"latest YoY inflation: {df['inflation_yoy'].iloc[-1]:.2f}%")


if __name__ == "__main__":
    main()
