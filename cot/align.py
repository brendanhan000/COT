"""Tuesday indexing + price alignment.

The CFTC ``report_date_as_yyyy_mm_dd`` is already the Tuesday as-of date. We
verify that, then align a daily price proxy to each Tuesday using a backward
``merge_asof`` (last close on/before the Tuesday) so holidays never misalign.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

TUESDAY = 1  # Monday=0 ... Sunday=6


@dataclass
class PriceResult:
    aligned: pd.DataFrame          # columns: report_date, price
    ticker: Optional[str]
    note: str
    ok: bool


def check_tuesday(dates: pd.Series) -> List[str]:
    """Return warnings for any as-of date that is not a Tuesday."""
    d = pd.to_datetime(dates)
    bad = d[d.dt.weekday != TUESDAY]
    if bad.empty:
        return []
    sample = ", ".join(pd.Series(bad.dt.date.unique()).astype(str)[:5])
    return [
        "%d as-of date(s) are not Tuesday (e.g. %s). Using the official as-of date "
        "as the index regardless." % (bad.nunique(), sample)
    ]


def normalize_tuesday_index(df: pd.DataFrame, date_col: str = "report_date") -> pd.DataFrame:
    """Ensure the date column is tz-naive midnight datetimes (the Tuesday key)."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.tz_localize(None).dt.normalize()
    return out


def fetch_price_series(
    ticker: str,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> pd.DataFrame:
    """Daily close for ``ticker`` from yfinance. Empty frame on any failure."""
    try:
        import yfinance as yf
    except Exception:  # yfinance not installed
        return pd.DataFrame(columns=["date", "price"])

    kwargs = dict(auto_adjust=False, progress=False)
    if start is not None:
        kwargs["start"] = pd.Timestamp(start).strftime("%Y-%m-%d")
    if end is not None:
        # pad a day so the final Tuesday's close is included
        kwargs["end"] = (pd.Timestamp(end) + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    try:
        raw = yf.download(ticker, **kwargs)
    except Exception:
        return pd.DataFrame(columns=["date", "price"])
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "price"])

    # yfinance may return single- or multi-index columns depending on version
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    out = close.reset_index()
    out.columns = ["date", "price"]
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    out = out.dropna(subset=["price"]).sort_values("date").reset_index(drop=True)
    return out


def align_price_to_tuesdays(
    price: pd.DataFrame, tuesdays: pd.Series
) -> pd.DataFrame:
    """Backward as-of join: each Tuesday -> last close on/before that Tuesday."""
    tue = (
        pd.DataFrame({"report_date": pd.to_datetime(pd.Series(tuesdays).unique())})
        .sort_values("report_date")
        .reset_index(drop=True)
    )
    tue["report_date"] = tue["report_date"].dt.tz_localize(None).dt.normalize()
    if price.empty:
        tue["price"] = pd.NA
        return tue
    merged = pd.merge_asof(
        tue,
        price.rename(columns={"date": "report_date"}),
        on="report_date",
        direction="backward",
    )
    return merged


def build_price_overlay(
    contract: str,
    ticker: Optional[str],
    tuesdays: pd.Series,
) -> PriceResult:
    """Fetch + align a proxy price for one contract's Tuesday series."""
    if not ticker:
        return PriceResult(
            aligned=pd.DataFrame(columns=["report_date", "price"]),
            ticker=None,
            note="no price proxy mapped for %r; overlay skipped" % contract,
            ok=False,
        )
    tdates = pd.to_datetime(pd.Series(tuesdays))
    price = fetch_price_series(ticker, tdates.min(), tdates.max())
    aligned = align_price_to_tuesdays(price, tdates)
    ok = aligned["price"].notna().any()
    note = (
        "PROXY price = %s (yfinance daily close, aligned backward to Tuesday)" % ticker
        if ok
        else "price fetch for proxy %s returned nothing; overlay skipped" % ticker
    )
    return PriceResult(aligned=aligned, ticker=ticker, note=note, ok=ok)
