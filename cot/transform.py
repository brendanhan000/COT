"""Derived metrics: net, COT Index (causal), %OI, WoW change, divergence, checks.

The two functions worth reading carefully are :func:`cot_index` (the headline
metric; trailing-window stochastic, no lookahead) and :func:`tidy_positions`
(reshape raw wide CFTC rows into a tidy long frame using tolerant field lookup).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import (
    CONTRACT_FIELD,
    DATE_FIELD,
    DEFAULT_WINDOW,
    OI_FIELD,
    ReportSpec,
)


# --------------------------------------------------------------------------
# field resolution
# --------------------------------------------------------------------------
def resolve_field(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    """First candidate present in ``columns`` (handles CFTC's naming quirks)."""
    colset = set(columns)
    for cand in candidates:
        if cand in colset:
            return cand
    return None


def _num(df: pd.DataFrame, col: Optional[str]) -> pd.Series:
    """Numeric view of a column; all-NaN if the column is absent."""
    if col is None or col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


# --------------------------------------------------------------------------
# COT Index (the headline metric)
# --------------------------------------------------------------------------
def cot_index(
    net: pd.Series,
    window: int = DEFAULT_WINDOW,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Causal stochastic normalisation of ``net`` to 0-100 over a trailing window.

        COT_t = 100 * (net_t - min(window)) / (max(window) - min(window))

    * Trailing window only: the value at t uses rows (t-window+1 .. t), never the
      future. (pandas ``rolling`` is inherently backward-looking.)
    * ``min_periods`` defaults to the full window (NaN until 3y of history).
    * Flat window (max == min) -> 50.0 (neutral) to avoid divide-by-zero.

    ``net`` is assumed already sorted ascending by date.
    """
    if min_periods is None:
        min_periods = window
    roll = net.rolling(window=window, min_periods=min_periods)
    lo = roll.min()
    hi = roll.max()
    rng = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        idx = (net - lo) / rng * 100.0
    # where the window is flat (rng==0) but populated -> neutral 50
    idx = idx.mask(rng == 0, 50.0)
    # leave NaN where the window is not yet full (rng is NaN)
    return idx


# --------------------------------------------------------------------------
# tidy reshape
# --------------------------------------------------------------------------
def tidy_positions(raw: pd.DataFrame, spec: ReportSpec) -> pd.DataFrame:
    """Wide CFTC rows -> tidy long frame: one row per (date, contract, category)."""
    if raw.empty:
        return pd.DataFrame(
            columns=[
                "report_date",
                "contract",
                "report_type",
                "fut_combined",
                "category",
                "category_label",
                "long",
                "short",
                "spread",
                "net",
                "open_interest",
                "traders_long",
                "traders_short",
                "pct_oi_long",
                "pct_oi_short",
            ]
        )

    cols = list(raw.columns)
    base = pd.DataFrame(
        {
            "report_date": pd.to_datetime(raw[DATE_FIELD]),
            "contract": raw[CONTRACT_FIELD].astype(str),
            "open_interest": _num(raw, OI_FIELD),
        }
    )
    fut_combined_col = "futonly_or_combined"
    base["fut_combined"] = (
        raw[fut_combined_col].astype(str) if fut_combined_col in raw.columns else ""
    )
    base["report_type"] = spec.report_type

    frames: List[pd.DataFrame] = []
    for cat in spec.categories:
        long_col = resolve_field(cols, cat.long)
        short_col = resolve_field(cols, cat.short)
        spread_col = resolve_field(cols, cat.spread)
        tl_col = resolve_field(cols, cat.traders_long)
        ts_col = resolve_field(cols, cat.traders_short)
        pl_col = resolve_field(cols, cat.pct_oi_long)
        ps_col = resolve_field(cols, cat.pct_oi_short)

        longs = _num(raw, long_col)
        shorts = _num(raw, short_col)
        part = base.copy()
        part["category"] = cat.key
        part["category_label"] = cat.label
        part["long"] = longs
        part["short"] = shorts
        part["spread"] = _num(raw, spread_col)
        part["net"] = longs - shorts  # spreading deliberately excluded from net
        part["traders_long"] = _num(raw, tl_col)
        part["traders_short"] = _num(raw, ts_col)
        part["pct_oi_long"] = _num(raw, pl_col)
        part["pct_oi_short"] = _num(raw, ps_col)
        frames.append(part)

    tidy = pd.concat(frames, ignore_index=True)
    tidy = tidy.sort_values(["contract", "category", "report_date"]).reset_index(
        drop=True
    )
    return tidy


# --------------------------------------------------------------------------
# enrich: COT index, WoW change, %OI of net
# --------------------------------------------------------------------------
def enrich(
    tidy: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """Add cot_index, net_chg_wow, pct_oi_net per (contract, category) series."""
    if tidy.empty:
        out = tidy.copy()
        for c in ("cot_index", "net_chg_wow", "pct_oi_net"):
            out[c] = pd.Series(dtype="float64")
        return out

    out = tidy.sort_values(["contract", "category", "report_date"]).copy()
    grp = out.groupby(["contract", "category"], sort=False)
    out["cot_index"] = grp["net"].transform(
        lambda s: cot_index(s, window=window, min_periods=min_periods)
    )
    out["net_chg_wow"] = grp["net"].transform(lambda s: s.diff())
    out["pct_oi_net"] = np.where(
        out["open_interest"] > 0, out["net"] / out["open_interest"] * 100.0, np.nan
    )
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# divergence (descriptive only)
# --------------------------------------------------------------------------
def divergence(tidy: pd.DataFrame, spec: ReportSpec) -> pd.DataFrame:
    """Net(a) - Net(b) per (contract, date). DESCRIPTIVE, not predictive."""
    a, b = spec.divergence_pair
    wide = tidy.pivot_table(
        index=["contract", "report_date"], columns="category", values="net"
    )
    if a not in wide.columns or b not in wide.columns:
        return pd.DataFrame(columns=["contract", "report_date", "divergence"])
    out = wide.reset_index()[["contract", "report_date"]].copy()
    out["divergence"] = (wide[a] - wide[b]).values
    out["pair"] = "%s_minus_%s" % (a, b)
    return out


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
@dataclass
class Validation:
    warnings: List[str]

    def ok(self) -> bool:
        return not self.warnings


def reconcile_open_interest(
    raw: pd.DataFrame, spec: ReportSpec, tolerance: float = 0.01
) -> List[str]:
    """Check long-side & short-side accounting identity vs open interest.

    OI(long) = sum(category longs) + sum(category spreads); same for short.
    Spreading counts equally on both sides, so it is added to each side.
    """
    warns: List[str] = []
    if raw.empty:
        return warns
    cols = list(raw.columns)
    oi = _num(raw, OI_FIELD)

    long_total = pd.Series(0.0, index=raw.index)
    short_total = pd.Series(0.0, index=raw.index)
    spread_total = pd.Series(0.0, index=raw.index)
    for cat in spec.categories:
        long_total = long_total.add(_num(raw, resolve_field(cols, cat.long)), fill_value=0)
        short_total = short_total.add(
            _num(raw, resolve_field(cols, cat.short)), fill_value=0
        )
        sp = resolve_field(cols, cat.spread)
        if sp is not None:
            spread_total = spread_total.add(_num(raw, sp), fill_value=0)

    long_side = long_total + spread_total
    short_side = short_total + spread_total
    denom = oi.replace(0, np.nan)
    long_err = (long_side - oi).abs() / denom
    short_err = (short_side - oi).abs() / denom

    bad = raw.loc[(long_err > tolerance) | (short_err > tolerance)]
    if not bad.empty:
        n = len(bad)
        worst = float(pd.concat([long_err, short_err], axis=1).max(axis=1).max())
        warns.append(
            "OI reconciliation: %d row(s) where long/short+spreading deviates from "
            "open interest by >%.1f%% (worst %.2f%%). Check for footnoted/adjusted weeks."
            % (n, tolerance * 100, worst * 100)
        )
    return warns


def validate(
    tidy: pd.DataFrame,
    raw: pd.DataFrame,
    spec: ReportSpec,
    requested_contracts: Sequence[str],
    resolved_contracts: Sequence[Optional[str]],
) -> Validation:
    """Collect non-fatal data-quality warnings."""
    warns: List[str] = []

    # contracts that resolved to nothing
    for req, res in zip(requested_contracts, resolved_contracts):
        if not res:
            warns.append("Contract %r matched no official contract_market_name." % req)

    if not tidy.empty:
        # contracts present in request but with zero rows
        present = set(tidy["contract"].unique())
        for res in resolved_contracts:
            if res and res not in present:
                warns.append("Contract %r returned no rows." % res)

        # weekly gaps per contract (expected 7-day cadence)
        for contract, sub in tidy.groupby("contract"):
            dates = (
                sub[["report_date"]]
                .drop_duplicates()
                .sort_values("report_date")["report_date"]
            )
            if len(dates) < 2:
                continue
            deltas = dates.diff().dropna().dt.days
            gaps = int((deltas > 7).sum())
            if gaps:
                biggest = int(deltas.max())
                warns.append(
                    "%s: %d gap(s) in the weekly series (largest %d days). "
                    "Holiday weeks can cause this." % (contract, gaps, biggest)
                )

        # non-Tuesday as-of dates
        weekdays = tidy["report_date"].dt.weekday
        non_tue = int((weekdays != 1).sum())
        if non_tue:
            warns.append(
                "%d row(s) have an as-of date that is not a Tuesday (holiday shift?)."
                % non_tue
            )

    warns.extend(reconcile_open_interest(raw, spec))
    return Validation(warns)


def latest_snapshot(enriched: pd.DataFrame) -> pd.DataFrame:
    """Most recent row per (contract, category) for labelling/printing."""
    if enriched.empty:
        return enriched
    idx = enriched.groupby(["contract", "category"])["report_date"].idxmax()
    return enriched.loc[idx].reset_index(drop=True)
