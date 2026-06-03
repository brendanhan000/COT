"""Tidy reshape (net = long - short, tolerant field lookup) + OI reconciliation."""
import pandas as pd

from cot.config import get_report_spec
from cot.transform import (
    enrich,
    reconcile_open_interest,
    resolve_field,
    tidy_positions,
)

SPEC = get_report_spec("legacy")


def _raw(open_interest=180):
    # Long side  = 100+50+20 + spread 10 = 180
    # Short side = 40 +90+40 + spread 10 = 180
    return pd.DataFrame(
        {
            "report_date_as_yyyy_mm_dd": pd.to_datetime(["2026-05-26"]),
            "contract_market_name": ["E-MINI S&P 500"],
            "futonly_or_combined": ["FutOnly"],
            "open_interest_all": [open_interest],
            "noncomm_positions_long_all": [100],
            "noncomm_positions_short_all": [40],
            "noncomm_postions_spread_all": [10],  # NOTE: real CFTC typo column
            "comm_positions_long_all": [50],
            "comm_positions_short_all": [90],
            "nonrept_positions_long_all": [20],
            "nonrept_positions_short_all": [40],
        }
    )


def test_resolve_field_prefers_first_present_candidate():
    cols = ["noncomm_postions_spread_all", "x"]
    cand = ["noncomm_positions_spread_all", "noncomm_postions_spread_all"]
    assert resolve_field(cols, cand) == "noncomm_postions_spread_all"
    assert resolve_field(["a"], ["nope"]) is None


def test_tidy_net_and_typo_spread():
    tidy = tidy_positions(_raw(), SPEC)
    nc = tidy[tidy["category"] == "noncommercial"].iloc[0]
    assert nc["net"] == 60  # 100 - 40
    assert nc["spread"] == 10  # resolved from the typo'd column
    comm = tidy[tidy["category"] == "commercial"].iloc[0]
    assert comm["net"] == -40  # 50 - 90
    nonrept = tidy[tidy["category"] == "nonreportable"].iloc[0]
    assert nonrept["net"] == -20


def test_reconcile_passes_when_identity_holds():
    assert reconcile_open_interest(_raw(open_interest=180), SPEC) == []


def test_reconcile_warns_when_oi_broken():
    warns = reconcile_open_interest(_raw(open_interest=200), SPEC)
    assert len(warns) == 1
    assert "OI reconciliation" in warns[0]


def test_enrich_adds_wow_change():
    a = _raw()
    b = _raw()
    b["report_date_as_yyyy_mm_dd"] = pd.to_datetime(["2026-06-02"])
    b["noncomm_positions_long_all"] = [130]  # net 130-40 = 90 (+30 WoW)
    tidy = tidy_positions(pd.concat([a, b], ignore_index=True), SPEC)
    enriched = enrich(tidy, window=4, min_periods=1)
    nc = enriched[enriched["category"] == "noncommercial"].sort_values("report_date")
    assert pd.isna(nc["net_chg_wow"].iloc[0])
    assert nc["net_chg_wow"].iloc[1] == 30
