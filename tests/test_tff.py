"""TFF report: asymmetric column names must resolve and process correctly.

TFF mixes ``dealer_positions_long_all`` (with _all) and
``lev_money_positions_long`` / ``asset_mgr_positions_long`` (no _all). If field
resolution slips, nets silently become NaN - these tests guard that.
"""
import numpy as np
import pandas as pd

from cot.align import PriceResult
from cot.config import get_report_spec
from cot.transform import (
    enrich,
    reconcile_open_interest,
    resolve_field,
    tidy_positions,
)
from cot.viz_interactive import build_contract_figure

TFF = get_report_spec("tff")


def test_resolve_picks_correct_all_vs_no_all_columns(make_tff_raw):
    cols = list(make_tff_raw(n_weeks=1).columns)
    # dealer carries _all
    assert resolve_field(cols, TFF.category("dealer").long) == "dealer_positions_long_all"
    # leveraged funds + asset manager do NOT carry _all
    assert resolve_field(cols, TFF.category("leveraged_funds").long) == "lev_money_positions_long"
    assert resolve_field(cols, TFF.category("asset_manager").long) == "asset_mgr_positions_long"
    assert resolve_field(cols, TFF.category("other_reportable").long) == "other_rept_positions_long"


def test_tff_nets_match_raw(make_tff_raw):
    raw = make_tff_raw(n_weeks=6).reset_index(drop=True)
    tidy = tidy_positions(raw, TFF)
    pairs = {
        "dealer": ("dealer_positions_long_all", "dealer_positions_short_all"),
        "asset_manager": ("asset_mgr_positions_long", "asset_mgr_positions_short"),
        "leveraged_funds": ("lev_money_positions_long", "lev_money_positions_short"),
        "other_reportable": ("other_rept_positions_long", "other_rept_positions_short"),
        "nonreportable": ("nonrept_positions_long_all", "nonrept_positions_short_all"),
    }
    for key, (lc, sc) in pairs.items():
        sub = tidy[tidy["category"] == key].sort_values("report_date").reset_index(drop=True)
        expect = raw[lc].to_numpy(dtype=float) - raw[sc].to_numpy(dtype=float)
        np.testing.assert_array_equal(sub["net"].to_numpy(dtype=float), expect)
        assert sub["net"].notna().all()  # would be NaN if resolution failed


def test_tff_spread_and_traders_resolve(make_tff_raw):
    raw = make_tff_raw(n_weeks=4).reset_index(drop=True)
    tidy = tidy_positions(raw, TFF)
    lev = tidy[tidy["category"] == "leveraged_funds"].sort_values("report_date").reset_index(drop=True)
    np.testing.assert_array_equal(
        lev["spread"].to_numpy(dtype=float),
        raw["lev_money_positions_spread"].to_numpy(dtype=float),
    )
    np.testing.assert_array_equal(
        lev["traders_long"].to_numpy(dtype=float),
        raw["traders_lev_money_long_all"].to_numpy(dtype=float),
    )


def test_tff_reconciliation_passes(make_tff_raw):
    raw = make_tff_raw(n_weeks=10)
    assert reconcile_open_interest(raw, TFF) == []


def test_tff_reconciliation_flags_break(make_tff_raw):
    raw = make_tff_raw(n_weeks=10)
    raw.loc[3, "lev_money_positions_long"] = raw.loc[3, "lev_money_positions_long"] + 500
    warns = reconcile_open_interest(raw, TFF)
    assert len(warns) == 1 and "OI reconciliation" in warns[0]


def test_tff_figure_primary_is_leveraged_funds(make_tff_raw):
    raw = make_tff_raw(n_weeks=8)
    enriched = enrich(tidy_positions(raw, TFF), window=4, min_periods=1)
    price = PriceResult(pd.DataFrame(columns=["report_date", "price"]), None, "off", False)
    fig = build_contract_figure(enriched, TFF, "E-MINI S&P 500", price)
    names = {t.name: t for t in fig.data}
    # four reportable net categories drawn
    for key in TFF.net_chart_keys:
        assert TFF.category(key).label in names
    # the row-3 primary line is the Leveraged Funds net, and matches the data
    lev_label = TFF.category("leveraged_funds").label
    exp = enriched[enriched["category"] == "leveraged_funds"].sort_values("report_date")
    np.testing.assert_allclose(
        np.array(names["%s net" % lev_label].y, dtype=float),
        exp["net"].to_numpy(dtype=float),
    )
