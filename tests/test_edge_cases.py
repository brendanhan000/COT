"""Edge cases that protect contextual accuracy: missing weeks (NaN), all-negative
nets, single observations, a missing category, the reconciliation tolerance band,
and empty input. None of these may silently produce a wrong-but-plausible value.
"""
import numpy as np
import pandas as pd

from cot.align import check_tuesday
from cot.config import get_report_spec
from cot.transform import (
    cot_index,
    divergence,
    enrich,
    reconcile_open_interest,
    tidy_positions,
)

LEGACY = get_report_spec("legacy")
DATE = "report_date_as_yyyy_mm_dd"
NAME = "contract_market_name"


def test_cot_index_all_negative_nets_stay_0_100():
    net = pd.Series([-100.0, -80.0, -60.0, -40.0, -20.0])  # rising toward zero
    idx = cot_index(net, window=5, min_periods=1)
    assert idx.iloc[-1] == 100.0               # least-short point is the window max
    assert (idx.dropna() >= 0).all() and (idx.dropna() <= 100).all()


def test_cot_index_skips_nan_week_without_fabricating():
    # a missing week (NaN) must yield NaN at that row, not an interpolated number
    net = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    idx = cot_index(net, window=3, min_periods=1)
    assert pd.isna(idx.iloc[2])                 # the gap stays a gap
    assert idx.iloc[1] == 100.0                 # 2 is max of {1,2}
    assert idx.iloc[3] == 100.0                 # 4 is max of {2,4}
    assert idx.iloc[4] == 100.0                 # 5 is max of {4,5}


def test_cot_index_single_observation():
    assert cot_index(pd.Series([5.0]), window=3, min_periods=1).tolist() == [50.0]  # flat -> neutral
    assert cot_index(pd.Series([5.0]), window=3).isna().all()                       # window unfilled -> NaN


def test_divergence_returns_empty_when_a_category_is_missing(make_legacy_raw):
    tidy = tidy_positions(make_legacy_raw(n_weeks=4), LEGACY)
    only_nc = tidy[tidy["category"] == "noncommercial"]   # drop the 'commercial' leg
    div = divergence(only_nc, LEGACY)
    assert div.empty
    assert list(div.columns) == ["contract", "report_date", "divergence"]


def _one_row(oi, nc_l, nc_s, cm_l, cm_s, nr_l, nr_s, spread=0):
    return pd.DataFrame(
        {
            DATE: [pd.Timestamp("2026-05-26")],
            NAME: ["E-MINI S&P 500"],
            "open_interest_all": [oi],
            "noncomm_positions_long_all": [nc_l],
            "noncomm_positions_short_all": [nc_s],
            "noncomm_postions_spread_all": [spread],
            "comm_positions_long_all": [cm_l],
            "comm_positions_short_all": [cm_s],
            "nonrept_positions_long_all": [nr_l],
            "nonrept_positions_short_all": [nr_s],
        }
    )


def test_reconcile_respects_tolerance_band():
    # long side 0.5% off (under 1% tol), short side exact -> no warning
    ok = _one_row(oi=1000, nc_l=600, cm_l=300, nr_l=105, nc_s=400, cm_s=400, nr_s=200)
    assert reconcile_open_interest(ok, LEGACY) == []
    # long side 2% off (over tol) -> warned
    bad = _one_row(oi=1000, nc_l=600, cm_l=300, nr_l=120, nc_s=400, cm_s=400, nr_s=200)
    assert len(reconcile_open_interest(bad, LEGACY)) == 1


def test_enrich_empty_frame_still_has_metric_columns():
    empty = tidy_positions(pd.DataFrame(), LEGACY)
    out = enrich(empty)
    assert len(out) == 0
    for col in ("cot_index", "net_chg_wow", "pct_oi_net"):
        assert col in out.columns


def test_check_tuesday_flags_each_holiday_shift():
    dates = pd.to_datetime(["2026-05-26", "2026-05-25", "2026-05-27"])  # Tue, Mon, Wed
    warns = check_tuesday(pd.Series(dates))
    assert len(warns) == 1 and "not Tuesday" in warns[0]
