"""Shared synthetic fixtures.

The builders emit wide CFTC-shaped frames using the REAL Socrata column names
(including the ``noncomm_postions_spread_all`` typo and TFF's no-``_all``
asset-mgr / lev-money / other-rept columns). Values are constructed so the
long-side and short-side accounting identities both equal open interest, i.e.
``reconcile_open_interest`` must pass. Tests derive their expectations from the
returned raw frame itself, so they verify the transform against the source data
rather than against duplicated formulas.
"""
import numpy as np
import pandas as pd
import pytest


def _tuesdays(n, start="2026-01-06"):  # 2026-01-06 is a Tuesday
    return pd.to_datetime(start) + pd.to_timedelta(np.arange(n) * 7, unit="D")


@pytest.fixture
def make_legacy_raw():
    def _make(n_weeks=8, contract="E-MINI S&P 500", start="2026-01-06"):
        i = np.arange(n_weeks)
        spread = np.full(n_weeks, 5)
        nc_long = 100 + 10 * i
        cm_long = 50 + 5 * i
        nr_long = 20 + i
        oi = nc_long + cm_long + nr_long + spread          # long side == OI
        nr_short = 25 + i
        cm_short = 90 + 9 * i
        nc_short = oi - spread - cm_short - nr_short        # forces short side == OI
        return pd.DataFrame(
            {
                "report_date_as_yyyy_mm_dd": _tuesdays(n_weeks, start),
                "contract_market_name": contract,
                "futonly_or_combined": "FutOnly",
                "open_interest_all": oi,
                "noncomm_positions_long_all": nc_long,
                "noncomm_positions_short_all": nc_short,
                "noncomm_postions_spread_all": spread,       # real CFTC typo column
                "comm_positions_long_all": cm_long,
                "comm_positions_short_all": cm_short,
                "nonrept_positions_long_all": nr_long,
                "nonrept_positions_short_all": nr_short,
                "traders_noncomm_long_all": 10 + i,
                "traders_noncomm_short_all": 8 + i,
                "traders_comm_long_all": 6 + i,
                "traders_comm_short_all": 7 + i,
                "pct_of_oi_noncomm_long_all": np.round(nc_long / oi * 100, 1),
                "pct_of_oi_noncomm_short_all": np.round(nc_short / oi * 100, 1),
                "pct_of_oi_comm_long_all": np.round(cm_long / oi * 100, 1),
                "pct_of_oi_comm_short_all": np.round(cm_short / oi * 100, 1),
                "pct_of_oi_nonrept_long_all": np.round(nr_long / oi * 100, 1),
                "pct_of_oi_nonrept_short_all": np.round(nr_short / oi * 100, 1),
            }
        )

    return _make


@pytest.fixture
def make_tff_raw():
    def _make(n_weeks=6, contract="E-MINI S&P 500", start="2026-01-06"):
        i = np.arange(n_weeks)
        d_sp, a_sp, l_sp, o_sp = 2, 3, 4, 1
        spread_total = d_sp + a_sp + l_sp + o_sp
        d_long = 30 + i
        a_long = 40 + 2 * i
        l_long = 50 + 3 * i
        o_long = np.full(n_weeks, 10)
        nr_long = 20 + i
        oi = d_long + a_long + l_long + o_long + nr_long + spread_total
        d_short = 60 + i
        a_short = 20 + i
        l_short = 25 + 2 * i
        o_short = 15 + i
        nr_short = oi - spread_total - (d_short + a_short + l_short + o_short)
        return pd.DataFrame(
            {
                "report_date_as_yyyy_mm_dd": _tuesdays(n_weeks, start),
                "contract_market_name": contract,
                "futonly_or_combined": "FutOnly",
                "open_interest_all": oi,
                # dealer/nonrept carry "_all"; asset_mgr/lev_money/other_rept do NOT
                "dealer_positions_long_all": d_long,
                "dealer_positions_short_all": d_short,
                "dealer_positions_spread_all": np.full(n_weeks, d_sp),
                "asset_mgr_positions_long": a_long,
                "asset_mgr_positions_short": a_short,
                "asset_mgr_positions_spread": np.full(n_weeks, a_sp),
                "lev_money_positions_long": l_long,
                "lev_money_positions_short": l_short,
                "lev_money_positions_spread": np.full(n_weeks, l_sp),
                "other_rept_positions_long": o_long,
                "other_rept_positions_short": o_short,
                "other_rept_positions_spread": np.full(n_weeks, o_sp),
                "nonrept_positions_long_all": nr_long,
                "nonrept_positions_short_all": nr_short,
                "traders_lev_money_long_all": 5 + i,
                "traders_asset_mgr_long_all": 4 + i,
                "traders_dealer_long_all": 3 + i,
            }
        )

    return _make
