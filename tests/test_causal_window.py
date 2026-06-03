"""Causality: the COT Index at time t must not use any future week."""
import numpy as np
import pandas as pd

from cot.transform import cot_index


def test_index_at_t_ignores_future():
    rng = np.random.default_rng(123)
    net = pd.Series(rng.normal(size=40).cumsum())
    window, mp = 8, 8
    full = cot_index(net, window=window, min_periods=mp)

    # For every t, recomputing on the truncated prefix must match the full value.
    for t in range(len(net)):
        prefix = cot_index(net.iloc[: t + 1], window=window, min_periods=mp)
        a, b = full.iloc[t], prefix.iloc[t]
        if pd.isna(a) or pd.isna(b):
            assert pd.isna(a) and pd.isna(b)
        else:
            assert abs(a - b) < 1e-9


def test_appending_future_does_not_change_past():
    base = pd.Series([1.0, 5.0, 3.0, 9.0, 2.0, 8.0, 4.0, 7.0, 6.0, 10.0])
    window, mp = 4, 4
    before = cot_index(base, window=window, min_periods=mp)
    extended = pd.concat([base, pd.Series([999.0, -999.0])], ignore_index=True)
    after = cot_index(extended, window=window, min_periods=mp)
    # the original positions must be identical despite wild future values
    pd.testing.assert_series_equal(
        before, after.iloc[: len(base)], check_names=False
    )
