"""COT Index math: known min/max -> known 0-100 anchors, and edge cases."""
import numpy as np
import pandas as pd

from cot.transform import cot_index


def test_known_anchors_min_mid_max():
    # window covers the whole series; min_periods=1 so every point is defined.
    net = pd.Series([50.0, 10.0, 30.0])
    idx = cot_index(net, window=3, min_periods=1)
    # t0: single value -> flat window -> neutral 50
    # t1: current(10) is the window min -> 0
    # t2: window {50,10,30}, current 30 is exact midpoint -> 50
    assert idx.tolist() == [50.0, 0.0, 50.0]


def test_max_anchor_is_100():
    net = pd.Series([10.0, 50.0, 30.0])
    idx = cot_index(net, window=3, min_periods=1)
    # t1: current(50) is the window max -> 100
    assert idx.iloc[1] == 100.0
    # t2: midpoint -> 50
    assert idx.iloc[2] == 50.0


def test_exact_fraction():
    # window {0,100} then value 25 -> (25-0)/(100-0)*100 = 25
    net = pd.Series([0.0, 100.0, 25.0])
    idx = cot_index(net, window=3, min_periods=1)
    assert idx.iloc[2] == 25.0


def test_flat_window_is_neutral_50():
    net = pd.Series([7.0, 7.0, 7.0, 7.0])
    idx = cot_index(net, window=3, min_periods=1)
    assert (idx == 50.0).all()


def test_insufficient_window_is_nan_not_50():
    # default min_periods == window; series shorter than window -> all NaN
    net = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    idx = cot_index(net, window=10)  # min_periods defaults to 10
    assert idx.isna().all()


def test_matches_manual_rolling_reference():
    rng = np.random.default_rng(42)
    net = pd.Series(rng.normal(size=60).cumsum())
    window, mp = 12, 12
    got = cot_index(net, window=window, min_periods=mp)
    # independent reference implementation
    lo = net.rolling(window, min_periods=mp).min()
    hi = net.rolling(window, min_periods=mp).max()
    ref = (net - lo) / (hi - lo) * 100.0
    ref = ref.mask(hi == lo, 50.0)
    pd.testing.assert_series_equal(got, ref, check_names=False)


def test_bounds_within_0_100():
    rng = np.random.default_rng(7)
    net = pd.Series(rng.normal(size=300).cumsum())
    idx = cot_index(net, window=52, min_periods=52).dropna()
    assert idx.min() >= 0.0
    assert idx.max() <= 100.0
