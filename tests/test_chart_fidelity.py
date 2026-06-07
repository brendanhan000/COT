"""Chart fidelity: what is DRAWN must equal the processed data, exactly.

Pipeline integrity proves raw -> enriched. These tests prove enriched -> chart
(both the interactive Plotly traces and the static matplotlib lines), so the
numbers you read off the chart are provably the CFTC source numbers.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import cot.viz as mviz
from cot.align import PriceResult
from cot.config import get_report_spec
from cot.disclaimers import EXTREME_LABEL
from cot.transform import enrich, tidy_positions
from cot.viz_interactive import build_contract_figure

LEGACY = get_report_spec("legacy")
CONTRACT = "E-MINI S&P 500"


def by_name(fig):
    return {t.name: t for t in fig.data}


def xdates(trace):
    return list(pd.to_datetime(list(trace.x)))


def no_price():
    return PriceResult(
        aligned=pd.DataFrame(columns=["report_date", "price"]),
        ticker=None,
        note="off",
        ok=False,
    )


def _enriched(make_legacy_raw, n_weeks=8, window=4, min_periods=1):
    raw = make_legacy_raw(n_weeks=n_weeks)
    return enrich(tidy_positions(raw, LEGACY), window=window, min_periods=min_periods)


# --------------------------------------------------------------------------
# net panel
# --------------------------------------------------------------------------
def test_net_traces_match_enriched_x_and_y(make_legacy_raw):
    enriched = _enriched(make_legacy_raw)
    fig = build_contract_figure(enriched, LEGACY, CONTRACT, no_price())
    tr = by_name(fig)
    for key in LEGACY.net_chart_keys:
        label = LEGACY.category(key).label
        exp = enriched[enriched["category"] == key].sort_values("report_date")
        assert label in tr, "missing net trace for %s" % label
        assert xdates(tr[label]) == list(exp["report_date"])
        np.testing.assert_allclose(
            np.array(tr[label].y, dtype=float), exp["net"].to_numpy(dtype=float)
        )


def test_net_current_value_annotation_matches_last_point(make_legacy_raw):
    enriched = _enriched(make_legacy_raw)
    fig = build_contract_figure(enriched, LEGACY, CONTRACT, no_price())
    texts = [a.text for a in fig.layout.annotations]
    for key in LEGACY.net_chart_keys:
        last = enriched[enriched["category"] == key].sort_values("report_date").iloc[-1]
        assert "{:,.0f}".format(last["net"]) in texts


# --------------------------------------------------------------------------
# COT index panel
# --------------------------------------------------------------------------
def test_cot_index_traces_match_enriched(make_legacy_raw):
    enriched = _enriched(make_legacy_raw)
    fig = build_contract_figure(enriched, LEGACY, CONTRACT, no_price())
    tr = by_name(fig)
    for key in LEGACY.headline_keys:
        name = LEGACY.category(key).label + " COT idx"
        exp = enriched[enriched["category"] == key].sort_values("report_date")
        exp = exp[exp["cot_index"].notna()]
        assert name in tr
        assert xdates(tr[name]) == list(exp["report_date"])
        np.testing.assert_allclose(
            np.array(tr[name].y, dtype=float), exp["cot_index"].to_numpy(dtype=float)
        )


def test_cot_traces_drop_only_nan_points(make_legacy_raw):
    # window not yet full -> leading NaNs must be omitted from the line, not zero-filled
    enriched = _enriched(make_legacy_raw, n_weeks=10, window=5, min_periods=5)
    fig = build_contract_figure(enriched, LEGACY, CONTRACT, no_price())
    tr = by_name(fig)
    name = LEGACY.category(LEGACY.primary_spec).label + " COT idx"
    n_valid = enriched[enriched["category"] == LEGACY.primary_spec]["cot_index"].notna().sum()
    assert len(tr[name].y) == n_valid
    assert not np.isnan(np.array(tr[name].y, dtype=float)).any()


# --------------------------------------------------------------------------
# net-vs-price panel
# --------------------------------------------------------------------------
def test_price_overlay_trace_matches_aligned_price(make_legacy_raw):
    enriched = _enriched(make_legacy_raw)
    prim = enriched[enriched["category"] == LEGACY.primary_spec].sort_values("report_date")
    dates = prim["report_date"].tolist()
    prices = [1000.0 + k for k in range(len(dates))]
    aligned = pd.DataFrame({"report_date": dates, "price": prices})
    price = PriceResult(aligned=aligned, ticker="ES=F", note="proxy", ok=True)

    fig = build_contract_figure(enriched, LEGACY, CONTRACT, price)
    tr = by_name(fig)
    plabel = LEGACY.category(LEGACY.primary_spec).label
    # primary net (row 3) equals the enriched primary net
    np.testing.assert_allclose(
        np.array(tr["%s net" % plabel].y, dtype=float), prim["net"].to_numpy(dtype=float)
    )
    # price trace equals the aligned proxy price, in Tuesday order
    pname = "price (PROXY ES=F)"
    assert pname in tr
    assert xdates(tr[pname]) == dates
    np.testing.assert_allclose(np.array(tr[pname].y, dtype=float), np.array(prices))


def test_no_price_trace_when_overlay_disabled(make_legacy_raw):
    enriched = _enriched(make_legacy_raw)
    fig = build_contract_figure(enriched, LEGACY, CONTRACT, no_price())
    assert not any(n.startswith("price (PROXY") for n in by_name(fig))
    # the primary net line is still drawn
    plabel = LEGACY.category(LEGACY.primary_spec).label
    assert "%s net" % plabel in by_name(fig)


# --------------------------------------------------------------------------
# divergence panel
# --------------------------------------------------------------------------
def test_divergence_trace_equals_net_difference(make_legacy_raw):
    enriched = _enriched(make_legacy_raw)
    fig = build_contract_figure(enriched, LEGACY, CONTRACT, no_price())
    a, b = LEGACY.divergence_pair
    name = "%s - %s net" % (a, b)
    tr = by_name(fig)[name]
    na = enriched[enriched["category"] == a].set_index("report_date")["net"]
    nb = enriched[enriched["category"] == b].set_index("report_date")["net"]
    expect = (na - nb).sort_index()
    assert xdates(tr) == list(expect.index)
    np.testing.assert_allclose(np.array(tr.y, dtype=float), expect.to_numpy(dtype=float))


# --------------------------------------------------------------------------
# extreme tagging is data-driven (and never a signal)
# --------------------------------------------------------------------------
def _hand_enriched(dates, per_cat):
    rows = []
    for key, (label, nets, cots) in per_cat.items():
        for d, n, c in zip(dates, nets, cots):
            rows.append(
                dict(
                    contract="X",
                    report_date=d,
                    category=key,
                    category_label=label,
                    net=float(n),
                    cot_index=(np.nan if c is None else float(c)),
                    open_interest=1000.0,
                )
            )
    return pd.DataFrame(rows)


def _legacy_hand(cot_last_primary):
    dates = pd.to_datetime(["2026-01-06", "2026-01-13", "2026-01-20"])
    lab = lambda k: LEGACY.category(k).label
    return _hand_enriched(
        dates,
        {
            "noncommercial": (lab("noncommercial"), [110, 120, 130], [40, 60, cot_last_primary]),
            "commercial": (lab("commercial"), [-10, -20, -30], [60, 40, 45]),
            "nonreportable": (lab("nonreportable"), [5, 5, 5], [50, 50, 50]),
        },
    )


def test_extreme_tag_present_when_cot_index_stretched():
    fig = build_contract_figure(_legacy_hand(100.0), LEGACY, "X", no_price())
    texts = [a.text for a in fig.layout.annotations]
    assert any(EXTREME_LABEL in t for t in texts)         # stretched -> tagged
    assert any(t.startswith("100") for t in texts)        # value drawn = data value


def test_no_extreme_tag_when_cot_index_mid():
    fig = build_contract_figure(_legacy_hand(50.0), LEGACY, "X", no_price())
    texts = [a.text for a in fig.layout.annotations]
    assert not any(EXTREME_LABEL in t for t in texts)     # mid -> no tag
    assert "50" in texts                                  # primary COT value drawn


# --------------------------------------------------------------------------
# robustness: must not invent data when the COT window is not yet full
# --------------------------------------------------------------------------
def test_figure_builds_with_all_nan_cot_index(make_legacy_raw):
    enriched = _enriched(make_legacy_raw, n_weeks=8, window=200, min_periods=200)
    assert enriched["cot_index"].isna().all()
    fig = build_contract_figure(enriched, LEGACY, CONTRACT, no_price())  # must not raise
    names = list(by_name(fig))
    assert not any(n.endswith("COT idx") for n in names)  # no fabricated COT line
    assert LEGACY.category("noncommercial").label in names  # net lines still drawn


# --------------------------------------------------------------------------
# static matplotlib PNG path draws the same values
# --------------------------------------------------------------------------
def test_matplotlib_net_panel_ydata_matches(monkeypatch, make_legacy_raw):
    enriched = _enriched(make_legacy_raw)
    captured = {}
    monkeypatch.setattr(mviz, "_save", lambda fig, path: captured.setdefault("fig", fig))
    mviz.panel_net_positions(enriched, LEGACY, CONTRACT, Path("ignored.png"))
    ax = captured["fig"].axes[0]
    lines = {ln.get_label(): ln for ln in ax.get_lines()}
    for key in LEGACY.net_chart_keys:
        label = LEGACY.category(key).label
        exp = enriched[enriched["category"] == key].sort_values("report_date")
        np.testing.assert_allclose(
            lines[label].get_ydata(), exp["net"].to_numpy(dtype=float)
        )
    mviz.plt.close(captured["fig"])
