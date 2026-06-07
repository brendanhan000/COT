"""Export + report assembly fidelity.

The dashboard consumes cot_tidy.parquet/csv; the report is cot_report.html. These
tests confirm the exported numbers survive a round-trip unchanged and that the
combined HTML actually contains every tracked contract plus the lag/guardrail
notes (so no contract is silently dropped from what you read).
"""
import pandas as pd

from cot.align import PriceResult
from cot.config import get_report_spec
from cot.transform import enrich, tidy_positions
from cot.viz_interactive import build_contract_figure, build_interactive_html

LEGACY = get_report_spec("legacy")
A = "E-MINI S&P 500"
B = "CRUDE OIL, LIGHT SWEET-WTI"

# the exact column set the CLI exports
EXPORT_COLS = [
    "report_date", "contract", "report_type", "fut_combined", "category",
    "category_label", "long", "short", "spread", "net", "net_chg_wow",
    "open_interest", "pct_oi_long", "pct_oi_short", "pct_oi_net",
    "traders_long", "traders_short", "cot_index",
]


def _enriched(make_legacy_raw, names=(A, B), weeks=8):
    raw = pd.concat([make_legacy_raw(n_weeks=weeks, contract=n) for n in names], ignore_index=True)
    return enrich(tidy_positions(raw, LEGACY), window=4, min_periods=1)


def test_parquet_export_is_lossless(tmp_path, make_legacy_raw):
    export = _enriched(make_legacy_raw)[EXPORT_COLS].reset_index(drop=True)
    p = tmp_path / "cot_tidy.parquet"
    export.to_parquet(p, index=False)
    back = pd.read_parquet(p)
    pd.testing.assert_frame_equal(export, back)


def test_csv_export_values_reparse_identically(tmp_path, make_legacy_raw):
    export = _enriched(make_legacy_raw)[EXPORT_COLS].reset_index(drop=True)
    p = tmp_path / "cot_tidy.csv"
    export.to_csv(p, index=False)
    back = pd.read_csv(p, parse_dates=["report_date"])
    # dates and labels survive
    pd.testing.assert_series_equal(export["report_date"], back["report_date"], check_names=False)
    assert list(export["contract"]) == list(back["contract"])
    assert list(export["category"]) == list(back["category"])
    # every numeric column matches to float precision (NaN-aware)
    for col in ["long", "short", "spread", "net", "net_chg_wow", "open_interest",
                "pct_oi_long", "pct_oi_short", "pct_oi_net", "cot_index"]:
        pd.testing.assert_series_equal(
            export[col].astype(float), back[col].astype(float), check_names=False
        )


def test_export_has_no_unexpected_nans_in_core_columns(make_legacy_raw):
    """net / open_interest / pct_oi_net must be fully populated for every row."""
    export = _enriched(make_legacy_raw)[EXPORT_COLS]
    for col in ["report_date", "contract", "category", "net", "open_interest", "pct_oi_net"]:
        assert export[col].notna().all(), "unexpected NaN in %s" % col


def test_interactive_html_contains_every_contract_and_disclaimers(tmp_path, make_legacy_raw):
    enriched = _enriched(make_legacy_raw, names=(A, B))
    items = []
    no_price = PriceResult(pd.DataFrame(columns=["report_date", "price"]), None, "off", False)
    for name in (A, B):
        as_of = pd.Timestamp(enriched[enriched["contract"] == name]["report_date"].max())
        items.append((name, as_of, build_contract_figure(enriched, LEGACY, name, no_price)))

    out = tmp_path / "cot_report.html"
    meta = {"report": "legacy/futonly", "dataset": "6dca-aqww"}
    build_interactive_html(items, out, meta, warnings=["E-MINI S&P 500: 2 gap(s) in the weekly series"], embed=False)
    html = out.read_text(encoding="utf-8")

    assert html.count("plotly-graph-div") == 2           # one interactive chart per contract
    assert A in html and B in html                       # neither contract dropped
    assert "2026-" in html                               # as-of dates rendered
    assert "3:30pm ET" in html and "lag" in html         # lag note present
    assert "does NOT emit buy/sell signals" in html      # guardrail
    assert "6dca-aqww" in html                           # dataset/endpoint provenance
    assert "2 gap(s)" in html                            # validation warnings surfaced


def test_html_marks_self_contained_vs_cdn(tmp_path, make_legacy_raw):
    enriched = _enriched(make_legacy_raw, names=(A,))
    no_price = PriceResult(pd.DataFrame(columns=["report_date", "price"]), None, "off", False)
    as_of = pd.Timestamp(enriched["report_date"].max())
    items = [(A, as_of, build_contract_figure(enriched, LEGACY, A, no_price))]

    embedded = tmp_path / "embed.html"
    cdn = tmp_path / "cdn.html"
    build_interactive_html(items, embedded, {}, [], embed=True)
    build_interactive_html(list(items), cdn, {}, [], embed=False)
    # embedding plotly.js makes the self-contained file much larger than the CDN one
    assert embedded.stat().st_size > cdn.stat().st_size * 5
