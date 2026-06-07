"""End-to-end data integrity: raw CFTC rows -> tidy/enriched, value-for-value.

These tests assert that NOTHING is silently dropped, mangled, mis-signed, or
cross-contaminated between contracts/categories on the way to the numbers the
chart later draws. Expectations are derived from the raw frame itself.
"""
import numpy as np
import pandas as pd

from cot.config import get_report_spec
from cot.transform import (
    cot_index,
    divergence,
    enrich,
    latest_snapshot,
    reconcile_open_interest,
    tidy_positions,
    validate,
)

LEGACY = get_report_spec("legacy")


def test_no_rows_dropped_or_added(make_legacy_raw):
    raw = make_legacy_raw(n_weeks=8)
    tidy = tidy_positions(raw, LEGACY)
    # one tidy row per (week, category); 3 legacy categories
    assert len(tidy) == 8 * len(LEGACY.categories)
    for key in ("noncommercial", "commercial", "nonreportable"):
        assert (tidy["category"] == key).sum() == 8
    # every Tuesday survived, in order, per category
    for _, sub in tidy.groupby("category"):
        assert list(sub.sort_values("report_date")["report_date"]) == list(
            pd.to_datetime(raw["report_date_as_yyyy_mm_dd"])
        )


def test_every_raw_value_roundtrips(make_legacy_raw):
    """long/short/spread/OI/traders/%OI carry through unchanged for every week."""
    raw = make_legacy_raw(n_weeks=8).reset_index(drop=True)
    tidy = tidy_positions(raw, LEGACY)
    fields = {
        "noncommercial": dict(
            long="noncomm_positions_long_all",
            short="noncomm_positions_short_all",
            spread="noncomm_postions_spread_all",
            traders_long="traders_noncomm_long_all",
            traders_short="traders_noncomm_short_all",
            pct_oi_long="pct_of_oi_noncomm_long_all",
            pct_oi_short="pct_of_oi_noncomm_short_all",
        ),
        "commercial": dict(
            long="comm_positions_long_all",
            short="comm_positions_short_all",
            traders_long="traders_comm_long_all",
            traders_short="traders_comm_short_all",
            pct_oi_long="pct_of_oi_comm_long_all",
            pct_oi_short="pct_of_oi_comm_short_all",
        ),
        "nonreportable": dict(
            long="nonrept_positions_long_all",
            short="nonrept_positions_short_all",
            pct_oi_long="pct_of_oi_nonrept_long_all",
            pct_oi_short="pct_of_oi_nonrept_short_all",
        ),
    }
    for key, cols in fields.items():
        sub = tidy[tidy["category"] == key].sort_values("report_date").reset_index(drop=True)
        for tidy_col, raw_col in cols.items():
            np.testing.assert_array_equal(
                sub[tidy_col].to_numpy(dtype=float),
                raw[raw_col].to_numpy(dtype=float),
                err_msg="%s/%s mismatch" % (key, tidy_col),
            )
        # open interest is the same contract-level value on every category row
        np.testing.assert_array_equal(
            sub["open_interest"].to_numpy(dtype=float),
            raw["open_interest_all"].to_numpy(dtype=float),
        )
    # nonreportable has no per-category trader counts -> must be NaN, not 0
    nonrept = tidy[tidy["category"] == "nonreportable"]
    assert nonrept["traders_long"].isna().all()
    assert nonrept["traders_short"].isna().all()


def test_net_is_long_minus_short_with_correct_sign(make_legacy_raw):
    raw = make_legacy_raw(n_weeks=8).reset_index(drop=True)
    tidy = tidy_positions(raw, LEGACY)
    for key, lc, sc in [
        ("noncommercial", "noncomm_positions_long_all", "noncomm_positions_short_all"),
        ("commercial", "comm_positions_long_all", "comm_positions_short_all"),
        ("nonreportable", "nonrept_positions_long_all", "nonrept_positions_short_all"),
    ]:
        sub = tidy[tidy["category"] == key].sort_values("report_date").reset_index(drop=True)
        expect = raw[lc].to_numpy(dtype=float) - raw[sc].to_numpy(dtype=float)
        np.testing.assert_array_equal(sub["net"].to_numpy(dtype=float), expect)
    # commercial is net SHORT in this fixture -> strictly negative everywhere
    comm = tidy[tidy["category"] == "commercial"]
    assert (comm["net"] < 0).all()
    # spreading is excluded from net (net != long-short+spread)
    nc = tidy[tidy["category"] == "noncommercial"].sort_values("report_date")
    assert (nc["net"] == raw["noncomm_positions_long_all"].values - raw["noncomm_positions_short_all"].values).all()


def test_pct_oi_net_is_exact(make_legacy_raw):
    raw = make_legacy_raw(n_weeks=8)
    enriched = enrich(tidy_positions(raw, LEGACY), window=4, min_periods=1)
    expect = enriched["net"] / enriched["open_interest"] * 100.0
    np.testing.assert_allclose(enriched["pct_oi_net"], expect, rtol=0, atol=1e-12)


def test_pct_oi_net_nan_when_oi_zero(make_legacy_raw):
    raw = make_legacy_raw(n_weeks=3)
    raw.loc[1, "open_interest_all"] = 0
    enriched = enrich(tidy_positions(raw, LEGACY), window=3, min_periods=1)
    zero_oi = enriched[enriched["open_interest"] == 0]
    assert zero_oi["pct_oi_net"].isna().all()


def test_wow_change_exact_and_first_is_nan(make_legacy_raw):
    raw = make_legacy_raw(n_weeks=6)
    enriched = enrich(tidy_positions(raw, LEGACY), window=6, min_periods=1)
    for key in ("noncommercial", "commercial", "nonreportable"):
        sub = enriched[enriched["category"] == key].sort_values("report_date")
        assert pd.isna(sub["net_chg_wow"].iloc[0])
        np.testing.assert_allclose(
            sub["net_chg_wow"].to_numpy()[1:], np.diff(sub["net"].to_numpy())
        )


def test_enrich_cot_index_matches_per_group_reference(make_legacy_raw):
    """The grouped cot_index equals cot_index computed on each isolated series."""
    raw = make_legacy_raw(n_weeks=12)
    enriched = enrich(tidy_positions(raw, LEGACY), window=6, min_periods=6)
    for key in ("noncommercial", "commercial", "nonreportable"):
        sub = enriched[enriched["category"] == key].sort_values("report_date")
        ref = cot_index(sub["net"].reset_index(drop=True), window=6, min_periods=6)
        np.testing.assert_allclose(
            sub["cot_index"].to_numpy(), ref.to_numpy(), equal_nan=True
        )


def test_no_cross_contamination_between_contracts(make_legacy_raw):
    """Two contracts with mirror-image nets must keep independent COT indices."""
    a = make_legacy_raw(n_weeks=5, contract="AAA")
    # build BBB as a decreasing series by flipping long growth into shrinkage
    b = make_legacy_raw(n_weeks=5, contract="BBB")
    b["noncomm_positions_long_all"] = b["noncomm_positions_long_all"].values[::-1]
    raw = pd.concat([a, b], ignore_index=True)
    enriched = enrich(tidy_positions(raw, LEGACY), window=5, min_periods=1)

    def last_cot(contract):
        s = enriched[
            (enriched["contract"] == contract)
            & (enriched["category"] == "noncommercial")
        ].sort_values("report_date")
        return s["cot_index"].iloc[-1]

    # AAA noncomm net increases -> last point is window max -> 100
    assert last_cot("AAA") == 100.0
    # BBB noncomm net decreases -> last point is window min -> 0
    assert last_cot("BBB") == 0.0


def test_numeric_coercion_strings_none_and_missing_columns():
    """Socrata returns strings; missing/None must become NaN, not 0 or error."""
    raw = pd.DataFrame(
        {
            "report_date_as_yyyy_mm_dd": ["2026-05-26T00:00:00.000"],
            "contract_market_name": ["E-MINI S&P 500"],
            "open_interest_all": ["180"],
            "noncomm_positions_long_all": ["100"],
            "noncomm_positions_short_all": ["-40"],   # negative as string
            "noncomm_postions_spread_all": [None],     # missing value -> NaN
            "comm_positions_long_all": ["50"],
            "comm_positions_short_all": ["90"],
            # nonrept_* columns entirely absent -> NaN, no crash
        }
    )
    tidy = tidy_positions(raw, LEGACY)
    nc = tidy[tidy["category"] == "noncommercial"].iloc[0]
    assert nc["net"] == 140.0            # 100 - (-40)
    assert pd.isna(nc["spread"])         # None coerced to NaN
    assert nc["open_interest"] == 180.0
    nonrept = tidy[tidy["category"] == "nonreportable"].iloc[0]
    assert pd.isna(nonrept["long"]) and pd.isna(nonrept["short"])  # absent columns
    assert pd.isna(nonrept["net"])
    # report_date parsed from ISO string
    assert nc["report_date"] == pd.Timestamp("2026-05-26")


def test_reconciliation_passes_across_all_weeks(make_legacy_raw):
    raw = make_legacy_raw(n_weeks=20)
    assert reconcile_open_interest(raw, LEGACY) == []


def test_reconciliation_flags_single_broken_week(make_legacy_raw):
    raw = make_legacy_raw(n_weeks=20)
    raw.loc[7, "open_interest_all"] = raw.loc[7, "open_interest_all"] + 999
    warns = reconcile_open_interest(raw, LEGACY)
    assert len(warns) == 1 and "1 row(s)" in warns[0]


def test_divergence_is_exact_and_per_contract(make_legacy_raw):
    a = make_legacy_raw(n_weeks=4, contract="AAA")
    b = make_legacy_raw(n_weeks=4, contract="BBB")
    tidy = tidy_positions(pd.concat([a, b], ignore_index=True), LEGACY)
    div = divergence(tidy, LEGACY)
    for contract in ("AAA", "BBB"):
        nc = tidy[(tidy.contract == contract) & (tidy.category == "noncommercial")].set_index("report_date")["net"]
        cm = tidy[(tidy.contract == contract) & (tidy.category == "commercial")].set_index("report_date")["net"]
        got = div[div.contract == contract].set_index("report_date")["divergence"]
        np.testing.assert_allclose(got.sort_index().to_numpy(), (nc - cm).sort_index().to_numpy())


def test_latest_snapshot_picks_max_date(make_legacy_raw):
    raw = make_legacy_raw(n_weeks=10)
    enriched = enrich(tidy_positions(raw, LEGACY), window=5, min_periods=1)
    snap = latest_snapshot(enriched)
    last_tue = pd.to_datetime(raw["report_date_as_yyyy_mm_dd"]).max()
    assert (snap["report_date"] == last_tue).all()
    assert len(snap) == len(LEGACY.categories)
    # snapshot net equals the final-week net for each category
    for _, row in snap.iterrows():
        expect = enriched[
            (enriched.category == row["category"]) & (enriched.report_date == last_tue)
        ]["net"].iloc[0]
        assert row["net"] == expect


def test_validate_clean_weekly_data_has_no_warnings(make_legacy_raw):
    raw = make_legacy_raw(n_weeks=12)
    tidy = tidy_positions(raw, LEGACY)
    v = validate(tidy, raw, LEGACY, ["E-MINI S&P 500"], ["E-MINI S&P 500"])
    assert v.ok(), v.warnings


def test_validate_flags_gap_and_non_tuesday(make_legacy_raw):
    raw = make_legacy_raw(n_weeks=12)
    raw = raw.drop(index=5).reset_index(drop=True)            # missing week -> gap
    raw.loc[3, "report_date_as_yyyy_mm_dd"] = pd.Timestamp("2026-01-28")  # a Wednesday
    tidy = tidy_positions(raw, LEGACY)
    v = validate(tidy, raw, LEGACY, ["E-MINI S&P 500"], ["E-MINI S&P 500"])
    text = " ".join(v.warnings)
    assert "gap(s)" in text
    assert "not a Tuesday" in text


def test_validate_flags_unresolved_contract(make_legacy_raw):
    raw = make_legacy_raw(n_weeks=4)
    tidy = tidy_positions(raw, LEGACY)
    v = validate(tidy, raw, LEGACY, ["TYPO NAME"], [None])
    assert any("matched no official" in w for w in v.warnings)
