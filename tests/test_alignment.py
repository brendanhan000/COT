"""Tuesday indexing + backward price alignment."""
import pandas as pd

from cot.align import (
    align_price_to_tuesdays,
    check_tuesday,
    normalize_tuesday_index,
)


def test_check_tuesday_all_good():
    # 2026-05-12, 19, 26 are Tuesdays
    dates = pd.to_datetime(["2026-05-12", "2026-05-19", "2026-05-26"])
    assert check_tuesday(pd.Series(dates)) == []


def test_check_tuesday_flags_non_tuesday():
    dates = pd.to_datetime(["2026-05-12", "2026-05-20"])  # 20th is a Wednesday
    warns = check_tuesday(pd.Series(dates))
    assert len(warns) == 1
    assert "not Tuesday" in warns[0]


def test_normalize_strips_tz_and_time():
    df = pd.DataFrame(
        {"report_date": pd.to_datetime(["2026-05-26T00:00:00.000Z"], utc=True)}
    )
    out = normalize_tuesday_index(df)
    assert out["report_date"].iloc[0] == pd.Timestamp("2026-05-26")
    assert out["report_date"].dt.tz is None


def test_price_aligns_backward_over_holiday():
    # Daily closes Mon-Fri across two weeks, but the 2nd Tuesday (19th) is missing
    # (simulated holiday). Backward as-of must pick Monday the 18th's close.
    price = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-05-11",  # Mon
                    "2026-05-12",  # Tue  -> close 101
                    "2026-05-13",
                    "2026-05-18",  # Mon  -> close 110
                    # 2026-05-19 (Tue) intentionally absent
                    "2026-05-20",  # Wed
                ]
            ),
            "price": [100.0, 101.0, 102.0, 110.0, 120.0],
        }
    )
    tuesdays = pd.Series(pd.to_datetime(["2026-05-12", "2026-05-19"]))
    aligned = align_price_to_tuesdays(price, tuesdays)
    by_date = dict(zip(aligned["report_date"], aligned["price"]))
    assert by_date[pd.Timestamp("2026-05-12")] == 101.0  # exact Tuesday close
    assert by_date[pd.Timestamp("2026-05-19")] == 110.0  # last close before holiday


def test_price_alignment_never_uses_future_close():
    price = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-12", "2026-05-26"]),
            "price": [101.0, 130.0],
        }
    )
    # 2026-05-19 has no close on/before it except the 12th -> must be 101, not 130
    aligned = align_price_to_tuesdays(price, pd.Series(pd.to_datetime(["2026-05-19"])))
    assert aligned["price"].iloc[0] == 101.0
