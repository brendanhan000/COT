"""Incremental cache integrity (no network): the weekly pipeline must never
drift from source.

A FakeClient serves wide rows from an in-memory store and honours the same
``$where`` contract+date filtering the real Socrata endpoint would. These tests
pin down: full-history first pull, new-week-only incremental pulls, "not
published yet" (zero new), new-contract backfill, dedupe-keep-last on revisions,
cache scoping to tracked contracts, refresh-full, and discovery skipping a
non-tabular duplicate dataset.
"""
import re

import pandas as pd
import pytest

from cot.config import get_report_spec
from cot.fetch import (
    SocrataError,
    cache_path,
    discover_dataset,
    fetch_incremental,
    load_cache,
)

LEGACY = get_report_spec("legacy")
A = "E-MINI S&P 500"
B = "CRUDE OIL, LIGHT SWEET-WTI"
DATE = "report_date_as_yyyy_mm_dd"
NAMECOL = "contract_market_name"


class FakeClient:
    """In-memory stand-in for SocrataClient with $where filtering."""

    def __init__(
        self,
        store,
        dataset_id="6dca-aqww",
        name="Legacy - Futures Only",
        extra_datasets=None,
        bad_ids=None,
        honor_date_filter=True,
    ):
        self.store = store.copy()
        self.store[DATE] = pd.to_datetime(self.store[DATE])
        self.dataset_id = dataset_id
        self.name = name
        self.extra_datasets = extra_datasets or []
        self.bad_ids = set(bad_ids or [])
        self.honor_date_filter = honor_date_filter
        self.domain = "fake.cftc"
        self.has_token = False
        self.where_calls = []

    # discovery
    def list_datasets(self):
        return list(self.extra_datasets) + [{"id": self.dataset_id, "name": self.name}]

    def probe(self, cid):
        if cid in self.bad_ids:
            raise SocrataError("non-tabular table: %s" % cid)
        if cid != self.dataset_id:
            raise SocrataError("unknown dataset: %s" % cid)
        return sorted(self.store.columns)

    # data
    def get(self, cid, params):
        sel = params.get("$select", "")
        if "distinct" in sel and NAMECOL in sel:
            return [{NAMECOL: n} for n in sorted(self.store[NAMECOL].unique())]
        return self._records(self.store)

    def get_all(self, cid, where=None, select=None, order=None):
        self.where_calls.append(where or "")
        df = self.store
        m = re.search(r"contract_market_name='((?:[^']|'')*)'", where or "")
        if m:
            df = df[df[NAMECOL] == m.group(1).replace("''", "'")]
        if self.honor_date_filter:
            m2 = re.search(r"report_date_as_yyyy_mm_dd > '([^']+)'", where or "")
            if m2:
                df = df[df[DATE] > pd.Timestamp(m2.group(1))]
        return self._records(df.sort_values(DATE))

    @staticmethod
    def _records(df):
        out = df.copy()
        out[DATE] = out[DATE].dt.strftime("%Y-%m-%dT%H:%M:%S.000")  # like real Socrata
        return out.to_dict("records")


def _store(make, weeks, names):
    return pd.concat([make(n_weeks=weeks, contract=nm) for nm in names], ignore_index=True)


def _where_for(client, name):
    return [w for w in client.where_calls if "='%s'" % name in w][0]


# --------------------------------------------------------------------------
def test_first_run_pulls_full_history_no_threshold(tmp_path, make_legacy_raw):
    client = FakeClient(_store(make_legacy_raw, 6, [A, B]))
    fr = fetch_incremental(client, LEGACY, "futonly", [A, B], tmp_path)
    assert fr.new_rows == 12
    assert len(fr.raw) == 12
    assert set(fr.raw[NAMECOL].unique()) == {A, B}
    # no incremental threshold on a cold cache
    assert all(" > '" not in w for w in client.where_calls)
    assert fr.latest_as_of == client.store[DATE].max()
    assert cache_path(tmp_path, "legacy", "futonly").exists()
    assert fr.discovery.dataset_id == "6dca-aqww"
    assert fr.discovery.matched_canonical is True


def test_second_run_fetches_only_new_week(tmp_path, make_legacy_raw):
    fetch_incremental(FakeClient(_store(make_legacy_raw, 6, [A, B])), LEGACY, "futonly", [A, B], tmp_path)
    client2 = FakeClient(_store(make_legacy_raw, 7, [A, B]))  # one extra week each
    fr2 = fetch_incremental(client2, LEGACY, "futonly", [A, B], tmp_path)
    assert fr2.new_rows == 2                         # exactly one new Tuesday per contract
    assert len(fr2.raw) == 14
    assert all(" > '" in w for w in client2.where_calls)  # incremental threshold used
    assert fr2.latest_as_of == client2.store[DATE].max()


def test_no_new_report_is_zero_new_rows(tmp_path, make_legacy_raw):
    fetch_incremental(FakeClient(_store(make_legacy_raw, 6, [A])), LEGACY, "futonly", [A], tmp_path)
    client2 = FakeClient(_store(make_legacy_raw, 6, [A]))      # nothing newer
    fr2 = fetch_incremental(client2, LEGACY, "futonly", [A], tmp_path)
    assert fr2.new_rows == 0
    assert len(fr2.raw) == 6
    assert fr2.latest_as_of == client2.store[DATE].max()       # unchanged


def test_new_contract_backfills_full_history(tmp_path, make_legacy_raw):
    fetch_incremental(FakeClient(_store(make_legacy_raw, 6, [A])), LEGACY, "futonly", [A], tmp_path)
    client2 = FakeClient(_store(make_legacy_raw, 6, [A, B]))
    fr2 = fetch_incremental(client2, LEGACY, "futonly", [A, B], tmp_path)
    assert fr2.new_rows == 6                          # B's whole history, A already cached
    assert " > '" in _where_for(client2, A)           # A pulled incrementally
    assert " > '" not in _where_for(client2, B)       # B backfilled in full
    assert len(fr2.raw) == 12 and set(fr2.raw[NAMECOL].unique()) == {A, B}


def test_dedupe_keeps_last_on_revision(tmp_path, make_legacy_raw):
    fetch_incremental(FakeClient(_store(make_legacy_raw, 6, [A])), LEGACY, "futonly", [A], tmp_path)
    revised = _store(make_legacy_raw, 6, [A])
    revised.loc[2, "noncomm_positions_long_all"] = 999999    # CFTC revises an old week
    # honor_date_filter=False -> the source re-sends overlapping rows
    client2 = FakeClient(revised, honor_date_filter=False)
    fr2 = fetch_incremental(client2, LEGACY, "futonly", [A], tmp_path)
    assert not fr2.raw.duplicated(subset=[NAMECOL, DATE]).any()   # one row per (contract, date)
    assert len(fr2.raw) == 6
    week2 = pd.to_datetime(revised.loc[2, DATE])
    row = fr2.raw[(fr2.raw[NAMECOL] == A) & (fr2.raw[DATE] == week2)]
    assert row["noncomm_positions_long_all"].iloc[0] == 999999   # last (revised) wins


def test_cache_scopes_to_tracked_contracts(tmp_path, make_legacy_raw):
    fetch_incremental(FakeClient(_store(make_legacy_raw, 6, [A, B])), LEGACY, "futonly", [A, B], tmp_path)
    fr2 = fetch_incremental(FakeClient(_store(make_legacy_raw, 6, [A, B])), LEGACY, "futonly", [A], tmp_path)
    assert set(fr2.raw[NAMECOL].unique()) == {A}     # B dropped from this run
    cached = load_cache(cache_path(tmp_path, "legacy", "futonly"))
    assert set(cached[NAMECOL].unique()) == {A}      # and removed from disk cache


def test_unresolved_contract_is_skipped(tmp_path, make_legacy_raw):
    client = FakeClient(_store(make_legacy_raw, 6, [A]))
    fr = fetch_incremental(client, LEGACY, "futonly", [A, "TOTALLY MADE UP"], tmp_path)
    by_q = {m.query: m for m in fr.matches}
    assert by_q["TOTALLY MADE UP"].how == "none" and by_q["TOTALLY MADE UP"].resolved is None
    assert set(fr.raw[NAMECOL].unique()) == {A}
    assert not any("MADE UP" in w for w in client.where_calls)


def test_refresh_full_ignores_cache_and_threshold(tmp_path, make_legacy_raw):
    fetch_incremental(FakeClient(_store(make_legacy_raw, 6, [A])), LEGACY, "futonly", [A], tmp_path)
    revised = _store(make_legacy_raw, 6, [A])
    revised["noncomm_positions_long_all"] = 1234
    client2 = FakeClient(revised)
    fr2 = fetch_incremental(client2, LEGACY, "futonly", [A], tmp_path, refresh_full=True)
    assert all(" > '" not in w for w in client2.where_calls)      # no incremental threshold
    assert (fr2.raw["noncomm_positions_long_all"] == 1234).all()  # fully re-pulled
    assert len(fr2.raw) == 6


def test_discovery_skips_non_tabular_duplicate(tmp_path, make_legacy_raw):
    # two datasets share the official name; the first is non-tabular (probe raises)
    client = FakeClient(
        _store(make_legacy_raw, 4, [A]),
        dataset_id="good-1111",
        extra_datasets=[{"id": "bad-0000", "name": "Legacy - Futures Only"}],
        bad_ids={"bad-0000"},
    )
    disc = discover_dataset(client, LEGACY, "futonly")
    assert disc.dataset_id == "good-1111"            # skipped the unqueryable duplicate
    assert "bad-0000" in disc.candidates             # but it was considered
    assert disc.matched_canonical is False


def test_discovery_raises_when_nothing_queryable(make_legacy_raw):
    client = FakeClient(_store(make_legacy_raw, 2, [A]), dataset_id="x", bad_ids={"x", "6dca-aqww"})
    with pytest.raises(SocrataError):
        discover_dataset(client, LEGACY, "futonly")
