"""Socrata client (requests-based), runtime dataset discovery, incremental cache.

Design notes
------------
* No sodapy dependency: a thin requests client gives us explicit control over
  pagination, retries, the app-token header and incremental ``$where`` filters.
* Discovery: list datasets via ``/api/views/metadata/v1`` (the cross-domain
  catalog at api.us.socrata.com does NOT federate CFTC), match by official name,
  prefer a known-canonical queryable id (the parallel duplicate ids are
  non-tabular views that error on query), then probe one row to confirm.
* Incremental: cache is per (report_type, fut_combined) parquet. For each
  contract we pull only rows newer than that contract's max cached Tuesday, so
  adding a brand-new contract backfills its full history without re-pulling the
  rest.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

try:  # urllib3 ships with requests; Retry import path is stable
    from urllib3.util.retry import Retry
except Exception:  # pragma: no cover - extremely defensive
    Retry = None  # type: ignore

from .config import (
    CONTRACT_FIELD,
    DATE_FIELD,
    SOCRATA_DOMAIN,
    ReportSpec,
)

PAGE_SIZE = 50000
SOCRATA_TS_FMT = "%Y-%m-%dT%H:%M:%S.000"


# --------------------------------------------------------------------------
# low-level client
# --------------------------------------------------------------------------
class SocrataError(RuntimeError):
    pass


class SocrataClient:
    """Minimal Socrata SoDA client over requests with retry + app token."""

    def __init__(
        self,
        domain: str = SOCRATA_DOMAIN,
        app_token: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.domain = domain
        self.timeout = timeout
        # env fallback so it "just works" with or without a token
        self.app_token = app_token or os.environ.get("SODA_APP_TOKEN")
        self.session = requests.Session()
        if Retry is not None:
            retry = Retry(
                total=4,
                backoff_factor=0.8,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=frozenset(["GET"]),
                respect_retry_after_header=True,
            )
            adapter = HTTPAdapter(max_retries=retry)
            self.session.mount("https://", adapter)
            self.session.mount("http://", adapter)

    # -- helpers
    def _headers(self) -> Dict[str, str]:
        h = {"Accept": "application/json", "User-Agent": "cot-positioning/0.1"}
        if self.app_token:
            h["X-App-Token"] = self.app_token
        return h

    @property
    def has_token(self) -> bool:
        return bool(self.app_token)

    def _get(self, url: str, params: Optional[Dict[str, str]] = None):
        try:
            resp = self.session.get(
                url, params=params, headers=self._headers(), timeout=self.timeout
            )
        except requests.RequestException as exc:  # network failure
            raise SocrataError("request to %s failed: %s" % (url, exc)) from exc
        if resp.status_code != 200:
            raise SocrataError(
                "HTTP %s from %s: %s" % (resp.status_code, resp.url, resp.text[:300])
            )
        return resp.json()

    # -- discovery
    def list_datasets(self) -> List[dict]:
        """All datasets on the domain (id + name + type)."""
        url = "https://%s/api/views/metadata/v1" % self.domain
        data = self._get(url, {"limit": "400"})
        if not isinstance(data, list):
            raise SocrataError("unexpected metadata payload")
        return data

    def probe(self, dataset_id: str) -> List[str]:
        """Fetch one row; return its column names. Raises if non-tabular/empty."""
        url = "https://%s/resource/%s.json" % (self.domain, dataset_id)
        data = self._get(url, {"$limit": "1"})
        if not isinstance(data, list) or not data:
            raise SocrataError("dataset %s returned no probe rows" % dataset_id)
        return sorted(data[0].keys())

    # -- data
    def get(self, dataset_id: str, params: Dict[str, str]) -> List[dict]:
        url = "https://%s/resource/%s.json" % (self.domain, dataset_id)
        return self._get(url, params)

    def get_all(
        self,
        dataset_id: str,
        where: Optional[str] = None,
        select: Optional[str] = None,
        order: str = DATE_FIELD,
    ) -> List[dict]:
        """Paginated fetch of every row matching ``where``."""
        rows: List[dict] = []
        offset = 0
        while True:
            params: Dict[str, str] = {
                "$limit": str(PAGE_SIZE),
                "$offset": str(offset),
                "$order": order,
            }
            if where:
                params["$where"] = where
            if select:
                params["$select"] = select
            page = self.get(dataset_id, params)
            rows.extend(page)
            if len(page) < PAGE_SIZE:
                break
            offset += PAGE_SIZE
            time.sleep(0.05 if self.has_token else 0.2)  # be polite when untokened
        return rows


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------
@dataclass
class DiscoveryResult:
    dataset_id: str
    dataset_name: str
    endpoint: str
    candidates: List[str]
    matched_canonical: bool
    columns: List[str]


def discover_dataset(
    client: SocrataClient, spec: ReportSpec, fut_combined: str
) -> DiscoveryResult:
    """Resolve the dataset id at runtime, preferring a queryable canonical id."""
    target_name = spec.dataset_name[fut_combined]
    canonical = spec.canonical_id[fut_combined]

    candidates: List[str] = []
    try:
        metas = client.list_datasets()
        for m in metas:
            if str(m.get("name", "")).strip() == target_name:
                mid = m.get("id")
                if mid:
                    candidates.append(mid)
    except SocrataError:
        candidates = []  # discovery endpoint unreachable; fall back to canonical

    # Ordering: canonical first (if discovered or as a hard fallback), then the rest.
    ordered: List[str] = []
    if canonical in candidates:
        ordered.append(canonical)
    ordered.extend([c for c in candidates if c != canonical])
    if canonical not in ordered:
        ordered.append(canonical)  # last-resort even if discovery missed it

    # Probe in order; first tabular one wins (skips the non-tabular duplicate views).
    last_err: Optional[Exception] = None
    for cid in ordered:
        try:
            cols = client.probe(cid)
        except SocrataError as exc:
            last_err = exc
            continue
        return DiscoveryResult(
            dataset_id=cid,
            dataset_name=target_name,
            endpoint="https://%s/resource/%s.json" % (client.domain, cid),
            candidates=candidates,
            matched_canonical=(cid == canonical),
            columns=cols,
        )
    raise SocrataError(
        "could not resolve a queryable dataset for %s/%s (tried %s): %s"
        % (spec.report_type, fut_combined, ordered, last_err)
    )


# --------------------------------------------------------------------------
# contract-name matching
# --------------------------------------------------------------------------
@dataclass
class ContractMatch:
    query: str
    resolved: Optional[str]
    candidates: List[str]
    how: str  # "exact" | "tokens" | "ambiguous" | "none"


def list_contract_names(client: SocrataClient, dataset_id: str) -> List[str]:
    rows = client.get(
        dataset_id,
        {"$select": "distinct %s" % CONTRACT_FIELD, "$limit": str(PAGE_SIZE)},
    )
    names = sorted({r[CONTRACT_FIELD] for r in rows if r.get(CONTRACT_FIELD)})
    return names


def match_contract(query: str, available: Sequence[str]) -> ContractMatch:
    """Resolve a user-typed contract to an official name.

    exact (case-insensitive) -> all-tokens-substring -> ambiguous(shortest) -> none.
    """
    q = query.strip()
    qup = q.upper()
    exact = [a for a in available if a.upper() == qup]
    if len(exact) == 1:
        return ContractMatch(q, exact[0], exact, "exact")
    if len(exact) > 1:
        return ContractMatch(q, min(exact, key=len), exact, "ambiguous")

    tokens = [t for t in qup.replace(",", " ").split() if t]
    subset = [a for a in available if all(t in a.upper() for t in tokens)]
    if len(subset) == 1:
        return ContractMatch(q, subset[0], subset, "tokens")
    if len(subset) > 1:
        # deterministic: shortest name is usually the primary/parent contract
        return ContractMatch(q, min(subset, key=len), subset, "ambiguous")
    return ContractMatch(q, None, [], "none")


# --------------------------------------------------------------------------
# incremental cache
# --------------------------------------------------------------------------
def cache_path(cachedir: Path, report_type: str, fut_combined: str) -> Path:
    return Path(cachedir) / ("%s_%s_raw.parquet" % (report_type, fut_combined))


def load_cache(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _escape(value: str) -> str:
    return value.replace("'", "''")


@dataclass
class FetchResult:
    raw: pd.DataFrame                 # full cached history (all tracked contracts)
    matches: List[ContractMatch]     # name resolution per requested contract
    new_rows: int                    # how many rows appended this run
    latest_as_of: Optional[pd.Timestamp]
    discovery: DiscoveryResult


def fetch_incremental(
    client: SocrataClient,
    spec: ReportSpec,
    fut_combined: str,
    contracts: Sequence[str],
    cachedir: Path,
    refresh_full: bool = False,
) -> FetchResult:
    """Discover dataset, resolve names, pull only new weeks, append, return all."""
    discovery = discover_dataset(client, spec, fut_combined)
    available = list_contract_names(client, discovery.dataset_id)
    matches = [match_contract(c, available) for c in contracts]

    cpath = cache_path(cachedir, spec.report_type, fut_combined)
    cached = pd.DataFrame() if refresh_full else load_cache(cpath)
    if not cached.empty:
        cached[DATE_FIELD] = pd.to_datetime(cached[DATE_FIELD])

    frames: List[pd.DataFrame] = []
    if not cached.empty:
        frames.append(cached)
    new_rows = 0

    for m in matches:
        if not m.resolved:
            continue
        name = m.resolved
        where = "%s='%s'" % (CONTRACT_FIELD, _escape(name))
        max_date = None
        if not cached.empty:
            sub = cached[cached[CONTRACT_FIELD] == name]
            if not sub.empty:
                max_date = sub[DATE_FIELD].max()
        if max_date is not None and pd.notna(max_date):
            where += " AND %s > '%s'" % (
                DATE_FIELD,
                pd.Timestamp(max_date).strftime(SOCRATA_TS_FMT),
            )
        rows = client.get_all(discovery.dataset_id, where=where)
        if rows:
            df = pd.DataFrame(rows)
            df[DATE_FIELD] = pd.to_datetime(df[DATE_FIELD])
            frames.append(df)
            new_rows += len(df)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame()

    if not combined.empty:
        combined = (
            combined.drop_duplicates(subset=[CONTRACT_FIELD, DATE_FIELD], keep="last")
            .sort_values([CONTRACT_FIELD, DATE_FIELD])
            .reset_index(drop=True)
        )
        # keep cache scoped to currently-tracked, resolved contracts
        resolved_names = {m.resolved for m in matches if m.resolved}
        scoped = combined[combined[CONTRACT_FIELD].isin(resolved_names)].copy()
        cpath.parent.mkdir(parents=True, exist_ok=True)
        scoped.to_parquet(cpath, index=False)
        combined = scoped

    latest = (
        pd.Timestamp(combined[DATE_FIELD].max()) if not combined.empty else None
    )
    return FetchResult(
        raw=combined,
        matches=matches,
        new_rows=new_rows,
        latest_as_of=latest,
        discovery=discovery,
    )
