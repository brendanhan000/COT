"""Static configuration: report-type registry, category->field maps, defaults.

Field-name candidate lists are intentionally tolerant. CFTC's Socrata columns
are inconsistent (e.g. the real typo ``noncomm_postions_spread_all``,
``tot_rept_positions_short`` with no ``_all``, and TFF mixing
``dealer_positions_long_all`` with ``lev_money_positions_long``). The transform
layer resolves the first candidate that actually exists in the returned columns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# --- global defaults -------------------------------------------------------

DEFAULT_WINDOW = 156  # weeks (~3 years) for the COT Index stochastic window
EXTREME_HIGH = 80.0
EXTREME_LOW = 20.0
SOCRATA_DOMAIN = "publicreporting.cftc.gov"

# Official contract_market_name values (corrected from the user's shorthand:
# there is no literal "CRUDE OIL WTI"; MICRO E-MINI carries an " INDEX" suffix).
DEFAULT_CONTRACTS: List[str] = [
    "E-MINI S&P 500",
    "MICRO E-MINI S&P 500 INDEX",
    "CRUDE OIL, LIGHT SWEET-WTI",
]

# yfinance proxy tickers. Always labelled a proxy on the chart.
DEFAULT_PRICE_MAP: Dict[str, str] = {
    "E-MINI S&P 500": "ES=F",
    "MICRO E-MINI S&P 500 INDEX": "ES=F",
    "CRUDE OIL, LIGHT SWEET-WTI": "CL=F",
}


@dataclass(frozen=True)
class CategoryFields:
    """How one trader category maps onto Socrata columns (tolerant candidates)."""

    key: str               # stable internal key, e.g. "noncommercial"
    label: str             # human label for charts/exports
    long: List[str]        # candidate column names for longs
    short: List[str]       # candidate column names for shorts
    spread: List[str] = field(default_factory=list)        # spreading (optional)
    traders_long: List[str] = field(default_factory=list)  # # traders long (optional)
    traders_short: List[str] = field(default_factory=list)
    pct_oi_long: List[str] = field(default_factory=list)    # % of OI long (optional)
    pct_oi_short: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReportSpec:
    """Everything the pipeline needs to know about a report type."""

    report_type: str                      # "legacy" | "tff"
    label: str
    dataset_name: Dict[str, str]          # fut_combined -> official dataset name
    canonical_id: Dict[str, str]          # fut_combined -> known-good queryable id
    categories: List[CategoryFields]
    primary_spec: str                     # category key used for price overlay / headline
    primary_hedger: str                   # opposite side for divergence
    headline_keys: List[str]              # categories shown on the COT-Index panel
    net_chart_keys: List[str]             # categories shown on the net-position panel
    divergence_pair: Tuple[str, str]      # (a, b) for the descriptive divergence

    def category(self, key: str) -> CategoryFields:
        for c in self.categories:
            if c.key == key:
                return c
        raise KeyError(key)


# --- LEGACY (Futures-Only / Combined) -------------------------------------
# Confirmed columns from resource 6dca-aqww. Note the "postions" typo candidate.
_LEGACY = ReportSpec(
    report_type="legacy",
    label="Legacy (Non-Commercial / Commercial / Non-Reportable)",
    dataset_name={
        "futonly": "Legacy - Futures Only",
        "combined": "Legacy - Combined",
    },
    canonical_id={"futonly": "6dca-aqww", "combined": "jun7-fc8e"},
    categories=[
        CategoryFields(
            key="noncommercial",
            label="Non-Commercial (Large Specs)",
            long=["noncomm_positions_long_all"],
            short=["noncomm_positions_short_all"],
            spread=[
                "noncomm_postions_spread_all",   # <- real CFTC typo, listed first
                "noncomm_positions_spread_all",
                "noncomm_positions_spread",
            ],
            traders_long=["traders_noncomm_long_all"],
            traders_short=["traders_noncomm_short_all"],
            pct_oi_long=["pct_of_oi_noncomm_long_all"],
            pct_oi_short=["pct_of_oi_noncomm_short_all"],
        ),
        CategoryFields(
            key="commercial",
            label="Commercial (Hedgers)",
            long=["comm_positions_long_all"],
            short=["comm_positions_short_all"],
            traders_long=["traders_comm_long_all"],
            traders_short=["traders_comm_short_all"],
            pct_oi_long=["pct_of_oi_comm_long_all"],
            pct_oi_short=["pct_of_oi_comm_short_all"],
        ),
        CategoryFields(
            key="nonreportable",
            label="Non-Reportable (Small Specs)",
            long=["nonrept_positions_long_all"],
            short=["nonrept_positions_short_all"],
            pct_oi_long=["pct_of_oi_nonrept_long_all"],
            pct_oi_short=["pct_of_oi_nonrept_short_all"],
        ),
    ],
    primary_spec="noncommercial",
    primary_hedger="commercial",
    headline_keys=["noncommercial", "commercial"],
    net_chart_keys=["noncommercial", "commercial", "nonreportable"],
    divergence_pair=("noncommercial", "commercial"),
)


# --- TFF (Traders in Financial Futures) -----------------------------------
# Confirmed columns from resource gpe5-46if. Asset-mgr / lev-money / other-rept
# position columns have NO "_all" suffix; dealer + nonrept DO. Hence candidates.
_TFF = ReportSpec(
    report_type="tff",
    label="TFF (Dealer / Asset Mgr / Leveraged Funds / Other / Non-Reportable)",
    dataset_name={
        "futonly": "TFF - Futures Only",
        "combined": "TFF - Combined",
    },
    canonical_id={"futonly": "gpe5-46if", "combined": "yw9f-hn96"},
    categories=[
        CategoryFields(
            key="dealer",
            label="Dealer / Intermediary",
            long=["dealer_positions_long_all"],
            short=["dealer_positions_short_all"],
            spread=["dealer_positions_spread_all"],
            traders_long=["traders_dealer_long_all"],
            traders_short=["traders_dealer_short_all"],
            pct_oi_long=["pct_of_oi_dealer_long_all"],
            pct_oi_short=["pct_of_oi_dealer_short_all"],
        ),
        CategoryFields(
            key="asset_manager",
            label="Asset Manager / Institutional",
            long=["asset_mgr_positions_long", "asset_mgr_positions_long_all"],
            short=["asset_mgr_positions_short", "asset_mgr_positions_short_all"],
            spread=["asset_mgr_positions_spread", "asset_mgr_positions_spread_all"],
            traders_long=["traders_asset_mgr_long_all"],
            traders_short=["traders_asset_mgr_short_all"],
            pct_oi_long=["pct_of_oi_asset_mgr_long"],
            pct_oi_short=["pct_of_oi_asset_mgr_short"],
        ),
        CategoryFields(
            key="leveraged_funds",
            label="Leveraged Funds",
            long=["lev_money_positions_long", "lev_money_positions_long_all"],
            short=["lev_money_positions_short", "lev_money_positions_short_all"],
            spread=["lev_money_positions_spread", "lev_money_positions_spread_all"],
            traders_long=["traders_lev_money_long_all"],
            traders_short=["traders_lev_money_short_all"],
            pct_oi_long=["pct_of_oi_lev_money_long"],
            pct_oi_short=["pct_of_oi_lev_money_short"],
        ),
        CategoryFields(
            key="other_reportable",
            label="Other Reportable",
            long=["other_rept_positions_long", "other_rept_positions_long_all"],
            short=["other_rept_positions_short", "other_rept_positions_short_all"],
            spread=["other_rept_positions_spread", "other_rept_positions_spread_all"],
            traders_long=["traders_other_rept_long_all"],
            traders_short=["traders_other_rept_short", "traders_other_rept_short_all"],
            pct_oi_long=["pct_of_oi_other_rept_long"],
            pct_oi_short=["pct_of_oi_other_rept_short"],
        ),
        CategoryFields(
            key="nonreportable",
            label="Non-Reportable (Small Specs)",
            long=["nonrept_positions_long_all"],
            short=["nonrept_positions_short_all"],
            pct_oi_long=["pct_of_oi_nonrept_long_all"],
            pct_oi_short=["pct_of_oi_nonrept_short_all"],
        ),
    ],
    primary_spec="leveraged_funds",
    primary_hedger="asset_manager",
    headline_keys=["leveraged_funds", "asset_manager"],
    net_chart_keys=["dealer", "asset_manager", "leveraged_funds", "other_reportable"],
    divergence_pair=("leveraged_funds", "asset_manager"),
)


REPORTS: Dict[str, ReportSpec] = {"legacy": _LEGACY, "tff": _TFF}

# Common identifier columns we always want to carry through.
DATE_FIELD = "report_date_as_yyyy_mm_dd"
CONTRACT_FIELD = "contract_market_name"
OI_FIELD = "open_interest_all"
TOTAL_TRADERS_FIELD = "traders_tot_all"
ID_FIELDS = [
    CONTRACT_FIELD,
    DATE_FIELD,
    "market_and_exchange_names",
    "cftc_contract_market_code",
    "futonly_or_combined",
    OI_FIELD,
    TOTAL_TRADERS_FIELD,
]


def get_report_spec(report_type: str) -> ReportSpec:
    rt = report_type.lower().strip()
    if rt not in REPORTS:
        raise ValueError(
            "Unknown report type %r; choose from %s" % (report_type, sorted(REPORTS))
        )
    return REPORTS[rt]


def guess_price_ticker(
    contract_name: str, overrides: Optional[Dict[str, str]] = None
) -> Optional[str]:
    """Best-effort yfinance proxy ticker for a contract; None if unknown.

    Exact map first, then token heuristics so user-renamed contracts still work.
    """
    overrides = overrides or {}
    if contract_name in overrides:
        return overrides[contract_name]
    if contract_name in DEFAULT_PRICE_MAP:
        return DEFAULT_PRICE_MAP[contract_name]
    upper = contract_name.upper()
    if "S&P 500" in upper:
        return "ES=F"
    if "NASDAQ" in upper or "NDX" in upper:
        return "NQ=F"
    if "CRUDE" in upper and ("WTI" in upper or "LIGHT SWEET" in upper):
        return "CL=F"
    if "CRUDE" in upper and "BRENT" in upper:
        return "BZ=F"
    if "GOLD" in upper:
        return "GC=F"
    if "NATURAL GAS" in upper:
        return "NG=F"
    return None
