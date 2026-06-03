"""Command-line entry point: fetch -> transform -> align -> visualise -> export.

Run ``python -m cot.cli --help`` for options. The pipeline prints, on every run:
the dataset/endpoint actually used, the exact contract names matched, the
as-of/lag note, and any data-validation warnings.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from . import disclaimers
from .align import build_price_overlay, check_tuesday, normalize_tuesday_index
from .config import (
    DEFAULT_CONTRACTS,
    DEFAULT_WINDOW,
    get_report_spec,
    guess_price_ticker,
)
from .fetch import SocrataClient, fetch_incremental, list_contract_names, discover_dataset
from .transform import (
    divergence,
    enrich,
    latest_snapshot,
    tidy_positions,
    validate,
)
from .viz import build_html, render_contract


def _split_contracts(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return list(DEFAULT_CONTRACTS)
    out: List[str] = []
    for v in values:
        out.extend([p.strip() for p in v.split(",") if p.strip()])
    return out or list(DEFAULT_CONTRACTS)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cot",
        description="CFTC COT positioning context tool (NOT a signal generator).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--contracts",
        nargs="*",
        default=None,
        help="Contract names (official contract_market_name; comma or space separated). "
        "Fuzzy-matched and the resolved names are printed.",
    )
    p.add_argument("--report-type", choices=["legacy", "tff"], default="legacy")
    p.add_argument(
        "--fut-combined",
        choices=["futonly", "combined"],
        default="futonly",
        help="Futures-Only or Futures+Options Combined.",
    )
    p.add_argument("--start", default=None, help="Output start date (YYYY-MM-DD).")
    p.add_argument("--end", default=None, help="Output end date (YYYY-MM-DD).")
    p.add_argument(
        "--lookback-weeks",
        type=int,
        default=DEFAULT_WINDOW,
        help="COT Index trailing window in weeks.",
    )
    p.add_argument(
        "--min-periods",
        type=int,
        default=None,
        help="Min weeks before COT Index is computed (default: full window).",
    )
    p.add_argument("--outdir", default="output", help="Output directory.")
    p.add_argument("--cachedir", default=".cache", help="Parquet cache directory.")
    p.add_argument("--no-price", action="store_true", help="Skip the price overlay panel.")
    p.add_argument(
        "--app-token",
        default=None,
        help="Socrata app token (else env SODA_APP_TOKEN; works without one).",
    )
    p.add_argument(
        "--refresh-full",
        action="store_true",
        help="Ignore cache and re-pull full history.",
    )
    p.add_argument(
        "--list-contracts",
        action="store_true",
        help="Print available contract_market_name values for the dataset and exit.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    print(disclaimers.banner())

    spec = get_report_spec(args.report_type)
    client = SocrataClient(app_token=args.app_token)
    print(
        "Socrata app token: %s"
        % ("present (X-App-Token)" if client.has_token else "NONE (unauthenticated; may throttle)")
    )

    # --- discovery-only helper -------------------------------------------------
    if args.list_contracts:
        disc = discover_dataset(client, spec, args.fut_combined)
        print("Dataset used: %s (%s)  ->  %s" % (disc.dataset_id, disc.dataset_name, disc.endpoint))
        names = list_contract_names(client, disc.dataset_id)
        print("\n%d contract_market_name values:" % len(names))
        for n in names:
            print("  %s" % n)
        return 0

    contracts = _split_contracts(args.contracts)
    print("\nReport: %s | %s" % (spec.label, args.fut_combined))
    print("Requested contracts: %s" % ", ".join(repr(c) for c in contracts))

    # --- fetch (incremental) ---------------------------------------------------
    try:
        fr = fetch_incremental(
            client=client,
            spec=spec,
            fut_combined=args.fut_combined,
            contracts=contracts,
            cachedir=Path(args.cachedir),
            refresh_full=args.refresh_full,
        )
    except Exception as exc:
        print("\nERROR during fetch: %s" % exc, file=sys.stderr)
        return 2

    d = fr.discovery
    print(
        "\nDataset used: %s (%s)%s"
        % (
            d.dataset_id,
            d.dataset_name,
            "  [matched canonical id]" if d.matched_canonical else "  [discovered]",
        )
    )
    print("Endpoint: %s" % d.endpoint)
    if d.candidates:
        print("Discovery candidates by name: %s" % ", ".join(d.candidates))

    print("\nContract name resolution (matched official names):")
    resolved_names: List[Optional[str]] = []
    for m in fr.matches:
        resolved_names.append(m.resolved)
        if m.how == "exact":
            print("  %r -> %r  [exact]" % (m.query, m.resolved))
        elif m.how == "tokens":
            print("  %r -> %r  [token match]" % (m.query, m.resolved))
        elif m.how == "ambiguous":
            print(
                "  %r -> %r  [AMBIGUOUS: chose shortest of %s]"
                % (m.query, m.resolved, m.candidates)
            )
        else:
            print("  %r -> NO MATCH (check spelling; try --list-contracts)" % m.query)

    print("\nNew rows fetched this run: %d" % fr.new_rows)
    if fr.latest_as_of is None:
        print("No data available for the requested contracts. Nothing to do.")
        return 1
    print("Latest available as-of (Tuesday): %s" % fr.latest_as_of.strftime("%Y-%m-%d"))
    if fr.new_rows == 0:
        print(
            "  -> No newer report than the cache. This week's report may not be "
            "published yet (releases Fri 3:30pm ET)."
        )

    # --- transform -------------------------------------------------------------
    tidy = tidy_positions(fr.raw, spec)
    tidy = normalize_tuesday_index(tidy, "report_date")
    enriched = enrich(tidy, window=args.lookback_weeks, min_periods=args.min_periods)

    # validation (on full history)
    val = validate(enriched, fr.raw, spec, contracts, resolved_names)
    val.warnings = check_tuesday(enriched["report_date"]) + [
        w for w in val.warnings if "not a Tuesday" not in w
    ]
    if val.warnings:
        print("\nData validation warnings (%d):" % len(val.warnings))
        for w in val.warnings:
            print("  ! %s" % w)
    else:
        print("\nData validation: no warnings.")

    # --- date-range filter for OUTPUT (COT index already used full history) ----
    out = enriched
    if args.start:
        out = out[out["report_date"] >= pd.Timestamp(args.start)]
    if args.end:
        out = out[out["report_date"] <= pd.Timestamp(args.end)]
    out = out.reset_index(drop=True)

    outdir = Path(args.outdir)
    chartdir = outdir / "charts"
    outdir.mkdir(parents=True, exist_ok=True)

    # --- exports (tidy parquet + CSV; divergence; latest snapshot) -------------
    tidy_cols = [
        "report_date", "contract", "report_type", "fut_combined", "category",
        "category_label", "long", "short", "spread", "net", "net_chg_wow",
        "open_interest", "pct_oi_long", "pct_oi_short", "pct_oi_net",
        "traders_long", "traders_short", "cot_index",
    ]
    export = out[[c for c in tidy_cols if c in out.columns]].copy()
    export.to_parquet(outdir / "cot_tidy.parquet", index=False)
    export.to_csv(outdir / "cot_tidy.csv", index=False)

    div = divergence(out, spec)
    if not div.empty:
        div.to_csv(outdir / "cot_divergence.csv", index=False)

    snap = latest_snapshot(out)
    if not snap.empty:
        snap_cols = [
            "report_date", "contract", "category_label", "net", "net_chg_wow",
            "cot_index", "pct_oi_net", "traders_long", "traders_short",
        ]
        snap[[c for c in snap_cols if c in snap.columns]].to_csv(
            outdir / "cot_latest_snapshot.csv", index=False
        )

    # --- visuals ---------------------------------------------------------------
    charts = []
    present_contracts = [n for n in resolved_names if n and n in set(out["contract"])]
    for contract in present_contracts:
        tdates = out[out["contract"] == contract]["report_date"]
        if args.no_price:
            from .align import PriceResult

            price = PriceResult(
                aligned=pd.DataFrame(columns=["report_date", "price"]),
                ticker=None,
                note="price overlay disabled (--no-price)",
                ok=False,
            )
        else:
            ticker = guess_price_ticker(contract)
            price = build_price_overlay(contract, ticker, tdates)
        print("  [%s] %s" % (contract, price.note))
        charts.append(render_contract(out, spec, contract, price, chartdir))

    html_path = outdir / "cot_report.html"
    meta = {
        "report": "%s/%s" % (spec.report_type, args.fut_combined),
        "dataset": d.dataset_id,
        "COT window (wk)": str(args.lookback_weeks),
        "latest as-of": fr.latest_as_of.strftime("%Y-%m-%d"),
    }
    build_html(charts, html_path, meta, val.warnings)

    # --- summary ---------------------------------------------------------------
    print("\nWrote:")
    print("  %s" % (outdir / "cot_tidy.parquet"))
    print("  %s" % (outdir / "cot_tidy.csv"))
    if not div.empty:
        print("  %s" % (outdir / "cot_divergence.csv"))
    if not snap.empty:
        print("  %s" % (outdir / "cot_latest_snapshot.csv"))
    for ch in charts:
        for path in ch.pngs.values():
            print("  %s" % path)
    print("  %s  <- open this" % html_path)
    print("\n" + disclaimers.GUARDRAIL_NOTE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
