# CFTC COT Positioning Tool

A weekly, repeatable pipeline that fetches CFTC **Commitments of Traders** data
and renders clean visuals of trader positioning.

> **This is a context / situational-awareness tool, not a signal generator.**
> It shows positioning and flags extremes — then stops. It does **not** emit
> buy/sell signals and is not wired to trade entries. COT has weak,
> regime-dependent timing value; treat everything here as context only.

---

## What it does

- Pulls COT history from the CFTC Public Reporting Socrata API
  (`publicreporting.cftc.gov`), discovering the dataset id **at runtime**.
- Supports **Legacy** (Non-Commercial / Commercial / Non-Reportable) and **TFF**
  (Dealer / Asset Manager / Leveraged Funds / Other / Non-Reportable), each in
  **Futures-Only** or **Futures+Options Combined**.
- Indexes every row to the **Tuesday "as-of" date**, not the Friday release.
- Computes net position, a causal **COT Index (0–100)**, % of open interest,
  trader counts, week-over-week change, and a descriptive Commercial-vs-spec
  divergence.
- Caches full history to parquet and **fetches only new weeks** on later runs.
- Emits an **interactive** combined HTML report (Plotly: zoom / pan / hover,
  synchronized across panels), optional static PNGs (`--png`), and a tidy
  **parquet/CSV** export for your dashboard.
- Validates the data: weekly gaps, contracts that returned nothing, non-Tuesday
  as-of dates, and an open-interest reconciliation check.

## Reporting lag (read this)

CFTC releases each report **Friday 3:30pm ET** describing positions held the
**prior Tuesday** — a ~3 calendar-day lag at release, and up to ~1 week stale
before it can be acted on. Every row, chart, and the price overlay is aligned to
the Tuesday as-of date (`report_date_as_yyyy_mm_dd`). This note is printed on
every run and stamped on every chart.

## Datasets / endpoints

Discovered at runtime via `GET /api/views/metadata/v1` (the cross-domain
`api.us.socrata.com` catalog does **not** federate CFTC). Matched by official
name, tie-broken to a known-canonical **queryable** id (the parallel duplicate
ids on the domain are non-tabular views that error on query), then probed before
use. The id actually used is printed every run.

| report | futures-only | combined |
|---|---|---|
| legacy | `6dca-aqww` | `jun7-fc8e` |
| tff | `gpe5-46if` | `yw9f-hn96` |

Data is read from `GET /resource/{id}.json` with `$where`/`$select`/`$order` and
`$limit`/`$offset` pagination. An optional Socrata app token is read from
`SODA_APP_TOKEN` (or `--app-token`) and sent as `X-App-Token` to avoid
throttling; it works without one.

## COT Index formula

Causal stochastic normalisation of net position over a trailing window
(default **156 weeks / 3 years**):

```
COT_t = 100 · (net_t − min(net over [t−W+1 … t])) / (max(...) − min(...))
```

- Trailing window only — the value at *t* never uses future weeks.
- `min_periods` defaults to the full window (NaN until 3y of history exists).
- A flat window (max == min) maps to **50** (neutral) to avoid divide-by-zero.
- `net = long − short`; spreading is excluded from net but included in the OI
  reconciliation check.

## Install

```bash
pip install -e .
# runtime deps: requests, pandas, numpy, plotly, matplotlib, pyarrow, yfinance
```

Spec target is Python 3.11+. The source is kept syntactically 3.9-safe so it
also runs/tests on older interpreters; 3.11+ is a supported superset. This build
uses `requests` (not `sodapy`). The combined HTML report is **interactive**
(Plotly, plotly.js embedded so it works offline); optional static PNGs (`--png`)
are rendered with `matplotlib`.

## Usage

```bash
# default: legacy futures-only, E-MINI S&P 500 / MICRO E-MINI / WTI crude
python -m cot.cli

# TFF, financial contracts, custom window and output dir
python -m cot.cli --report-type tff --contracts "E-MINI S&P 500" \
    --lookback-weeks 104 --outdir output_tff

# futures+options combined, date-bounded output
python -m cot.cli --fut-combined combined --start 2018-01-01 --end 2026-05-26

# discover the exact official contract names for a dataset
python -m cot.cli --list-contracts
```

Key flags: `--contracts`, `--report-type {legacy,tff}`,
`--fut-combined {futonly,combined}`, `--start/--end`, `--lookback-weeks`,
`--min-periods`, `--outdir`, `--cachedir`, `--no-price`, `--png` (also write
static PNGs), `--plotly-cdn` (slimmer HTML, needs internet), `--app-token`,
`--refresh-full`, `--list-contracts`. See `python -m cot.cli --help`.

Contract names are fuzzy-matched against the official `contract_market_name`
values and the **resolved names are printed** so you can fix typos. Note: there
is no literal "CRUDE OIL WTI" (it's `CRUDE OIL, LIGHT SWEET-WTI`) and MICRO
carries an " INDEX" suffix.

## Outputs (in `--outdir`, default `output/`)

- `cot_tidy.parquet` / `cot_tidy.csv` — one row per (date, contract, category)
  with net, COT index, %OI, WoW change, trader counts.
- `cot_divergence.csv` — descriptive net(spec) − net(hedger).
- `cot_latest_snapshot.csv` — most recent week per contract/category.
- `cot_report.html` — **interactive** combined report; open this. Drag or scroll
  to zoom, toolbar for zoom in/out & reset, double-click to autoscale, unified
  hover, click legend to toggle series. The four panels share one time axis, so
  zooming one zooms all. Self-contained by default (`--plotly-cdn` for a smaller,
  CDN-linked file).
- `charts/<contract>_{1_net,2_cotindex,3_net_vs_price,4_divergence}.png` — static
  matplotlib panels, only written with `--png`.

## Modules

- `cot/fetch.py` — Socrata client, runtime discovery, contract matching,
  incremental parquet cache.
- `cot/transform.py` — net, **COT index** (causal), %OI, WoW, divergence,
  validation/reconciliation, tolerant field resolution.
- `cot/align.py` — Tuesday-index checks, yfinance proxy price, backward as-of
  merge.
- `cot/viz_interactive.py` — interactive Plotly panels + combined HTML (default).
- `cot/viz.py` — static matplotlib panels/PNGs (used with `--png`).
- `cot/cli.py` — orchestration and the printed transparency/guardrails.
- `cot/config.py` — report registry and category→column candidate maps.

## Tests

```bash
python -m pytest
```

Covers the COT-index math against known min/max anchors, causal windowing (the
index at *t* is unchanged by appending future weeks), Tuesday alignment + backward
price merge over a simulated holiday, the contract-name matcher, and the tidy
reshape + OI reconciliation.

## Caveats / honesty

- The Legacy **Commercial** bucket mixes genuine hedgers with swap dealers and is
  classified by predominant business purpose — "smart money" framing is
  approximate.
- COT timing value is weak and regime-dependent. Extremes are labelled
  *"positioning stretched — context only"*, never as entries.
- Price overlays are **proxies** (yfinance front-month/cash), labelled as such.
- Contract history depth depends on the official name: e.g.
  `CRUDE OIL, LIGHT SWEET-WTI` only exists from ~2009 in these datasets (earlier
  crude sits under a different legacy name), and `MICRO E-MINI S&P 500 INDEX`
  starts ~2020. The tool reports the available range rather than silently
  stitching names.
