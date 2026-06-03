"""Visualisation: 3 glanceable panels per contract + a combined HTML page.

Panels
------
1. Net position by category over time (current value labelled).
2. COT Index (0-100) with >80 / <20 extreme bands ("stretched - context only").
3. Net position (primary spec category) overlaid on a PROXY price, Tuesday-aligned.

All charts use the Agg backend (headless PNG). The HTML embeds PNGs as base64 so
it is a single portable file, prefaced with the lag note + guardrails.
"""
from __future__ import annotations

import base64
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import disclaimers  # noqa: E402
from .align import PriceResult  # noqa: E402
from .config import EXTREME_HIGH, EXTREME_LOW, ReportSpec  # noqa: E402

_COLORS = {
    0: "#1f77b4",
    1: "#d62728",
    2: "#2ca02c",
    3: "#9467bd",
    4: "#8c564b",
}


def _fmt_k(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return "{:,.0f}".format(value)


def _asof_str(date) -> str:
    return pd.Timestamp(date).strftime("%Y-%m-%d")


def _footer(fig) -> None:
    fig.text(
        0.5,
        0.005,
        disclaimers.CHART_FOOTER,
        ha="center",
        va="bottom",
        fontsize=7,
        color="#555555",
        wrap=True,
    )


def _date_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))


@dataclass
class ContractCharts:
    contract: str
    as_of: pd.Timestamp
    pngs: Dict[str, Path]            # panel name -> file path
    note: str                       # price overlay note


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def panel_net_positions(
    enriched: pd.DataFrame, spec: ReportSpec, contract: str, out_png: Path
) -> pd.Timestamp:
    sub = enriched[enriched["contract"] == contract]
    as_of = pd.Timestamp(sub["report_date"].max())
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, key in enumerate(spec.net_chart_keys):
        cat = sub[sub["category"] == key].sort_values("report_date")
        if cat.empty:
            continue
        label = cat["category_label"].iloc[0]
        ax.plot(
            cat["report_date"],
            cat["net"],
            label=label,
            color=_COLORS.get(i, None),
            linewidth=1.4,
        )
        last = cat.iloc[-1]
        ax.annotate(
            _fmt_k(last["net"]),
            xy=(last["report_date"], last["net"]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8,
            color=_COLORS.get(i, None),
            va="center",
        )
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
    ax.set_title(
        "%s - Net Position by Category  (as-of Tue %s)" % (contract, _asof_str(as_of)),
        fontsize=11,
    )
    ax.set_ylabel("Net contracts (long - short)")
    ax.legend(loc="best", fontsize=8, frameon=False)
    _date_axis(ax)
    fig.subplots_adjust(bottom=0.13)
    _footer(fig)
    _save(fig, out_png)
    return as_of


def panel_cot_index(
    enriched: pd.DataFrame, spec: ReportSpec, contract: str, out_png: Path
) -> None:
    sub = enriched[enriched["contract"] == contract]
    as_of = pd.Timestamp(sub["report_date"].max())
    fig, ax = plt.subplots(figsize=(10, 4.6))

    # extreme bands
    ax.axhspan(EXTREME_HIGH, 100, color="#d62728", alpha=0.08)
    ax.axhspan(0, EXTREME_LOW, color="#2ca02c", alpha=0.08)
    ax.axhline(EXTREME_HIGH, color="#d62728", linewidth=0.8, linestyle=":")
    ax.axhline(EXTREME_LOW, color="#2ca02c", linewidth=0.8, linestyle=":")
    ax.axhline(50, color="#aaaaaa", linewidth=0.6, linestyle="--")

    # Primary spec category is drawn bold; the (near-mirror) hedger is faded so
    # the panel stays glanceable instead of a two-line scribble.
    for key in spec.headline_keys:
        cat = sub[sub["category"] == key].sort_values("report_date")
        cat = cat[cat["cot_index"].notna()]
        if cat.empty:
            continue
        primary = key == spec.primary_spec
        label = cat["category_label"].iloc[0]
        color = _COLORS[0] if primary else "#777777"
        ax.plot(
            cat["report_date"],
            cat["cot_index"],
            label=label,
            color=color,
            linewidth=1.5 if primary else 0.9,
            alpha=0.95 if primary else 0.45,
            zorder=4 if primary else 2,
        )
        last = cat.iloc[-1]
        val = last["cot_index"]
        ax.scatter([last["report_date"]], [val], color=color, zorder=5, s=26 if primary else 14)
        tag = "  %s" % disclaimers.EXTREME_LABEL if (val >= EXTREME_HIGH or val <= EXTREME_LOW) else ""
        ax.annotate(
            "%.0f%s" % (val, tag),
            xy=(last["report_date"], val),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8 if primary else 7,
            fontweight="bold" if primary else "normal",
            color=color,
            va="center",
        )

    ax.set_ylim(-2, 102)
    ax.set_title(
        "%s - COT Index 0-100 (trailing window)  (as-of Tue %s)"
        % (contract, _asof_str(as_of)),
        fontsize=11,
    )
    ax.set_ylabel("COT Index (stochastic of net)")
    ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.85)
    _date_axis(ax)
    fig.subplots_adjust(bottom=0.13)
    _footer(fig)
    _save(fig, out_png)


def panel_net_vs_price(
    enriched: pd.DataFrame,
    spec: ReportSpec,
    contract: str,
    price: PriceResult,
    out_png: Path,
) -> str:
    sub = enriched[
        (enriched["contract"] == contract) & (enriched["category"] == spec.primary_spec)
    ].sort_values("report_date")
    as_of = pd.Timestamp(sub["report_date"].max())
    cat_label = sub["category_label"].iloc[0] if not sub.empty else spec.primary_spec

    fig, ax1 = plt.subplots(figsize=(10, 4.6))
    ax1.plot(
        sub["report_date"], sub["net"], color=_COLORS[0], linewidth=1.4, label="%s net" % cat_label
    )
    ax1.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("%s net contracts" % cat_label, color=_COLORS[0])
    ax1.tick_params(axis="y", labelcolor=_COLORS[0])
    _date_axis(ax1)

    note = price.note
    if price.ok and not price.aligned.empty:
        merged = sub.merge(price.aligned, on="report_date", how="left")
        ax2 = ax1.twinx()
        ax2.plot(
            merged["report_date"],
            merged["price"],
            color="#444444",
            linewidth=1.1,
            alpha=0.8,
            label="price (proxy)",
        )
        ax2.set_ylabel("PROXY price: %s" % price.ticker, color="#444444")
        ax2.tick_params(axis="y", labelcolor="#444444")

    ax1.set_title(
        "%s - %s Net vs Price (PROXY)  (as-of Tue %s)"
        % (contract, cat_label, _asof_str(as_of)),
        fontsize=11,
    )
    fig.text(0.5, 0.045, note, ha="center", fontsize=7, color="#444444")
    fig.subplots_adjust(bottom=0.16)
    _footer(fig)
    _save(fig, out_png)
    return note


def panel_divergence(
    enriched: pd.DataFrame, spec: ReportSpec, contract: str, out_png: Path
) -> None:
    """DESCRIPTIVE net(a) - net(b). The two sides usually oppose; not predictive."""
    a, b = spec.divergence_pair
    sub = enriched[enriched["contract"] == contract]
    as_of = pd.Timestamp(sub["report_date"].max())
    wide = sub.pivot_table(index="report_date", columns="category", values="net").sort_index()
    fig, ax = plt.subplots(figsize=(10, 4.4))
    if a in wide.columns and b in wide.columns:
        div = (wide[a] - wide[b]).dropna()
        la, lb = spec.category(a).label, spec.category(b).label
        ax.plot(div.index, div.values, color="#6a3d9a", linewidth=1.3)
        ax.fill_between(div.index, div.values, 0, where=(div.values >= 0), color="#6a3d9a", alpha=0.12)
        ax.fill_between(div.index, div.values, 0, where=(div.values < 0), color="#b15928", alpha=0.12)
        ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
        if len(div):
            ax.annotate(
                _fmt_k(div.iloc[-1]),
                xy=(div.index[-1], div.iloc[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=8,
                color="#6a3d9a",
                va="center",
            )
        ax.set_title(
            "%s - Divergence: %s minus %s net  (DESCRIPTIVE, not predictive)  (as-of Tue %s)"
            % (contract, la, lb, _asof_str(as_of)),
            fontsize=10.5,
        )
        ax.set_ylabel("net(%s) - net(%s)" % (a, b))
    else:
        ax.text(0.5, 0.5, "divergence pair unavailable", ha="center", transform=ax.transAxes)
    _date_axis(ax)
    fig.subplots_adjust(bottom=0.13)
    _footer(fig)
    _save(fig, out_png)


def render_contract(
    enriched: pd.DataFrame,
    spec: ReportSpec,
    contract: str,
    price: PriceResult,
    outdir: Path,
) -> ContractCharts:
    safe = "".join(c if c.isalnum() else "_" for c in contract).strip("_")
    p1 = outdir / ("%s_1_net.png" % safe)
    p2 = outdir / ("%s_2_cotindex.png" % safe)
    p3 = outdir / ("%s_3_net_vs_price.png" % safe)
    p4 = outdir / ("%s_4_divergence.png" % safe)
    as_of = panel_net_positions(enriched, spec, contract, p1)
    panel_cot_index(enriched, spec, contract, p2)
    note = panel_net_vs_price(enriched, spec, contract, price, p3)
    panel_divergence(enriched, spec, contract, p4)
    return ContractCharts(
        contract=contract,
        as_of=as_of,
        pngs={"net": p1, "cot_index": p2, "net_vs_price": p3, "divergence": p4},
        note=note,
    )


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def build_html(
    charts: List[ContractCharts],
    out_html: Path,
    meta: Dict[str, str],
    warnings: List[str],
) -> None:
    """Single self-contained HTML embedding every PNG + the disclaimers."""
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts: List[str] = []
    parts.append(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>CFTC COT Positioning</title>"
        "<style>"
        "body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;"
        "max-width:1100px;margin:24px auto;padding:0 16px;color:#222;}"
        "h1{font-size:22px;margin-bottom:2px;} h2{margin-top:34px;border-bottom:1px solid #eee;}"
        ".note{background:#fff8e1;border:1px solid #ffe0a3;padding:12px 14px;border-radius:8px;"
        "font-size:13px;line-height:1.45;margin:10px 0;}"
        ".guard{background:#fde8e8;border:1px solid #f5b5b5;}"
        ".meta{font-size:12px;color:#555;background:#f5f7fa;border:1px solid #e3e8ee;"
        "padding:10px 12px;border-radius:8px;}"
        ".warn{background:#fff0f0;border:1px solid #f3c2c2;padding:10px 12px;border-radius:8px;"
        "font-size:12px;} img{width:100%;height:auto;margin:6px 0 14px;border:1px solid #eee;}"
        "code{background:#eef;padding:1px 4px;border-radius:4px;}"
        "</style></head><body>"
    )
    parts.append("<h1>CFTC Commitments of Traders - Positioning</h1>")
    parts.append(
        "<div class='meta'>Generated %s &nbsp;|&nbsp; %s</div>"
        % (generated, " &nbsp;|&nbsp; ".join("%s: <code>%s</code>" % (k, v) for k, v in meta.items()))
    )
    parts.append("<div class='note'><b>As-of / lag:</b> %s</div>" % disclaimers.LAG_NOTE)
    parts.append("<div class='note'><b>What this is:</b> %s</div>" % disclaimers.METHODOLOGY_NOTE)
    parts.append("<div class='note guard'><b>Guardrail:</b> %s</div>" % disclaimers.GUARDRAIL_NOTE)
    if warnings:
        parts.append(
            "<div class='warn'><b>Data validation warnings (%d):</b><ul>%s</ul></div>"
            % (len(warnings), "".join("<li>%s</li>" % w for w in warnings))
        )

    for ch in charts:
        parts.append("<h2>%s &mdash; as-of Tuesday %s</h2>" % (ch.contract, _asof_str(ch.as_of)))
        for key in ("net", "cot_index", "net_vs_price", "divergence"):
            png = ch.pngs.get(key)
            if png and png.exists():
                parts.append("<img alt='%s %s' src='data:image/png;base64,%s'/>" % (ch.contract, key, _b64(png)))
    parts.append("</body></html>")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text("".join(parts), encoding="utf-8")
