"""Interactive Plotly output: one zoomable figure per contract + combined HTML.

Each contract is a single figure with four vertically stacked panels that SHARE
the time axis, so zooming/panning one panel zooms them all together:

1. Net position by category
2. COT Index 0-100 with >80 / <20 extreme bands
3. Net (primary spec) vs PROXY price (secondary y-axis)
4. Commercial-vs-spec divergence (descriptive)

Interactions: box-zoom + scroll-zoom + the modebar zoom in/out & reset buttons,
unified hover, and click-to-toggle series in the legend. The combined HTML is
self-contained (plotly.js embedded once) unless ``embed=False`` (CDN).
"""
from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import disclaimers
from .align import PriceResult
from .config import EXTREME_HIGH, EXTREME_LOW, ReportSpec

# zoom/pan UX: enable scroll zoom, drop the plotly logo, keep it responsive
PLOTLY_CONFIG: Dict[str, object] = {
    "scrollZoom": True,
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}

_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]
_FADE = "#888888"
_DIV = "#6a3d9a"


def _asof(date) -> str:
    return pd.Timestamp(date).strftime("%Y-%m-%d")


def build_contract_figure(
    enriched: pd.DataFrame,
    spec: ReportSpec,
    contract: str,
    price: PriceResult,
) -> go.Figure:
    sub = enriched[enriched["contract"] == contract]
    as_of = pd.Timestamp(sub["report_date"].max())

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        row_heights=[0.27, 0.25, 0.26, 0.22],
        specs=[[{}], [{}], [{"secondary_y": True}], [{}]],
        subplot_titles=(
            "Net position by category (long - short)",
            "COT Index 0-100 (trailing window) - >80/<20 = stretched, context only",
            "Net (%s) vs PROXY price" % spec.category(spec.primary_spec).label,
            "Divergence: %s minus %s net (DESCRIPTIVE, not predictive)"
            % (spec.category(spec.divergence_pair[0]).label,
               spec.category(spec.divergence_pair[1]).label),
        ),
    )

    # -- row 1: net by category ------------------------------------------------
    for i, key in enumerate(spec.net_chart_keys):
        cat = sub[sub["category"] == key].sort_values("report_date")
        if cat.empty:
            continue
        label = cat["category_label"].iloc[0]
        color = _COLORS[i % len(_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=cat["report_date"], y=cat["net"], name=label, legendgroup=label,
                mode="lines", line=dict(color=color, width=1.3),
                hovertemplate=label + ": %{y:,.0f}<extra></extra>",
            ),
            row=1, col=1,
        )
        last = cat.iloc[-1]
        fig.add_annotation(
            x=last["report_date"], y=last["net"], text="{:,.0f}".format(last["net"]),
            showarrow=False, xanchor="left", xshift=4, font=dict(size=10, color=color),
            row=1, col=1,
        )
    fig.add_hline(y=0, line=dict(color="#999", width=0.8, dash="dash"), row=1, col=1)

    # -- row 2: COT index with extreme bands -----------------------------------
    fig.add_hrect(y0=EXTREME_HIGH, y1=100, fillcolor="#d62728", opacity=0.08, line_width=0, row=2, col=1)
    fig.add_hrect(y0=0, y1=EXTREME_LOW, fillcolor="#2ca02c", opacity=0.08, line_width=0, row=2, col=1)
    for yb in (EXTREME_LOW, EXTREME_HIGH):
        fig.add_hline(y=yb, line=dict(color="#bbbbbb", width=0.7, dash="dot"), row=2, col=1)
    fig.add_hline(y=50, line=dict(color="#cccccc", width=0.6, dash="dash"), row=2, col=1)

    for key in spec.headline_keys:
        cat = sub[sub["category"] == key].sort_values("report_date")
        cat = cat[cat["cot_index"].notna()]
        if cat.empty:
            continue
        primary = key == spec.primary_spec
        label = cat["category_label"].iloc[0]
        color = _COLORS[0] if primary else _FADE
        fig.add_trace(
            go.Scatter(
                x=cat["report_date"], y=cat["cot_index"],
                name=label + " COT idx", legendgroup=label + " COT idx",
                mode="lines", line=dict(color=color, width=1.6 if primary else 1.0),
                opacity=0.95 if primary else 0.5,
                hovertemplate=label + " COT idx: %{y:.1f}<extra></extra>",
            ),
            row=2, col=1,
        )
        last = cat.iloc[-1]
        val = float(last["cot_index"])
        tag = "  " + disclaimers.EXTREME_LABEL if (val >= EXTREME_HIGH or val <= EXTREME_LOW) else ""
        fig.add_annotation(
            x=last["report_date"], y=val, text="%.0f%s" % (val, tag),
            showarrow=False, xanchor="left", xshift=4,
            font=dict(size=11 if primary else 9, color=color), row=2, col=1,
        )
    fig.update_yaxes(range=[-2, 102], row=2, col=1)

    # -- row 3: net (primary spec) vs proxy price ------------------------------
    prim = sub[sub["category"] == spec.primary_spec].sort_values("report_date")
    plabel = spec.category(spec.primary_spec).label
    fig.add_trace(
        go.Scatter(
            x=prim["report_date"], y=prim["net"], name="%s net" % plabel,
            mode="lines", line=dict(color=_COLORS[0], width=1.3), showlegend=False,
            hovertemplate="net: %{y:,.0f}<extra></extra>",
        ),
        row=3, col=1, secondary_y=False,
    )
    fig.add_hline(y=0, line=dict(color="#999", width=0.8, dash="dash"), row=3, col=1, secondary_y=False)
    if price.ok and not price.aligned.empty:
        merged = prim.merge(price.aligned, on="report_date", how="left")
        fig.add_trace(
            go.Scatter(
                x=merged["report_date"], y=merged["price"], name="price (PROXY %s)" % price.ticker,
                mode="lines", line=dict(color="#444444", width=1.0),
                hovertemplate="price: %{y:,.2f}<extra></extra>",
            ),
            row=3, col=1, secondary_y=True,
        )
        fig.update_yaxes(title_text="PROXY: %s" % price.ticker, row=3, col=1, secondary_y=True)
    fig.update_yaxes(title_text="%s net" % plabel, row=3, col=1, secondary_y=False)

    # -- row 4: divergence -----------------------------------------------------
    a, b = spec.divergence_pair
    wide = sub.pivot_table(index="report_date", columns="category", values="net").sort_index()
    if a in wide.columns and b in wide.columns:
        div = (wide[a] - wide[b]).dropna()
        fig.add_trace(
            go.Scatter(
                x=div.index, y=div.values, name="%s - %s net" % (a, b),
                mode="lines", line=dict(color=_DIV, width=1.3), fill="tozeroy",
                fillcolor="rgba(106,61,154,0.12)", showlegend=False,
                hovertemplate="divergence: %{y:,.0f}<extra></extra>",
            ),
            row=4, col=1,
        )
        fig.add_hline(y=0, line=dict(color="#999", width=0.8, dash="dash"), row=4, col=1)
        if len(div):
            fig.add_annotation(
                x=div.index[-1], y=float(div.iloc[-1]), text="{:,.0f}".format(div.iloc[-1]),
                showarrow=False, xanchor="left", xshift=4, font=dict(size=10, color=_DIV),
                row=4, col=1,
            )

    fig.update_layout(
        title=dict(
            text="<b>%s</b>   |   as-of Tuesday %s   |   %s   |   context only, not a signal"
            % (contract, _asof(as_of), spec.report_type.upper()),
            font=dict(size=14), x=0.0, xanchor="left",
        ),
        height=1200,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.06, xanchor="center", x=0.5, font=dict(size=10)),
        margin=dict(l=72, r=30, t=66, b=120),
        template="plotly_white",
    )
    # smaller subplot-title font (the 4 subplot titles are the only annotations)
    for ann in fig.layout.annotations[:4]:
        ann.font.size = 12
    fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1)
    return fig


def build_interactive_html(
    items: List[Tuple[str, pd.Timestamp, go.Figure]],
    out_html,
    meta: Dict[str, str],
    warnings: List[str],
    embed: bool = True,
) -> None:
    """Concatenate per-contract figures into one self-contained interactive page."""
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    head = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>CFTC COT Positioning (interactive)</title><style>"
        "body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;"
        "max-width:1180px;margin:22px auto;padding:0 16px;color:#222;}"
        "h1{font-size:22px;margin-bottom:2px;}"
        ".note{background:#fff8e1;border:1px solid #ffe0a3;padding:11px 13px;border-radius:8px;"
        "font-size:13px;line-height:1.45;margin:9px 0;}"
        ".guard{background:#fde8e8;border:1px solid #f5b5b5;}"
        ".meta{font-size:12px;color:#555;background:#f5f7fa;border:1px solid #e3e8ee;"
        "padding:9px 12px;border-radius:8px;}"
        ".tip{font-size:12px;color:#345;background:#eef6ff;border:1px solid #cfe2f7;"
        "padding:9px 12px;border-radius:8px;}"
        ".warn{background:#fff0f0;border:1px solid #f3c2c2;padding:9px 12px;border-radius:8px;font-size:12px;}"
        "code{background:#eef;padding:1px 4px;border-radius:4px;}"
        "</style></head><body>"
    )
    parts: List[str] = [head, "<h1>CFTC Commitments of Traders - Positioning (interactive)</h1>"]
    parts.append(
        "<div class='meta'>Generated %s &nbsp;|&nbsp; %s</div>"
        % (generated, " &nbsp;|&nbsp; ".join("%s: <code>%s</code>" % (k, v) for k, v in meta.items()))
    )
    parts.append(
        "<div class='tip'><b>Zoom:</b> drag a box to zoom, scroll to zoom, use the toolbar "
        "(top-right of each chart) for zoom in/out &amp; reset, and double-click to autoscale. "
        "The four panels share one time axis, so zooming one zooms all. Click legend items to toggle series.</div>"
    )
    parts.append("<div class='note'><b>As-of / lag:</b> %s</div>" % disclaimers.LAG_NOTE)
    parts.append("<div class='note'><b>What this is:</b> %s</div>" % disclaimers.METHODOLOGY_NOTE)
    parts.append("<div class='note guard'><b>Guardrail:</b> %s</div>" % disclaimers.GUARDRAIL_NOTE)
    if warnings:
        parts.append(
            "<div class='warn'><b>Data validation warnings (%d):</b><ul>%s</ul></div>"
            % (len(warnings), "".join("<li>%s</li>" % w for w in warnings))
        )

    first = True
    for contract, as_of, fig in items:
        parts.append("<h2>%s &mdash; as-of Tuesday %s</h2>" % (contract, _asof(as_of)))
        if first:
            include = True if embed else "cdn"  # embed plotly.js once
            first = False
        else:
            include = False
        parts.append(fig.to_html(full_html=False, include_plotlyjs=include, config=PLOTLY_CONFIG))
    parts.append("</body></html>")

    from pathlib import Path

    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    Path(out_html).write_text("".join(parts), encoding="utf-8")
