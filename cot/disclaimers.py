"""Honesty / guardrail text printed on every run and stamped onto every output.

Kept in one place so the CLI, the charts, and the combined HTML all say the
same thing. None of this is investment advice and none of it is a signal.
"""
from __future__ import annotations

# Shown wherever positioning is reported. The lag wording is deliberately blunt.
LAG_NOTE = (
    "Data is as-of TUESDAY (report_date_as_yyyy_mm_dd). CFTC releases each report "
    "Friday 3:30pm ET describing positions held the PRIOR Tuesday: ~3 calendar-day "
    "lag at release, and up to ~1 week stale before it can be acted on. Every row "
    "and any price overlay is indexed to the Tuesday as-of date."
)

# The "this is not smart money / not a signal" caveat.
METHODOLOGY_NOTE = (
    "COT is a positioning / situational-awareness tool with WEAK, regime-dependent "
    "timing value. The Legacy 'Commercial' bucket mixes true hedgers with swap "
    "dealers and is classified by predominant business purpose, so 'smart money' "
    "framing is approximate."
)

GUARDRAIL_NOTE = (
    "This tool does NOT emit buy/sell signals. A COT Index above 80 or below 20 is "
    "flagged as 'positioning stretched - context only', never as an entry."
)

EXTREME_LABEL = "positioning stretched - context only"

# Compact one-liner for chart footers.
CHART_FOOTER = (
    "As-of Tuesday; released Fri 3:30pm ET (~3-day lag). Positioning context only - "
    "not a signal. COT Index >80/<20 = stretched, not an entry."
)


def banner() -> str:
    """Full multi-line disclaimer block for stdout at the top of a run."""
    bar = "=" * 78
    return "\n".join(
        [
            bar,
            "CFTC COT POSITIONING TOOL - context / situational awareness, NOT signals",
            bar,
            "* " + LAG_NOTE,
            "* " + METHODOLOGY_NOTE,
            "* " + GUARDRAIL_NOTE,
            bar,
        ]
    )
