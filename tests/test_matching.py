"""Contract-name matching against official contract_market_name values."""
from cot.fetch import match_contract

# A realistic slice of the live legacy contract list.
AVAILABLE = [
    "E-MINI S&P 500",
    "MICRO E-MINI S&P 500 INDEX",
    "S&P 500 STOCK INDEX",
    "CRUDE OIL, LIGHT SWEET-WTI",
    "WTI CRUDE OIL 1ST LINE",
    "WTI CRUDE OIL FINANCIAL",
    "E-MINI CRUDE OIL, LIGHT SWEET",
]


def test_exact_match():
    m = match_contract("E-MINI S&P 500", AVAILABLE)
    assert m.how == "exact"
    assert m.resolved == "E-MINI S&P 500"


def test_case_insensitive_exact():
    m = match_contract("e-mini s&p 500", AVAILABLE)
    assert m.how == "exact"
    assert m.resolved == "E-MINI S&P 500"


def test_token_match_recovers_micro_suffix():
    # user omits the trailing " INDEX"
    m = match_contract("MICRO E-MINI S&P 500", AVAILABLE)
    assert m.how == "tokens"
    assert m.resolved == "MICRO E-MINI S&P 500 INDEX"


def test_ambiguous_picks_shortest_and_lists_candidates():
    # "CRUDE OIL WTI" tokens appear in several names -> ambiguous
    m = match_contract("CRUDE OIL WTI", AVAILABLE)
    assert m.how == "ambiguous"
    assert len(m.candidates) > 1
    # shortest candidate chosen deterministically
    assert m.resolved == min(m.candidates, key=len)


def test_no_match():
    m = match_contract("SOYBEAN OIL", AVAILABLE)
    assert m.how == "none"
    assert m.resolved is None
