"""
corporate_actions.py
--------------------
Handles corporate actions that change symbol-to-holding relationships:
  - Demergers  (one symbol → multiple new symbols)
  - Stock splits (1 share → N shares at 1/N price)
  - Bonus issues (N additional shares credited per M held)

Add new events to DEMERGER_REGISTRY / SPLIT_REGISTRY as they happen.
"""

from datetime import date
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# DEMERGER REGISTRY
# ---------------------------------------------------------------------------

DEMERGER_REGISTRY: List[Dict[str, Any]] = [

    {
        "effective_date": date(2025, 1, 2),
        "original": "TATAMOTORS.NS",
        "raw_symbol": "TATAMOTORS",
        "bse_scrip_code": "500570",
        "children": [
            {
                "symbol": "TMPV.NS",
                "bse_symbol": "TMPV.BO",
                "company_name": "Tata Motors Passenger Vehicles",
                "ratio": 1.0,
                "price_ratio": 0.6885,      # 68.85 % of cost allocated here
                "keep_original": True,
            },
            {
                "symbol": "TMCV.NS",
                "bse_symbol": "TMCV.BO",
                "company_name": "Tata Motors Commercial Vehicles",
                "ratio": 1.0,
                "price_ratio": 0.3115,      # 31.15 % of cost allocated here
                "keep_original": False,
            },
        ],
    },
]

# ---------------------------------------------------------------------------
# STOCK SPLIT REGISTRY
# ---------------------------------------------------------------------------
# ratio > 1 means a split  (e.g. 1 share → 2 shares, ratio = 2)
# ratio < 1 means a reverse-split (e.g. 10 shares → 1 share, ratio = 0.1)
SPLIT_REGISTRY: List[Dict[str, Any]] = [
    # Example:
    # {"effective_date": date(2024, 3, 15), "symbol": "INFY.NS", "ratio": 2},
]

# ---------------------------------------------------------------------------
# BONUS REGISTRY
# ---------------------------------------------------------------------------
# For every "per" shares held, "bonus" additional shares are credited.
BONUS_REGISTRY: List[Dict[str, Any]] = [
    # Example: 1 bonus share for every 2 held
    # {"effective_date": date(2024, 6, 1), "symbol": "HDFCBANK.NS", "bonus": 1, "per": 2},
]


# ---------------------------------------------------------------------------
# Helper: look up the demerger record that applies to a given symbol / date
# ---------------------------------------------------------------------------

def get_demerger(symbol: str, transaction_date: date) -> Optional[Dict[str, Any]]:
    """
    Return the demerger record if *symbol* was involved in a demerger
    that became effective on or before *transaction_date*.
    Matches against the resolved 'original' field (e.g. "TATAMOTORS.NS").
    Returns None if no matching demerger found.
    """
    for record in DEMERGER_REGISTRY:
        if record["original"].upper() == symbol.upper():
            if transaction_date >= record["effective_date"]:
                return record
    return None


def get_demerger_by_raw_symbol(raw_symbol: str, transaction_date: date) -> Optional[Dict[str, Any]]:
    """
    Return the demerger record matching the *raw_symbol* exactly as it appears
    in the exchange PDF (e.g. "TATAMOTORS" without the .NS suffix), on or
    before *transaction_date*.
    Returns None if no matching demerger found.
    """
    for record in DEMERGER_REGISTRY:
        if record.get("raw_symbol", "").upper() == raw_symbol.upper():
            if transaction_date >= record["effective_date"]:
                return record
    return None


def get_demerger_by_bse_code(scrip_code: str, transaction_date: date) -> Optional[Dict[str, Any]]:
    """
    Return the demerger record matching the BSE *scrip_code* (e.g. "500570"
    for Tata Motors), on or before *transaction_date*.
    Returns None if no matching demerger found.
    """
    for record in DEMERGER_REGISTRY:
        if record.get("bse_scrip_code", "") == str(scrip_code).strip():
            if transaction_date >= record["effective_date"]:
                return record
    return None


def get_split(symbol: str, transaction_date: date) -> Optional[Dict[str, Any]]:
    """Return the split record applicable to *symbol* on *transaction_date*."""
    for record in SPLIT_REGISTRY:
        if record["symbol"].upper() == symbol.upper():
            if transaction_date >= record["effective_date"]:
                return record
    return None


def get_bonus(symbol: str, transaction_date: date) -> Optional[Dict[str, Any]]:
    """Return the bonus record applicable to *symbol* on *transaction_date*."""
    for record in BONUS_REGISTRY:
        if record["symbol"].upper() == symbol.upper():
            if transaction_date >= record["effective_date"]:
                return record
    return None
