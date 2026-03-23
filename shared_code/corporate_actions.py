from datetime import date
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# DEMERGER REGISTRY
# ---------------------------------------------------------------------------

from datetime import date
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# DEMERGER_REGISTRY
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
                "company_name": "Tata Motors Ltd",
                "ratio": 1.0,
                "price_ratio": 0.3115,      # 31.15 % of cost allocated here
                "keep_original": False,
            },
        ],
    },
    {
        "effective_date": date(2025, 1, 1),
        "original": "ITC.NS",
        "raw_symbol": "ITC",
        "bse_scrip_code": "500875",
        "children": [
            {
                "symbol": "ITCHOTELS.NS",
                "company_name": "ITC Hotels Ltd",
                "ratio": 0.1,               # 1 share for every 10 held
                "price_ratio": 0.8649,
                "keep_original": True,
            },
            {
                "symbol": "ITC.NS",
                "company_name": "ITC Ltd",
                "ratio": 1,
                "price_ratio": 0.1351,
                "keep_original": True,
            },
        ],
    },
]

# ---------------------------------------------------------------------------
# STOCK SPLIT REGISTRY
# ---------------------------------------------------------------------------
SPLIT_REGISTRY: List[Dict[str, Any]] = [
    {"effective_date": date(2024, 10, 28), "symbol": "DRREDDY.NS", "ratio": 5},
]

# ---------------------------------------------------------------------------
# BONUS REGISTRY
# ---------------------------------------------------------------------------
BONUS_REGISTRY: List[Dict[str, Any]] = [
    {"effective_date": date(2025, 8, 26), "symbol": "KARURVYSYA.NS", "bonus": 1, "per": 5},
    {"effective_date": date(2025, 7, 18), "symbol": "MOTHERSON.NS", "bonus": 1, "per": 2},
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
