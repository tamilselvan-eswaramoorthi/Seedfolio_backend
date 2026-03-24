from datetime import date, datetime
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# DEMERGER_REGISTRY
# ---------------------------------------------------------------------------
DEMERGER_REGISTRY: List[Dict[str, Any]] = [
    {
        "effective_date": date(2025, 10, 1),
        "original": "TATAMOTORS.NS",
        "raw_symbol": "TATAMOTORS",
        "bse_scrip_code": "500570",
        "children": [
            {
                "symbol": "TMPV",
                "bse_symbol": "TMPV.BO",
                "company_name": "Tata Motors Passenger Vehicles",
                "ratio": 1.0,
                "price_ratio": 0.6885,      # 68.85 % of cost allocated here
                "keep_original": True,
            },
            {
                "symbol": "TMCV",
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
    {"effective_date": date(2024, 10, 28), "symbol": "DRREDDY", "ratio": 5},
]

# ---------------------------------------------------------------------------
# BONUS REGISTRY
# ---------------------------------------------------------------------------
BONUS_REGISTRY: List[Dict[str, Any]] = [
    {"effective_date": date(2025, 8, 26), "symbol": "KARURVYSYA", "bonus": 1, "per": 5},
    {"effective_date": date(2025, 7, 18), "symbol": "MOTHERSON", "bonus": 1, "per": 2},
]

def get_demerger_by_raw_symbol(raw_symbol: str, transaction_date: date) -> Optional[Dict[str, Any]]:
    for record in DEMERGER_REGISTRY:
        if record.get("raw_symbol", "").upper() == raw_symbol.upper():
            if isinstance(transaction_date, str):
                transaction_date = datetime.strptime(transaction_date, "%Y-%m-%d").date()
            if transaction_date <= record["effective_date"]:    
                return record
    return None


def get_demerger_by_bse_code(scrip_code: str, transaction_date: date) -> Optional[Dict[str, Any]]:
    for record in DEMERGER_REGISTRY:
        if record.get("bse_scrip_code", "") == str(scrip_code).strip():
            if transaction_date <= record["effective_date"]:
                return record
    return None


def get_split(symbol: str, transaction_date: date) -> Optional[Dict[str, Any]]:
    for record in SPLIT_REGISTRY:
        if record["symbol"].upper() == symbol.upper().split(".")[0]:
            if isinstance(transaction_date, str):
                transaction_date = datetime.strptime(transaction_date, "%Y-%m-%d").date()
            if transaction_date <= record["effective_date"]:
                return record
    return None


def get_bonus(symbol: str, transaction_date: date) -> Optional[Dict[str, Any]]:
    for record in BONUS_REGISTRY:
        if record["symbol"].upper() == symbol.upper().split(".")[0]:
            if isinstance(transaction_date, str):
                transaction_date = datetime.strptime(transaction_date, "%Y-%m-%d").date()
            if transaction_date <= record["effective_date"]:
                return record
    return None
