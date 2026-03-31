from datetime import date, datetime
from typing import Any, Dict, List, Optional

from sqlmodel import select

# ---------------------------------------------------------------------------
# STATIC FALLBACK REGISTRIES  (used when DB has no matching record)
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

SPLIT_REGISTRY: List[Dict[str, Any]] = [
    {"effective_date": date(2024, 10, 28), "symbol": "DRREDDY", "ratio": 5},
]

BONUS_REGISTRY: List[Dict[str, Any]] = [
    {"effective_date": date(2025, 8, 26), "symbol": "KARURVYSYA", "bonus": 1, "per": 5},
    {"effective_date": date(2025, 7, 18), "symbol": "MOTHERSON", "bonus": 1, "per": 2},
]


# ---------------------------------------------------------------------------
# DB-backed helpers
# ---------------------------------------------------------------------------

def _rows_to_demerger_dict(rows) -> Optional[Dict[str, Any]]:
    """Convert a list of Demerger DB rows (same original symbol) into the
    legacy dict shape expected by callers."""
    if not rows:
        return None
    first = rows[0]
    return {
        "effective_date": first.effective_date,
        "raw_symbol": first.original_symbol,
        "bse_scrip_code": first.bse_scrip_code,
        "children": [
            {
                "symbol": r.child_symbol,
                "bse_symbol": r.child_bse_symbol,
                "company_name": r.company_name,
                "ratio": r.ratio,
                "price_ratio": r.price_ratio,
                "keep_original": r.keep_original,
            }
            for r in rows
        ],
    }


def _get_demerger_from_db(*, raw_symbol: Optional[str] = None, bse_scrip_code: Optional[str] = None,
                           transaction_date: Optional[date] = None) -> Optional[Dict[str, Any]]:
    try:
        from database import db_handler
        from database.models import Demerger
        with db_handler.get_session() as session:
            stmt = select(Demerger)
            if raw_symbol:
                stmt = stmt.where(Demerger.original_symbol == raw_symbol.upper())
            elif bse_scrip_code:
                stmt = stmt.where(Demerger.bse_scrip_code == str(bse_scrip_code).strip())
            else:
                return None
            rows = session.exec(stmt).all()
            if transaction_date:
                rows = [r for r in rows if transaction_date <= r.effective_date]
            return _rows_to_demerger_dict(rows)
    except Exception:
        return None


def _get_split_from_db(symbol: str, transaction_date: date) -> Optional[Dict[str, Any]]:
    try:
        from database import db_handler
        from database.models import StockSplit
        base = symbol.upper().split(".")[0]
        with db_handler.get_session() as session:
            stmt = select(StockSplit).where(StockSplit.symbol == base)
            rows = session.exec(stmt).all()
            for r in rows:
                if transaction_date <= r.effective_date:
                    return {"effective_date": r.effective_date, "symbol": r.symbol, "ratio": r.ratio}
    except Exception:
        pass
    return None


def _get_bonus_from_db(symbol: str, transaction_date: date) -> Optional[Dict[str, Any]]:
    try:
        from database import db_handler
        from database.models import Bonus
        base = symbol.upper().split(".")[0]
        with db_handler.get_session() as session:
            stmt = select(Bonus).where(Bonus.symbol == base)
            rows = session.exec(stmt).all()
            for r in rows:
                if transaction_date <= r.effective_date:
                    return {"effective_date": r.effective_date, "symbol": r.symbol,
                            "bonus": r.bonus, "per": r.per}
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Public helpers (DB-first, static fallback)
# ---------------------------------------------------------------------------

def get_demerger_by_raw_symbol(raw_symbol: str, transaction_date: date) -> Optional[Dict[str, Any]]:
    if isinstance(transaction_date, str):
        transaction_date = datetime.strptime(transaction_date, "%Y-%m-%d").date()
    result = _get_demerger_from_db(raw_symbol=raw_symbol, transaction_date=transaction_date)
    if result:
        return result


def get_demerger_by_bse_code(scrip_code: str, transaction_date: date) -> Optional[Dict[str, Any]]:
    if isinstance(transaction_date, str):
        transaction_date = datetime.strptime(transaction_date, "%Y-%m-%d").date()
    result = _get_demerger_from_db(bse_scrip_code=scrip_code, transaction_date=transaction_date)
    if result:
        return result


def get_split(symbol: str, transaction_date: date) -> Optional[Dict[str, Any]]:
    if isinstance(transaction_date, str):
        transaction_date = datetime.strptime(transaction_date, "%Y-%m-%d").date()
    result = _get_split_from_db(symbol, transaction_date)
    if result:
        return result


def get_bonus(symbol: str, transaction_date: date) -> Optional[Dict[str, Any]]:
    if isinstance(transaction_date, str):
        transaction_date = datetime.strptime(transaction_date, "%Y-%m-%d").date()
    result = _get_bonus_from_db(symbol, transaction_date)
    if result:
        return result
