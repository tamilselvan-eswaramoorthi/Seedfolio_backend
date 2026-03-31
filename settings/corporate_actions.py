import logging
import traceback
import requests
from datetime import datetime, timedelta

from sqlmodel import select

from config import Config
from database import db_handler, StockSplit, Bonus, Demerger



logger = logging.getLogger(__name__)


def _date_range_params(years_ago: int = 20):
    today = datetime.now().date()
    past = today - timedelta(days=365 * years_ago)
    return  past.strftime("%d-%m-%Y"), today.strftime("%d-%m-%Y")

def _fetch_all_pages(event_type: str, min_date: str, max_date: str) -> list:
    """Fetch all pages from the TradeBrains corporate actions API."""
    records = []
    page = 1
    while True:
        params = {
            "event_type": event_type,
            "page": page,
            "per_page": 100,
            "min_date": min_date,
            "max_date": max_date,
        }
        resp = requests.get(Config.TRADEBRAINS_BASE_URL, params=params, timeout=20)
        if resp.status_code != 200:
            logger.error(f"TradeBrains API error {resp.status_code}: {resp.text[:200]}")
            break
        data = resp.json()
        # API shape: {"results": [...], "next": <url|null>, ...}
        results = data.get("results") or data.get("data") or (data if isinstance(data, list) else [])
        if not results:
            break
        records.extend(results)
        # stop if no next page
        if not data.get("next") and not isinstance(data, list):
            break
        if isinstance(data, list) or len(results) < 100:
            break
        page += 1
    return records


def _parse_date(value: str):
    """Parse dates returned by the TradeBrains API (DD-MM-YYYY or YYYY-MM-DD)."""
    for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except (ValueError, TypeError):
            continue
    return None


# ---------------------------------------------------------------------------
# Upsert helpers
# ---------------------------------------------------------------------------

def _upsert_splits(records: list) -> int:
    """Insert or update StockSplit rows."""
    saved = 0
    with db_handler.get_session() as session:
        for rec in records:
            # Expected API fields (adjust keys if TradeBrains response differs)
            symbol = (rec.get("nse_symbol") or rec.get("symbol") or "").strip().upper()
            eff_date = _parse_date(rec.get("ex_date") or rec.get("effective_date") or "")
            ratio_raw = rec.get("split_ratio") or rec.get("ratio") or "1:1"
            if not symbol or not eff_date:
                continue
            # ratio like "1:5" means 5 new shares per 1 old
            try:
                parts = str(ratio_raw).split(":")
                ratio = int(parts[1]) if len(parts) == 2 else int(ratio_raw)
            except (ValueError, IndexError):
                ratio = 1

            existing = session.exec(
                select(StockSplit).where(StockSplit.symbol == symbol,
                                        StockSplit.effective_date == eff_date)
            ).first()
            if existing:
                existing.ratio = ratio
                existing.last_updated = datetime.now()
                session.add(existing)
            else:
                session.add(StockSplit(
                    symbol=symbol,
                    effective_date=eff_date,
                    ratio=ratio,
                    last_updated=datetime.now(),
                ))
            saved += 1
        session.commit()
    return saved


def _upsert_bonus(records: list) -> int:
    """Insert or update Bonus rows."""
    saved = 0
    with db_handler.get_session() as session:
        for rec in records:
            symbol = (rec.get("nse_symbol") or rec.get("symbol") or "").strip().upper()
            eff_date = _parse_date(rec.get("ex_date") or rec.get("effective_date") or "")
            ratio_raw = rec.get("bonus_ratio") or rec.get("ratio") or "1:1"
            if not symbol or not eff_date:
                continue
            # ratio like "1:2" means 1 bonus share per 2 held
            try:
                parts = str(ratio_raw).split(":")
                bonus_shares = int(parts[0])
                per_shares = int(parts[1]) if len(parts) == 2 else 1
            except (ValueError, IndexError):
                bonus_shares, per_shares = 1, 1

            existing = session.exec(
                select(Bonus).where(Bonus.symbol == symbol,
                                    Bonus.effective_date == eff_date)
            ).first()
            if existing:
                existing.bonus = bonus_shares
                existing.per = per_shares
                existing.last_updated = datetime.now()
                session.add(existing)
            else:
                session.add(Bonus(
                    symbol=symbol,
                    effective_date=eff_date,
                    bonus=bonus_shares,
                    per=per_shares,
                    last_updated=datetime.now(),
                ))
            saved += 1
        session.commit()
    return saved


def _upsert_demergers(records: list) -> int:
    """Insert or update Demerger rows (one row per child)."""
    saved = 0
    with db_handler.get_session() as session:
        for rec in records:
            original_symbol = str(rec.get("nse_symbol") or rec.get("symbol") or "").strip().upper()
            bse_code = str(rec.get("bse_code") or rec.get("bse_scrip_code") or None)
            eff_date = _parse_date(rec.get("ex_date") or rec.get("effective_date") or "")
            company_name = str(rec.get("company_name") or rec.get("name") or "")
            child_symbol = str(rec.get("child_symbol") or rec.get("new_symbol") or original_symbol).strip().upper()
            child_bse = str(rec.get("child_bse_symbol") or None)
            ratio = float(rec.get("ratio") or 1.0)
            price_ratio = float(rec.get("price_ratio") or 1.0)
            keep_original = bool(rec.get("keep_original", True))
            print (f"Processing demerger: {original_symbol} → {child_symbol} on {eff_date} (ratio={ratio}, price_ratio={price_ratio})")

            if not original_symbol or not eff_date:
                continue

            existing = session.exec(
                select(Demerger).where(
                    Demerger.original_symbol == original_symbol,
                    Demerger.child_symbol == child_symbol,
                    Demerger.effective_date == eff_date,
                )
            ).first()
            if existing:
                existing.ratio = ratio
                existing.price_ratio = price_ratio
                existing.keep_original = keep_original
                existing.company_name = company_name
                existing.bse_scrip_code = bse_code
                existing.last_updated = datetime.now()
                session.add(existing)
            else:
                session.add(Demerger(
                    original_symbol=original_symbol,
                    bse_scrip_code=bse_code,
                    child_symbol=child_symbol,
                    child_bse_symbol=child_bse,
                    company_name=company_name,
                    effective_date=eff_date,
                    ratio=ratio,
                    price_ratio=price_ratio,
                    keep_original=keep_original,
                    last_updated=datetime.now(),
                ))
            saved += 1
        session.commit()
    return saved


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def sync_corporate_actions(event_type: str):
    """
    Fetch upcoming corporate actions from TradeBrains and upsert them into the DB.

    Args:
        event_type: One of "Splits", "Bonus", or "Merger/Demerger"
        days_ahead: How many days into the future to look (default 180)

    Returns:
        (response_dict, http_status_code)
    """
    logger.info(f"sync_corporate_actions triggered for event_type={event_type}")
    try:
        min_date, max_date = _date_range_params()
        records = _fetch_all_pages(event_type, min_date, max_date)

        if event_type == "Splits":
            count = _upsert_splits(records)
        elif event_type == "Bonus":
            count = _upsert_bonus(records)
        elif event_type in ("Merger/Demerger", "Demerger"):
            count = _upsert_demergers(records)
        else:
            return {"error": f"Unsupported event_type: {event_type}"}, 400

        return {
            "message": f"Successfully synced {count} '{event_type}' records "
                       f"(date range: {min_date} – {max_date}).",
            "count": count,
        }, 200

    except Exception as e:
        logger.error(traceback.format_exc())
        return {"error": f"Failed to sync corporate actions: {str(e)}"}, 500
