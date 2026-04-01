import logging
import traceback
from typing import Optional
import requests
from collections import defaultdict
from datetime import datetime, timedelta

from sqlmodel import select

from config import Config
from database import db_handler, StockSplit, Bonus, Demerger, Stock



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
            symbol = str(rec.get("symbol") or "").strip().upper()
            eff_date = _parse_date(rec.get("split_date"))
            ratio_raw = rec.get("split_ratio") or rec.get("ratio") or "1:1"
            if not symbol or not eff_date:
                continue
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
            symbol = str(rec.get("symbol") or "").strip().upper()
            eff_date = _parse_date(rec.get("ex_date") or "")
            ratio_raw = rec.get("bonus_ratio") or "1:1"
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
    """
    Insert or update Demerger rows.
    """

    # ── 1. Filter to demerger rows only and group them ──────────────────────
    groups: dict = defaultdict(list)
    for rec in records:
        if rec.get("types", "").lower() != "demerger":
            continue
        symbol = str(rec.get("symbol") or "").strip().upper()
        eff_date = _parse_date(str(rec.get("date") or "").strip())
        if not symbol or not eff_date:
            continue
        groups[(symbol, eff_date)].append(rec)

    if not groups:
        return 0

    saved = 0
    with db_handler.get_session() as session:

        for (original_symbol, eff_date), group_recs in groups.items():

            # ── 2. Resolve original stock → ISIN ────────────────────────────
            if original_symbol.isdigit():
                orig_stock = session.exec(
                    select(Stock).where(Stock.bse_symbol == original_symbol)
                ).first()
            else:
                orig_stock = session.exec(
                    select(Stock).where(Stock.nse_symbol == original_symbol)
                ).first()
            original_isin_code = orig_stock.isin_code if orig_stock else None

            # ── 3. Collect unique children (by instname) ────────────────────
            # Each API row has:
            #   comp_name  = the company being split / that holds the share
            #   instname   = the new entity created (child)
            seen_children: set = set()
            children: list = []          # list of instname strings, deduped
            for rec in group_recs:
                child_name = str(rec.get("instname") or "").strip()
                if child_name and child_name not in seen_children:
                    seen_children.add(child_name)
                    children.append(child_name)

            # ── 4. Map children → ISIN codes ────────────────────────────────
            def isin_for(name: str) -> Optional[str]:
                stock = session.exec(
                    select(Stock).where(Stock.name == name)
                ).first()
                return stock.isin_code if stock else None

            child_1_isin = isin_for(children[0]) if len(children) > 0 else None
            child_2_isin = isin_for(children[1]) if len(children) > 1 else None
            child_1_name = children[0] if len(children) > 0 else None
            child_2_name = children[1] if len(children) > 1 else None

            # ── 5. Parse split ratio (e.g. "1:1" → 1.0) ────────────────────
            def parse_ratio(raw) -> float:
                try:
                    parts = str(raw or "1:1").split(":")
                    if len(parts) == 2:
                        return float(parts[0]) / float(parts[1])
                    return float(parts[0])
                except (ValueError, ZeroDivisionError):
                    return 1.0

            ratio_raw = group_recs[0].get("merger_ratio", "1:1")
            split_ratio = parse_ratio(ratio_raw)

            # ── 6. Upsert ────────────────────────────────────────────────────
            existing = session.exec(
                select(Demerger).where(
                    Demerger.original_isin_code == original_isin_code,
                    Demerger.effective_date == eff_date,
                )
            ).first()

            if existing:
                existing.child_1_isin_code = child_1_isin
                existing.child_1_split_ratio = split_ratio
                existing.child_2_isin_code = child_2_isin
                existing.child_2_split_ratio = split_ratio
                existing.child_1_name = child_1_name
                existing.child_2_name = child_2_name

                existing.last_updated = datetime.now()
                session.add(existing)
            else:
                session.add(Demerger(
                    effective_date=eff_date,
                    original_symbol=original_symbol,
                    original_isin_code=original_isin_code,
                    child_1_name=child_1_name,
                    child_1_isin_code=child_1_isin,
                    child_1_split_ratio=split_ratio,
                    child_1_price_percentage=100.0,
                    child_2_name=child_2_name,
                    child_2_isin_code=child_2_isin,
                    child_2_split_ratio=split_ratio,
                    child_2_price_percentage=100.0,
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
