import logging
from datetime import datetime, timedelta, date

import yfinance as yf
from sqlmodel import select

from database import db_handler, Holdings, Transaction
from warnings import filterwarnings

filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# Period lookback definitions (days)
PERIOD_DAYS = {
    "today":     1,
    "yesterday": 2,
    "1w":        7,
    "1m":        30,
    "3m":        90,
    "6m":        180,
    "1y":        365,
    "5y":        365 * 5,
}


def _fetch_current_prices(symbols: list[str]) -> dict[str, float]:
    """
    Fetch the latest close prices for a batch of NSE symbols in ONE HTTP call.
    Returns {symbol: price} — 0.0 for any symbol that couldn't be resolved.
    """
    if not symbols:
        return {}
    tickers = [f"{s}.NS" for s in symbols]
    try:
        # Single batched download — much faster than one Ticker() call per symbol
        hist = yf.download(
            tickers,
            period="2d",          # last 2 trading sessions is enough
            progress=False,
            auto_adjust=True,
            threads=True,         # yfinance parallelises internally
        )
        if hist.empty:
            return {s: 0.0 for s in symbols}

        close = hist["Close"] if "Close" in hist.columns else hist

        prices: dict[str, float] = {}
        for sym, ticker in zip(symbols, tickers):
            try:
                # Take the last available row for this ticker
                col = ticker if ticker in close.columns else close.columns[0]
                prices[sym] = float(close[col].dropna().iloc[-1])
            except Exception:
                prices[sym] = 0.0
        return prices
    except Exception:
        return {s: 0.0 for s in symbols}


def _fetch_history_batch(symbols: list[str], start: date, end: date):
    """
    Download OHLCV history for all symbols in ONE request covering [start, end].
    Returns a pandas DataFrame with a MultiIndex or single-level Close column.
    """
    if not symbols:
        return None
    tickers = [f"{s}.NS" for s in symbols]
    try:
        hist = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
            threads=True,
        )
        return hist
    except Exception:
        return None


def _price_at_date(hist_df, ticker: str, target: date, fallback: float) -> float:
    """
    Extract the closing price for `ticker` on or just before `target` from
    the already-downloaded `hist_df`. Falls back to `fallback` if not found.
    """
    try:
        close = hist_df["Close"]
        # Multi-ticker download → columns are ticker strings
        col = ticker if ticker in close.columns else None
        series = close[col] if col else close
        # Slice up to and including target date
        target_ts = str(target)
        sliced = series.loc[:target_ts].dropna()
        if sliced.empty:
            return fallback
        return float(sliced.iloc[-1])
    except Exception:
        return fallback


def _portfolio_value_from_cache(
    symbol_data: dict, hist_df, target: date
) -> float:
    """
    Compute portfolio value at `target` using the pre-fetched `hist_df`.
    No extra HTTP calls — just slice the cached DataFrame.
    """
    total = 0.0
    for sym, data in symbol_data.items():
        ticker = f"{sym}.NS"
        price = _price_at_date(hist_df, ticker, target, fallback=data["avg_buy"])
        total += data["qty"] * price
    return total


def _pct_return(current_value: float, past_value: float) -> float:
    if past_value == 0:
        return 0.0
    return float(round((current_value - past_value) / past_value * 100, 2))


def gain_returns(user_id: str):
    """
    Calculate gain returns for the user based on holdings and transactions.

    Returns:
        ({
            "gain_returns": {
                "unrealized_gain": float,   # current value - cost basis (active holdings)
                "realized_gain":  float,    # cumulative realized P&L across all transactions
                "today":     float,         # % return vs close 1 trading day ago
                "yesterday": float,         # % return vs close 2 trading days ago
                "1w":  float,               # % return vs 7 days ago
                "1m":  float,
                "3m":  float,
                "6m":  float,
                "1y":  float,
                "ytd": float,
                "5y":  float,
                "max": float,               # % return from earliest transaction
            }
        }, http_status_code)
    """
    try:
        with db_handler.get_session() as session:
            holdings: list[Holdings] = session.exec(
                select(Holdings).where(Holdings.user_id == user_id)
            ).all()
            transactions: list[Transaction] = session.exec(
                select(Transaction).where(Transaction.user_id == user_id)
            ).all()

        if not holdings and not transactions:
            return {"message": "No holdings or transactions found for user."}, 404

        # ── 1. Realized gain: sum of realized_pl across all transactions ────────
        realized_gain = sum(t.realized_pl for t in transactions)

        # ── 2. Active holdings (qty > 0) ────────────────────────────────────────
        active_holdings = [h for h in holdings if h.quantity > 0]

        # ── 3. Build per-symbol aggregates ──────────────────────────────────────
        # {symbol: {"qty": int, "avg_buy": float}}
        symbol_data: dict = {}
        for h in active_holdings:
            sym = h.stock_symbol
            if sym not in symbol_data:
                symbol_data[sym] = {"qty": h.quantity, "avg_buy": h.avg_buy}
            else:
                # Shouldn't happen (one row per symbol), but handle gracefully
                old = symbol_data[sym]
                total_qty = old["qty"] + h.quantity
                if total_qty > 0:
                    blended_avg = (old["qty"] * old["avg_buy"] + h.quantity * h.avg_buy) / total_qty
                else:
                    blended_avg = 0.0
                symbol_data[sym] = {"qty": total_qty, "avg_buy": blended_avg}

        # ── 4. Fetch current prices in one batch call ───────────────────────────
        current_prices = _fetch_current_prices(list(symbol_data.keys()))

        cost_basis      = sum(symbol_data[s]["qty"] * symbol_data[s]["avg_buy"] for s in symbol_data)
        current_value   = sum(symbol_data[s]["qty"] * current_prices[s]         for s in symbol_data)
        unrealized_gain = current_value - cost_basis

        # ── 5. Period returns — ONE batch history download ─────────────────────
        today = datetime.now().date()

        # Collect all target dates we need
        target_dates: dict[str, date] = {
            period: today - timedelta(days=days)
            for period, days in PERIOD_DAYS.items()
        }
        target_dates["ytd"] = date(today.year, 1, 1)

        earliest_date = None
        if transactions:
            earliest_dt = min(t.transaction_datetime for t in transactions)
            earliest_date = earliest_dt.date() if isinstance(earliest_dt, datetime) else earliest_dt
            target_dates["max"] = earliest_date

        # Single download covering from the furthest-back date to today
        history_start = min(target_dates.values())
        hist_df = _fetch_history_batch(list(symbol_data.keys()), history_start, today)

        period_returns: dict = {}
        if hist_df is not None and not hist_df.empty:
            for period, target in target_dates.items():
                past_val = _portfolio_value_from_cache(symbol_data, hist_df, target)
                period_returns[period] = _pct_return(current_value, past_val)
        else:
            # Fallback: all returns = 0 if history unavailable
            period_returns = {p: 0.0 for p in list(PERIOD_DAYS.keys()) + ["ytd", "max"]}

        if earliest_date is None:
            period_returns["max"] = 0.0

        return {
            "gain_returns": {
                "unrealized_gain": round(unrealized_gain, 2),
                "realized_gain":   round(realized_gain, 2),
                **period_returns,
            }
        }, 200

    except Exception as e:
        logger.exception("gain_returns failed for user_id=%s", user_id)
        return {"message": str(e)}, 500
