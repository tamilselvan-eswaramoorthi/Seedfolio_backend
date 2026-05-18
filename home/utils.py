import logging
import datetime
import pandas as pd
import yfinance as yf
from datetime import timedelta

logger = logging.getLogger(__name__)


def yahoo_ticker(symbol):
    return symbol if str(symbol).startswith("^") or "." in str(symbol) else f"{symbol}.NS"


def xirr(cashflows):
    if not cashflows or not any(v > 0 for _, v in cashflows) or not any(v < 0 for _, v in cashflows):
        return None
    start = cashflows[0][0]

    def xnpv(rate):
        return sum(v / ((1 + rate) ** ((d - start).days / 365.0)) for d, v in cashflows)

    low, high = -0.9999, 10.0
    if xnpv(low) * xnpv(high) > 0:
        return None
    for _ in range(80):
        mid = (low + high) / 2
        if xnpv(low) * xnpv(mid) <= 0:
            high = mid
        else:
            low = mid
    return round(((low + high) / 2) * 100, 2)


def fetch_history_batch(symbols, start, end, *, already_yf_tickers: bool = False, interval = "1d", get_day_data = False) -> pd.DataFrame | None:
    """
    Download OHLCV history for all symbols in a single yfinance request.
    yfinance handles internal parallelism when threads=True.

    Parameters
    ----------
    already_yf_tickers : bool
        Skip the yahoo_ticker() conversion (for index tickers like '^NSEI').
    """
    if not symbols:
        return None
    tickers = list(symbols) if already_yf_tickers else [yahoo_ticker(s) for s in symbols]
    # Deduplicate preserving order
    seen: set[str] = set()
    unique = [t for t in tickers if not (t in seen or seen.add(t))]
    try:
        hist = yf.download(
            unique,
            start=start.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
            threads=True,
            interval=interval,
        )
        if get_day_data:
            if not hist.empty:
                try:
                    if hist.index.tz is None:
                        hist.index = hist.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
                    else:
                        hist.index = hist.index.tz_convert('Asia/Kolkata')
                except Exception as e:
                    logger.warning("Timezone conversion failed: %s", e)
                hist = hist.between_time(datetime.time(9, 15), datetime.time(16, 30))
        return hist if hist is not None and not hist.empty else None
    except Exception as exc:
        logger.warning("fetch_history_batch failed: %s", exc)
        return None


def fetch_recent_prices(symbols, *, already_yf_tickers: bool = False) -> pd.DataFrame | None:
    """
    Fetch only the last 5 trading days — used for the slim/fast code path.
    This is typically 10-50x faster than fetching full history.
    """
    if not symbols:
        return None
    tickers = list(symbols) if already_yf_tickers else [yahoo_ticker(s) for s in symbols]
    seen: set[str] = set()
    unique = [t for t in tickers if not (t in seen or seen.add(t))]
    try:
        hist = yf.download(
            unique,
            period="5d",
            progress=False,
            auto_adjust=True,
            threads=True,
        )
        return hist if hist is not None and not hist.empty else None
    except Exception as exc:
        logger.warning("fetch_recent_prices failed: %s", exc)
        return None


def fetch_current_prices(symbols: list[str]) -> dict[str, float]:
    """
    Fetch the latest close prices for a batch of NSE symbols.
    Returns {symbol: price} — 0.0 for any symbol that couldn't be resolved.
    """
    if not symbols:
        return {}
    tickers = [yahoo_ticker(s) for s in symbols]
    try:
        hist = yf.download(tickers, period="2d", progress=False, auto_adjust=True, threads=True)
        if hist is None or hist.empty:
            return {s: 0.0 for s in symbols}
        close = hist["Close"] if "Close" in hist.columns else hist
        prices: dict[str, float] = {}
        for sym, ticker in zip(symbols, tickers):
            try:
                series = close[ticker] if hasattr(close, "columns") and ticker in close.columns else close
                if hasattr(series, "columns"):
                    series = series.iloc[:, 0]
                prices[sym] = float(series.dropna().iloc[-1])
            except Exception:
                prices[sym] = 0.0
        return prices
    except Exception:
        return {s: 0.0 for s in symbols}


def price_at_date(hist_df, ticker, target, fallback=None):
    try:
        close = hist_df["Close"]
        if hasattr(close, "columns"):
            series = close[ticker] if ticker in close.columns else close.iloc[:, 0]
        else:
            series = close
        sliced = series.loc[:str(target)].dropna()
        if sliced.empty:
            return fallback
        return float(sliced.iloc[-1])
    except Exception:
        return fallback


def close_series(hist_df, ticker):
    try:
        close = hist_df["Close"]
        series = close[ticker] if hasattr(close, "columns") and ticker in close.columns else close
        if hasattr(series, "columns"):
            series = series.iloc[:, 0]
        return series.dropna()
    except Exception:
        return None


def portfolio_value_series(symbol_data, hist_df):
    total = None
    for sym, data in symbol_data.items():
        series = close_series(hist_df, yahoo_ticker(sym))
        if series is None or series.empty:
            continue
        value = series * data["qty"]
        total = value if total is None else total.add(value, fill_value=0)
    return total.dropna() if total is not None else None


def portfolio_value_from_cache(symbol_data, hist_df, target) -> float:
    total = 0.0
    for sym, data in symbol_data.items():
        ticker = yahoo_ticker(sym)
        price = price_at_date(hist_df, ticker, target, fallback=data["avg_buy"])
        total += data["qty"] * price
    return total


def pct_return(current_value: float, past_value: float, round_off: int= 2) -> float:
    if past_value == 0:
        return 0.0
    return float(round((current_value - past_value) / past_value * 100, round_off))
