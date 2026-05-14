import yfinance as yf
from datetime import timedelta


def yahoo_ticker(symbol):
    return symbol if str(symbol).startswith("^") or "." in str(symbol) else f"{symbol}.NS"


def fetch_current_prices(symbols: list[str]) -> dict[str, float]:
    """
    Fetch the latest close prices for a batch of NSE symbols in ONE HTTP call.
    Returns {symbol: price} — 0.0 for any symbol that couldn't be resolved.
    """
    if not symbols:
        return {}
    tickers = [yahoo_ticker(s) for s in symbols]
    try:
        # Single batched download — much faster than one Ticker() call per symbol
        hist = yf.download(
            tickers,
            period="2d",          # last 2 trading sessions is enough
            progress=False,
            auto_adjust=True,
            threads=True,         # yfinance parallelises internally
        )
        if hist is None:
            return {s: 0.0 for s in symbols}
        
        if hist.empty:
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


def fetch_history_batch(symbols, start, end):
    """
    Download OHLCV history for all symbols in ONE request covering [start, end].
    Returns a pandas DataFrame with a MultiIndex or single-level Close column.
    """
    if not symbols:
        return None
    tickers = [yahoo_ticker(s) for s in symbols]
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


def price_at_date(hist_df, ticker, target, fallback=None):
    """
    Extract the closing price for `ticker` on or just before `target` from
    the already-downloaded `hist_df`. Falls back to `fallback` if not found.
    """
    try:
        close = hist_df["Close"]
        if hasattr(close, "columns"):
            series = close[ticker] if ticker in close.columns else close.iloc[:, 0]
        else:
            series = close
        target_ts = str(target)
        sliced = series.loc[:target_ts].dropna()
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
    """
    Compute portfolio value at `target` using the pre-fetched `hist_df`.
    No extra HTTP calls — just slice the cached DataFrame.
    """
    total = 0.0
    for sym, data in symbol_data.items():
        ticker = yahoo_ticker(sym)
        price = price_at_date(hist_df, ticker, target, fallback=data["avg_buy"])
        total += data["qty"] * price
    return total


def pct_return(current_value: float, past_value: float) -> float:
    if past_value == 0:
        return 0.0
    return float(round((current_value - past_value) / past_value * 100, 2))
