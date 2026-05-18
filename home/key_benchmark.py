import logging
from math import e
import numpy as np
from datetime import date, datetime, timedelta

from sqlmodel import select
import pandas as pd
from database import Holdings, Transaction, db_handler
from .cache import get_or_create
from .constants import BENCHMARKS, PERIOD_DAYS
from .utils import (
    close_series,
    fetch_history_batch,
    pct_return,
    portfolio_value_from_cache,
    portfolio_value_series,
    fetch_recent_prices,
    price_at_date,
    yahoo_ticker,
    xirr,
)    

logger = logging.getLogger(__name__)


def get_portfolio_performance(user_id: str) -> "PortfolioPerformance":
    """
    Return a cached PortfolioPerformance for today.
    The yfinance download happens once per user per calendar day and is
    reused by all endpoints (/performance, /risk_and_benchmarks, /get_holdings, etc.).
    """
    return get_or_create("portfolio", user_id, lambda: PortfolioPerformance(user_id))

def get_market_data() -> dict:
    """
    Fetch and cache Nifty 50 & Sensex current levels + 30-day sparklines.
    Result is shared across all users and refreshed once per calendar day.
    """
    return get_or_create("market", "indices", _build_market_data)

def _build_market_data() -> dict:
    tickers = {
        "nifty_50":   {"ticker": "^NSEI",  "name": "Nifty 50"},
        "sensex":     {"ticker": "^BSESN", "name": "S&P BSE Sensex"},
        "nifty_bank": {"ticker": "^NSEBANK", "name": "Nifty Bank"},
        "nifty_100":  {"ticker": "^CNX100",  "name": "Nifty 100"},
    }
    
    # Fetch today's intraday data (e.g., 5-minute intervals for the last 1 day)
    # This guarantees a rich dataset to build today's sparkline.
    raw = fetch_history_batch(
        [v["ticker"] for v in tickers.values()],
        start=date.today(),
        end=date.today(),
        already_yf_tickers=True,
        interval="30m",
        get_day_data = True
    )

    # Fetch recent daily prices to get the previous close
    daily_raw = fetch_recent_prices(
        [v["ticker"] for v in tickers.values()],
        already_yf_tickers=True
    )

    result = {}
    for key, meta in tickers.items():
        series = close_series(raw, meta["ticker"]) if raw is not None and not raw.empty else None
        if series is None or series.empty:
            result[key] = {"name": meta["name"], "ticker": meta["ticker"], "error": "no data"}
            continue
            
        current = float(series.iloc[-1])
        open_price = float(series.iloc[0]) 

        # Determine previous close and today's official close
        prev_close = open_price # fallback
        today_close = current # fallback to intraday last tick
        if daily_raw is not None and not daily_raw.empty:
            daily_series = close_series(daily_raw, meta["ticker"])
            if daily_series is not None and not daily_series.empty:
                # filter to dates strictly before today
                today_ts = pd.Timestamp(date.today())
                if daily_series.index.tz:
                    today_ts = today_ts.tz_localize(daily_series.index.tz)
                prev_series = daily_series[daily_series.index < today_ts]
                if not prev_series.empty:
                    prev_close = float(prev_series.iloc[-1])
                
                # Check if today's daily close is available
                today_series = daily_series[daily_series.index >= today_ts]
                if not today_series.empty:
                    today_close = float(today_series.iloc[-1])
                    # If market is closed, today_close from daily data is the official close.
                    # We can update `current` to be this official close if preferred, 
                    # but we'll include it as a separate field.
        
        result[key] = {
            "name":          meta["name"],
            "ticker":        meta["ticker"],
            "current":       round(current, 2),
            "today_close":   round(today_close, 2),
            "previous_close": round(prev_close, 2),
            "change":        round(today_close - prev_close, 3),
            "change_pct":    pct_return(today_close, prev_close),
            # Note: YTD change requires historical daily data. 
            # If you only fetch '1d', omit YTD or fetch it separately.
            "sparkline":     [round(float(v), 2) for v in series.values],
            # Timestamps will now include hours/minutes for intraday tracking
            "sparkline_dates": [str(d.strftime("%H:%M")) if hasattr(d, "strftime") else str(d) for d in series.index],
        }
    return result

class PortfolioPerformance:
    MAX_HISTORY_DAYS = 365 * 5

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.today = datetime.now().date()
        self.holdings, self.transactions = self._load_data()
        self.symbol_data = self._symbol_data()
        self.earliest_date = self._earliest_transaction_date()
        self.target_dates = self._target_dates()
        self.applicable = {p: self.earliest_date is None or d >= self.earliest_date for p, d in self.target_dates.items()}
        self.history = self._fetch_history()
        self.portfolio_series = self._portfolio_series()
        self.symbol_first_dates = self._symbol_first_dates()

    def _load_data(self):
        with db_handler.get_session() as session:
            holdings = session.exec(select(Holdings).where(Holdings.user_id == self.user_id)).all()
            transactions = session.exec(select(Transaction).where(Transaction.user_id == self.user_id)).all()
        return holdings, sorted(transactions, key=lambda t: t.transaction_datetime)

    def _symbol_data(self):
        data = {}
        for h in [h for h in self.holdings if h.quantity > 0]:
            old = data.get(h.stock_symbol, {"qty": 0, "avg_buy": 0.0})
            qty = old["qty"] + h.quantity
            avg = ((old["qty"] * old["avg_buy"]) + (h.quantity * h.avg_buy)) / qty if qty > 0 else 0.0
            data[h.stock_symbol] = {"qty": qty, "avg_buy": avg, "company_name": h.company_name}
        return data

    def _earliest_transaction_date(self):
        if not self.transactions:
            return None
        earliest = min(t.transaction_datetime for t in self.transactions)
        return earliest.date() if isinstance(earliest, datetime) else earliest

    def _symbol_first_dates(self):
        dates = {}
        for t in self.transactions:
            trade_date = t.transaction_datetime.date() if isinstance(t.transaction_datetime, datetime) else t.transaction_datetime
            symbol = t.stock_symbol.split(".")[0]
            dates[symbol] = min(dates.get(symbol, trade_date), trade_date)
        return dates

    def _target_dates(self):
        targets = {p: self.today - timedelta(days=d) for p, d in PERIOD_DAYS.items()}
        targets["ytd"] = date(self.today.year, 1, 1)
        if self.earliest_date:
            targets["max"] = self.earliest_date
        return targets

    def _fetch_history(self):
        """Download full history (up to 5 years) for all portfolio symbols + benchmarks."""
        portfolio_syms = list(self.symbol_data.keys())
        if not portfolio_syms:
            return None
        dates = [d for p, d in self.target_dates.items() if self.applicable[p]]
        if self.earliest_date:
            dates.append(self.earliest_date)
        raw_start = min(dates) if dates else self.today
        capped_start = max(raw_start, self.today - timedelta(days=self.MAX_HISTORY_DAYS))
        all_symbols = portfolio_syms + [b["ticker"] for b in BENCHMARKS.values()]
        return fetch_history_batch(all_symbols, capped_start, self.today)

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    def current_value(self):
        if self.history is None or self.history.empty:
            return sum(d["qty"] * d["avg_buy"] for d in self.symbol_data.values())
        return portfolio_value_from_cache(self.symbol_data, self.history, self.today)

    def _portfolio_series(self):
        if self.history is None or self.history.empty:
            return None
        series = portfolio_value_series(self.symbol_data, self.history)
        if series is None or series.empty or self.earliest_date is None:
            return series
        return series.loc[str(self.earliest_date):]

    def period_returns(self, current_value):
        periods = {}
        if self.history is None or self.history.empty:
            return {p: 0.0 if self.applicable.get(p, True) else None for p in list(PERIOD_DAYS.keys()) + ["ytd", "max"]}
        for period, target in self.target_dates.items():
            if not self.applicable[period]:
                periods[period] = None
                continue
            past = portfolio_value_from_cache(self.symbol_data, self.history, target)
            periods[period] = [round(current_value - past, 2), pct_return(current_value, past)]
        if self.earliest_date is None:
            periods["max"] = 0.0
        return periods

    def cashflows(self, terminal_value):
        flows = []
        for t in self.transactions:
            trade_date = t.transaction_datetime.date() if isinstance(t.transaction_datetime, datetime) else t.transaction_datetime
            amount = float(t.quantity) * float(t.price)
            if t.transaction_type.upper() == "BUY":
                flows.append((trade_date, -amount))
            elif t.transaction_type.upper() == "SELL":
                flows.append((trade_date, amount))
        flows.append((self.today, terminal_value))
        return flows

    def benchmark_returns(self):
        results = {}
        if self.history is None or self.history.empty:
            return results
        for key, benchmark in BENCHMARKS.items():
            ticker, units, invested, withdrawn, flows = benchmark["ticker"], 0.0, 0.0, 0.0, []
            for t in self.transactions:
                trade_date = t.transaction_datetime.date() if isinstance(t.transaction_datetime, datetime) else t.transaction_datetime
                price = price_at_date(self.history, ticker, trade_date)
                if not price:
                    continue
                amount = float(t.quantity) * float(t.price)
                if t.transaction_type.upper() == "BUY":
                    units += amount / price
                    invested += amount
                    flows.append((trade_date, -amount))
                elif t.transaction_type.upper() == "SELL":
                    units -= amount / price
                    withdrawn += amount
                    flows.append((trade_date, amount))
            current_price = price_at_date(self.history, ticker, self.today, 0.0)
            if current_price == 0.0 or current_price is None:
                continue
            if units == 0 and invested == 0 and withdrawn == 0:
                continue
            value = int(units) * float(current_price)
            gain = value + withdrawn - invested
            flows.append((self.today, value))
            results[key] = {
                "name": benchmark["name"],
                "ticker": ticker,
                "current_value": round(value, 2),
                "total_gain": round(gain, 2),
                "return_pct": pct_return(value + withdrawn, invested),
                "xirr": xirr(flows),
            }
        return results

    def risk_metrics(self):
        s = self.portfolio_series
        if s is None or s.empty:
            return {}
        last_52w = s.loc[str(self.today - timedelta(days=365)):]
        rolling_max = s.cummax()
        drawdown = (s / rolling_max - 1) * 100
        daily_returns = s.pct_change().dropna()
        benchmark = close_series(self.history, "^NSEI")
        benchmark_returns = benchmark.pct_change().dropna() if benchmark is not None else None
        aligned = daily_returns.align(benchmark_returns, join="inner") if benchmark_returns is not None else (daily_returns, None)
        beta = None
        if aligned[1] is not None and len(aligned[0]) > 1 and aligned[1].var() != 0:
            beta = round(aligned[0].cov(aligned[1]) / aligned[1].var(), 2)
        return {
            "all_time_high": round(float(s.max()), 2),
            "fifty_two_week_high": round(float((last_52w if not last_52w.empty else s).max()), 2),
            "max_drawdown": round(float(drawdown.min()), 2),
            "max_drawdown_duration_days": self._max_drawdown_duration_days(s),
            "sharpe_ratio": round((daily_returns.mean() / daily_returns.std()) * (252 ** 0.5), 2) if len(daily_returns) > 1 and daily_returns.std() else None,
            "beta": beta,
        }

    def _max_drawdown_duration_days(self, series):
        max_duration, peak_date = 0, None
        peak = float("-inf")
        for idx, value in series.items():
            if value >= peak:
                peak, peak_date = value, idx
            elif peak_date is not None:
                max_duration = max(max_duration, (idx - peak_date).days)
        return max_duration

    def _cagr_and_value(self):
        """
        Returns per-holding CAGR alongside overall portfolio CAGR and value.

        CAGR formula: (current_value / cost_basis) ^ (1 / years) - 1
        where years = days_held / 365.
        """
        current_value = self.current_value()
        cost_basis = sum(d["qty"] * d["avg_buy"] for d in self.symbol_data.values())


        # Overall portfolio CAGR
        if self.earliest_date and cost_basis > 0:
            total_years = (self.today - self.earliest_date).days / 365.0
            portfolio_cagr = round((((current_value / cost_basis) ** (1 / total_years)) - 1) * 100, 2) if total_years > 0 else 0.0
        else:
            portfolio_cagr = 0.0

        return portfolio_cagr

    # ------------------------------------------------------------------
    # Value at Risk (95% confidence, parametric & historical)
    # ------------------------------------------------------------------

    def value_at_risk(self):
        """
        Compute 1-day VaR at 95% confidence level using:
          - Parametric (normal distribution) method
          - Historical simulation method
        Both expressed as absolute rupee values and as % of portfolio.
        """
        if not self.holdings and not self.transactions:
            return {"message": "No holdings or transactions found for user."}, 404

        s = self.portfolio_series
        current_value = self.current_value()

        if s is None or len(s) < 30:
            return {
                "message": "Insufficient history for VaR calculation (need ≥ 30 days).",
                "current_value": round(current_value, 2),
            }, 200

        daily_returns = s.pct_change().dropna()

        # --- Parametric VaR (assumes normal distribution) ---
        mean   = float(daily_returns.mean())
        std    = float(daily_returns.std())
        # z-score for 95% confidence (one-tail) = 1.645
        z_95   = 1.6449
        param_var_pct  = round((mean - z_95 * std) * 100, 4)    # negative = loss
        param_var_abs  = round(abs(param_var_pct / 100) * current_value, 2)

        # --- Historical VaR (5th percentile of actual returns) ---
        hist_var_pct   = round(float(np.percentile(daily_returns, 5)) * 100, 4)
        hist_var_abs   = round(abs(hist_var_pct / 100) * current_value, 2)

        # --- Conditional VaR / Expected Shortfall (CVaR) ---
        cutoff         = np.percentile(daily_returns, 5)
        tail_returns   = daily_returns[daily_returns <= cutoff]
        cvar_pct       = round(float(tail_returns.mean()) * 100, 4) if len(tail_returns) > 0 else hist_var_pct
        cvar_abs       = round(abs(cvar_pct / 100) * current_value, 2)

        # --- Per-holding contribution ---
        holding_var = []
        for symbol, data in self.symbol_data.items():
            series = close_series(self.history, yahoo_ticker(symbol))
            if series is None or len(series) < 30:
                continue
            ret = series.pct_change().dropna()
            h_current_value = data["qty"] * float(series.iloc[-1])
            h_hist_var_pct  = round(float(np.percentile(ret, 5)) * 100, 4)
            h_hist_var_abs  = round(abs(h_hist_var_pct / 100) * h_current_value, 2)
            holding_var.append({
                "stock_symbol":  symbol,
                "company_name":  data.get("company_name", ""),
                "current_value": round(h_current_value, 2),
                "var_95_pct":    h_hist_var_pct,
                "var_95_abs":    h_hist_var_abs,
            })

        return {
            "confidence_level": "95%",
            "horizon":          "1 day",
            "current_value":    round(current_value, 2),
            "observations":     len(daily_returns),
            "parametric": {
                "var_pct": param_var_pct,
                "var_abs": param_var_abs,
                "description": "Max expected 1-day loss (normal distribution, 95% confidence)",
            },
            "historical": {
                "var_pct": hist_var_pct,
                "var_abs": hist_var_abs,
                "description": "Max expected 1-day loss (historical simulation, 95% confidence)",
            },
            "conditional_var": {
                "cvar_pct": cvar_pct,
                "cvar_abs": cvar_abs,
                "description": "Expected loss beyond 95% VaR threshold (Expected Shortfall)",
            },
            "holdings": holding_var,
        }, 200

    # ------------------------------------------------------------------
    # Existing endpoints (unchanged logic)
    # ------------------------------------------------------------------

    def _holding_period_gains(self, symbol, data, current_value):
        gains = {}
        first_date = self.symbol_first_dates.get(symbol, self.earliest_date)
        for period, target in self.target_dates.items():
            if first_date and target < first_date:
                gains[period] = None
                continue
            past_price = price_at_date(self.history, yahoo_ticker(symbol), target, data["avg_buy"])
            past_value = data["qty"] * past_price if past_price is not None else 0.0
            gains[period] = [round(current_value - past_value, 2), pct_return(current_value, past_value)]
        return gains

    def gain_returns(self):
        """
        Portfolio KPI summary — excludes risk_metrics and benchmark_returns.
        Call get_risk_and_benchmarks() separately; it reuses the same cached history.
        """
        if not self.holdings and not self.transactions:
            return {"message": "No holdings or transactions found for user."}, 404
        realized_gain = sum(t.realized_pl for t in self.transactions)
        cost_basis = sum(d["qty"] * d["avg_buy"] for d in self.symbol_data.values())
        current_value = self.current_value()
        unrealized_gain = current_value - cost_basis
        total_invested = cost_basis + sum(t.realized_pl for t in self.transactions if t.realized_pl < 0)
        total_gain = unrealized_gain + realized_gain
        return {
            "gain_returns": {
                "unrealized_gain": round(unrealized_gain, 2),
                "realized_gain": round(realized_gain, 2),
                "total_invested": round(total_invested, 2),
                "total_gain": round(total_gain, 2),
                "total_return_pct": round((total_gain / total_invested * 100) if total_invested > 0 else 0.0, 2),
                "total_realized_return_pct": round((realized_gain / total_invested * 100) if total_invested > 0 else 0.0, 2),
                "xirr": xirr(self.cashflows(current_value)),
                "periods": self.period_returns(current_value),
                "cagr": self._cagr_and_value(),
                "days_since_last_contribution": (
                    self.today - self.transactions[-1].transaction_datetime.date()
                ).days if self.transactions else None,
            }
        }, 200

    def get_risk_and_benchmarks(self):
        """
        Risk metrics + benchmark comparison.
        Reuses self.history and self.portfolio_series already cached in memory
        — no additional yfinance call needed.
        """
        if not self.holdings and not self.transactions:
            return {"message": "No holdings or transactions found for user."}, 404
        return {
            "risk_metrics": self.risk_metrics(),
            "benchmarks": self.benchmark_returns(),
        }, 200

    def get_holdings(self):
        if not self.holdings and not self.transactions:
            return {"message": "No holdings or transactions found for user."}, 404

        holdings_list = []
        portfolio_current_value = self.current_value()
        portfolio_cost = sum(d["qty"] * d["avg_buy"] for d in self.symbol_data.values())
        for symbol, data in self.symbol_data.items():
            current_price = price_at_date(self.history, yahoo_ticker(symbol), self.today, data["avg_buy"])
            if current_price is None:
                current_price = data["avg_buy"]
            current_value = data["qty"] * (current_price if current_price is not None else data["avg_buy"])
            cost_value = data["qty"] * data["avg_buy"]
            total_gain = current_value - cost_value

            holdings_list.append({
                "stock_symbol": symbol,
                "company_name": data.get("company_name", ""),
                "quantity": data["qty"],
                "avg_buy_price": round(data["avg_buy"], 2),
                "current_price": round(current_price, 2),
                "current_value": round(current_value, 2),
                "cost_value": round(cost_value, 2),
                "allocation_pct": round(current_value / portfolio_current_value * 100, 2) if portfolio_current_value else 0.0,
                "cost_allocation_pct": round(cost_value / portfolio_cost * 100, 2) if portfolio_cost else 0.0,
                "total_gain": [round(total_gain, 2), pct_return(current_value, cost_value)],
                "periods": self._holding_period_gains(symbol, data, current_value),
            })
        holdings_list.sort(key=lambda x: x["cost_allocation_pct"], reverse=True)
        return {"holdings": holdings_list}, 200

    def daily_performance(self):
        if not self.transactions:
            return {"message": "No transactions found for user."}, 404
        if self.history is None or self.history.empty:
            return {"daily_performance": []}, 200

        stock_units, nifty_units, cash_deposit, rows = {}, 0.0, 0.0, []
        nifty = close_series(self.history, "^NSEI")
        if nifty is None or nifty.empty:
            return {"daily_performance": []}, 200

        trade_idx = 0

        for idx in sorted(nifty.loc[str(self.earliest_date):].index):
            day = idx.date() if hasattr(idx, "date") else idx
            while trade_idx < len(self.transactions):
                t = self.transactions[trade_idx]
                trade_date = t.transaction_datetime.date() if isinstance(t.transaction_datetime, datetime) else t.transaction_datetime
                if trade_date > day:
                    break
                amount = float(t.quantity) * float(t.price)
                symbol = t.stock_symbol.split(".")[0]
                if t.transaction_type.upper() == "BUY":
                    stock_units[symbol] = stock_units.get(symbol, 0.0) + float(t.quantity)
                    nifty_price = price_at_date(self.history, "^NSEI", day)
                    nifty_units += amount / nifty_price if nifty_price else 0.0
                    cash_deposit += amount
                elif t.transaction_type.upper() == "SELL":
                    stock_units[symbol] = stock_units.get(symbol, 0.0) - float(t.quantity)
                    nifty_price = price_at_date(self.history, "^NSEI", day)
                    nifty_units -= amount / nifty_price if nifty_price else 0.0
                    cash_deposit -= amount
                trade_idx += 1

            portfolio_value = sum(
                qty * (price_at_date(self.history, yahoo_ticker(symbol), day, 0.0) or 0.0)
                for symbol, qty in stock_units.items()
            )
            nifty_value = nifty_units * float(nifty.loc[idx])
            rows.append({
                "date": str(day),
                "portfolio_value": round(portfolio_value, 2),
                "cash_deposit": round(cash_deposit, 2),
                "cash_gain": round(portfolio_value - cash_deposit, 2),
                "cash_return_pct": pct_return(portfolio_value, cash_deposit),
                "nifty_50_value": round(nifty_value, 2),
                "nifty_50_gain": round(portfolio_value - nifty_value, 2),
                "nifty_50_return_pct": pct_return(portfolio_value, nifty_value),
            })
        return {"daily_performance": rows}, 200

    def daily_movers(self, is_today=True, count=None):
        """
        Get daily movers — top gainers and losers from holdings with today's % change.
        """
        if self.history is None or self.history.empty:
            return {"daily_performance": []}, 200

        end_date = self.today if is_today else self.today - timedelta(days=1)

        all_movers = []
        for symbol, data in self.symbol_data.items():
            series = close_series(self.history, yahoo_ticker(symbol))
            if series is None or series.empty:
                current_price = data["avg_buy"]
                prev_price = data["avg_buy"]
            else:
                sliced = series.loc[:str(end_date)].dropna()
                if len(sliced) >= 2:
                    current_price = float(sliced.iloc[-1])
                    prev_price = float(sliced.iloc[-2])
                elif len(sliced) == 1:
                    current_price = float(sliced.iloc[-1])
                    prev_price = data["avg_buy"]
                else:
                    current_price = data["avg_buy"]
                    prev_price = data["avg_buy"]

            current_value = data["qty"] * current_price
            prev_value = data["qty"] * prev_price
            today_gain = current_value - prev_value

            all_movers.append({
                "stock_symbol":   symbol,
                "company_name":   data.get("company_name", ""),
                "current_price":  round(current_price, 2),
                "current_value":  round(current_value, 2),
                "gain":           round(today_gain, 2),
                "gain_precentage": pct_return(current_value, prev_value),
            })

        all_movers.sort(key=lambda x: x["gain_precentage"], reverse=True)

        positive_movers = [m for m in all_movers if m["gain"] > 0]
        negative_movers = [m for m in all_movers if m["gain"] < 0]

        if count is not None:
            positive_movers = positive_movers[:count]
            negative_movers = sorted(negative_movers, key=lambda x: x["gain_precentage"])[:count]

        return {
            "top_gainers": positive_movers,
            "top_losers":  sorted(negative_movers, key=lambda x: x["gain_precentage"]),
        }, 200
    
    def alpha(self):
        """
        Alpha of the overall portfolio vs Nifty 50, Sensex, and Nifty Bank.
         - Alpha represents excess return over benchmark; Beta represents sensitivity to benchmark movements.
         - Calculation uses daily returns and is annualized.
         - Requires at least 2 overlapping return observations between portfolio and benchmark.
         - Returns None for alpha/beta if insufficient data or zero variance in benchmark returns.
         - Reuses self.history and self.portfolio_series already cached in memory.
         - No additional yfinance call needed.
         - Endpoint: /performance/alpha
        """
        if not self.holdings and not self.transactions:
            return {"message": "No holdings or transactions found for user."}, 404
        s = self.portfolio_series
        if s is None or s.empty:
            return {"message": "Insufficient history for alpha calculation."}, 200

        daily_returns = s.pct_change().dropna()
        benchmarks = {
            "nifty_50": close_series(self.history, "^NSEI"),
            "sensex": close_series(self.history, "^BSESN"),
            "nifty_bank": close_series(self.history, "^NSEBANK"),
        }
        results = {}
        for key, benchmark in benchmarks.items():
            key = key.replace("_", " ").title()  # e.g. "nifty_50" -> "Nifty 50"
            if benchmark is None or benchmark.empty:
                results[key] = {"alpha_pct": None, "beta": None, "message": "No data"}
                continue
            benchmark_returns = benchmark.pct_change().dropna()
            aligned = daily_returns.align(benchmark_returns, join="inner")
            if len(aligned[0]) < 2 or aligned[1].var() == 0:
                results[key] = {"alpha_pct": None, "beta": None, "message": "Insufficient overlapping returns"}
                continue
            beta = round(aligned[0].cov(aligned[1]) / aligned[1].var(), 2)
            results[key] = {"beta": beta}
        return {"beta_vs_benchmarks": results}, 200
    
    def what_if_analysis(self):
        """
        What-if analysis for investment into benchmark indices.
        Simulates investing the exact same amounts at the exact same transaction dates
        into the benchmarks instead of the portfolio.
        """
        if not self.holdings and not self.transactions:
            return {"message": "No holdings or transactions found for user."}, 404
        
        benchmarks = {
            "nifty_50": {"ticker": "^NSEI", "name": "Nifty 50"},
            "sensex": {"ticker": "^BSESN", "name": "Sensex"},
            "nifty_bank": {"ticker": "^NSEBANK", "name": "Nifty Bank"},
        }
        results = {}
        
        for key, bench_info in benchmarks.items():
            ticker = bench_info["ticker"]
            name = bench_info["name"]
            
            units = 0.0
            invested = 0.0
            withdrawn = 0.0
            
            for t in self.transactions:
                trade_date = t.transaction_datetime.date() if hasattr(t.transaction_datetime, 'date') else t.transaction_datetime
                price = price_at_date(self.history, ticker, trade_date)
                if not price:
                    continue
                amount = float(t.quantity) * float(t.price)
                if t.transaction_type.upper() == "BUY":
                    units += amount / price
                    invested += amount
                elif t.transaction_type.upper() == "SELL":
                    units -= amount / price
                    withdrawn += amount
            
            current_price = price_at_date(self.history, ticker, self.today, fallback=0.0)
            if not current_price or (units == 0 and invested == 0 and withdrawn == 0):
                results[key] = {"name": name, "current_value_if_invested": None, "gain": 0.0, "return_pct": 0.0, "message": "No data"}
                continue
                
            value = units * float(current_price)
            gain = value + withdrawn - invested
            return_pct = pct_return(value + withdrawn, invested)
            
            results[key] = {
                "name": name,
                "current_value_if_invested": round(value, 2),
                "gain": round(gain, 2),
                "return_pct": return_pct,
            }
            
        return {"what_if_analysis": results}, 200