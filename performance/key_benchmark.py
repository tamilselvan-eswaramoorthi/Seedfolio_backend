import logging
from datetime import date, datetime, timedelta

from sqlmodel import select

from database import Holdings, Transaction, db_handler
from .constants import BENCHMARKS, PERIOD_DAYS
from .utils import close_series, fetch_history_batch, pct_return, portfolio_value_from_cache, portfolio_value_series, price_at_date, yahoo_ticker

logger = logging.getLogger(__name__)


def _xirr(cashflows):
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


class PortfolioPerformance:
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
        dates = [d for p, d in self.target_dates.items() if self.applicable[p]]
        if self.earliest_date:
            dates.append(self.earliest_date)
        start = min(dates) if dates else self.today
        symbols = list(self.symbol_data.keys()) + [b["ticker"] for b in BENCHMARKS.values()]
        return fetch_history_batch(symbols, start, self.today)

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
                "xirr": _xirr(flows),
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
                "xirr": _xirr(self.cashflows(current_value)),
                "risk_metrics": self.risk_metrics(),
                "benchmarks": self.benchmark_returns(),
                "periods": self.period_returns(current_value),
            }
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
