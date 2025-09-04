import logging
import time
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import os
import pickle

from config import Config

class StockData:
    def __init__(self, cache_dir='stock_cache', cache_expiry_days=1):
        self.cache_dir = cache_dir
        self.cache_expiry = timedelta(days=cache_expiry_days)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _get_cache_path(self, cache_key):
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")

    def get_data(self, tickers, start_date, end_date, group_by='ticker'):
        if isinstance(tickers, str):
            tickers = [tickers]
        
        cache_key = f"bulk_{'_'.join(sorted(tickers))}_{start_date.date()}_{end_date.date()}"
        cache_path = self._get_cache_path(cache_key)

        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                data, timestamp = pickle.load(f)
            if datetime.now() - timestamp < self.cache_expiry:
                logging.info(f"Loading from cache: {cache_key}")
                return data

        logging.info(f"Fetching from yfinance: {tickers}")
        data = yf.download(tickers, start=start_date, end=end_date, group_by=group_by)
        with open(cache_path, 'wb') as f:
            pickle.dump((data, datetime.now()), f)
        return data

    def refresh_cache(self, tickers, start_date, end_date, group_by='ticker'):
        if isinstance(tickers, str):
            tickers = [tickers]
            
        cache_key = f"bulk_{'_'.join(sorted(tickers))}_{start_date.date()}_{end_date.date()}"
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            os.remove(cache_path)
        
        return self.get_data(tickers, start_date, end_date, group_by)

class PortfolioAnalysis:
    def __init__(self):
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=730) # 2 years to be safe
        self.today = self.end_date.date()
        self.benchmark_tickers = list(Config.BENCHMARK_NAMES.keys())
        self.stock_data_fetcher = StockData()
        self.benchmark_data = self.stock_data_fetcher.get_data(self.benchmark_tickers, start_date=self.start_date, end_date=self.end_date)

    def convert_transactions(self, transactions):
        """Convert list of Transaction objects to DataFrame"""
        
        transactions_data = [
            {
                "Ticker": t.stock_symbol,
                "Shares": t.quantity if t.transaction_type.lower() == 'buy' else -t.quantity,
                "Price": t.price,
                "Date": t.transaction_datetime,
                "Type": t.transaction_type.lower()
            } for t in transactions
        ]
        
        transactions_df = pd.DataFrame(transactions_data)
        transactions_df['Date'] = pd.to_datetime(transactions_df['Date'])
        return transactions_df

    def get_benchmark_and_what_if_analysis(self, buy_transactions_df, total_investment):
        benchmarks = {}
        for ticker, name in Config.BENCHMARK_NAMES.items():
            if ticker in self.benchmark_data.columns.levels[0]:
                bm_hist = self.benchmark_data[ticker]['Close'].dropna()
                if len(bm_hist) >= 2:
                    benchmark = {
                        "last": round(bm_hist.iloc[-1], 2),
                        "change_1d": round(bm_hist.iloc[-1] - bm_hist.iloc[-2], 2),
                        "pct_change_1d": round(((bm_hist.iloc[-1] - bm_hist.iloc[-2]) / bm_hist.iloc[-2]), 4) * 100 if bm_hist.iloc[-2] != 0 else 0
                    }


                    total_bm_units = 0
                    for _, row in buy_transactions_df.iterrows():
                        investment_date = row['Date']
                        investment_amount = row['Shares'] * row['Price']
                        
                        try:
                            # Find the closest available date in benchmark history
                            bm_price_on_date = bm_hist.asof(investment_date)
                        except KeyError:
                            continue

                        if bm_price_on_date > 0:
                            bm_units = investment_amount / bm_price_on_date
                            total_bm_units += bm_units

                    current_bm_price = bm_hist.iloc[-1]
                    prev_day_bm_price = bm_hist.iloc[-2] if len(bm_hist) > 1 else current_bm_price
                    
                    what_if_bm_value = total_bm_units * current_bm_price
                    what_if_bm_value_prev_day = total_bm_units * prev_day_bm_price

                    what_if_bm_change_1d = what_if_bm_value - what_if_bm_value_prev_day
                    what_if_bm_pct_change_1d = (what_if_bm_change_1d / what_if_bm_value_prev_day) * 100 if what_if_bm_value_prev_day > 0 else 0
                    
                    what_if_bm_change_total = what_if_bm_value - total_investment
                    what_if_bm_pct_change_total = (what_if_bm_change_total / total_investment) * 100 if total_investment > 0 else 0

                    what_if_investing_in_benchmark = {
                        "value": round(what_if_bm_value, 2),
                        "change_1d": round(what_if_bm_change_1d, 2),
                        "pct_change_1d": round(what_if_bm_pct_change_1d, 2),
                        "change_total": round(what_if_bm_change_total, 2),
                        "pct_change_total": round(what_if_bm_pct_change_total, 2)
                    }

                    benchmarks[name] = {
                        "ticker": ticker,
                        'name': name,
                        "last": benchmark["last"],
                        "change_1d": benchmark["change_1d"],
                        "pct_change_1d": benchmark["pct_change_1d"],
                        "what_if_not_invested": total_investment,
                        "what_if_investing_in_benchmark": what_if_investing_in_benchmark
                    }

        return benchmarks

    def get_portfolio_value_on_date(self, portfolio_df, data, target_date):
        """Calculate total portfolio value at a specific date"""
        portfolio_value = 0

        for index, row in portfolio_df.iterrows():
            symbol = row['Ticker']
            shares = row['Shares']

            if symbol in data:
                symbol_data = data[symbol]
                # Find the closest trading day to target_date
                available_dates = symbol_data.index
                closest_date = min(available_dates, key=lambda x: abs((x.date() - target_date).days))

                if 'Close' in symbol_data.columns:
                    price = symbol_data.loc[closest_date, 'Close']
                else:
                    price = symbol_data.loc[closest_date]

                portfolio_value += shares * price

        return portfolio_value

    def get_current_portfolio_value(self, portfolio_df, history_data):
        """Get current portfolio value"""
        current_value = 0
        for symbol in portfolio_df['Ticker']:
            if symbol in history_data:
                current_price = history_data[symbol]['Close'].iloc[-1]
                shares = portfolio_df.loc[portfolio_df['Ticker'] == symbol, 'Shares'].values[0]
                current_value += shares * current_price
        return current_value

    def get_portfolio_performance(self, current_value, portfolio_df, hist_data):

        periods = {
            '1W': self.today - timedelta(weeks=1),
            '1M': self.today - timedelta(days=30),
            '3M': self.today - timedelta(days=90),
            '6M': self.today - timedelta(days=180),
            'YTD': datetime(self.today.year, 1, 1).date(),
            '1Y': self.today - timedelta(days=365),
            'All': self.today - timedelta(days=365 * 100)
        }

        results = {}
        for period_name, past_date in periods.items():
            try:
                past_value = self.get_portfolio_value_on_date(portfolio_df, hist_data, past_date)
                gain_amount = current_value - past_value
                gain_percentage = (gain_amount / past_value) * 100
                
                results[period_name] = {
                    'gain_amount': round(gain_amount, 2),
                    'gain_percentage': round(gain_percentage, 2)
                }
                
            except Exception as e:
                print(f"Error calculating {period_name}: {e}")
                results[period_name] = {'gain_amount': 0, 'gain_percentage': 0}

        portfolio_performance = {
            "current_value": round(current_value, 2),
            "performance": results
        }
        # # All-time and 52-week high
        shares_series = portfolio_df.set_index('Ticker')['Shares']
        daily_portfolio_values = (pd.DataFrame({ticker: hist_data[ticker]['Close'] for ticker in portfolio_df['Ticker']}) * shares_series).sum(axis=1)
        all_time_high = daily_portfolio_values.max()
        one_year_ago = self.today - timedelta(days=365)
        fifty_two_week_high = daily_portfolio_values.loc[one_year_ago:].max()

        portfolio_performance['all_time_high'] = round(all_time_high, 2)
        portfolio_performance['fifty_two_week_high'] = round(fifty_two_week_high, 2)
        return portfolio_performance

    def calculate_advanced_metrics(self, daily_portfolio_values, portfolio_df):
        """
        Calculates advanced portfolio metrics like Beta, Sharpe Ratio, and Max Drawdown.
        """
        if daily_portfolio_values.empty or len(daily_portfolio_values) < 2:
            return {}

        # Portfolio daily returns
        portfolio_returns = daily_portfolio_values.pct_change().dropna()

        # --- Beta Calculation for Portfolio ---
        betas = {}
        for ticker in self.benchmark_tickers:
            if ticker in self.benchmark_data.columns.levels[0]:
                bm_hist = self.benchmark_data[ticker]['Close'].dropna()
                bm_returns = bm_hist.pct_change().dropna()
                
                # Align data
                aligned_returns = pd.concat([portfolio_returns, bm_returns], axis=1, join='inner').dropna()
                aligned_returns.columns = ['portfolio', 'benchmark']

                if len(aligned_returns) > 1:
                    # Covariance matrix
                    cov_matrix = np.cov(aligned_returns['portfolio'], aligned_returns['benchmark'])
                    # Variance of benchmark
                    bm_variance = np.var(aligned_returns['benchmark'])
                    if bm_variance > 0:
                        beta = cov_matrix[0, 1] / bm_variance
                        betas[ticker] = round(beta, 2)
        

        # --- Sharpe Ratio ---
        # Assuming risk-free rate is 0
        sharpe_ratio = 0
        if portfolio_returns.std() > 0:
            # Annualized Sharpe Ratio
            sharpe_ratio = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)

        # --- Max Drawdown and Duration ---
        running_max = daily_portfolio_values.cummax()
        drawdown = (daily_portfolio_values - running_max) / running_max
        max_drawdown = drawdown.min()
        
        max_drawdown_duration = 0
        if not drawdown.empty and max_drawdown < 0:
            end_date = drawdown.idxmin()
            start_date = daily_portfolio_values.loc[:end_date].idxmax()
            max_drawdown_duration = (end_date - start_date).days

        return {
            "beta": betas,
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown": round(max_drawdown * 100, 2), # in percentage
            "max_drawdown_duration_days": max_drawdown_duration
        }

    def get_current_holdings(self, transactions, tickers = None, portfolio_df = None, avg_buy_prices = None, sell_transactions = None):
        """Get current holdings with unrealized and realized gains"""
        transactions_df = self.convert_transactions(transactions)
        if tickers is None:
            portfolio_df = transactions_df.groupby('Ticker')['Shares'].sum().reset_index()
            portfolio_df = portfolio_df[portfolio_df['Shares'] > 0]
            tickers = portfolio_df['Ticker'].unique().tolist()
            # Calculate total realized and unrealized gains
            buy_transactions = transactions_df[transactions_df['Type'] == 'buy']
            sell_transactions = transactions_df[transactions_df['Type'] == 'sell']

            # Calculate average buy price for each ticker
            avg_buy_prices = buy_transactions.groupby('Ticker').apply(lambda x: (x['Price'] * x['Shares']).sum() / x['Shares'].sum()).to_dict()


        current_holdings = []
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            long_name = stock.info.get('longName', ticker)
            current_price = stock.history(period='1d')['Close'].iloc[0]
            shares = int(portfolio_df.loc[portfolio_df['Ticker'] == ticker, 'Shares'].values[0])
            market_value = current_price * shares
            avg_buy_price = avg_buy_prices.get(ticker, 0)

            unrealized_gain = (current_price - avg_buy_price) * shares if avg_buy_price else 0
            unrealized_gain_pct = (unrealized_gain / (avg_buy_price * shares)) * 100 if avg_buy_price and shares else 0

            ticker_sells = sell_transactions[sell_transactions['Ticker'] == ticker]
            realized_gain_for_ticker = 0
            if not ticker_sells.empty:
                realized_gain_for_ticker = ticker_sells.apply(
                    lambda row: (row['Price'] - avg_buy_price) * (-row['Shares']),
                    axis=1
                ).sum()

            realized_gain_for_ticker_pct = (realized_gain_for_ticker / (avg_buy_price * (-ticker_sells['Shares']).sum())) * 100 if avg_buy_price and not ticker_sells.empty else 0

            current_holdings.append({
                "ticker": ticker,
                'name': long_name,
                "shares": shares,
                "current_price": round(current_price, 2),
                "market_value": round(market_value, 2),
                "average_buy_price": round(avg_buy_price, 2),
                "unrealized_gain": round(unrealized_gain, 2),
                "unrealized_gain_pct": round(unrealized_gain_pct, 2),
                "realized_gain": round(realized_gain_for_ticker, 2),
                "realized_gain_pct": round(realized_gain_for_ticker_pct, 2),
            })
        return current_holdings
    
    def calculate_portfolio_performance(self, transactions):
        """
        Calculates detailed portfolio performance metrics based on transactions.
        """
        if not transactions:
            return {}
        
        transactions_df = self.convert_transactions(transactions)

        # Portfolio holdings
        portfolio_df = transactions_df.groupby('Ticker')['Shares'].sum().reset_index()
        portfolio_df = portfolio_df[portfolio_df['Shares'] > 0]
        tickers = portfolio_df['Ticker'].unique().tolist()

        
        # Calculate total realized and unrealized gains
        buy_transactions = transactions_df[transactions_df['Type'] == 'buy']
        sell_transactions = transactions_df[transactions_df['Type'] == 'sell']

        # Calculate average buy price for each ticker
        avg_buy_prices = buy_transactions.groupby('Ticker').apply(lambda x: (x['Price'] * x['Shares']).sum() / x['Shares'].sum()).to_dict()

        # Calculate total realized gain
        total_realized_gain = 0
        if not sell_transactions.empty:
            realized_gains_per_sale = sell_transactions.apply(
                lambda row: (row['Price'] - avg_buy_prices.get(row['Ticker'], 0)) * (-row['Shares']),
                axis=1
            )
            total_realized_gain = realized_gains_per_sale.sum()
        
        # Total Investment
        buy_transactions_df = transactions_df[transactions_df['Type'] == 'buy']
        total_investment_gross = (buy_transactions_df['Shares'] * buy_transactions_df['Price']).sum()

        # Cash from sales
        sell_transactions_df_for_cash = transactions_df[transactions_df['Type'] == 'sell']
        cash_from_sales = (-sell_transactions_df_for_cash['Shares'] * sell_transactions_df_for_cash['Price']).sum()

        # Cost of sold assets
        cost_of_sold_assets = 0
        if not sell_transactions.empty:
            cost_of_sold_assets = sell_transactions.apply(
                lambda row: avg_buy_prices.get(row['Ticker'], 0) * (-row['Shares']),
                axis=1
            ).sum()

        total_current_investment = total_investment_gross - cost_of_sold_assets
        
        # Download historical data for all tickers at once
        s = time.time()
        hist_data = self.stock_data_fetcher.get_data(tickers, start_date=self.start_date, end_date=self.end_date, group_by='ticker')
        logging.info(f"Downloaded historical data in {time.time() - s} seconds")
        
        current_value = self.get_current_portfolio_value(portfolio_df, hist_data)

        portfolio_performance = self.get_portfolio_performance(current_value, portfolio_df, hist_data)

        benchmarks = self.get_benchmark_and_what_if_analysis(buy_transactions_df, total_investment_gross)

        # Calculate advanced metrics
        shares_series = portfolio_df.set_index('Ticker')['Shares']
        daily_portfolio_values = (pd.DataFrame({ticker: hist_data[ticker]['Close'] for ticker in portfolio_df['Ticker']}) * shares_series).sum(axis=1).dropna()
        advanced_metrics = self.calculate_advanced_metrics(daily_portfolio_values, portfolio_df)


        summary = {
                    "current_value": {
                        "total_gross_investment": round(total_investment_gross, 2),
                        'total_current_investment': round(total_current_investment, 2),
                        "current_value": round(current_value, 2),
                        "cash_from_sales": round(cash_from_sales, 2),
                        "realized_gain": round(total_realized_gain, 2),
                    },
                    "portfolio_performance": portfolio_performance,
                    "advanced_metrics": advanced_metrics,
                    "benchmarks": benchmarks,
                    'current_datetime': self.end_date.isoformat()
                }
    
        
        return summary
