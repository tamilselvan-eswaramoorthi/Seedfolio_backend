import pandas as pd
from datetime import date
from home.utils import fetch_recent_prices, close_series

tickers = ["^NSEI", "^BSESN"]
daily_raw = fetch_recent_prices(tickers, already_yf_tickers=True)
for t in tickers:
    s = close_series(daily_raw, t)
    # Get previous close
    today_ts = pd.Timestamp(date.today()).tz_localize(s.index.tz)
    prev = s[s.index < today_ts]
    print(f"{t}: last={s.iloc[-1]}, prev={prev.iloc[-1]}")
