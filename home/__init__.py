from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from auth_decorator import auth_required

from .key_benchmark import get_market_data, get_portfolio_performance

perf_router = APIRouter(tags=["home"], prefix="/home")

@perf_router.get("/performance")
@auth_required
def performance(request: Request):
    """
    Portfolio KPI summary — unrealized/realized gain, XIRR, period returns.

    Intentionally excludes risk_metrics and benchmark_returns.
    Call /risk_and_benchmarks separately (reuses the same cached download).
    """
    response, status_code = get_portfolio_performance(request.user_id).gain_returns()
    return JSONResponse(content=response, status_code=status_code)


@perf_router.get("/risk_and_benchmarks")
@auth_required
def risk_and_benchmarks(request: Request):
    """
    Risk metrics (Sharpe, beta, drawdown, ATH) and benchmark comparison
    (Nifty 50, Sensex, etc.) — uses the same cached download as /performance,
    so this call is instant after /performance has already been served.
    """
    response, status_code = get_portfolio_performance(request.user_id).get_risk_and_benchmarks()
    return JSONResponse(content=response, status_code=status_code)
    

@perf_router.get("/get_holdings")
@auth_required
def get_holdings(request: Request):
    """All current holdings with allocation and period returns."""
    response, status_code = get_portfolio_performance(request.user_id).get_holdings()
    return JSONResponse(content=response, status_code=status_code)

@perf_router.get("/get_risk_and_benchmarks")
@auth_required
def get_risk_and_benchmarks(request: Request):
    """Current allocation breakdown by sector and market cap."""
    response, status_code = get_portfolio_performance(request.user_id).get_risk_and_benchmarks()
    return JSONResponse(content=response, status_code=status_code)


@perf_router.get("/get_daily_data")
@auth_required
def get_daily_data(request: Request):
    """Day-by-day portfolio value vs Nifty 50."""
    response, status_code = get_portfolio_performance(request.user_id).daily_performance()
    return JSONResponse(content=response, status_code=status_code)


@perf_router.get("/get_daily_movers")
@auth_required
def get_daily_movers(request: Request):
    """Top gainers and losers from holdings for today."""
    response, status_code = get_portfolio_performance(request.user_id).daily_movers()
    return JSONResponse(content=response, status_code=status_code)

@perf_router.get("/market_data")
def market_data():
    """
    Current levels and 30-day sparklines for Nifty 50, Sensex, Nifty Bank,
    and Nifty 100. Cached for the calendar day, no authentication required.
    """
    data = get_market_data()
    return JSONResponse(content=data, status_code=200)

@perf_router.get("/var")
@auth_required
def value_at_risk(request: Request):
    """
    1-day Value at Risk at 95% confidence — parametric, historical, and CVaR.
    Per-holding VaR contribution included. Reuses the cached download.
    """
    response, status_code = get_portfolio_performance(request.user_id).value_at_risk()
    return JSONResponse(content=response, status_code=status_code)


@perf_router.get("/alpha")
@auth_required
def alpha(request: Request):
    """
    Alpha and beta of the overall portfolio vs Nifty 50. Reuses the cached download.
    """
    response, status_code = get_portfolio_performance(request.user_id).alpha()
    return JSONResponse(content=response, status_code=status_code)

@perf_router.get("/what_if")
@auth_required
def what_if(request: Request):
    """
    What-if analysis for investment into benchmark indices.
    """ 
    response, status_code = get_portfolio_performance(request.user_id).what_if_analysis()
    return JSONResponse(content=response, status_code=status_code)