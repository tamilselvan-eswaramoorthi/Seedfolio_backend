
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from auth_decorator import auth_required

from .key_benchmark import PortfolioPerformance

perf_router = APIRouter(tags=["performance"], prefix="/performance")

@perf_router.get("/performance")
@auth_required
async def performance(request: Request):
    """
    Sync IPO details endpoint
    """
    user_id = request.user_id
    if not isinstance(user_id, str) or not user_id:
        return JSONResponse(content={"message": "Unauthorized"}, status_code=401)
    response, status_code = PortfolioPerformance(user_id).gain_returns()
    return JSONResponse(content=response, status_code=status_code)


@perf_router.get("/get_holdings")
@auth_required
async def get_holdings(request: Request):
    """
    Sync IPO details endpoint
    """
    user_id = request.user_id
    if not isinstance(user_id, str) or not user_id:
        return JSONResponse(content={"message": "Unauthorized"}, status_code=401)
    response, status_code = PortfolioPerformance(user_id).get_holdings()
    return JSONResponse(content=response, status_code=status_code)


@perf_router.get("/get_daily_data")
@auth_required
async def get_daily_data(request: Request):
    """
    Sync IPO details endpoint
    """
    user_id = request.user_id
    if not isinstance(user_id, str) or not user_id:
        return JSONResponse(content={"message": "Unauthorized"}, status_code=401)
    response, status_code = PortfolioPerformance(user_id).daily_performance()
    return JSONResponse(content=response, status_code=status_code)


@perf_router.get("/get_daily_movers")
@auth_required
async def get_daily_movers(request: Request):
    """
    Get daily movers
    """
    user_id = request.user_id
    if not isinstance(user_id, str) or not user_id:
        return JSONResponse(content={"message": "Unauthorized"}, status_code=401)
    response, status_code = PortfolioPerformance(user_id).daily_movers()
    return JSONResponse(content=response, status_code=status_code)
