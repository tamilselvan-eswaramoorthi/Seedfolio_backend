
from typing import List, Optional
from fastapi import APIRouter, File, Request, UploadFile, Form, Query
from fastapi.responses import JSONResponse
from auth_decorator import auth_required

from .ipo import sync_ipo_details
from .stock import sync_market_data
from .corporate_actions import sync_corporate_actions

settings_router = APIRouter(tags=["Settings"], prefix="/settings")

@settings_router.get("/sync_ipo_details")
@auth_required
async def sync_ipo(request: Request):
    """
    Sync IPO details endpoint
    """
    response, status_code = sync_ipo_details()
    return JSONResponse(content=response, status_code=status_code)

@settings_router.post("/sync_market_data")
@auth_required
async def sync_market(
    request: Request,
    params: Optional[dict] = Form(None),
    file: UploadFile = File(None)
):
    """
    Sync market data endpoint
    """
    if not params:
        params = {}
    response, status_code = await sync_market_data(params=params, file=file)
    return JSONResponse(content=response, status_code=status_code)

@settings_router.get("/sync_corporate_actions")
@auth_required
async def sync_corp_actions(
    request: Request,
    event_type: str = Query(..., description="One of: Splits, Bonus, Merger/Demerger")
):
    """
    Fetch upcoming corporate actions from TradeBrains and persist them to DB.

    - **event_type**: `Splits`, `Bonus`, or `Merger/Demerger`
    """
    response, status_code = sync_corporate_actions(event_type=event_type)
    return JSONResponse(content=response, status_code=status_code)