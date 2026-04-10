
from typing import List, Optional
from fastapi import APIRouter, File, Request, UploadFile, Form, Query
from fastapi.responses import JSONResponse
from auth_decorator import auth_required

from .key_benchmark import gain_returns

perf_router = APIRouter(tags=["performance"], prefix="/performance")

@perf_router.get("/gain_returns")
@auth_required
async def sync_ipo(request: Request):
    """
    Sync IPO details endpoint
    """
    response, status_code = gain_returns(request.user_id)
    return JSONResponse(content=response, status_code=status_code)
