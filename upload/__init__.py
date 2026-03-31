from fastapi import APIRouter
from fastapi.responses import JSONResponse

from auth_decorator import auth_required
from .transactions import sync_transactions

upload_router = APIRouter(tags=["Gmail"], prefix="/gmail")

@upload_router.post("/sync_transactions")
@auth_required
def sync_transactions_endpoint(user_id):
    """
    Sync transactions from gmail
    """
    response_data, status_code = sync_transactions(user_id)
    return JSONResponse(content=response_data, status_code=status_code)