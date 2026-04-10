from typing import Optional
from fastapi import APIRouter, File, Form, UploadFile, Request
from fastapi.responses import JSONResponse

from auth_decorator import auth_required
from .transactions import sync_transactions
from .file_upload import upload_transactions

upload_router = APIRouter(tags=["Gmail"], prefix="/upload")

@upload_router.post("/sync_transactions")
@auth_required
def sync_transactions_endpoint(user_id):
    """
    Sync transactions from gmail
    """
    response_data, status_code = sync_transactions(user_id)
    return JSONResponse(content=response_data, status_code=status_code)


@upload_router.post('/upload_transactions')
@auth_required
def upload_transactions_endpoint( 
    request: Request,
    broker: Optional[str] = Form(None),
    file: UploadFile = File(None)
):
    """
    Upload transactions from UI
    """
    user_id = request.user_id # type: ignore
    response_data, status_code = upload_transactions(broker, file, user_id)
    return JSONResponse(content=response_data, status_code=status_code)