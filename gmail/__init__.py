
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from .oauth_callback import exchange_code_for_token
from .oauth_url import generate_google_oauth_url

gmail_router = APIRouter(tags=["Gmail"], prefix="/gmail")

@gmail_router.post("/oauth-callback")
async def oauth_callback(payload: dict = Body(...)):
    """  
    OAuth callback endpoint using the logic from oauth_callback.py
    """
    response_data, status_code = exchange_code_for_token(payload)
    return JSONResponse(content=response_data, status_code=status_code)

@gmail_router.get("/oauth-url")
async def oauth_url():
    """
    OAuth URL endpoint using the logic from oauth_url.py
    """
    response_data, status_code = generate_google_oauth_url()
    return JSONResponse(content=response_data, status_code=status_code)