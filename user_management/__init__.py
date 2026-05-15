from fastapi import APIRouter, Body, Request

from fastapi.responses import JSONResponse
from .login import login_user
from .register import register_user
from .login import decode_access_token
from auth_decorator import auth_required
from database import db_handler, User
from sqlmodel import select

user_mgmt_router = APIRouter(tags=["user-management"], prefix="/user")

@user_mgmt_router.post("/login")
async def login(payload: dict = Body(...)):
    """
    Login endpoint using the logic from login.py
    """
    response_data, status_code = login_user(payload)
    return JSONResponse(content=response_data, status_code=status_code)

@user_mgmt_router.get("/userinfo")
@auth_required
async def userinfo(request: Request):
    """
    User info endpoint that returns the authenticated user's information.
    """
    # Get the Authorization header
    user_info = {
        "username": request.user_info['username'],
        "email": request.user_info['email'],
        "user_id": request.user_info['user_id'],
        "last_login": request.user_info['last_login'].isoformat() if request.user_info['last_login'] else None
    }
    return JSONResponse(content=user_info, status_code=200)

@user_mgmt_router.post("/register")
async def register(payload: dict = Body(...)):
    """
    Register endpoint using the logic from register.py
    """
    response_data, status_code = register_user(payload)
    return JSONResponse(content=response_data, status_code=status_code)
