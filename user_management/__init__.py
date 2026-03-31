from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from .login import login_user
from .register import register_user

user_mgmt_router = APIRouter(tags=["user-management"], prefix="/user")

@user_mgmt_router.post("/login")
async def login(payload: dict = Body(...)):
    """
    Login endpoint using the logic from login.py
    """
    response_data, status_code = login_user(payload)
    return JSONResponse(content=response_data, status_code=status_code)

@user_mgmt_router.post("/register")
async def register(payload: dict = Body(...)):
    """
    Register endpoint using the logic from register.py
    """
    response_data, status_code = register_user(payload)
    return JSONResponse(content=response_data, status_code=status_code)
