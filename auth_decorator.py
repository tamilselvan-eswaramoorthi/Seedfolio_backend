import logging
from functools import wraps
import asyncio
import inspect
from fastapi import Request, HTTPException, status
from jose import jwt
from sqlmodel import select

from config import Config
from database import db_handler, User

def decode_access_token(token: str):
    """
    Decodes the JWT access token and returns the payload.
    """
    try:
        # Handle "Bearer <token>" format
        if token.startswith("Bearer "):
            token = token.split(" ")[1]
        
        payload = jwt.decode(token, Config.JWT_SECRET, algorithms=[Config.JWT_ALGORITHM])
        return payload
    except Exception as e:
        logging.error(f"Token decoding error: {e}")
        return None

def auth_required(f):
    """
    Authentication decorator.
    Can be used on both async and sync route handlers.
    Attaches user information to request.state.
    """
    @wraps(f)
    async def decorated_function(request: Request, *args, **kwargs):
        logging.info('Executing authentication decorator.')
        
        # 1. Get the Authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: No Authorization header provided"
            )
        
        # 2. Decode and validate the token
        payload = decode_access_token(auth_header)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        
        username = payload.get("sub")
        if not username:
             raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )

        # 3. Retrieve user from database
        with db_handler.get_session() as session:
            user = session.exec(select(User).where(User.username == username)).first()
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
        
        # 4. Attach user information to the request state
        # In FastAPI, request.state is the standard place for custom data
        request.state.user = user
        request.state.user_id = user.user_id
        
        # Legacy support for attributes directly on request object
        request.user_id = user.user_id  # type: ignore
        request.user_info = { # type: ignore
            "user_id": user.user_id,
            "username": user.username,
            "created_at": user.created_at,
            "updated_at": user.updated_at,
            "last_login": user.last_login
        }

        # 5. Dynamically handle arguments for the decorated function
        # This allows the route to optionally accept 'request' or 'user_id'
        sig = inspect.signature(f)
        if 'request' in sig.parameters:
            kwargs['request'] = request
        if 'user_id' in sig.parameters:
            kwargs['user_id'] = user.user_id
        
        # 6. Execute the decorated function
        if asyncio.iscoroutinefunction(f):
            return await f(*args, **kwargs)
        else:
            return f(*args, **kwargs)
    
    return decorated_function
