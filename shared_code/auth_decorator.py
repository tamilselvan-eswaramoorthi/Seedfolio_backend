import logging
from functools import wraps
import azure.functions as func
from sqlmodel import select
from shared_code.database import db_handler
from shared_code.auth import decode_access_token
from shared_code.models import User

def auth_required(f):
    @wraps(f)
    def decorated_function(req: func.HttpRequest, *args, **kwargs):
        logging.info('Executing authentication decorator.')
        
        token = req.headers.get('Authorization')
        if not token:
            return func.HttpResponse("Unauthorized", status_code=401)
        
        try:
            token = token.split(" ")[1]
            payload = decode_access_token(token)
            if not payload:
                return func.HttpResponse("Invalid token", status_code=401)
            username = payload.get("sub")
        except Exception as e:
            return func.HttpResponse(f"Token error: {e}", status_code=401)

        with db_handler.get_session() as session:
            user = session.exec(select(User).where(User.username == username)).first()
            if not user:
                return func.HttpResponse("User not found", status_code=404)
        
        # Attach user to the request object
        req.user_info = {
            "user_id": user.user_id,
            "username": user.username,
            "created_at": user.created_at,
            "updated_at": user.updated_at,
            "last_login": user.last_login
        }
        return f(req, *args, **kwargs)
    
    return decorated_function
