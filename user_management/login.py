import logging
from datetime import datetime, timedelta
from sqlmodel import select
from jose import jwt

from config import Config
from database import db_handler, User

def verify_password(plain_password, hashed_password):
    return Config.PWD_CONTEXT.verify(plain_password, hashed_password)

def get_password_hash(password):
    return Config.PWD_CONTEXT.hash(password.strip())

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(seconds=int(Config.JWT_EXP_DELTA_SECONDS))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, Config.JWT_SECRET, algorithm=Config.JWT_ALGORITHM)
    return encoded_jwt

def login_user(body: dict):
    """
    Standard login logic, independent of web framework.
    """
    logging.info('Processing login request.')

    username = body.get('username')
    password = body.get('password')

    if not username or not password:
        return {"error": "Please provide both username and password."}, 400

    with db_handler.get_session() as session:
        user = session.exec(select(User).where(User.username == username)).first()

        if user and verify_password(password, user.hashed_password):
            access_token = create_access_token(data={"sub": username})
            
            # update last login time
            user.last_login = datetime.now()
            session.add(user)
            session.commit()
            session.refresh(user)
            
            return {"access_token": access_token, "token_type": "bearer"}, 200
        else:
            return {"error": "Invalid username or password."}, 401
