import re
import logging
import uuid
import traceback
from datetime import datetime
from sqlmodel import select

from config import Config
from database import db_handler, User

def get_password_hash(password):
    return Config.PWD_CONTEXT.hash(password.strip())

def register_user(body: dict):
    """
    Standard registration logic, independent of web framework.
    """
    logging.info('Processing registration request.')

    try:
        username = body.get('username')
        email = body.get('email')
        password = body.get('password')
        pan_card = body.get('pan_card')

        if not username or not password or not email or not pan_card:
            return {"error": "Please provide username, password, email, and pan_card."}, 400

        # email validation
        if not re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", email):
            return {"error": "Invalid email address."}, 400

        with db_handler.get_session() as session:
            existing_user = session.exec(select(User).where(User.email == email)).first()
            if existing_user:
                return {"error": "Email already exists."}, 409

            user = User(
                user_id=str(uuid.uuid4()), 
                username=username, 
                email=email,
                pan_card=pan_card,
                hashed_password=get_password_hash(password),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )

            session.add(user)
            try:
                session.commit()
                session.refresh(user)
                return {"message": "User created successfully.", "user_id": user.user_id}, 201
            except Exception as e:
                session.rollback()
                return {"error": f"Error creating user: {e}"}, 409

    except Exception as e:
        logging.error(f"Error during registration: {traceback.format_exc()}")
        return {"error": f"Internal server error: {str(e)}"}, 500
