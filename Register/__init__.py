import logging
import json
import uuid
from datetime import datetime
import azure.functions as func
from shared_code.database import db_handler
from shared_code.auth import get_password_hash
from shared_code.models import User
from sqlmodel import select

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request for registration.')

    try:
        body = req.get_json()
        username = body.get('username')
        password = body.get('password')

        if not username or not password:
            return func.HttpResponse(
                "Please provide both username and password.",
                status_code=400
            )

        with db_handler.get_session() as session:
            existing_user = session.exec(select(User).where(User.username == username)).first()
            if existing_user:
                return func.HttpResponse(
                    "Username already exists.",
                    status_code=409
                )

            user = User(user_id=str(uuid.uuid4()), 
                        username=username, 
                        hashed_password=get_password_hash(password),
                        created_at=datetime.now(),
                        updated_at=datetime.now()
            )

            session.add(user)
            try:
                session.commit()
                session.refresh(user)
                return func.HttpResponse(
                    body=json.dumps({"message": "User created successfully.", "user_id": user.user_id}),
                    mimetype="application/json",
                    status_code=201
                )
            except Exception as e:
                session.rollback()
                return func.HttpResponse(
                    f"Error creating user: {e}",
                    status_code=409
                )

    except Exception as e:
        return func.HttpResponse(
            f"Error during registration: {e}",
            status_code=500
        )
