import logging
import json
from datetime import datetime
import azure.functions as func
from sqlmodel import select
from shared_code.database import db_handler
from shared_code.auth import verify_password, create_access_token
from shared_code.models import User

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request for login.')

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
            user = session.exec(select(User).where(User.username == username)).first()

            if user and verify_password(password, user.hashed_password):
                access_token = create_access_token(data={"sub": username})
                #update last login time
                user.last_login = datetime.now()
                session.add(user)
                session.commit()
                session.refresh(user)
                return func.HttpResponse(
                    body=json.dumps({"access_token": access_token, "token_type": "bearer"}),
                    mimetype="application/json"
                )
            else:
                return func.HttpResponse(
                    "Invalid username or password.",
                    status_code=401
                )

    except Exception as e:
        return func.HttpResponse(
            f"Error during login: {e}",
            status_code=500
        )
