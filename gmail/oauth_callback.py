import logging
import json
import uuid
import traceback
from datetime import datetime
from sqlmodel import select
from google_auth_oauthlib.flow import Flow

from config import Config
from database import db_handler, GoogleOAuthToken

def create_token_from_code(user_id: str, code: str, redirect_uri: str = Config.REDIRECT_URI) -> str:
    SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
    flow = Flow.from_client_secrets_file(
        "credentials.json",
        scopes=SCOPES,
        redirect_uri=redirect_uri,
        autogenerate_code_verifier=False
    )
    flow.fetch_token(code=code)
    creds = flow.credentials
    token_dict = json.loads(creds.to_json())
    
    with db_handler.get_session() as session:
        expiry_date = None
        if "expiry" in token_dict and token_dict["expiry"]:
            expiry_date = datetime.fromisoformat(token_dict["expiry"].replace("Z", "+00:00"))

        existing_token = session.exec(select(GoogleOAuthToken).where(GoogleOAuthToken.user_id == user_id)).first()
        
        if existing_token:
            existing_token.token = token_dict.get("token", existing_token.token)
            if token_dict.get("refresh_token"):
                existing_token.refresh_token = token_dict["refresh_token"]
            existing_token.expiry = expiry_date
            session.add(existing_token)
        else:
            new_token = GoogleOAuthToken(
                id=str(uuid.uuid4()),
                user_id=user_id,
                token=token_dict["token"],
                refresh_token=token_dict.get("refresh_token"),
                token_uri=token_dict["token_uri"],
                client_id=token_dict["client_id"],
                client_secret=token_dict["client_secret"],
                scopes=",".join(token_dict.get("scopes", [])),
                universe_domain=token_dict.get("universe_domain", "googleapis.com"),
                account=token_dict.get("account", ""),
                expiry=expiry_date
            )
            session.add(new_token)
        
        session.commit()
    return "Token successfully stored in database!"


def exchange_code_for_token(code, state):
    logging.info('Python HTTP trigger function processed a request to exchange auth code for token.')

    try:
        if not code or not state:
            return {"error": "Please provide a valid authorization 'code' and 'user_id' (can be passed via 'state' query parameter)."}, 400

        result_msg = create_token_from_code(user_id=state, code=code)
        return {"message": result_msg}, 200

    except Exception as e:
        logging.error(traceback.format_exc())
        return {"error": f"Error exchanging OAuth code: {str(e)}"}, 500