import logging
import json
import azure.functions as func
from google_auth_oauthlib.flow import Flow

from config import Config
from shared_code.auth_decorator import auth_required

def generate_google_oauth_url(redirect_uri, state):
    SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
    flow = Flow.from_client_secrets_file(
        "credentials.json",
        scopes=SCOPES,
        redirect_uri=redirect_uri,
        autogenerate_code_verifier=False
    )
    
    kwargs = {'prompt': 'consent', 'access_type': 'offline'}
    if state:
        kwargs['state'] = state
        
    auth_url, _ = flow.authorization_url(**kwargs)
    return auth_url

@auth_required
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request to generate Gmail OAuth URL.')

    try:
        url = generate_google_oauth_url(redirect_uri=Config.REDIRECT_URI, state=req.user_id) # type: ignore
        return func.HttpResponse(
            body=json.dumps({"oauth_url": url}),
            mimetype="application/json",
            status_code=200
        )
    except Exception as e:
        logging.error(f"Error generating OAuth URL: {str(e)}")
        return func.HttpResponse(
            f"Error generating OAuth URL: {str(e)}",
            status_code=500
        )
