from google_auth_oauthlib.flow import Flow

from config import Config
from database import db_handler, User

def generate_google_oauth_url():
    SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

    with db_handler.get_session() as session:
        user = session.query(User).filter(User.username == 'Transactions').first() # type: ignore
        if user:
            state = user.user_id
        else:
            state = None

    redirect_uri = Config.REDIRECT_URI
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


