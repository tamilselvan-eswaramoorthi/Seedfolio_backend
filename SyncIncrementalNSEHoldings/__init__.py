import logging
import json
import uuid
import math
from datetime import datetime, timezone
import azure.functions as func

from sqlmodel import select
from shared_code.database import db_handler
from shared_code.models import Holdings
from shared_code.gmail import GetHoldingsFromGmail
from sqlalchemy import func as sa_func
from shared_code.auth_decorator import auth_required

@auth_required
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Triggered SyncIncrementalNSEHoldings API')
    try:
        user_id = req.user_id

        with db_handler.get_session() as session:
            # Find max date
            max_date = session.query(sa_func.max(Holdings.holding_datetime)).filter(Holdings.user_id == user_id).scalar()
            
            if max_date is not None:

                max_date = int(max_date.replace(tzinfo=timezone.utc).timestamp())

            gmail_client = GetHoldingsFromGmail(user_id=user_id)
            gmail_client.fetch_incremental_emails(max_date)

        return func.HttpResponse(
            body=json.dumps({"message": f"Successfully synchronized NSE and BSE holdings for user {user_id}."}),
            mimetype="application/json",
            status_code=200
        )
    except Exception as e:
        logging.error(str(e))
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)
