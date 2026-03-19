import logging
import json
import uuid
from datetime import datetime
import azure.functions as func

from sqlmodel import select
from shared_code.database import db_handler
from shared_code.models import Holdings
from shared_code.gmail import GetHoldingsFromGmail
from shared_code.auth_decorator import auth_required

@auth_required
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Triggered FetchAndStoreHoldings API')
    try:
        user_id = req.user_id  # type: ignore
        gmail_client = GetHoldingsFromGmail(user_id=user_id)
        holdings_list = gmail_client.fetch_from_emails(user_id=user_id)

        with db_handler.get_session() as session:
            # Delete old holdings for user
            session.query(Holdings).filter(Holdings.user_id == user_id).delete()
            
            # Insert new ones
            for item in holdings_list:
                new_holding = Holdings(
                    holding_id=str(uuid.uuid4()),
                    user_id=user_id,
                    holding_datetime=item["timestamp"],
                    stock_symbol=item["symbol"],
                    company_name=item["company_name"],
                    quantity=int(item["quantity"]),
                    avg_rate=float(item["rate"])
                )
                session.add(new_holding)
            
            session.commit()

        return func.HttpResponse(
            body=json.dumps({"message": f"Successfully synced {len(holdings_list)} holdings."}),
            mimetype="application/json",
            status_code=200
        )
    except Exception as e:
        import traceback
        logging.error(str(traceback.format_exc()))
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)
