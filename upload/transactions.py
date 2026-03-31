import logging
import traceback
from sqlalchemy import func
from datetime import timezone

from database import db_handler, Transaction
from shared_code.gmail import GetHoldingsFromGmail

def sync_transactions(user_id) -> tuple[dict, int]:
    logging.info('Triggered SyncIncrementalNSEHoldings API')
    try:
        with db_handler.get_session() as session:
            # Find max date
            max_date = session.query(func.max(Transaction.transaction_datetime)).filter(Transaction.user_id == user_id).scalar()

            if max_date is not None:

                max_date = int(max_date.replace(tzinfo=timezone.utc).timestamp())

            gmail_client = GetHoldingsFromGmail(user_id=user_id)
            gmail_client.fetch_incremental_emails(max_date)

        return {"message": f"Successfully synchronized NSE and BSE transactions for user {user_id}."}, 200
    except Exception as e:
        traceback_str = traceback.format_exc()
        logging.error(traceback_str)
        return {"error": f"Failed to sync transactions: {str(e)}"}, 500
