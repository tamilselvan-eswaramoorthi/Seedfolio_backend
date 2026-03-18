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
            
            if not max_date:
                # Fallback if no prior run
                return func.HttpResponse("No existing holdings found. Run full sync first.", status_code=400)

            after_timestamp_sec = int(max_date.replace(tzinfo=timezone.utc).timestamp())

            gmail_client = GetHoldingsFromGmail(user_id=user_id)
            nse_items = gmail_client.fetch_incremental_nse_emails(after_timestamp_sec)
            logging.info (nse_items)

            processed_count = 0
            for item in nse_items:
                symbol = item["symbol"]
                qty = float(item["quantity"])
                rate = float(item["rate"])

                existing = session.exec(select(Holdings).where(Holdings.user_id == user_id, Holdings.stock_symbol == symbol)).first()
                if existing:
                    old_qty = existing.quantity
                    old_rate = existing.avg_rate
                    new_qty = old_qty + qty
                    new_rate = ((old_qty * old_rate) + (qty * rate)) / new_qty if new_qty > 0 else 0
                    
                    existing.quantity = int(new_qty)
                    existing.avg_rate = float(f"{new_rate:.2f}")
                    existing.holding_datetime = item["timestamp"]
                    session.add(existing)
                else:
                    new_holding = Holdings(
                        holding_id=str(uuid.uuid4()),
                        user_id=user_id,
                        holding_datetime=item["timestamp"],
                        stock_symbol=symbol,
                        company_name=item["company_name"],
                        quantity=int(qty),
                        avg_rate=float(f"{rate:.2f}")
                    )
                    session.add(new_holding)
                processed_count += 1

            session.commit()

        return func.HttpResponse(
            body=json.dumps({"message": f"Successfully merged {processed_count} incremental NSE rows."}),
            mimetype="application/json",
            status_code=200
        )
    except Exception as e:
        logging.error(str(e))
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)
