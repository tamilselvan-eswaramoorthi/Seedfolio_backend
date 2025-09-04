import logging
import json
import pandas as pd
from datetime import datetime
import azure.functions as func
from shared_code.database import db_handler
from shared_code.models import Transaction
from shared_code.auth_decorator import auth_required

@auth_required
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    user_id = req.user_info['user_id']

    try:
        files = req.files.values()
        file = next(files, None)

        if not file:
            return func.HttpResponse(
                "Please upload a file.",
                status_code=400
            )

        df = pd.read_csv(file)
        
        with db_handler.get_session() as session:
            for index, row in df.iterrows():
                logging.info(f"Processing row {index}: {row.to_dict()}")
                row['transaction_date'] = datetime.fromisoformat(row['transaction_date'])
                transaction = Transaction(
                    transaction_id=row['tl_id'],
                    user_id=user_id,
                    transaction_datetime=row['transaction_date'],
                    stock_symbol=row['ticker'],
                    transaction_type=row['action'],
                    quantity=row['shares'],
                    price=row['price_local'],
                    commission_local=row['commission_local'],
                    processed=row['processed']
                )
                session.add(transaction)
            
            session.commit()
        
        return func.HttpResponse(
            body=json.dumps({"message": "Transactions uploaded successfully."}),
            mimetype='application/json'
        )

    except Exception as e:
        return func.HttpResponse(
            f"Error processing file: {e}",
            status_code=500
        )
