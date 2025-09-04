import logging
import json
import azure.functions as func
from sqlmodel import select
from shared_code.database import db_handler
from shared_code.models import Transaction
from shared_code.auth_decorator import auth_required
from shared_code.portfolio_utils import PortfolioAnalysis


portfolio_analysis = PortfolioAnalysis()

@auth_required
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        with db_handler.get_session() as session:
            transactions = session.exec(select(Transaction).where(Transaction.user_id == req.user_info['user_id'])).all()

            if not transactions:
                return func.HttpResponse(
                    body=json.dumps({"message": "No transactions found for this user."}),
                    mimetype='application/json',
                    status_code=200
                )

            portfolio_summary = portfolio_analysis.calculate_portfolio_performance(transactions)

            return func.HttpResponse(
                body=json.dumps(portfolio_summary, default=str),
                mimetype='application/json'
            )

    except Exception as e:
        logging.error(f"Error processing request: {e}", exc_info=True)
        return func.HttpResponse(
            f"Error processing request: {e}",
            status_code=500
        )

