import logging
import json
import azure.functions as func
from sqlmodel import select
from shared_code.database import db_handler
from shared_code.models import Holdings, Transaction
from shared_code.auth_decorator import auth_required
from shared_code.portfolio_utils import PortfolioAnalysis

portfolio_analysis = PortfolioAnalysis()

@auth_required
def main(req: func.HttpRequest) -> func.HttpResponse:
    # try:
        with db_handler.get_session() as session:
            holdings = session.exec(select(Holdings).where(Holdings.user_id == req.user_info['user_id'])).all()

            if not holdings:
                return func.HttpResponse(
                    body=json.dumps({"message": "No holdings found for this user."}),
                    mimetype='application/json',
                    status_code=200
                )

        # Get current holdings
        current_holdings = portfolio_analysis.get_current_holdings(holdings)
        
        return func.HttpResponse(
            json.dumps(current_holdings),
            mimetype="application/json",
            status_code=200
        )

    # except Exception as e:
    #     logging.error(f"Error in GetCurrentHoldings: {e}")
    #     return func.HttpResponse("Error processing request", status_code=500)
