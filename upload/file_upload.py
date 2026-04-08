import io
from uuid import uuid4
import pandas as pd

from sqlmodel import select
from database import db_handler, Transaction, Stock, Holdings

def upload_transactions(broker, file, user_id):
    try: 
        df = None
        transactions = []
        if broker.lower() == "zerodha":
            if file.filename.endswith(".csv"):
                df = pd.read_csv(file.file)
            elif file.filename.endswith(".xlsx"):
                content = file.file.read()
                file_copy = io.BytesIO(content)
                df = pd.read_excel(file_copy, engine='openpyxl', sheet_name='Equity', skiprows=14)
                df = df.rename(columns={
                    'Order Execution Time': 'order_execution_time',
                    'Symbol': 'symbol',
                    'Trade Type': 'trade_type',
                    'Quantity': 'quantity',
                    'Price': 'price',
                    'Exchange': 'exchange'
                })

            else:
                return {"message": "Invalid file format"}, 400
        
        elif broker.lower() == 'groww':
            if file.filename.endswith(".xlsx"):
                content = file.file.read()
                file_copy = io.BytesIO(content)
                df = pd.read_excel(file_copy, engine='openpyxl', skiprows=5)
                df = df.rename(columns={
                    'Execution date and time': 'order_execution_time',
                    'Symbol': 'symbol',
                    'Type': 'trade_type',
                    'Quantity': 'quantity',
                    'Value': 'value',
                    'Exchange': 'exchange'
                })
                if 'quantity' in df.columns and 'value' in df.columns:
                    df['price'] = df['value'] / df['quantity']
                else:
                    return {"message": "Missing required columns in the file"}, 400

            else:
                return {"message": "Invalid file format"}, 400
        if df is not None:
            with db_handler.get_session() as session:
                for _, row in df.iterrows():
                    if pd.isna(row['order_execution_time']):
                        continue
                    stock_details = session.query(Stock).filter(Stock.nse_symbol == row['symbol']).first()
                    stock_name = stock_details.name if stock_details else ""
                    transaction = Transaction(
                        transaction_id=str(uuid4()),
                        user_id=user_id,
                        transaction_datetime=pd.to_datetime(row['order_execution_time']),
                        stock_symbol=row['symbol'],
                        stock_name=stock_name,
                        transaction_type=row['trade_type'].upper(),
                        quantity=int(row['quantity']),
                        price=float(row['price']),
                        exchange=row['exchange'],
                        broker=broker.lower(),
                        commission_local=0.0,
                        realized_pl=0.0
                    )
                    transactions.append(transaction)

                if transactions:
                    # Sort transactions by date to ensure proper holding calculation
                    transactions.sort(key=lambda x: x.transaction_datetime)

                    # Update holdings
                    # Use base symbol (without .NS/.BO) for holdings cache
                    symbols = list(set(t.stock_symbol.split(".")[0] for t in transactions))
                    holdings_cache = {}
                    if symbols:
                        existing_holdings = session.exec(
                            select(Holdings).where(
                                Holdings.user_id == user_id,
                                Holdings.stock_symbol.in_(symbols)
                            )
                        ).all()
                        for h in existing_holdings:
                            holdings_cache[h.stock_symbol] = h

                    for transaction in transactions:
                        base_symbol = transaction.stock_symbol.split(".")[0]
                        qty_val = transaction.quantity
                        rate = transaction.price
                        transaction_type = transaction.transaction_type.upper()

                        if base_symbol in holdings_cache:
                            holding = holdings_cache[base_symbol]
                            old_qty = holding.quantity
                            old_rate = holding.avg_buy
                            
                            if transaction_type == "BUY":
                                new_qty = old_qty + qty_val
                                new_rate = ((old_qty * old_rate) + (qty_val * rate)) / new_qty if new_qty > 0 else 0
                                holding.quantity = new_qty
                                holding.avg_buy = float(f"{new_rate:.2f}")
                            elif transaction_type == "SELL":
                                new_qty = old_qty - qty_val
                                realized_pl_val = (rate - old_rate) * qty_val
                                transaction.realized_pl = float(f"{realized_pl_val:.2f}")
                                holding.quantity = new_qty
                                holding.realized_pl = float(f"{(holding.realized_pl + realized_pl_val):.2f}")
                            
                            holding.holding_datetime = transaction.transaction_datetime
                            session.add(holding)
                        else:
                            if transaction_type == "BUY":
                                holding_qty = qty_val
                                avg_buy = float(f"{rate:.2f}")
                                realized_pl = 0.0
                            else:
                                # SELL without history
                                holding_qty = -qty_val
                                avg_buy = rate
                                realized_pl = 0.0
                            
                            holding = Holdings(
                                holding_id=str(uuid4()),
                                user_id=user_id,
                                stock_symbol=base_symbol,
                                company_name=transaction.stock_name,
                                quantity=holding_qty,
                                avg_buy=avg_buy,
                                realized_pl=realized_pl,
                                holding_datetime=transaction.transaction_datetime
                            )
                            holdings_cache[base_symbol] = holding
                            session.add(holding)
                        
                        session.add(transaction)
                    
                    session.commit()
                return {"message": f"Successfully processed {len(transactions)} transactions"}, 200
    except Exception as e:
        return {"message": str(e)}, 500
