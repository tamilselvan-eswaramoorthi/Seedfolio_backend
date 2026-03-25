import base64
import re
import json
import uuid
import yfinance as yf
from sqlmodel import select

from datetime import datetime, timezone, timedelta

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from shared_code.pdf_processor import ExtractHoldings
from shared_code.database import db_handler
from shared_code.models import GoogleOAuthToken, User, Holdings, Transaction, IPO

class GetHoldingsFromGmail:
    def __init__(self, user_id: str):
        self.service = None
        self.user_id = user_id
        self.extractor = ExtractHoldings()
        self.SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
        self._get_user_details()
        self.authenticate()
        self.zerodha_query = "from:no-reply-transaction-with-holding-statement@reportsmailer.zerodha.net"
        self.cas_query = "from:eCAS@cdslstatement.com"
        self.nse_query = "from:nse-direct@nse.co.in"
        self.bse_query = "from:mgrpt@bseindia.com"

    def _get_user_details(self):
        with db_handler.get_session() as session:
            user = session.exec(select(User).where(User.user_id == self.user_id)).first()
            self.PASSWORD = user.pan_card # type: ignore

    def authenticate(self):
        creds = None
        with db_handler.get_session() as session:
            user_token = session.exec(select(GoogleOAuthToken).where(GoogleOAuthToken.user_id == self.user_id)).first()
            if user_token:
                creds = Credentials(
                    token=user_token.token,
                    refresh_token=user_token.refresh_token,
                    token_uri=user_token.token_uri,
                    client_id=user_token.client_id,
                    client_secret=user_token.client_secret,
                    scopes=user_token.scopes.split(",") if user_token.scopes else self.SCOPES
                )
            
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                    # Update DB with new tokens
                    token_dict = json.loads(creds.to_json())
                    user_token.token = token_dict.get("token", user_token.token) # type: ignore
                    if "expiry" in token_dict and token_dict["expiry"]:
                        user_token.expiry = datetime.fromisoformat(token_dict["expiry"].replace("Z", "+00:00")) # type: ignore
                    session.add(user_token)
                    session.commit()
                else:
                    raise Exception("No valid credentials found for user. User must re-authenticate.")
                    
        self.service = build("gmail", "v1", credentials=creds)

    def get_attachments_in_memory(self, user_id, msg_id, payload):
        attachments = []
        
        parts = [payload] if "parts" not in payload else payload["parts"]
        queue = parts[:]
        
        while queue:
            part = queue.pop(0)
            
            if part.get("filename"):
                filename = part["filename"]
                attachment_id = part["body"].get("attachmentId")
                data = part["body"].get("data")
                
                if attachment_id:
                    attachment = self.service.users().messages().attachments().get( # type: ignore
                        userId=user_id, messageId=msg_id, id=attachment_id).execute()
                    data = attachment.get("data")
                
                if data:
                    file_data = base64.urlsafe_b64decode(data.encode('UTF-8'))
                    # Store the bytes directly instead of writing to disk!
                    attachments.append({"filename": filename, "data": file_data})
                    
            if "parts" in part:
                queue.extend(part["parts"])
                
        return attachments

    def _get_body_text(self, payload):
        text = ""
        if 'body' in payload and 'data' in payload['body']:
            try:
                text = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='ignore')
                text = text.split('\n')[5]
            except:
                pass
        elif 'parts' in payload:
            for part in payload['parts']:
                text += self._get_body_text(part)
        return text

    def _get_full_body_text(self, payload):
        text = ""
        if 'body' in payload and 'data' in payload['body']:
            try:
                text = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="ignore")
            except:
                pass
        elif 'parts' in payload:
            for part in payload['parts']:
                text += self._get_full_body_text(part)
        return text

    def _detect_broker_from_email(self, msg):
        payload = msg.get("payload", {})
        body_text = self._get_full_body_text(payload).lower()
        if "zerodha" in body_text:
            return "Zerodha"
        if "groww" in body_text:
            return "Groww"
        if "angleone" in body_text or "angelone" in body_text:
            return "AngelOne"
        return None

    def _extract_nse_date(self, msg):
        payload = msg.get('payload', {})
        full_text = self._get_body_text(payload)
        
        match = re.search(r"for   (\d{1,2}-[A-Z]{3,}-\d{4})", full_text)
        if match:
            date_str = match.group(1)
            try:
                # Convert 13-MAR-2026 to 2026-03-13 for standard ISO format
                from datetime import datetime
                dt = datetime.strptime(date_str, "%d-%b-%Y")
                return dt.strftime("%Y-%m-%d")
            except:
                return date_str
        return None
        

    # ------------------------------------------------------------------
    # Core incremental processing
    # ------------------------------------------------------------------

    def _fetch_ipo_price(self, isin: str) -> float:
        with db_handler.get_session() as session:
            ipo_detail = session.exec(
                        select(IPO).where(
                            IPO.isin_code == isin
                        )
                    ).first()
            if ipo_detail:
                return ipo_detail.offer_price, ipo_detail.ipo_listing_date
        return 0, None # type: ignore

    def _calculate_holdings(self, extractions):
        if not extractions:
            return [], []

        # 1. Sort transactions by date to ensure proper order (e.g., IPO allotment before sell)
        def get_dt(item):
            dt = item.get("trade_datetime")
            if not dt:
                return datetime.min.replace(tzinfo=timezone.utc)
            if isinstance(dt, str):
                try:
                    return datetime.fromisoformat(dt.replace("Z", "+00:00")).replace(tzinfo=timezone.utc)
                except:
                    return datetime.min.replace(tzinfo=timezone.utc)
            if isinstance(dt, datetime) and dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt

        extractions.sort(key=get_dt)

        transactions_to_add = []
        holdings_cache = {} # Key: symbol, Value: holding object

        with db_handler.get_session() as session:
            # 2. Pre-fetch existing holdings for all symbols involved to minimize DB hits
            symbols = list(set(item.get("symbol").split(".")[0] for item in extractions if item.get("symbol")))
            if symbols:
                existing_holdings = session.exec(
                    select(Holdings).where(
                        Holdings.user_id == self.user_id,
                        Holdings.stock_symbol.in_(symbols) # type: ignore
                    )
                ).all()
                for h in existing_holdings:
                    holdings_cache[h.stock_symbol] = h

            for item in extractions:
                symbol = item.get("symbol")
                isin = item.get('isin')
                transaction_type = item.get("transaction_type", "BUY").upper()
                if not symbol:
                    continue
                try:
                    qty_val = float(str(item.get("quantity", 0)).replace(",", ""))
                    rate = float(str(item.get("rate", 0)).replace(",", ""))
                except ValueError:
                    continue

                trade_datetime = item.get("trade_datetime")
                base_symbol = symbol.split(".")[0]

                # 3. Add Transaction to list
                transaction = Transaction(
                    transaction_id=str(uuid.uuid4()),
                    user_id=self.user_id,
                    transaction_datetime=trade_datetime if trade_datetime else datetime.now(),
                    stock_symbol=symbol,
                    stock_name=item.get("company_name", ""),
                    transaction_type=transaction_type,
                    quantity=int(qty_val),
                    broker=item.get("broker", ""),
                    exchange=item.get("exchange", ""),
                    price=rate
                )
                transactions_to_add.append(transaction)

                # 4. Update/Create Holding in cache
                if base_symbol in holdings_cache:
                    holding = holdings_cache[base_symbol]
                    old_qty = holding.quantity
                    old_rate = holding.avg_buy
                    if transaction_type == "BUY":
                        new_qty = old_qty + int(qty_val)
                        new_rate = ((old_qty * old_rate) + (int(qty_val) * rate)) / new_qty if new_qty > 0 else 0
                        holding.quantity = new_qty
                        holding.avg_buy = float(f"{new_rate:.2f}")
                    elif transaction_type == "SELL":
                        new_qty = old_qty - int(qty_val)
                        realized_pl = (rate - old_rate) * int(qty_val)
                        transaction.realized_pl = float(f"{realized_pl:.2f}")
                        holding.quantity = new_qty
                        holding.realized_pl = float(f"{(holding.realized_pl + realized_pl):.2f}")

                    if trade_datetime:
                        holding.holding_datetime = trade_datetime
                else:
                    # For a new stock (e.g., IPO BUY), or a SELL with no history
                    if transaction_type == "BUY":
                        holding_qty = int(qty_val)
                        realized_pl = 0.0
                        avg_buy = float(f"{rate:.2f}")
                    else:
                        # SELL without history - check if it's an IPO or missing history
                        ipo_price, listing_date = self._fetch_ipo_price(isin)
                        if ipo_price > 0:
                            realized_pl = (rate - ipo_price) * int(qty_val)
                            avg_buy = ipo_price
                            ipo_transaction = Transaction(
                                transaction_id=str(uuid.uuid4()),
                                user_id=self.user_id,
                                transaction_datetime=listing_date,
                                stock_symbol=symbol,
                                stock_name=item.get("company_name", ""),
                                transaction_type='BUY',
                                quantity=int(qty_val),
                                broker=item.get("broker", ""),
                                exchange=item.get("exchange", ""),
                                price=ipo_price
                            )
                            transactions_to_add.append(ipo_transaction)

                        else:
                            # If no IPO price found, fallback to 0 gain or total value as gain
                            realized_pl = 0.0 
                            avg_buy = rate
                        
                        holding_qty = -int(qty_val) # Negative quantity if we only have the sell

                    holding = Holdings(
                        holding_id=str(uuid.uuid4()),
                        user_id=self.user_id,
                        stock_symbol=base_symbol,
                        company_name=item.get("company_name", ""),
                        quantity=holding_qty,
                        avg_buy=avg_buy,
                        realized_pl=realized_pl,
                        holding_datetime=trade_datetime if trade_datetime else datetime.now()
                    )
                    holdings_cache[base_symbol] = holding


        return transactions_to_add, list(holdings_cache.values())

    def _upload_extractions(self, transactions, holdings):
        if not transactions and not holdings:
            return

        with db_handler.get_session() as session:
            # Add all objects to session and commit once
            for t in transactions:
                session.add(t)
            for h in holdings:
                session.add(h)
            session.commit()

    def fetch_cas_details(self, after_timestamp_sec: int):
        if after_timestamp_sec is not None:
            cas_query = f"{self.cas_query} after:{after_timestamp_sec}"
        else:
            cas_query = self.cas_query
        print(f"\nQuerying CAS emails incrementally with: {cas_query}")

        cas_results = self.service.users().messages().list( # type: ignore
            userId="me", q=cas_query, maxResults=1).execute()
        holdings = None
        if "messages" in cas_results and cas_results["messages"]:
            cas_msg_ref = cas_results["messages"][0]
            cas_msg = self.service.users().messages().get(userId="me", id=cas_msg_ref["id"]).execute() # type: ignore
            cas_attachments = self.get_attachments_in_memory("me", cas_msg['id'], cas_msg.get("payload", {}))
            for att in cas_attachments:
                if str(att["filename"]).lower().endswith(".pdf"):
                    holdings = self.extractor.extract_cas_pdf(att["data"], self.PASSWORD)
        return holdings

    def fetch_incremental_nse_emails(self, after_timestamp_sec: int):
        if after_timestamp_sec is not None:
            nse_query = f"{self.nse_query} after:{after_timestamp_sec}"
        else:
            nse_query = self.nse_query
        print(f"\nQuerying NSE emails incrementally with: {nse_query}")
        
        page_token = None
        nse_messages = []
        while True:
            nse_results = self.service.users().messages().list( # type: ignore
                userId="me", q=nse_query, pageToken=page_token).execute()
            
            if "messages" in nse_results:
                nse_messages.extend(nse_results["messages"])
            
            page_token = nse_results.get("nextPageToken")
            if not page_token:
                break
        print (f"Found {len(nse_messages)} NSE emails to process incrementally.")
        all_extractions = []
        if nse_messages:
            for index, nse_msg_ref in enumerate(nse_messages):
                nse_msg = self.service.users().messages().get(userId="me", id=nse_msg_ref["id"]).execute() # type: ignore
                nse_attachments = self.get_attachments_in_memory("me", nse_msg['id'], nse_msg.get("payload", {}))

                email_sent_time = nse_msg.get("internalDate")
                email_sent_date = datetime.fromtimestamp(int(email_sent_time) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

                for att in nse_attachments:
                    if str(att["filename"]).lower().endswith(".pdf"):
                        email_date = self._extract_nse_date(nse_msg)
                        if email_date is None:
                            email_date = nse_msg.get("internalDate")
                            email_date = datetime.fromtimestamp(int(email_date) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                        nse_holdings = self.extractor.extract_nse_pdf(att["data"], self.PASSWORD, email_date)

                        if nse_holdings is not None:
                            all_extractions.extend(nse_holdings)
                print (f"Processed NSE email {index + 1}/{len(nse_messages)}")
        return all_extractions

    def fetch_incremental_bse_emails(self, after_timestamp_sec: int):
        bse_query = self.bse_query
        if after_timestamp_sec is not None:
            bse_query = f"{self.bse_query} after:{after_timestamp_sec}"
        else:
            bse_query = self.bse_query  
        
        page_token = None
        bse_messages = []
        while True:
            bse_results = self.service.users().messages().list( # type: ignore
                userId="me", q=bse_query, pageToken=page_token).execute()

            if "messages" in bse_results:
                bse_messages.extend(bse_results["messages"])

            page_token = bse_results.get("nextPageToken")
            if not page_token:
                break
        print (f"Found {len(bse_messages)} BSE emails to process incrementally.")
        all_extractions = []
        if bse_messages:
            for index, bse_msg_ref in enumerate(bse_messages):
                bse_msg = self.service.users().messages().get(userId="me", id=bse_msg_ref["id"]).execute() # type: ignore
                bse_attachments = self.get_attachments_in_memory("me", bse_msg['id'], bse_msg.get("payload", {}))

                email_sent_time = bse_msg.get("internalDate")
                email_sent_date = datetime.fromtimestamp(int(email_sent_time) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

                broker_name = self._detect_broker_from_email(bse_msg)
                for att in bse_attachments:
                    if str(att["filename"]).lower().endswith(".pdf"):
                        extractions = self.extractor.extract_bse_pdf(att["data"], self.PASSWORD, broker_name)
                        if extractions is not None:
                            all_extractions.extend(extractions)
                print (f"Processed BSE email {index + 1}/{len(bse_messages)}")
        return all_extractions
        
    def _fill_missing_transactions(self, extractions, cas_holdings):
        """ For each holding in CAS, find the corresponding extractions and group by broker """
        statement_date = cas_holdings['statement_date']
        statement_date = datetime.strptime(statement_date, "%d-%m-%Y").date()
        ret_extractions = extractions[:]
        for broker, holdings in cas_holdings.items():
            if broker in ["mutual_funds", "statement_date", "transactions", "other"]:
                continue
            for holding in holdings:
                matched_extractions = []
                company_name = None
                symbol = None
                for e in extractions:
                    if holding['isin'].lower() == e['isin'].lower():
                        company_name = e['company_name']
                        symbol = e['symbol']
                        if 'broker' in e:
                            broker_name = e['broker'].lower()
                            trade_date = e['trade_datetime']
                            if isinstance(trade_date, str):
                                trade_dt = datetime.strptime(trade_date, "%Y-%m-%d").date()
                            else:
                                trade_dt = trade_date.date()
                            if trade_dt <= statement_date:
                                if broker_name == broker:
                                    matched_extractions.append(e)
                        else:
                            print (f"Extraction without broker info: {e}")
                
                qty_ext = sum(e['quantity'] if e['transaction_type'] == 'BUY' else -e['quantity'] for e in matched_extractions)
                cost_ext = sum(float(e['rate']) * e['quantity'] if e['transaction_type'] == 'BUY' else -float(e['rate']) * e['quantity'] for e in matched_extractions)
                
                qty_diff = holding['free_bal'] - qty_ext
                
                if abs(qty_diff) > 0.01:
                    mkt_price = holding.get('market_price')
                    if not mkt_price and holding.get('value') and holding.get('free_bal'):
                        mkt_price = float(holding['value']) / float(holding['free_bal'])
                    
                    # Calculate profit contributed by known extractions
                    known_profit = 0
                    for e in matched_extractions:
                        p = (mkt_price - float(e['rate'])) * (e['quantity'] if e['transaction_type'] == 'BUY' else -e['quantity'])
                        known_profit += p

                    
                    value_diff = holding['value'] - cost_ext
                    missing_profit = value_diff - known_profit

                    approx_rate = mkt_price - (missing_profit / qty_diff)

                    # Use yfinance to find the nearest date
                    if symbol:
                        ticker_symbol = f"{symbol}.NS"
                        try:
                            # Find first transaction date or go back 1 year
                            start_dt = statement_date - timedelta(days=365)
                            if matched_extractions:
                                for e in matched_extractions:
                                    if isinstance(e['trade_datetime'], str):
                                        e['trade_datetime'] = datetime.strptime(e['trade_datetime'], "%Y-%m-%d")
                                first_trade = min(e['trade_datetime'].date() for e in matched_extractions)
                                start_dt = min(start_dt, first_trade)

                            hist = yf.download(ticker_symbol, start=start_dt, end=statement_date + timedelta(days=1), progress=False)
                            if hist is not None and not hist.empty:
                                # Find date where Close is closest to approx_rate
                                hist['diff'] = (hist['Close'] - approx_rate).abs()
                                best_match_date = hist['diff'].idxmin()
                                best_match_price = hist.loc[best_match_date, 'Close'].values if best_match_date in hist.index else approx_rate
                                
                                print (best_match_price)

                                inferred_transaction = {
                                    'company_name': company_name,
                                    'symbol': symbol,
                                    'isin': holding['isin'],
                                    'rate': round(float(best_match_price), 2),
                                    'quantity': abs(qty_diff),
                                    'trade_datetime': best_match_date.strftime("%Y-%m-%d %H:%M:%S%z"), # type: ignore
                                    'transaction_type': "BUY" if qty_diff > 0 else "SELL",
                                    'exchange': 'NSE',
                                    'broker': broker,
                                    'is_inferred': True
                                }
                                print(f"  Inferred Transaction: {inferred_transaction}")
                                ret_extractions.append(inferred_transaction)
                        except Exception as ex:
                            import traceback
                            traceback_str = traceback.format_exc()
                            print(traceback_str)
                            print(f"  Error fetching data for {ticker_symbol}: {ex}")
        return ret_extractions
                    
    def fetch_incremental_emails(self, after_timestamp_sec: int):
        cas_holdings = self.fetch_cas_details(after_timestamp_sec)
        nse_extractions = self.fetch_incremental_nse_emails(after_timestamp_sec)
        bse_extractions = self.fetch_incremental_bse_emails(after_timestamp_sec)
        
        all_extractions = nse_extractions + bse_extractions
        
        if cas_holdings:
            all_extractions = self._fill_missing_transactions(all_extractions, cas_holdings)

        if all_extractions:
            # 1. Calculate holdings and transactions
            transactions, holdings = self._calculate_holdings(all_extractions)

            # # 2. Upload to database
            self._upload_extractions(transactions, holdings)
