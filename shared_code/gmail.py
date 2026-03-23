import base64
import logging
from math import e
import re
import json
import time
import uuid
from datetime import datetime, timezone

from sqlmodel import select

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from shared_code.pdf_processor import ExtractHoldings
from shared_code.database import db_handler
from shared_code.models import GoogleOAuthToken, User, Holdings, Transaction

class GetHoldingsFromGmail:
    def __init__(self, user_id: str):
        self.service = None
        self.user_id = user_id
        self.extractor = ExtractHoldings()
        self.SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
        self._get_user_details()
        self.authenticate()
        self.zerodha_query = "from:no-reply-transaction-with-holding-statement@reportsmailer.zerodha.net"
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

    def _process_extractions(self, extractions):
        if not extractions:
            return

        transactions_to_add = []
        holdings_cache = {} # Key: symbol, Value: holding object

        with db_handler.get_session() as session:
            # 1. Pre-fetch existing holdings for all symbols involved to minimize DB hits
            symbols = list(set(item.get("symbol").split(".")[0] for item in extractions if item.get("symbol")))
            if symbols:
                existing_holdings = session.exec(
                    select(Holdings).where(
                        Holdings.user_id == self.user_id,
                        Holdings.stock_symbol.in_(symbols)
                    )
                ).all()
                for h in existing_holdings:
                    holdings_cache[h.stock_symbol] = h

            for item in extractions:
                symbol = item.get("symbol")
                transaction_type = item.get("transaction_type", "BUY").upper()
                if not symbol:
                    continue
                try:
                    qty = float(str(item.get("quantity", 0)).replace(",", ""))
                    rate = float(str(item.get("rate", 0)).replace(",", ""))
                except ValueError:
                    continue

                if qty <= 0 or rate <= 0:
                    continue

                trade_datetime = item.get("trade_datetime")
                base_symbol = symbol.split(".")[0]

                # 1. Add Transaction to list
                transaction = Transaction(
                    transaction_id=str(uuid.uuid4()),
                    user_id=self.user_id,
                    transaction_datetime=trade_datetime if trade_datetime else datetime.now(),
                    stock_symbol=symbol,
                    stock_name=item.get("company_name", ""),
                    transaction_type=transaction_type,
                    quantity=int(qty),
                    broker=item.get("broker", ""),
                    exchange=item.get("exchange", ""),
                    price=rate
                )
                transactions_to_add.append(transaction)

                # 2. Update/Create Holding in cache
                if base_symbol in holdings_cache:
                    holding = holdings_cache[base_symbol]
                    old_qty = holding.quantity
                    old_rate = holding.avg_buy
                    if transaction_type == "BUY":
                        new_qty = old_qty + int(qty)
                        new_rate = ((old_qty * old_rate) + (int(qty) * rate)) / new_qty if new_qty > 0 else 0
                        holding.quantity = new_qty
                        holding.avg_buy = float(f"{new_rate:.2f}")
                    elif transaction_type == "SELL":
                        new_qty = old_qty - int(qty)
                        realized_pl = (rate - old_rate) * int(qty)
                        holding.quantity = new_qty
                        holding.realized_pl = (
                            holding.realized_pl + realized_pl if holding.realized_pl else realized_pl
                        )
                    if trade_datetime:
                        holding.holding_datetime = trade_datetime
                else:
                    holding = Holdings(
                        holding_id=str(uuid.uuid4()),
                        user_id=self.user_id,
                        stock_symbol=base_symbol,
                        company_name=item.get("company_name", ""),
                        quantity=int(qty) if transaction_type == "BUY" else -int(qty),
                        avg_buy=rate,
                        realized_pl=0.0,
                        holding_datetime=trade_datetime if trade_datetime else datetime.now()
                    )
                    holdings_cache[base_symbol] = holding

            # 3. Add all objects to session and commit once
            for t in transactions_to_add:
                session.add(t)
            for h in holdings_cache.values():
                session.add(h)
            session.commit()

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

                for att in bse_attachments:
                    if str(att["filename"]).lower().endswith(".pdf"):
                        extractions = self.extractor.extract_bse_pdf(att["data"], self.PASSWORD)
                        if extractions is not None:
                            all_extractions.extend(extractions)
                print (f"Processed BSE email {index + 1}/{len(bse_messages)}")
        return all_extractions
        
    def fetch_incremental_emails(self, after_timestamp_sec: int):
        nse_extractions = self.fetch_incremental_nse_emails(after_timestamp_sec)
        bse_extractions = self.fetch_incremental_bse_emails(after_timestamp_sec)
        
        all_extractions = nse_extractions + bse_extractions
        logging.info(f"Total extractions from NSE and BSE emails: {len(all_extractions)}")
        if all_extractions:
            self._process_extractions(all_extractions)
