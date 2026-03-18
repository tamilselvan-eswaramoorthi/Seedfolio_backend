import os.path
import base64
import re
import fitz
import json
import requests
import pandas as pd
from datetime import datetime, timezone

from sqlmodel import select
from sqlalchemy import func as sa_func

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow, Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from shared_code.zerodha import ExtractHoldings
from shared_code.database import db_handler
from shared_code.models import GoogleOAuthToken, User, Holdings

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

    def _get_user_details(self):
        with db_handler.get_session() as session:
            user = session.exec(select(User).where(User.user_id == self.user_id)).first()
            self.PASSWORD = user.pan_card

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
                    user_token.token = token_dict.get("token", user_token.token)
                    if "expiry" in token_dict and token_dict["expiry"]:
                        user_token.expiry = datetime.fromisoformat(token_dict["expiry"].replace("Z", "+00:00"))
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
                    attachment = self.service.users().messages().attachments().get(
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

    def fetch_from_emails(self, user_id):
        consolidated_holdings: dict = {}
        try:
            # 1. Fetch latest email from zerodha
            results = self.service.users().messages().list(userId="me", q=self.zerodha_query, maxResults=1).execute()
            messages = results.get("messages", [])

            z_internal_date_ms = 0

            if messages:
                msg = self.service.users().messages().get(userId="me", id=messages[0]["id"]).execute()
                z_internal_date_ms = int(msg["internalDate"])
                                
                # Fetch attachments IN-MEMORY
                attachments = self.get_attachments_in_memory("me", msg['id'], msg.get("payload", {}))
                
                for att in attachments:
                    if str(att["filename"]).lower().endswith(".pdf"):
                        print(f"Processing Zerodha PDF in-memory: {att['filename']}")
                        # Pass the raw byte stream directly to PyMuPDF!
                        zerodha_holdings = self.extractor.extract_last_zerodha_holdings(att["data"], self.PASSWORD)
                        if zerodha_holdings is not None:
                            for item in zerodha_holdings:
                                symbol = item.get("symbol")
                                if not symbol:
                                    continue
                                try:
                                    qty = float(str(item.get("quantity", 0)).replace(",", ""))
                                    rate = float(str(item.get("rate", 0)).replace(",", ""))
                                except ValueError:
                                    continue
                                
                                consolidated_holdings[symbol] = {
                                    "company_name": item.get("company_name", ""),
                                    "symbol": symbol.split(".")[0],
                                    "quantity": qty,
                                    "rate": rate,
                                    "timestamp": item.get("timestamp"),
                                }
                        else:
                            print("Failed to extract Zerodha holdings table.")
            else:
                print("No messages found from Zerodha.")
                return list(consolidated_holdings.values())

            # 2. Fetch all emails from NSE from that datetime
            with db_handler.get_session() as session:
                max_date = session.query(sa_func.max(Holdings.holding_datetime)).filter(Holdings.user_id == user_id).scalar()
                if max_date is not None:
                    after_timestamp_sec = int(max_date.replace(tzinfo=timezone.utc).timestamp())
                    nse_query = f"{self.nse_query} after:{after_timestamp_sec}"
                else:
                    nse_query = self.nse_query
            page_token = None
            nse_messages = []
            while True:
                nse_results = self.service.users().messages().list(
                    userId="me", q=nse_query, pageToken=page_token).execute()
                
                if "messages" in nse_results:
                    nse_messages.extend(nse_results["messages"])
                
                page_token = nse_results.get("nextPageToken")
                if not page_token:
                    break

            if nse_messages:
                for nse_msg_ref in nse_messages:
                    nse_msg = self.service.users().messages().get(userId="me", id=nse_msg_ref["id"]).execute()
                    
                    # Check timestamps effectively
                    if int(nse_msg["internalDate"]) < z_internal_date_ms:
                        continue
                        
                    nse_attachments = self.get_attachments_in_memory("me", nse_msg['id'], nse_msg.get("payload", {}))
                    
                    for att in nse_attachments:
                        if str(att["filename"]).lower().endswith(".pdf"):
                            email_date = self._extract_nse_date(nse_msg)
                            nse_holdings = self.extractor.extract_nse_pdf(att["data"], self.PASSWORD, email_date)
                            if nse_holdings is not None:
                                for item in nse_holdings:
                                    symbol = item.get("symbol")
                                    if not symbol:
                                        continue
                                    try:
                                        qty = float(str(item.get("quantity", 0)).replace(",", ""))
                                        rate = float(str(item.get("rate", 0)).replace(",", ""))
                                    except ValueError:
                                        continue
                                        
                                    if email_date and item.get("trade_time"):
                                        timestamp = f"{email_date} {item.get('trade_time')}"
                                        timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S %p").strftime("%Y-%m-%dT%H:%M:%SZ")
                                    else:
                                        timestamp = None

                                    print (f"stock brought on {timestamp} is {qty} of {symbol} at {rate}")
                                    
                                    if symbol in consolidated_holdings:
                                        holding = consolidated_holdings.get(symbol, {})
                                        old_qty = float(holding.get("quantity", 0.0))
                                        old_rate = float(holding.get("rate", 0.0))
                                        new_qty = old_qty + qty
                                        new_rate = ((old_qty * old_rate) + (qty * rate)) / new_qty if new_qty > 0 else 0
                                        holding["quantity"] = new_qty
                                        holding["rate"] = float(f"{new_rate:.2f}")
                                        holding["timestamp"] = timestamp
                                    else:
                                        consolidated_holdings[symbol] = {
                                            "company_name": item.get("company_name", ""),
                                            "symbol": symbol.split(".")[0],
                                            "quantity": qty,
                                            "rate": rate,
                                            "timestamp": timestamp
                                        }
                            else:
                                print("  Failed to extract NSE table.")
            else:
                print("No NSE messages found after that datetime.")

        except HttpError as error:
            print(f"An error occurred: {error}")
            
        return list(consolidated_holdings.values())

    def fetch_incremental_nse_emails(self, after_timestamp_sec: int):
        consolidated_nse = []
        nse_query = f"{self.nse_query} after:{after_timestamp_sec}"
        print(f"\nQuerying NSE emails incrementally with: {nse_query}")
        
        page_token = None
        nse_messages = []
        while True:
            nse_results = self.service.users().messages().list(
                userId="me", q=nse_query, pageToken=page_token).execute()
            
            if "messages" in nse_results:
                nse_messages.extend(nse_results["messages"])
            
            page_token = nse_results.get("nextPageToken")
            if not page_token:
                break

        if nse_messages:
            for nse_msg_ref in nse_messages:
                nse_msg = self.service.users().messages().get(userId="me", id=nse_msg_ref["id"]).execute()
                nse_attachments = self.get_attachments_in_memory("me", nse_msg['id'], nse_msg.get("payload", {}))
                
                for att in nse_attachments:
                    if str(att["filename"]).lower().endswith(".pdf"):
                        email_date = self._extract_nse_date(nse_msg)
                        nse_holdings = self.extractor.extract_nse_pdf(att["data"], self.PASSWORD, email_date)
                        if nse_holdings is not None:
                            for item in nse_holdings:
                                symbol = item.get("symbol")
                                if not symbol:
                                    continue
                                try:
                                    qty = float(str(item.get("quantity", 0)).replace(",", ""))
                                    rate = float(str(item.get("rate", 0)).replace(",", ""))
                                except ValueError:
                                    continue
                                print (email_date, item.get("trade_time"))
                                if email_date and item.get("trade_time"):
                                    timestamp = f"{email_date} {item.get('trade_time')}"
                                    timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S %p").strftime("%Y-%m-%dT%H:%M:%SZ")
                                else:
                                    timestamp = None
                                consolidated_nse.append({
                                    "company_name": item.get("company_name", ""),
                                    "symbol": symbol,
                                    "quantity": qty,
                                    "rate": rate,
                                    "timestamp": timestamp
                                })
        return consolidated_nse
