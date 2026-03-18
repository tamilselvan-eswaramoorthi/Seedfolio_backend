import os.path
import base64
import re
import fitz
import requests
import pandas as pd


class ExtractHoldings:
    def __init__(self):
        self.FINNHUB_TOKEN = "d6t7lt9r01qoqoisi740d6t7lt9r01qoqoisi74g"
        self.pattern = re.compile(
            r"(IN[A-Z0-9]{10})\s+"        
            r"(.*?)\s+"                   
            r"([\d\.]+)\s+"               
            r"([\d\.]+)\s+"               
            r"([\d\.]+)\s+"               
            r"([\d\.]+)\s+"               
            r"([\d\.]+)\s+"               
            r"([\d\.]+)\s+"               
            r"([\d\.]+)\s+"               
            r"([\d\.]+(?:,\d+)?)\s+"      
            r"([\d\.]+(?:,\d+)?)"         
        )
        self.columns = [
            "ISIN Code", "Company Name", "Curr. Bal", "Free Bal", 
            "Pldg. Bal", "Earmark Bal", "Demat", "Remat", 
            "Lockin", "Rate", "Value"
        ]

    def _open_doc(self, pdf_source, password):
        # Allow processing directly from a byte stream (in-memory) instead of file path
        if isinstance(pdf_source, bytes):
            doc = fitz.open(stream=pdf_source, filetype="pdf")
        else:
            doc = fitz.open(pdf_source)
            
        if doc.is_encrypted:
            doc.authenticate(password)
        return doc

    def extract_last_zerodha_holdings(self, pdf_source, password):
        doc = self._open_doc(pdf_source, password)
        holdings = []
        holding_date = None
        
        for page in doc:
            text = page.get_text("text").replace('\n', ' ')
            
            if not holding_date:
                date_match = re.search(r"Holdings as on (\d{4}-\d{2}-\d{2})", text)
                if date_match:
                    holding_date = date_match.group(1)

            matches = self.pattern.findall(text)
            for match in matches:
                holdings.append(match)

        doc.close()

        if holdings:
            df = pd.DataFrame(holdings, columns=self.columns)
            df['holding_date'] = holding_date
            
            for index, row in df.iterrows():
                try:
                    response = requests.get(f"https://finnhub.io/api/v1/search?q={row['ISIN Code']}&token={self.FINNHUB_TOKEN}")
                    if response.status_code == 200 and response.json().get('result'):
                        response = response.json()
                        if response['result']:
                            company_name = response['result'][0]['description']
                            symbol = response['result'][0]['symbol']
                            df.at[index, "Company Name"] = company_name
                            df.at[index, "Symbol"] = symbol
                except Exception as e:
                    print(f"Error fetching exact company name: {e}")
            df = df[['holding_date', 'Company Name', 'Symbol', 'Rate', 'Curr. Bal']]
            df.rename(columns={'holding_date': 'timestamp', 'Company Name': 'company_name', 'Symbol': 'symbol', 'Rate': 'rate', 'Curr. Bal': 'quantity'}, inplace=True)
            return df.to_dict(orient="records")
        return None

    def extract_nse_pdf(self, pdf_source, password, email_date=None):
        doc = self._open_doc(pdf_source, password)
        all_tables = []
        for page in doc:
            tabs = page.find_tables()
            if tabs and tabs.tables:
                for tab in tabs:
                    headers = [str(col).replace('\n', '') for col in tab.header.names]
                    if any("Trade No" in h or "Symbol" in h for h in headers) and len(headers) >= 10:
                        df = tab.to_pandas()
                        all_tables.append(df)
        
        doc.close()
        
        if all_tables:
            final_df = pd.concat(all_tables, ignore_index=True)
            final_df.columns = [str(c).replace('\n', ' ').strip() for c in final_df.columns]

            rows = []
            for index, row in final_df.iterrows():
                try:
                    response = requests.get(f"https://finnhub.io/api/v1/search?q={row['Symbol']}&token={self.FINNHUB_TOKEN}")
                    if response.status_code == 200 and response.json().get('result'):
                        response = response.json()
                        if response['result']:
                            company_name = response['result'][0]['description']
                            trade_time = row['Trade Time']
                            rows.append({
                                "company_name": company_name, 
                                "symbol": row['Symbol'], 
                                "rate": row['Price (Rs.)'], 
                                "quantity": row['Quantity'],
                                "trade_time": trade_time,
                                "trade_date": email_date
                            })
                except Exception as e:
                    print(f"Error fetching exact company name: {e}")
            return rows
        return None
