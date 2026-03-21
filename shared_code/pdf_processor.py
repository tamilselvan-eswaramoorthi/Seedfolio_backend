import re
import fitz
import requests
import pandas as pd
from datetime import date as date_type, datetime as datetime_type, timezone
import difflib

from shared_code.corporate_actions import (
    get_demerger, get_demerger_by_raw_symbol, get_demerger_by_bse_code,
    get_split, get_bonus
)


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
            r"([\d\.]+(?:,\d+)?)\s+"
            r"([\d\.]+(?:,\d+)?)"
        )
        self.nse_columns = [
            "ISIN Code", "Company Name", "Curr. Bal", "Free Bal",
            "Pldg. Bal", "Earmark Bal", "Demat", "Remat",
            "Lockin", "Rate", "Value"
        ]
        self.bse_columns = [
            "ISIN Code", "Company Name", "Curr. Bal", "Free Bal",
            "Pldg. Bal", "Earmark Bal", "Demat", "Remat",
            "Lockin", "Rate", "Value"
        ]
        self.formats = [
            "%Y-%m-%d %H:%M:%S",
            "%d-%m-%Y %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
            "%Y-%m-%d %I:%M:%S %p",
            "%d-%m-%Y %I:%M:%S %p",
            "%Y/%m/%d %I:%M:%S %p",
            "%d/%m/%Y %I:%M:%S %p",
        ]

    # ======================================================================
    # Private helpers
    # ======================================================================

    def _parse_date_str(self, value) -> date_type:
        """
        Convert *value* (string, datetime, date, or None) to a ``date`` object.
        Falls back to today if parsing fails.
        """
        try:
            if isinstance(value, datetime_type):
                return value.date()
            if isinstance(value, date_type):
                return value
            if isinstance(value, str) and value.strip():
                s = value.strip()
                # Try ISO format first (most common from email_date)
                for fmt in ["%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y"]:
                    try:
                        return datetime_type.strptime(s, fmt).date()
                    except ValueError:
                        continue
        except Exception:
            pass
        return date_type.today()

    def _parse_trade_datetime(self, date_val, time_val, nse_fixed_fmt=False):
        """
        Build a UTC-aware datetime from *date_val* and *time_val*.

        Parameters
        ----------
        date_val      : date string or object
        time_val      : time string from the PDF row
        nse_fixed_fmt : if True, use the known NSE format "%Y-%m-%d %I:%M:%S %p"
                        instead of trying all self.formats (faster for NSE).
        """
        try:
            combined = f"{date_val} {time_val}".strip()
            if nse_fixed_fmt:
                dt = pd.to_datetime(combined, format="%Y-%m-%d %I:%M:%S %p", errors="coerce")
                if dt is not None and str(dt) != "NaT":
                    return dt.replace(tzinfo=timezone.utc)
                return None

            for fmt in self.formats:
                try:
                    dt = pd.to_datetime(combined, format=fmt, errors="coerce")
                    if dt is not None and str(dt) != "NaT":
                        return dt.replace(tzinfo=timezone.utc)
                except Exception:
                    continue
        except Exception as e:
            print(f"Error parsing trade datetime: {e}")
        return None

    def _resolve_nse_symbol(self, query: str, company_name: str):
        """
        Query Finnhub for *query* and return the best-matching NSE symbol
        and company name using string-similarity against *company_name*.

        Returns (symbol, company_name) — unchanged if no good match found.
        """
        symbol = query
        try:
            resp = requests.get(
                f"https://finnhub.io/api/v1/search?q={query}&token={self.FINNHUB_TOKEN}"
            )
            if resp.status_code == 200:
                results = resp.json().get("result", [])
                best_match = None
                best_ratio = 0.0
                for res in results:
                    sym_candidate = res.get("symbol", "")
                    desc = res.get("description", "")
                    if ".ns" in sym_candidate.lower():
                        ratio = difflib.SequenceMatcher(
                            None, str(company_name).lower(), str(desc).lower()
                        ).ratio()
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_match = res
                if best_match and best_ratio > 0.6:
                    company_name = best_match.get("description", company_name)
                    symbol = best_match.get("symbol", symbol)
        except Exception as e:
            print(f"[Finnhub/NSE] Error resolving {query}: {e}")
        return symbol, company_name

    def _resolve_bse_symbol(self, scrip_code: str, company_name: str):
        """
        Query Finnhub for *scrip_code* and return the matching BSE symbol
        (suffix .bo or .bs) and company name.

        Returns (symbol, company_name) — scrip_code unchanged if no match.
        """
        symbol = scrip_code
        try:
            resp = requests.get(
                f"https://finnhub.io/api/v1/search?q={scrip_code}&token={self.FINNHUB_TOKEN}"
            )
            if resp.status_code == 200:
                results = resp.json().get("result", [])
                for res in results:
                    sym_candidate = res.get("symbol", "")
                    if ".bo" in sym_candidate.lower() or ".bs" in sym_candidate.lower():
                        company_name = res.get("description", company_name)
                        symbol = res.get("symbol", symbol)
                        break   # first BSE match is sufficient
        except Exception as e:
            print(f"[Finnhub/BSE] Error resolving {scrip_code}: {e}")
        return symbol, company_name

    def _expand_demerger_children(
        self, demerger: dict, common_fields: dict, symbol_key: str = "symbol"
    ) -> list:
        """
        Convert one demerger registry entry into a list of child row dicts.

        Parameters
        ----------
        demerger      : entry from DEMERGER_REGISTRY
        common_fields : dict with company_name, quantity, rate, trade_datetime,
                        transaction_type, exchange already populated
        symbol_key    : which child field to use as the output symbol
                        ("symbol" for NSE, "bse_symbol" for BSE — falls back to "symbol")
        """
        rows = []
        qty   = common_fields["quantity"]
        price = float(common_fields["rate"])
        for child in demerger["children"]:
            child_symbol = child.get(symbol_key) or child["symbol"]
            child_row = dict(common_fields)
            child_row["symbol"]       = child_symbol
            child_row["company_name"] = child.get("company_name", common_fields["company_name"])
            child_row["quantity"]     = int(qty * child["ratio"])
            child_row["rate"]         = round(price * child["price_ratio"] / child["ratio"], 2)
            print(
                f"[Demerger-PreFinnhub/{common_fields['exchange']}] "
                f"{common_fields.get('_raw', '?')} → {child_symbol} "
                f"({child_row['company_name']}): "
                f"qty={child_row['quantity']}, rate={child_row['rate']}"
            )
            rows.append(child_row)
        return rows

    # ======================================================================
    # Corporate-action expansion (post-Finnhub, for splits / bonuses)
    # ======================================================================

    def _expand_corporate_actions(self, row: dict, trade_date_str) -> list:
        """
        Check whether the already-resolved *row['symbol']* is subject to a
        split or bonus issue.  Demergers are handled pre-Finnhub via
        ``_expand_demerger_children``; this method is for the remaining cases.

        Returns a list of row dicts (normally just ``[row]``).
        """
        symbol = row.get("symbol", "")
        qty    = row.get("quantity", 0)
        price  = row.get("rate", 0)
        trade_date = self._parse_date_str(trade_date_str)

        # ── Split ──────────────────────────────────────────────────────────
        split = get_split(symbol, trade_date)
        if split:
            ratio     = split["ratio"]
            split_row = dict(row)
            split_row["quantity"] = int(qty * ratio)
            split_row["rate"]     = round(float(price) / ratio, 2)
            print(f"[Split] {symbol}: qty {qty}→{split_row['quantity']}, rate {price}→{split_row['rate']}")
            return [split_row]

        # ── Bonus ──────────────────────────────────────────────────────────
        bonus_rec = get_bonus(symbol, trade_date)
        if bonus_rec:
            bonus_qty = int(qty / bonus_rec["per"]) * bonus_rec["bonus"]
            if bonus_qty > 0:
                bonus_row = dict(row)
                bonus_row["quantity"] = bonus_qty
                bonus_row["rate"]     = 0.0
                print(f"[Bonus] {symbol}: +{bonus_qty} bonus shares at rate=0")
                return [row, bonus_row]

        return [row]

    # ======================================================================
    # PDF extraction helpers
    # ======================================================================

    def _open_doc(self, pdf_source, password):
        """Open a PDF from bytes or a file path, authenticating if encrypted."""
        if isinstance(pdf_source, bytes):
            doc = fitz.open(stream=pdf_source, filetype="pdf")
        else:
            doc = fitz.open(pdf_source)
        if doc.is_encrypted:
            doc.authenticate(password)
        return doc

    # ======================================================================
    # Public extraction methods
    # ======================================================================

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
            df = pd.DataFrame(holdings, columns=self.nse_columns)
            df['holding_date'] = holding_date
            for index, row in df.iterrows():
                try:
                    resp = requests.get(
                        f"https://finnhub.io/api/v1/search?q={row['ISIN Code']}&token={self.FINNHUB_TOKEN}"
                    )
                    if resp.status_code == 200 and resp.json().get('result'):
                        result = resp.json()['result']
                        if result:
                            df.at[index, "Company Name"] = result[0]['description']
                            df.at[index, "Symbol"]       = result[0]['symbol']
                except Exception as e:
                    print(f"Error fetching company name: {e}")
            df = df[['holding_date', 'Company Name', 'Symbol', 'Rate', 'Curr. Bal']]
            df.rename(columns={
                'holding_date': 'timestamp', 'Company Name': 'company_name',
                'Symbol': 'symbol', 'Rate': 'rate', 'Curr. Bal': 'quantity'
            }, inplace=True)
            return df.to_dict(orient="records")
        return None

    def extract_nse_pdf(self, pdf_source, password, email_date=None):
        """Extract trade rows from an NSE contract note PDF."""
        doc = self._open_doc(pdf_source, password)
        all_tables = []
        for page in doc:
            tabs = page.find_tables()
            if tabs and tabs.tables:
                for tab in tabs:
                    headers = [str(col).replace('\n', '') for col in tab.header.names]
                    if any("Trade No" in h or "Symbol" in h for h in headers) and len(headers) >= 10:
                        all_tables.append(tab.to_pandas())
        doc.close()

        if not all_tables:
            return None

        final_df = pd.concat(all_tables, ignore_index=True)
        final_df.columns = [str(c).replace('\n', ' ').strip() for c in final_df.columns]
        check_date = self._parse_date_str(email_date)

        rows = []
        for _, row in final_df.iterrows():
            try:
                raw_symbol   = str(row['Symbol']).strip()
                company_name = row.get('Name of the Security', raw_symbol)
                buy_or_sell  = "BUY" if row['Buy/ Sell'] == "B" else "SELL"
                try:
                    quantity = int(row['Quantity'])
                except (ValueError, TypeError):
                    quantity = 0
                price          = row['Price (Rs.)']
                trade_datetime = self._parse_trade_datetime(email_date, row['Trade Time'], nse_fixed_fmt=True)

                # ── Pre-Finnhub: demerger check ───────────────────────────
                early_demerger = get_demerger_by_raw_symbol(raw_symbol, check_date)
                if early_demerger:
                    common = {
                        "company_name":     company_name,
                        "quantity":         quantity,
                        "rate":             price,
                        "trade_datetime":   trade_datetime,
                        "transaction_type": buy_or_sell,
                        "exchange":         "NSE",
                        "_raw":             raw_symbol,   # for logging only
                    }
                    rows.extend(self._expand_demerger_children(early_demerger, common, symbol_key="symbol"))
                    continue  # skip Finnhub

                # ── Normal path: Finnhub symbol resolution ────────────────
                symbol, company_name = self._resolve_nse_symbol(raw_symbol, company_name)
                base_row = {
                    "company_name":     company_name,
                    "symbol":           symbol,
                    "rate":             price,
                    "quantity":         quantity,
                    "trade_datetime":   trade_datetime,
                    "transaction_type": buy_or_sell,
                    "exchange":         "NSE",
                }
                rows.extend(self._expand_corporate_actions(base_row, email_date))

            except Exception as e:
                print(f"[NSE] Error processing row: {e}")

        return rows

    def extract_bse_pdf(self, pdf_source, password):
        """Extract trade rows from a BSE contract note PDF."""
        doc = self._open_doc(pdf_source, password)
        all_tables = []
        for page in doc:
            tabs = page.find_tables()
            if tabs and tabs.tables:
                for tab in tabs:
                    headers = [str(col).replace('\n', '') for col in tab.header.names]
                    if len(headers) == 12:
                        all_tables.append(tab.to_pandas())
        doc.close()

        if not all_tables:
            return None

        final_df = pd.concat(all_tables, ignore_index=True)
        final_df.columns = [str(c).replace('\n', ' ').strip() for c in final_df.columns]

        rows = []
        for _, row in final_df.iterrows():
            try:
                if not str(row.get('Scrip Name', '')).strip():
                    continue

                company_name = str(row['Scrip Name']).replace('\n', ' ').strip()
                scrip_code   = str(row['Scrip Code']).strip()
                buy_or_sell  = "BUY" if row['Buy Sell'] == "B" else "SELL"
                try:
                    quantity = int(row['Qty'])
                except (ValueError, TypeError):
                    quantity = 0
                price          = row['Price']
                trade_datetime = self._parse_trade_datetime(row['Trade Date'], row['Trade Time'])
                bse_trade_date = self._parse_date_str(row['Trade Date'])

                # ── Pre-Finnhub: demerger check ───────────────────────────
                early_demerger = get_demerger_by_bse_code(scrip_code, bse_trade_date)
                if early_demerger:
                    common = {
                        "company_name":     company_name,
                        "quantity":         quantity,
                        "rate":             price,
                        "trade_datetime":   trade_datetime,
                        "transaction_type": buy_or_sell,
                        "exchange":         "BSE",
                        "_raw":             scrip_code,   # for logging only
                    }
                    rows.extend(self._expand_demerger_children(early_demerger, common, symbol_key="bse_symbol"))
                    continue  # skip Finnhub

                # ── Normal path: Finnhub symbol resolution ────────────────
                symbol, company_name = self._resolve_bse_symbol(scrip_code, company_name)
                base_row = {
                    "company_name":     company_name,
                    "symbol":           symbol,
                    "rate":             price,
                    "quantity":         quantity,
                    "trade_datetime":   trade_datetime,
                    "transaction_type": buy_or_sell,
                    "exchange":         "BSE",
                }
                rows.extend(self._expand_corporate_actions(base_row, None))

            except Exception as e:
                print(f"[BSE] Error processing row: {e}")

        return rows
