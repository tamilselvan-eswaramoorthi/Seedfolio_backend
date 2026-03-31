import logging
import time
import ijson
import requests
import datetime
import traceback

from database import db_handler, IPO

def sync_ipo_details():
    logging.info('Triggered SyncIncrementalNSEHoldings API')
    try:
        ipo_records = []
        for draw, start in enumerate(range(0, 800, 10)):
            length = 10
            timestamp = int(time.time() * 1000)

            url = f"""https://www.ipoplatform.com/main-board/index?draw={draw + 1}&columns%5B0%5D%5Bdata%5D=company_link&columns%5B0%5D%5Bname%5D=company_name&columns%5B0%5D%5Bsearchable%5D=true&columns%5B0%5D%5Borderable%5D=true&columns%5B0%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B0%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B1%5D%5Bdata%5D=ipo_year_mb&columns%5B1%5D%5Bname%5D=ipo_year&columns%5B1%5D%5Bsearchable%5D=true&columns%5B1%5D%5Borderable%5D=true&columns%5B1%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B1%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B2%5D%5Bdata%5D=city&columns%5B2%5D%5Bname%5D=company_location&columns%5B2%5D%5Bsearchable%5D=true&columns%5B2%5D%5Borderable%5D=true&columns%5B2%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B2%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B3%5D%5Bdata%5D=sector&columns%5B3%5D%5Bname%5D=sectors.name&columns%5B3%5D%5Bsearchable%5D=true&columns%5B3%5D%5Borderable%5D=true&columns%5B3%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B3%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B4%5D%5Bdata%5D=ipo_sizee&columns%5B4%5D%5Bname%5D=ipo_size&columns%5B4%5D%5Bsearchable%5D=true&columns%5B4%5D%5Borderable%5D=true&columns%5B4%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B4%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B5%5D%5Bdata%5D=pe%20Ratio&columns%5B5%5D%5Bname%5D=price_to_earning&columns%5B5%5D%5Bsearchable%5D=true&columns%5B5%5D%5Borderable%5D=true&columns%5B5%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B5%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B6%5D%5Bdata%5D=merchant_banker_mainboard&columns%5B6%5D%5Bname%5D=merchant_banker_mainboard&columns%5B6%5D%5Bsearchable%5D=true&columns%5B6%5D%5Borderable%5D=true&columns%5B6%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B6%5D%5Bsearch%5D%5Bregex%5D=false&start={start}&length={length}&search%5Bvalue%5D=&search%5Bregex%5D=false&ipo_type=MainBoard&_={timestamp}"""

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
                "X-Requested-With": "XMLHttpRequest",
                "Referer": "https://www.ipoplatform.com/main-board/all-ipo-list"
            }
            response = requests.get(url, headers=headers, timeout=20)
            if response.status_code == 200:
                data = ijson.items(response.text, 'data.item')
                if data:
                    try:
                        for record in data:
                            nse_symbol = record['nse_script_symbol']
                            bse_symbol = record['bse_script_code']
                            isin_code = record['isin']
                            company_name = record['company_name']
                            offer_price = record['offer_price']
                            ipo_year = record['ipo_year']

                            ipo_records.append(IPO(
                                nse_symbol=nse_symbol,
                                company_name=company_name,
                                offer_price=offer_price,
                                isin_code=isin_code,
                                bse_symbol=bse_symbol,
                                ipo_listing_date=ipo_year,
                                last_updated=datetime.datetime.now()
                            )) # type: ignore
                    except Exception as e:
                        logging.error(f"Error processing record: {str(e)}")
        if ipo_records:
            db_handler.bulk_save(ipo_records)

        return {"message": f"Successfully synced {len(ipo_records)} IPO records from NSE."}, 200
    except Exception as e:
        traceback_str = traceback.format_exc()
        logging.error(traceback_str)
        return {"error": f"Failed to sync IPO records: {str(e)}"}, 500
