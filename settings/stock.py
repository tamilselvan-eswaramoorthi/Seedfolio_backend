import logging
import pandas as pd
import requests
import io
import numpy as np
from datetime import datetime
from sqlmodel import select
from fastapi import UploadFile
from database import db_handler, Stock

import asyncio

async def sync_market_data(params: dict, file: UploadFile):
    logging.info('SyncStockData function processed a request.')

    # Get data type from request (default: stock)
    data_type = params.get('type', 'stock').lower()
    if data_type not in ['stock', 'mf']:
        return {"message": "Invalid 'type' parameter. Must be 'stock' or 'mf'."}, 400

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        # 1. Fetch NSE Data
        if data_type == 'stock':
            nse_url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
        else:
            nse_url = "https://nsearchives.nseindia.com/content/equities/eq_etfseclist.csv"

        logging.info(f"Fetching NSE {data_type} data from {nse_url}")
        nse_response = requests.get(nse_url, headers=headers)
        nse_response.raise_for_status()
        
        nse_df = pd.read_csv(io.StringIO(nse_response.text))
        nse_df.columns = nse_df.columns.str.strip()

        if data_type == 'stock':
            nse_df = nse_df[['ISIN NUMBER', 'SYMBOL', 'NAME OF COMPANY']].rename(columns={
                'ISIN NUMBER': 'isin_code',
                'SYMBOL': 'nse_symbol',
                'NAME OF COMPANY': 'name_nse'
            })
        else:
            nse_df = nse_df[['ISINNumber', 'Symbol', 'SecurityName', 'Underlying']].rename(columns={
                'ISINNumber': 'isin_code',
                'Symbol': 'nse_symbol',
                'SecurityName': 'name_nse',
                'Underlying': 'underlying'
            })
            #combine name and underlying for better matching
            nse_df['name_nse'] = nse_df['name_nse'].fillna('') + ' ' + nse_df['underlying'].fillna('')
            nse_df = nse_df[['isin_code', 'nse_symbol', 'name_nse']]

        # 2. Load BSE List from uploaded CSV
        if not file:
            logging.error(f"No BSE CSV file found in the request.")
            return {"message": "No BSE CSV file found in the request."}, 400

        try:
            # Read the uploaded file
            bse_bytes = await file.read()
            bse_content = bse_bytes.decode('utf-8-sig') # Handle potential BOM
            df_raw = pd.read_csv(io.StringIO(bse_content), skipinitialspace=True, index_col=False)
            df_raw.columns = df_raw.columns.str.strip()
            
            # Map BSE columns (assumed same structure for Equity and MF files)
            bse_df = df_raw[['Security Code', 'Security Name', 'ISIN No']].copy()
            bse_df.columns = ['bse_symbol', 'name_bse', 'isin_code']

        except Exception as e:
            logging.error(f"Error parsing uploaded BSE CSV: {str(e)}")
            return {"message": f"Error parsing uploaded BSE CSV: {str(e)}"}, 400
        
        # Pre-process ISIN codes - strip whitespace and ensure uppercase
        nse_df['isin_code'] = nse_df['isin_code'].astype(str).str.strip().str.upper()
        bse_df['isin_code'] = bse_df['isin_code'].astype(str).str.strip().str.upper()
    

        # 3. Merge on isin_code using an outer join
        logging.info(f"Merging NSE and BSE {data_type} data by ISIN")
        merged_df = pd.merge(nse_df, bse_df, on='isin_code', how='outer')

        # Set final name (prefer NSE over BSE)
        merged_df['name'] = merged_df['name_nse'].fillna(merged_df['name_bse'])
        
        # Remove entries without valid ISIN and deduplicate
        merged_df = merged_df[merged_df['isin_code'].notna() & (merged_df['isin_code'] != 'nan') & (merged_df['isin_code'] != '')]
        merged_df = merged_df.drop_duplicates(subset=['isin_code'])

        # Convert back to regular Python types for database insertion
        merged_df = merged_df.replace({np.nan: None})

        logging.info(f"Syncing {len(merged_df)} {data_type} records to database in batches")
        
        batch_size = 500
        items_to_sync = merged_df.to_dict('records')

        with db_handler.get_session() as session:
            for i in range(0, len(items_to_sync), batch_size):
                batch = items_to_sync[i:i + batch_size]
                batch_isins = [str(row['isin_code']).strip() for row in batch]
                
                # Fetch existing records for this batch by ISIN ONLY (regardless of type)
                existing_items = session.exec(select(Stock).where(Stock.isin_code.in_(batch_isins))).all()   # type: ignore
                existing_map = {s.isin_code: s for s in existing_items}
                
                for row in batch:
                    isin = str(row['isin_code']).strip()
                    
                    # If it already exists in the database, skip it as requested
                    if isin in existing_map:
                        logging.debug(f"Skipping existing ISIN: {isin}")
                        continue

                    name = str(row['name']).strip() if row['name'] else "Unknown"
                    nse_symbol = str(row['nse_symbol']).strip() if row['nse_symbol'] else None
                    
                    # BSE symbol can be a number (Security Code), ensure it's a string
                    bse_symbol = None
                    if row['bse_symbol'] is not None:
                        try:
                            # It might be a float if read from CSV
                            bse_symbol = str(int(float(row['bse_symbol'])))
                        except:
                            bse_symbol = str(row['bse_symbol']).strip()

                    new_stock = Stock(
                        isin_code=isin,
                        name=name,
                        nse_symbol=nse_symbol,
                        bse_symbol=bse_symbol,
                        type=data_type,
                        last_updated=datetime.now()
                    )
                    session.add(new_stock)
                
                # Commit after each batch
                session.commit()
                logging.info(f"Committed batch (processed {min(i + batch_size, len(items_to_sync))}/{len(items_to_sync)})")

        return {"message": f"Successfully synced {len(merged_df)} {data_type} records from NSE and BSE."}, 200

    except Exception as e:
        logging.error(f"Error syncing market data: {str(e)}", exc_info=True)
        return {"error": f"Failed to sync market data: {str(e)}"}, 500
