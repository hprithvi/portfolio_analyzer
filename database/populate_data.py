# database/populate_multi_asset.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from database.multi_asset_db_manager import MultiAssetDBManager
from utils.multi_asset_fetcher import MultiAssetDataFetcher
from datetime import datetime, timedelta
import time

def populate_mutual_funds(db, fetcher, limit=None):
    """Populate mutual fund data"""
    print("Fetching mutual fund schemes...")
    schemes_df = fetcher.fetch_all_mf_schemes()
    
    if limit:
        schemes_df = schemes_df.head(limit)
    
    for idx, scheme in schemes_df.iterrows():
        try:
            print(f"[{idx+1}/{len(schemes_df)}] Processing {scheme['scheme_name'][:50]}...")
            
            if pd.isna(scheme['isin']):
                print("  No ISIN, skipping...")
                continue
            
            # Insert fund metadata
            fund_data = {
                'isin': scheme['isin'],
                'scheme_code': scheme['scheme_code'],
                'scheme_name': scheme['scheme_name'],
                'amc_name': scheme['amc'],
                'category': scheme['category'],
                'sub_category': scheme['sub_category']
            }
            
            db.insert_mutual_fund(fund_data)
            
            # Fetch and insert NAV history
            start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
            nav_df = fetcher.fetch_mf_nav_history(scheme['scheme_code'], start_date)
            
            if nav_df is not None and not nav_df.empty:
                db.insert_price_history_bulk(nav_df.reset_index(), scheme['isin'])
                print(f"  ✓ Inserted {len(nav_df)} NAV records")
            
            time.sleep(0.5)
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

def populate_stocks(db, fetcher, limit=None):
    """Populate stock data"""
    print("Fetching NSE equity list...")
    stocks_df = fetcher.fetch_nse_equity_list()
    
    if limit:
        stocks_df = stocks_df.head(limit)
    
    # Download historical data
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
    
    print(f"Downloading historical data from {start_date.date()} to {end_date.date()}...")
    historical_df = fetcher.download_nse_historical(start_date, end_date)
    
    if not historical_df.empty:
        # Group by ISIN and insert
        for isin in historical_df['isin'].unique():
            try:
                isin_data = historical_df[historical_df['isin'] == isin]
                symbol = isin_data['symbol'].iloc[0]
                
                # Insert stock metadata
                stock_data = {
                    'isin': isin,
                    'symbol': symbol,
                    'company_name': symbol,  # Need to get actual name
                    'exchange': 'NSE'
                }
                
                db.insert_stock(stock_data)
                
                # Insert price history
                price_data = isin_data[['date', 'open_price', 'high_price', 'low_price', 
                                       'close_price', 'volume', 'value', 'trades']]
                db.insert_price_history_bulk(price_data, isin)
                
                print(f"✓ Inserted {symbol} with {len(price_data)} records")
            
            except Exception as e:
                print(f"✗ Error with {isin}: {e}")
                continue

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Populate multi-asset database')
    parser.add_argument('--asset-type', choices=['mf', 'stocks', 'bonds', 'all'], default='all')
    parser.add_argument('--limit', type=int, help='Limit number of assets')
    parser.add_argument('--db-url', help='Database URL')
    
    args = parser.parse_args()
    
    # Initialize
    db = MultiAssetDBManager(db_url=args.db_url, db_type='postgresql')
    fetcher = MultiAssetDataFetcher()
    
    if args.asset_type in ['mf', 'all']:
        populate_mutual_funds(db, fetcher, args.limit)
    
    if args.asset_type in ['stocks', 'all']:
        populate_stocks(db, fetcher, args.limit)
    
    print("\n✅ Population complete!")

if __name__ == "__main__":
    main()
