import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager
from utils.data_fetcher import MutualFundDataFetcher
import time
from datetime import datetime, timedelta

def populate_database(limit=None, update_existing=False):
    """
    Populate database with mutual fund data
    
    Args:
        limit: Number of funds to fetch (None for all)
        update_existing: Whether to update existing funds
    """
    db = DatabaseManager()
    fetcher = MutualFundDataFetcher()
    
    print("Fetching list of all schemes...")
    schemes_df = fetcher.get_all_schemes()
    
    if schemes_df.empty:
        print("No schemes found!")
        return
    
    print(f"Found {len(schemes_df)} schemes")
    
    # Limit if specified
    if limit:
        schemes_df = schemes_df.head(limit)
        print(f"Processing first {limit} schemes...")
    
    success_count = 0
    error_count = 0
    
    for idx, row in schemes_df.iterrows():
        scheme_code = row['schemeCode']
        scheme_name = row['schemeName']
        
        try:
            print(f"\n[{idx+1}/{len(schemes_df)}] Processing: {scheme_name[:50]}...")
            
            # Check if already exists
            existing = db.get_fund_details(str(scheme_code))
            if existing is not None and not update_existing:
                print(f"  ✓ Already exists, skipping...")
                continue
            
            # Fetch NAV history (last 3 years)
            start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
            nav_df, metadata = fetcher.get_nav_history(str(scheme_code), start_date)
            
            if nav_df is None or nav_df.empty:
                print(f"  ✗ No NAV data available")
                error_count += 1
                continue
            
            # Calculate returns
            returns = fetcher.calculate_returns(nav_df)
            
            # Prepare fund data
            fund_data = {
                'scheme_code': str(scheme_code),
                'scheme_name': metadata.get('scheme_name', scheme_name),
                'fund_house': metadata.get('fund_house', ''),
                'category': metadata.get('scheme_type', ''),
                'sub_category': metadata.get('scheme_category', ''),
                'current_nav': returns.get('current_nav'),
                'nav_date': returns.get('nav_date'),
                'returns_1m': returns.get('returns_1m'),
                'returns_3m': returns.get('returns_3m'),
                'returns_6m': returns.get('returns_6m'),
                'returns_1y': returns.get('returns_1y'),
                'returns_3y': returns.get('returns_3y'),
                'returns_5y': returns.get('returns_5y'),
                'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Insert fund metadata
            db.insert_mutual_fund(fund_data)
            
            # Insert NAV history
            db.insert_nav_bulk(nav_df, str(scheme_code))
            
            # Calculate and store statistics
            db.calculate_and_store_statistics(str(scheme_code))
            
            print(f"  ✓ Success - {len(nav_df)} NAV records inserted")
            success_count += 1
            
            # Rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            error_count += 1
            continue
    
    print(f"\n{'='*60}")
    print(f"Population complete!")
    print(f"Success: {success_count}")
    print(f"Errors: {error_count}")
    print(f"{'='*60}")

def update_navs():
    """Update NAVs for all funds in database"""
    db = DatabaseManager()
    fetcher = MutualFundDataFetcher()
    
    print("Fetching all funds from database...")
    funds = db.get_all_funds()
    
    print(f"Updating NAVs for {len(funds)} funds...")
    
    for idx, fund in funds.iterrows():
        scheme_code = fund['scheme_code']
        
        try:
            print(f"\n[{idx+1}/{len(funds)}] Updating: {fund['scheme_name'][:50]}...")
            
            # Get latest date in DB
            latest_date = db.get_latest_nav_date(scheme_code)
            
            # Fetch new data
            nav_df, _ = fetcher.get_nav_history(scheme_code, start_date=latest_date)
            
            if nav_df is not None and not nav_df.empty:
                db.insert_nav_bulk(nav_df, scheme_code)
                print(f"  ✓ Added {len(nav_df)} new NAV records")
            else:
                print(f"  - No new data")
            
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            continue

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Populate mutual fund database')
    parser.add_argument('--limit', type=int, help='Limit number of funds to fetch')
    parser.add_argument('--update', action='store_true', help='Update existing funds')
    parser.add_argument('--update-navs', action='store_true', help='Update NAVs only')
    
    args = parser.parse_args()
    
    if args.update_navs:
        update_navs()
    else:
        populate_database(limit=args.limit, update_existing=args.update)
