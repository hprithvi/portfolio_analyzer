# database/populate_data.py
# One-time batch load of all mutual fund and stock data for the last 3 years.

import sys
import os
import logging
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

sys.path.append(str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

if not os.getenv('DATABASE_URL'):
    print("WARNING: DATABASE_URL not found in environment!")
    print("Make sure you have a .env file with DATABASE_URL set")

from database.multi_asset_db_manager import MultiAssetDBManager
from utils.data_fetcher import MultiAssetDataFetcher
from sqlalchemy import text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('batch_load.log')
    ]
)
logger = logging.getLogger(__name__)

LOOKBACK_YEARS = 3


def populate_mutual_funds(db, fetcher, limit=None, max_workers=5, resume=True):
    """Populate all mutual fund data with concurrency and checkpointing."""
    logger.info("=== PHASE: MUTUAL FUNDS ===")

    # Step 1: Get scheme list (single API call, ISINs already included)
    logger.info("Fetching mutual fund scheme list...")
    schemes_df = fetcher.fetch_all_mf_schemes()

    if schemes_df.empty:
        logger.error("No schemes returned from API")
        return

    if limit:
        schemes_df = schemes_df.head(limit)

    logger.info(f"Total schemes to process: {len(schemes_df)}")

    # Step 2: Get already-completed checkpoints for resume
    completed = set()
    if resume:
        try:
            completed = db.get_completed_checkpoints('mf_nav')
            logger.info(f"Resuming: {len(completed)} schemes already completed")
        except Exception:
            logger.info("No checkpoint table found, processing all schemes")

    # Filter out completed
    pending = schemes_df[~schemes_df['scheme_code'].astype(str).isin(completed)]
    logger.info(f"{len(pending)} schemes remaining to process")

    if pending.empty:
        logger.info("All MF schemes already processed")
        return

    # Step 3: Process each scheme concurrently
    success_count = 0
    error_count = 0
    start_time = time.time()
    start_date_str = (datetime.now() - timedelta(days=LOOKBACK_YEARS * 365)).strftime('%Y-%m-%d')

    def process_one_scheme(row):
        """Process a single MF scheme: fetch NAV, insert metadata + prices."""
        scheme_code = str(row['scheme_code'])
        isin = row['isin']
        scheme_name = row.get('scheme_name', '')

        try:
            # fetch_mf_nav_history returns (df, meta) tuple
            nav_df, meta = fetcher.fetch_mf_nav_history(
                scheme_code, start_date=start_date_str, isin=isin
            )

            # Insert fund metadata (using meta from the detail API response)
            fund_data = {
                'isin': isin,
                'scheme_code': scheme_code,
                'scheme_name': meta.get('scheme_name', scheme_name),
                'amc_name': meta.get('fund_house', ''),
                'category': meta.get('scheme_type', ''),
                'sub_category': meta.get('scheme_category', '')
            }
            db.insert_mutual_fund(fund_data)

            # Insert price history
            inserted = 0
            if nav_df is not None and not nav_df.empty:
                inserted = db.insert_price_history_bulk(nav_df)

            db.mark_checkpoint('mf_nav', scheme_code, 'completed')
            return scheme_code, inserted, None

        except Exception as e:
            try:
                db.mark_checkpoint('mf_nav', scheme_code, 'failed', str(e)[:500])
            except Exception:
                pass
            return scheme_code, 0, str(e)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, row in pending.iterrows():
            future = executor.submit(process_one_scheme, row)
            futures[future] = str(row['scheme_code'])

        for i, future in enumerate(as_completed(futures), 1):
            scheme_code, inserted, error = future.result()
            if error:
                error_count += 1
                logger.warning(f"[{i}/{len(pending)}] FAILED {scheme_code}: {error[:200]}")
            else:
                success_count += 1
                if i % 100 == 0 or i == len(pending):
                    elapsed = time.time() - start_time
                    rate = i / elapsed * 60
                    logger.info(f"[{i}/{len(pending)}] {rate:.0f} schemes/min | "
                                f"Success: {success_count} | Errors: {error_count}")

    elapsed = time.time() - start_time
    logger.info(f"MF complete: {success_count} success, {error_count} errors in {elapsed/60:.1f} min")


# NSE bhav copy column name mappings (columns may have leading spaces)
BHAV_COLUMN_MAP = {
    'DATE1': 'date',
    'SYMBOL': 'symbol',
    'SERIES': 'series',
    'OPEN_PRICE': 'open_price',
    'HIGH_PRICE': 'high_price',
    'LOW_PRICE': 'low_price',
    'CLOSE_PRICE': 'close_price',
    'TTL_TRD_QNTY': 'volume',
    'TURNOVER_LACS': 'value',
    'NO_OF_TRADES': 'trades',
}


def populate_stocks(db, fetcher, limit=None, resume=True):
    """Populate all NSE stock data for the last 3 years."""
    logger.info("=== PHASE: STOCKS ===")

    # Step 1: Fetch equity master list for ISIN mapping
    logger.info("Fetching NSE equity master list...")
    equity_df = fetcher.fetch_nse_equity_list()

    if equity_df.empty:
        logger.error("Failed to fetch equity list")
        return

    logger.info(f"Found {len(equity_df)} equities in master list")

    # Build symbol -> ISIN lookup and insert stock metadata
    symbol_to_isin = {}
    for _, row in equity_df.iterrows():
        sym = str(row.get('symbol', '')).strip()
        isin = str(row.get('isin', '')).strip()
        if not sym or not isin or isin == 'nan':
            continue

        symbol_to_isin[sym] = isin

        # Insert stock metadata
        def safe_scalar(val, default=None):
            """Extract a scalar from a value that might be a Series (due to duplicate cols)."""
            if isinstance(val, pd.Series):
                val = val.iloc[0]
            if pd.isna(val):
                return default
            return val

        try:
            stock_data = {
                'isin': isin,
                'symbol': sym,
                'company_name': str(safe_scalar(row.get('company_name'), sym)).strip(),
                'exchange': 'NSE',
                'series': str(safe_scalar(row.get('series'), 'EQ')).strip(),
                'face_value': safe_scalar(row.get('face_value')),
                'listing_date': str(safe_scalar(row.get('listing_date'))).strip() if safe_scalar(row.get('listing_date')) else None,
            }
            db.insert_stock(stock_data)
        except Exception as e:
            logger.warning(f"Failed to insert stock metadata for {sym}: {e}")

    logger.info(f"Inserted/updated {len(symbol_to_isin)} stock records")

    # Step 2: Download bhav copies for 3 years
    start_date = datetime.now() - timedelta(days=LOOKBACK_YEARS * 365)
    end_date = datetime.now() - timedelta(days=1)

    # Get completed checkpoints for resume
    completed_dates = set()
    if resume:
        try:
            completed_dates = db.get_completed_checkpoints('stock_bhav')
            logger.info(f"Resuming: {len(completed_dates)} dates already completed")
        except Exception:
            logger.info("No checkpoint table found, processing all dates")

    # Generate business day range
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # B = business days
    pending_dates = [d for d in date_range if d.strftime('%Y-%m-%d') not in completed_dates]

    if limit:
        pending_dates = pending_dates[:limit]

    logger.info(f"{len(pending_dates)} trading days to process (of {len(date_range)} total business days)")

    if not pending_dates:
        logger.info("All stock dates already processed")
        return

    # Step 3: Process each bhav copy
    success_count = 0
    error_count = 0
    total_records = 0
    start_time = time.time()

    for i, date in enumerate(pending_dates, 1):
        date_str = date.strftime('%Y-%m-%d')
        try:
            bhav_df = fetcher.fetch_nse_bhav_copy(date)

            if bhav_df is None or (hasattr(bhav_df, 'empty') and bhav_df.empty):
                # Likely a holiday — mark as completed
                db.mark_checkpoint('stock_bhav', date_str, 'completed')
                continue

            # Clean column names (strip spaces)
            bhav_df.columns = bhav_df.columns.str.strip()

            # Rename columns to match price_history schema
            rename_map = {}
            for old_col in bhav_df.columns:
                if old_col in BHAV_COLUMN_MAP:
                    rename_map[old_col] = BHAV_COLUMN_MAP[old_col]
            bhav_df = bhav_df.rename(columns=rename_map)

            # Filter to EQ series only (skip BE, BZ, etc.)
            if 'series' in bhav_df.columns:
                bhav_df = bhav_df[bhav_df['series'].str.strip() == 'EQ']

            # Map symbols to ISINs
            if 'symbol' in bhav_df.columns:
                bhav_df['symbol'] = bhav_df['symbol'].str.strip()
                bhav_df['isin'] = bhav_df['symbol'].map(symbol_to_isin)

                # Drop rows with no ISIN mapping
                bhav_df = bhav_df.dropna(subset=['isin'])

            if bhav_df.empty:
                db.mark_checkpoint('stock_bhav', date_str, 'completed')
                continue

            # Parse date column or use the loop date
            if 'date' in bhav_df.columns:
                bhav_df['date'] = pd.to_datetime(bhav_df['date'], format='mixed', dayfirst=True)
            else:
                bhav_df['date'] = date

            # Select and clean columns for price_history
            price_cols = ['isin', 'date', 'open_price', 'high_price', 'low_price',
                          'close_price', 'volume', 'value', 'trades']
            available = [c for c in price_cols if c in bhav_df.columns]
            price_df = bhav_df[available].copy()

            # Convert numeric columns
            for col in ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'value', 'trades']:
                if col in price_df.columns:
                    price_df[col] = pd.to_numeric(price_df[col], errors='coerce')

            inserted = db.insert_price_history_bulk(price_df)
            total_records += inserted
            success_count += 1
            db.mark_checkpoint('stock_bhav', date_str, 'completed')

            if i % 50 == 0 or i == len(pending_dates):
                elapsed = time.time() - start_time
                logger.info(f"[{i}/{len(pending_dates)}] {date_str} | "
                            f"{total_records:,} total records | {elapsed/60:.1f} min elapsed")

            time.sleep(1)  # Rate limiting for NSE

        except Exception as e:
            error_count += 1
            try:
                db.mark_checkpoint('stock_bhav', date_str, 'failed', str(e)[:500])
            except Exception:
                pass
            logger.warning(f"[{i}/{len(pending_dates)}] FAILED {date_str}: {e}")
            time.sleep(2)  # Extra wait on error

    # Step 4: Mark delisted stocks
    logger.info("Identifying potentially delisted stocks...")
    try:
        engine = db.get_engine()
        historical_isins = pd.read_sql_query(
            text("SELECT DISTINCT isin FROM stocks"), engine
        )
        current_isins = set(symbol_to_isin.values())

        delisted_count = 0
        for _, row in historical_isins.iterrows():
            if row['isin'] not in current_isins:
                db.mark_asset_inactive(row['isin'])
                delisted_count += 1

        if delisted_count > 0:
            logger.info(f"Marked {delisted_count} stocks as inactive (delisted)")
    except Exception as e:
        logger.warning(f"Could not check for delisted stocks: {e}")

    elapsed = time.time() - start_time
    logger.info(f"Stocks complete: {success_count} days, {total_records:,} records, "
                f"{error_count} errors in {elapsed/60:.1f} min")


def apply_schema(db):
    """Apply the multi-asset schema to the database."""
    schema_path = Path(__file__).parent / 'schema.sql'
    if not schema_path.exists():
        logger.error(f"Schema file not found: {schema_path}")
        return

    engine = db.get_engine()
    with open(schema_path) as f:
        schema_sql = f.read()

    # Strip SQL comments before splitting into statements
    lines = [line for line in schema_sql.splitlines()
             if line.strip() and not line.strip().startswith('--')]
    clean_sql = '\n'.join(lines)

    with engine.connect() as conn:
        for statement in clean_sql.split(';'):
            statement = statement.strip()
            if statement:
                try:
                    conn.execute(text(statement))
                except Exception as e:
                    logger.warning(f"Schema statement warning: {e}")
        conn.commit()

    logger.info("Schema applied successfully")


def main():
    parser = argparse.ArgumentParser(description='Batch load Indian financial data (MF + stocks, 3 years)')
    parser.add_argument('--asset-type', choices=['mf', 'stocks', 'all'], default='all',
                        help='Which asset types to load (default: all)')
    parser.add_argument('--limit', type=int,
                        help='Limit number of items (for testing)')
    parser.add_argument('--workers', type=int, default=10,
                        help='Number of concurrent workers for MF NAV fetch (default: 10)')
    parser.add_argument('--no-resume', action='store_true',
                        help='Start fresh, ignore checkpoints')
    parser.add_argument('--apply-schema', action='store_true',
                        help='Apply schema.sql before loading data')
    parser.add_argument('--db-url',
                        help='Override DATABASE_URL')

    args = parser.parse_args()

    # Initialize DB
    db = MultiAssetDBManager(db_url=args.db_url, db_type='postgresql')

    # Test connection
    success, msg = db.test_connection()
    if not success:
        logger.error(f"Database connection failed: {msg}")
        sys.exit(1)
    logger.info("Database connection OK")

    # Apply schema if requested
    if args.apply_schema:
        apply_schema(db)

    resume = not args.no_resume
    overall_start = time.time()

    try:
        if args.asset_type in ['mf', 'all']:
            fetcher = MultiAssetDataFetcher()
            populate_mutual_funds(db, fetcher, args.limit, args.workers, resume)

        if args.asset_type in ['stocks', 'all']:
            fetcher = MultiAssetDataFetcher()  # Fresh session for NSE cookies
            populate_stocks(db, fetcher, args.limit, resume)

    finally:
        db.close()

    total_time = (time.time() - overall_start) / 3600
    logger.info(f"Batch load complete in {total_time:.2f} hours")


if __name__ == "__main__":
    main()
