# database/update_prices.py
# Incremental price update: fetches MF NAV and stock bhav copies
# from the last available date in the DB to today.
# Designed to be run daily or on a regular schedule.

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

from database.multi_asset_db_manager import MultiAssetDBManager
from utils.data_fetcher import MultiAssetDataFetcher
from sqlalchemy import text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('price_update.log')
    ]
)
logger = logging.getLogger(__name__)


def get_last_price_date(db, asset_type):
    """Get the most recent price date for a given asset type."""
    engine = db.get_engine()
    query = text("""
        SELECT MAX(ph.date) as last_date
        FROM price_history ph
        JOIN assets a ON ph.isin = a.isin
        WHERE a.asset_type = :asset_type
    """)
    df = pd.read_sql_query(query, engine, params={'asset_type': asset_type})
    last_date = df['last_date'].iloc[0]
    if last_date is None:
        return None
    if isinstance(last_date, str):
        return datetime.strptime(last_date, '%Y-%m-%d').date()
    return pd.Timestamp(last_date).date()


def get_active_mf_schemes(db):
    """Get all active mutual fund schemes with their ISIN and scheme_code."""
    engine = db.get_engine()
    query = text("""
        SELECT mf.isin, mf.scheme_code, mf.scheme_name
        FROM mutual_funds mf
        JOIN assets a ON mf.isin = a.isin
        WHERE a.is_active = TRUE
    """)
    return pd.read_sql_query(query, engine)


def get_active_stocks(db):
    """Get all active stocks with their ISIN and symbol."""
    engine = db.get_engine()
    query = text("""
        SELECT s.isin, s.symbol
        FROM stocks s
        JOIN assets a ON s.isin = a.isin
        WHERE a.is_active = TRUE
    """)
    return pd.read_sql_query(query, engine)


def update_mutual_funds(db, fetcher, max_workers=10):
    """Update MF NAV data from last available date to today."""
    logger.info("=== UPDATING MUTUAL FUND PRICES ===")

    last_date = get_last_price_date(db, 'mutual_fund')
    if last_date is None:
        logger.error("No existing MF price data found. Run populate_data.py first.")
        return

    start_date = last_date + timedelta(days=1)
    today = datetime.now().date()

    if start_date > today:
        logger.info("MF prices are already up to date.")
        return

    logger.info(f"Fetching MF NAV from {start_date} to {today}")

    schemes = get_active_mf_schemes(db)
    logger.info(f"Active MF schemes: {len(schemes)}")

    start_date_str = start_date.strftime('%Y-%m-%d')
    success_count = 0
    error_count = 0
    skip_count = 0
    start_time = time.time()

    def process_scheme(row):
        scheme_code = str(row['scheme_code'])
        isin = row['isin']
        try:
            nav_df, meta = fetcher.fetch_mf_nav_history(
                scheme_code, start_date=start_date_str, isin=isin
            )
            if nav_df is not None and not nav_df.empty:
                inserted = db.insert_price_history_bulk(nav_df)
                return scheme_code, inserted, None
            else:
                return scheme_code, 0, None  # No new data (holiday/weekend gap)
        except Exception as e:
            return scheme_code, 0, str(e)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_scheme, row): str(row['scheme_code'])
                   for _, row in schemes.iterrows()}

        for i, future in enumerate(as_completed(futures), 1):
            scheme_code, inserted, error = future.result()
            if error:
                error_count += 1
                if i <= 5 or error_count <= 10:
                    logger.warning(f"FAILED {scheme_code}: {error[:150]}")
            elif inserted == 0:
                skip_count += 1
            else:
                success_count += 1

            if i % 500 == 0 or i == len(futures):
                elapsed = time.time() - start_time
                rate = i / elapsed * 60 if elapsed > 0 else 0
                logger.info(f"[{i}/{len(futures)}] {rate:.0f}/min | "
                            f"Updated: {success_count} | Skipped: {skip_count} | Errors: {error_count}")

    elapsed = time.time() - start_time
    logger.info(f"MF update complete: {success_count} updated, {skip_count} skipped, "
                f"{error_count} errors in {elapsed/60:.1f} min")


# Column mapping for bhav copy (same as populate_data.py)
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


def get_active_etfs(db):
    """Get all active ETFs with their ISIN and symbol."""
    engine = db.get_engine()
    query = text("""
        SELECT s.isin, s.symbol
        FROM stocks s
        JOIN assets a ON s.isin = a.isin
        WHERE a.asset_type = 'etf' AND a.is_active = TRUE
    """)
    return pd.read_sql_query(query, engine)


def update_etfs(db, fetcher):
    """Update ETF prices from last available date to today."""
    logger.info("=== UPDATING ETF PRICES ===")

    last_date = get_last_price_date(db, 'etf')
    if last_date is None:
        logger.error("No existing ETF price data found. Run populate_data.py --asset-type etfs first.")
        return

    start_date = last_date + timedelta(days=1)
    today = datetime.now().date()

    if start_date > today:
        logger.info("ETF prices are already up to date.")
        return

    logger.info(f"Fetching ETF bhav copies from {start_date} to {today}")

    # Build symbol -> ISIN lookup from active ETFs
    etfs_df = get_active_etfs(db)
    symbol_to_isin = dict(zip(etfs_df['symbol'], etfs_df['isin']))
    etf_symbols = set(symbol_to_isin.keys())
    logger.info(f"Active ETFs: {len(symbol_to_isin)}")

    # Generate business day range
    date_range = pd.date_range(start=start_date, end=today, freq='B')
    logger.info(f"Business days to process: {len(date_range)}")

    if len(date_range) == 0:
        logger.info("No business days to process.")
        return

    success_count = 0
    error_count = 0
    total_records = 0
    start_time = time.time()

    for i, date in enumerate(date_range, 1):
        date_str = date.strftime('%Y-%m-%d')
        try:
            bhav_df = fetcher.fetch_nse_bhav_copy(date)

            if bhav_df is None or (hasattr(bhav_df, 'empty') and bhav_df.empty):
                continue

            bhav_df.columns = bhav_df.columns.str.strip()

            rename_map = {old: BHAV_COLUMN_MAP[old]
                          for old in bhav_df.columns if old in BHAV_COLUMN_MAP}
            bhav_df = bhav_df.rename(columns=rename_map)

            # Filter to ETF symbols (instead of series filter)
            if 'symbol' in bhav_df.columns:
                bhav_df['symbol'] = bhav_df['symbol'].str.strip()
                bhav_df = bhav_df[bhav_df['symbol'].isin(etf_symbols)]
                bhav_df['isin'] = bhav_df['symbol'].map(symbol_to_isin)
                bhav_df = bhav_df.dropna(subset=['isin'])

            if bhav_df.empty:
                continue

            if 'date' in bhav_df.columns:
                bhav_df['date'] = pd.to_datetime(bhav_df['date'], format='mixed', dayfirst=True)
            else:
                bhav_df['date'] = date

            price_cols = ['isin', 'date', 'open_price', 'high_price', 'low_price',
                          'close_price', 'volume', 'value', 'trades']
            available = [c for c in price_cols if c in bhav_df.columns]
            price_df = bhav_df[available].copy()

            for col in ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'value', 'trades']:
                if col in price_df.columns:
                    price_df[col] = pd.to_numeric(price_df[col], errors='coerce')

            inserted = db.insert_price_history_bulk(price_df)
            total_records += inserted
            success_count += 1

            if i % 10 == 0 or i == len(date_range):
                elapsed = time.time() - start_time
                logger.info(f"[{i}/{len(date_range)}] {date_str} | "
                            f"{total_records:,} records | {elapsed/60:.1f} min")

            time.sleep(1)

        except Exception as e:
            error_count += 1
            logger.warning(f"[{i}/{len(date_range)}] FAILED {date_str}: {e}")
            time.sleep(2)

    elapsed = time.time() - start_time
    logger.info(f"ETF update complete: {success_count} days, {total_records:,} records, "
                f"{error_count} errors in {elapsed/60:.1f} min")


def update_indices(db, fetcher):
    """Update index price data from last available date to today."""
    logger.info("=== UPDATING INDEX PRICES ===")

    indices = db.get_all_indices()
    if indices.empty:
        logger.info("No indices found in database. Run populate_data.py --asset-type indices first.")
        return

    today = datetime.now().date()

    for _, idx_row in indices.iterrows():
        symbol = idx_row['symbol']
        try:
            # Get last date for this index
            engine = db.get_engine()
            query = text("""
                SELECT MAX(date) as last_date
                FROM index_price_history
                WHERE symbol = :symbol
            """)
            result = pd.read_sql_query(query, engine, params={'symbol': symbol})
            last_date = result['last_date'].iloc[0]

            if last_date is None:
                start_date = datetime.now().date() - timedelta(days=3 * 365)
            else:
                if isinstance(last_date, str):
                    start_date = datetime.strptime(last_date, '%Y-%m-%d').date() + timedelta(days=1)
                else:
                    start_date = pd.Timestamp(last_date).date() + timedelta(days=1)

            if start_date > today:
                logger.info(f"Index {symbol} is already up to date.")
                continue

            logger.info(f"Fetching {symbol} data from {start_date} to {today}")
            data = fetcher.fetch_index_data(symbol, start_date, today)

            if data is not None and not data.empty:
                data['symbol'] = symbol
                inserted = db.insert_index_price_bulk(data)
                logger.info(f"Index {symbol}: inserted {inserted} records")
            else:
                logger.warning(f"No new data for index {symbol}")

        except Exception as e:
            logger.error(f"Failed to update index {symbol}: {e}")

    logger.info("Index update complete")


def update_stocks(db, fetcher):
    """Update stock prices from last available date to today."""
    logger.info("=== UPDATING STOCK PRICES ===")

    last_date = get_last_price_date(db, 'stock')
    if last_date is None:
        logger.error("No existing stock price data found. Run populate_data.py first.")
        return

    start_date = last_date + timedelta(days=1)
    today = datetime.now().date()

    if start_date > today:
        logger.info("Stock prices are already up to date.")
        return

    logger.info(f"Fetching stock bhav copies from {start_date} to {today}")

    # Build symbol -> ISIN lookup from active stocks
    stocks_df = get_active_stocks(db)
    symbol_to_isin = dict(zip(stocks_df['symbol'], stocks_df['isin']))
    logger.info(f"Active stocks: {len(symbol_to_isin)}")

    # Generate business day range
    date_range = pd.date_range(start=start_date, end=today, freq='B')
    logger.info(f"Business days to process: {len(date_range)}")

    if len(date_range) == 0:
        logger.info("No business days to process.")
        return

    success_count = 0
    error_count = 0
    total_records = 0
    start_time = time.time()

    for i, date in enumerate(date_range, 1):
        date_str = date.strftime('%Y-%m-%d')
        try:
            bhav_df = fetcher.fetch_nse_bhav_copy(date)

            if bhav_df is None or (hasattr(bhav_df, 'empty') and bhav_df.empty):
                # Likely a holiday
                continue

            # Clean column names
            bhav_df.columns = bhav_df.columns.str.strip()

            # Rename columns
            rename_map = {old: BHAV_COLUMN_MAP[old]
                          for old in bhav_df.columns if old in BHAV_COLUMN_MAP}
            bhav_df = bhav_df.rename(columns=rename_map)

            # Filter to EQ series
            if 'series' in bhav_df.columns:
                bhav_df = bhav_df[bhav_df['series'].str.strip() == 'EQ']

            # Map symbols to ISINs
            if 'symbol' in bhav_df.columns:
                bhav_df['symbol'] = bhav_df['symbol'].str.strip()
                bhav_df['isin'] = bhav_df['symbol'].map(symbol_to_isin)
                bhav_df = bhav_df.dropna(subset=['isin'])

            if bhav_df.empty:
                continue

            # Parse date
            if 'date' in bhav_df.columns:
                bhav_df['date'] = pd.to_datetime(bhav_df['date'], format='mixed', dayfirst=True)
            else:
                bhav_df['date'] = date

            # Select price columns
            price_cols = ['isin', 'date', 'open_price', 'high_price', 'low_price',
                          'close_price', 'volume', 'value', 'trades']
            available = [c for c in price_cols if c in bhav_df.columns]
            price_df = bhav_df[available].copy()

            # Convert numeric
            for col in ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'value', 'trades']:
                if col in price_df.columns:
                    price_df[col] = pd.to_numeric(price_df[col], errors='coerce')

            inserted = db.insert_price_history_bulk(price_df)
            total_records += inserted
            success_count += 1

            if i % 10 == 0 or i == len(date_range):
                elapsed = time.time() - start_time
                logger.info(f"[{i}/{len(date_range)}] {date_str} | "
                            f"{total_records:,} records | {elapsed/60:.1f} min")

            time.sleep(1)  # Rate limiting for NSE

        except Exception as e:
            error_count += 1
            logger.warning(f"[{i}/{len(date_range)}] FAILED {date_str}: {e}")
            time.sleep(2)

    elapsed = time.time() - start_time
    logger.info(f"Stock update complete: {success_count} days, {total_records:,} records, "
                f"{error_count} errors in {elapsed/60:.1f} min")


def main():
    parser = argparse.ArgumentParser(
        description='Incremental price update for MF and stock data'
    )
    parser.add_argument('--asset-type', choices=['mf', 'stocks', 'etfs', 'indices', 'all'], default='all',
                        help='Which asset types to update (default: all)')
    parser.add_argument('--workers', type=int, default=10,
                        help='Concurrent workers for MF fetch (default: 10)')
    parser.add_argument('--db-url', help='Override DATABASE_URL')

    args = parser.parse_args()

    db = MultiAssetDBManager(db_url=args.db_url, db_type='postgresql')

    success, msg = db.test_connection()
    if not success:
        logger.error(f"Database connection failed: {msg}")
        sys.exit(1)
    logger.info("Database connection OK")

    overall_start = time.time()

    try:
        if args.asset_type in ['mf', 'all']:
            fetcher = MultiAssetDataFetcher()
            update_mutual_funds(db, fetcher, max_workers=args.workers)

        if args.asset_type in ['stocks', 'all']:
            fetcher = MultiAssetDataFetcher()  # Fresh session for NSE cookies
            update_stocks(db, fetcher)

        if args.asset_type in ['etfs', 'all']:
            fetcher = MultiAssetDataFetcher()
            update_etfs(db, fetcher)

        if args.asset_type in ['indices', 'all']:
            fetcher = MultiAssetDataFetcher()
            update_indices(db, fetcher)
    finally:
        db.close()

    total_min = (time.time() - overall_start) / 60
    logger.info(f"Price update complete in {total_min:.1f} minutes")


if __name__ == "__main__":
    main()
