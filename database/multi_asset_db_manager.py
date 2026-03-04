# database/multi_asset_db_manager.py

from pathlib import Path

from database.db_manager import DatabaseManager
import pandas as pd
from datetime import datetime
from sqlalchemy import text, inspect
from psycopg2.extras import execute_values

class MultiAssetDBManager(DatabaseManager):
    """Extended database manager for multi-asset support"""

    def __init__(self, db_url=None, db_type='postgresql'):
        super().__init__(db_url, db_type)

    def init_database(self):
        """Initialize database with multi-asset schema if needed."""
        engine = self.get_engine()
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        print(f"[init_database] Existing tables: {existing_tables}")

        if 'assets' not in existing_tables:
            print("[init_database] 'assets' table not found, applying schema...")
            self._apply_multi_asset_schema()

            # Verify
            inspector = inspect(engine)
            tables_after = inspector.get_table_names()
            print(f"[init_database] Tables after schema: {tables_after}")

    def _apply_multi_asset_schema(self):
        """Apply the full multi-asset schema from schema.sql."""
        schema_path = Path(__file__).parent / 'schema.sql'
        print(f"[schema] Looking for: {schema_path} (exists: {schema_path.exists()})")
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        with open(schema_path) as f:
            schema_sql = f.read()

        # Strip SQL comments before splitting into statements
        lines = [line for line in schema_sql.splitlines()
                 if line.strip() and not line.strip().startswith('--')]
        clean_sql = '\n'.join(lines)

        engine = self.get_engine()

        # Drop old conflicting tables that may have incompatible schemas
        # (e.g. old mutual_funds from db_manager.py with different columns)
        old_tables = [
            'fund_correlations', 'fund_statistics', 'nav_history',  # old schema dependents
            'asset_correlations', 'asset_statistics', 'data_quality_log',
            'corporate_actions', 'price_history', 'batch_checkpoint',
            'mutual_funds', 'stocks', 'bonds', 'assets'
        ]
        with engine.connect() as conn:
            for table in old_tables:
                try:
                    conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
                except Exception:
                    pass
            conn.commit()
        print("[schema] Dropped old tables")

        statements = [s.strip() for s in clean_sql.split(';') if s.strip()]
        print(f"[schema] Executing {len(statements)} SQL statements...")

        with engine.connect() as conn:
            for i, statement in enumerate(statements):
                try:
                    conn.execute(text(statement))
                    print(f"[schema] Statement {i+1}/{len(statements)}: OK")
                except Exception as e:
                    print(f"[schema] Statement {i+1}/{len(statements)}: FAILED - {e}")
                    print(f"[schema]   SQL: {statement[:100]}...")
            conn.commit()
            print("[schema] Committed")

    # ==========================================
    # ASSET OPERATIONS
    # ==========================================

    def insert_asset(self, isin, asset_type, symbol, name, exchange=None):
        """Insert or update asset in master table"""
        engine = self.get_engine()
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO assets (isin, asset_type, symbol, name, exchange, updated_at)
                VALUES (:isin, :asset_type, :symbol, :name, :exchange, NOW())
                ON CONFLICT (isin) DO UPDATE SET
                    symbol = EXCLUDED.symbol,
                    name = EXCLUDED.name,
                    exchange = EXCLUDED.exchange,
                    updated_at = NOW()
            """), {
                'isin': isin, 'asset_type': asset_type,
                'symbol': symbol, 'name': name, 'exchange': exchange
            })
            conn.commit()

    def get_assets_by_type(self, asset_type, limit=1000):
        """Get all assets of a specific type"""
        engine = self.get_engine()
        query = text("""
            SELECT * FROM assets
            WHERE asset_type = :asset_type AND is_active = TRUE
            ORDER BY name
            LIMIT :limit
        """)
        return pd.read_sql_query(query, engine, params={'asset_type': asset_type, 'limit': limit})

    def mark_asset_inactive(self, isin):
        """Mark a delisted asset as inactive"""
        engine = self.get_engine()
        with engine.connect() as conn:
            conn.execute(text("UPDATE assets SET is_active = FALSE, updated_at = NOW() WHERE isin = :isin"),
                         {'isin': isin})
            conn.commit()

    # ==========================================
    # MUTUAL FUND OPERATIONS
    # ==========================================

    def insert_mutual_fund(self, fund_data):
        """Insert or update mutual fund specific data (single connection for both tables)"""
        engine = self.get_engine()
        with engine.connect() as conn:
            # Upsert into assets table
            if 'isin' in fund_data:
                conn.execute(text("""
                    INSERT INTO assets (isin, asset_type, symbol, name, exchange, updated_at)
                    VALUES (:isin, 'mutual_fund', :symbol, :name, 'AMFI', NOW())
                    ON CONFLICT (isin) DO UPDATE SET
                        symbol = EXCLUDED.symbol,
                        name = EXCLUDED.name,
                        updated_at = NOW()
                """), {
                    'isin': fund_data['isin'],
                    'symbol': str(fund_data.get('scheme_code', '')),
                    'name': fund_data.get('scheme_name', '')
                })

            # Upsert into mutual_funds table
            conn.execute(text("""
                INSERT INTO mutual_funds (isin, scheme_code, scheme_name, amc_name, category, sub_category)
                VALUES (:isin, :scheme_code, :scheme_name, :amc_name, :category, :sub_category)
                ON CONFLICT (isin) DO UPDATE SET
                    scheme_name = EXCLUDED.scheme_name,
                    amc_name = EXCLUDED.amc_name,
                    category = EXCLUDED.category,
                    sub_category = EXCLUDED.sub_category
            """), {
                'isin': fund_data.get('isin'),
                'scheme_code': str(fund_data.get('scheme_code', '')),
                'scheme_name': fund_data.get('scheme_name', ''),
                'amc_name': fund_data.get('amc_name', ''),
                'category': fund_data.get('category', ''),
                'sub_category': fund_data.get('sub_category', '')
            })
            conn.commit()

    # ==========================================
    # STOCK OPERATIONS
    # ==========================================

    def insert_stock(self, stock_data):
        """Insert or update stock specific data"""
        # Upsert into assets table
        self.insert_asset(
            isin=stock_data['isin'],
            asset_type='stock',
            symbol=stock_data['symbol'],
            name=stock_data['company_name'],
            exchange=stock_data.get('exchange', 'NSE')
        )

        # Upsert into stocks table
        engine = self.get_engine()
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO stocks (isin, symbol, company_name, exchange, series, face_value, listing_date)
                VALUES (:isin, :symbol, :company_name, :exchange, :series, :face_value, :listing_date)
                ON CONFLICT (isin) DO UPDATE SET
                    symbol = EXCLUDED.symbol,
                    company_name = EXCLUDED.company_name,
                    series = EXCLUDED.series
            """), {
                'isin': stock_data['isin'],
                'symbol': stock_data['symbol'],
                'company_name': stock_data['company_name'],
                'exchange': stock_data.get('exchange', 'NSE'),
                'series': stock_data.get('series'),
                'face_value': stock_data.get('face_value'),
                'listing_date': stock_data.get('listing_date')
            })
            conn.commit()

    # ==========================================
    # BOND OPERATIONS
    # ==========================================

    def insert_bond(self, bond_data):
        """Insert bond specific data"""
        engine = self.get_engine()

        # Insert into assets table
        self.insert_asset(
            isin=bond_data['isin'],
            asset_type='bond',
            symbol=bond_data.get('isin'),  # Bonds typically use ISIN as symbol
            name=bond_data['bond_name'],
            exchange='NSE'
        )

        # Insert into bonds table
        df = pd.DataFrame([bond_data])
        df.to_sql('bonds', engine, if_exists='append', index=False)

    # ==========================================
    # PRICE HISTORY (Common for all assets)
    # ==========================================

    def insert_price_history_bulk(self, price_df, chunk_size=5000, max_retries=3):
        """Insert bulk price history using execute_values for speed."""
        if price_df is None or price_df.empty:
            return 0

        required = ['isin', 'date', 'close_price']
        for col in required:
            if col not in price_df.columns:
                raise ValueError(f"Missing required column: {col}")

        price_df = price_df.copy()
        price_df['date'] = pd.to_datetime(price_df['date']).dt.strftime('%Y-%m-%d')

        # Fill optional columns with defaults
        for col in ['open_price', 'high_price', 'low_price']:
            if col not in price_df.columns:
                price_df[col] = price_df['close_price']
        for col in ['volume', 'value', 'trades']:
            if col not in price_df.columns:
                price_df[col] = None

        cols = ['isin', 'date', 'open_price', 'high_price', 'low_price',
                'close_price', 'volume', 'value', 'trades']
        # Convert to list of tuples for execute_values
        tuples = list(price_df[cols].itertuples(index=False, name=None))

        engine = self.get_engine()
        sql = """
            INSERT INTO price_history (isin, date, open_price, high_price, low_price,
                                       close_price, volume, value, trades)
            VALUES %s
            ON CONFLICT (isin, date) DO NOTHING
        """

        for attempt in range(max_retries):
            raw_conn = engine.raw_connection()
            try:
                cursor = raw_conn.cursor()
                inserted = 0
                for i in range(0, len(tuples), chunk_size):
                    chunk = tuples[i:i+chunk_size]
                    execute_values(cursor, sql, chunk, page_size=chunk_size)
                    inserted += len(chunk)
                raw_conn.commit()
                return inserted
            except Exception as e:
                try:
                    raw_conn.rollback()
                except Exception:
                    pass
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s
                    continue
                raise
            finally:
                try:
                    raw_conn.close()
                except Exception:
                    pass

        return 0

    def get_price_history(self, isin, start_date=None, end_date=None):
        """Get price history for any asset by ISIN"""
        engine = self.get_engine()

        conditions = ['isin = :isin']
        params = {'isin': isin}

        if start_date:
            conditions.append('date >= :start_date')
            params['start_date'] = start_date

        if end_date:
            conditions.append('date <= :end_date')
            params['end_date'] = end_date

        where_clause = ' AND '.join(conditions)
        query = text(f"SELECT * FROM price_history WHERE {where_clause} ORDER BY date")

        return pd.read_sql_query(query, engine, params=params, parse_dates=['date'])

    # ==========================================
    # CHECKPOINT OPERATIONS
    # ==========================================

    def mark_checkpoint(self, job_type, identifier, status='completed', error_message=None):
        """Mark a checkpoint as completed or failed"""
        engine = self.get_engine()
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO batch_checkpoint (job_type, identifier, status, error_message, updated_at)
                VALUES (:job_type, :identifier, :status, :error_message, NOW())
                ON CONFLICT (job_type, identifier) DO UPDATE SET
                    status = EXCLUDED.status,
                    error_message = EXCLUDED.error_message,
                    updated_at = NOW()
            """), {
                'job_type': job_type, 'identifier': identifier,
                'status': status, 'error_message': error_message
            })
            conn.commit()

    def get_completed_checkpoints(self, job_type):
        """Get set of completed identifiers for a job type"""
        engine = self.get_engine()
        df = pd.read_sql_query(
            text("SELECT identifier FROM batch_checkpoint WHERE job_type = :jt AND status = 'completed'"),
            engine, params={'jt': job_type}
        )
        return set(df['identifier'].tolist())

    # ==========================================
    # USER EMAIL OPERATIONS
    # ==========================================

    def ensure_user_emails_table(self):
        """Create user_emails table if it doesn't exist."""
        engine = self.get_engine()
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS user_emails (
                    id SERIAL PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()

    def email_exists(self, email):
        """Check if an email already exists in the database."""
        engine = self.get_engine()
        df = pd.read_sql_query(
            text("SELECT 1 FROM user_emails WHERE email = :email"),
            engine, params={'email': email.strip().lower()}
        )
        return not df.empty

    def insert_user_email(self, email):
        """Insert a new user email. Returns (success, message)."""
        email = email.strip().lower()
        self.ensure_user_emails_table()
        if self.email_exists(email):
            return False, "This email is already registered. Please use a different email address."
        engine = self.get_engine()
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO user_emails (email) VALUES (:email)
            """), {'email': email})
            conn.commit()
        return True, "Email registered successfully."

    # ==========================================
    # SEARCH OPERATIONS
    # ==========================================

    def search_all_assets(self, search_term, asset_types=None, limit=100):
        """Search across all asset types"""
        engine = self.get_engine()

        query = """
            SELECT a.*,
                   COALESCE(mf.scheme_name, s.company_name, b.bond_name) as full_name,
                   COALESCE(mf.amc_name, s.sector, b.issuer) as additional_info
            FROM assets a
            LEFT JOIN mutual_funds mf ON a.isin = mf.isin
            LEFT JOIN stocks s ON a.isin = s.isin
            LEFT JOIN bonds b ON a.isin = b.isin
            WHERE (a.name ILIKE %(search)s
                   OR a.symbol ILIKE %(search)s
                   OR a.isin ILIKE %(search)s)
        """

        params = {'search': f'%{search_term}%', 'limit': limit}

        if asset_types:
            placeholders = ','.join([f"'{t}'" for t in asset_types])
            query += f" AND a.asset_type IN ({placeholders})"

        query += " LIMIT %(limit)s"

        return pd.read_sql_query(query, engine, params=params)
