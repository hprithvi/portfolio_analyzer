import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.pool import QueuePool
import streamlit as st

class DatabaseManager:
    def __init__(self, db_url=None, db_type='postgressql'):
        """
        Initialize database connection
        
        Args:
            db_url: Database connection URL
            db_type: 'sqlite' or 'postgresql'
        """
        self.db_type = db_type
        self.db_url = db_url or self._get_db_url()
        self.engine = None
        self.init_database()
    
    def _get_db_url(self):
        """Get database URL from environment or default"""
        if self.db_type == 'postgresql':
            # Try to get from environment variables
            db_url = os.getenv('DATABASE_URL')
            
            if not db_url:
                # Build from individual components
                db_host = os.getenv('DB_HOST', 'localhost')
                db_port = os.getenv('DB_PORT', '5432')
                db_name = os.getenv('DB_NAME', 'mutual_funds')
                db_user = os.getenv('DB_USER', 'postgres')
                db_password = os.getenv('DB_PASSWORD', '')
                
                db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            
            # Fix for Heroku postgres URLs
            if db_url.startswith('postgres://'):
                db_url = db_url.replace('postgres://', 'postgresql://', 1)
            
            return db_url
        else:
            # SQLite default
            return 'sqlite:///mutual_funds.db'
    
    def get_engine(self):
        """Get or create SQLAlchemy engine"""
        if self.engine is None:
            if self.db_type == 'postgresql':
                # PostgreSQL with connection pooling
                self.engine = create_engine(
                    self.db_url,
                    poolclass=QueuePool,
                    pool_size=15,
                    max_overflow=20,
                    pool_pre_ping=True,  # Verify connections before using
                    pool_recycle=3600,   # Recycle connections after 1 hour
                    echo=False
                )
            else:
                # SQLite
                self.engine = create_engine(
                    self.db_url,
                    echo=False,
                    connect_args={'check_same_thread': False}
                )
        return self.engine
    
    def test_connection(self):
        """Test database connection"""
        try:
            engine = self.get_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True, "Connection successful"
        except Exception as e:
            return False, str(e)
    
    def init_database(self):
        """Initialize database with schema"""
        engine = self.get_engine()
        
        # Check if tables exist
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        
        if 'mutual_funds' not in existing_tables:
            self._create_schema()
    
    def _create_schema(self):
        """Create database schema"""
        engine = self.get_engine()
        
        # Use appropriate syntax for each database
        if self.db_type == 'postgresql':
            schema_sql = self._get_postgres_schema()
        else:
            schema_sql = self._get_sqlite_schema()
        
        with engine.connect() as conn:
            # Execute schema
            for statement in schema_sql.split(';'):
                statement = statement.strip()
                if statement:
                    conn.execute(text(statement))
            conn.commit()
    
    def _get_postgres_schema(self):
        """PostgreSQL schema"""
        return """
        CREATE TABLE IF NOT EXISTS mutual_funds (
            scheme_code TEXT PRIMARY KEY,
            scheme_name TEXT NOT NULL,
            fund_house TEXT,
            category TEXT,
            sub_category TEXT,
            launch_date DATE,
            nav_date DATE,
            current_nav REAL,
            aum REAL,
            expense_ratio REAL,
            min_investment REAL,
            exit_load TEXT,
            returns_1m REAL,
            returns_3m REAL,
            returns_6m REAL,
            returns_1y REAL,
            returns_3y REAL,
            returns_5y REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS nav_history (
            id SERIAL PRIMARY KEY,
            scheme_code TEXT NOT NULL,
            date DATE NOT NULL,
            nav REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (scheme_code) REFERENCES mutual_funds(scheme_code) ON DELETE CASCADE,
            UNIQUE(scheme_code, date)
        );
        
        CREATE INDEX IF NOT EXISTS idx_nav_history_scheme_date 
        ON nav_history(scheme_code, date);
        
        CREATE INDEX IF NOT EXISTS idx_mutual_funds_category 
        ON mutual_funds(category);
        
        CREATE INDEX IF NOT EXISTS idx_mutual_funds_name 
        ON mutual_funds(scheme_name);
        
        CREATE TABLE IF NOT EXISTS fund_statistics (
            scheme_code TEXT PRIMARY KEY,
            mean_daily_return REAL,
            std_daily_return REAL,
            annual_volatility REAL,
            sharpe_ratio REAL,
            max_drawdown REAL,
            avg_30d_volume REAL,
            beta REAL,
            alpha REAL,
            calculated_date DATE,
            FOREIGN KEY (scheme_code) REFERENCES mutual_funds(scheme_code) ON DELETE CASCADE
        );
        
        CREATE TABLE IF NOT EXISTS fund_correlations (
            scheme_code_1 TEXT NOT NULL,
            scheme_code_2 TEXT NOT NULL,
            correlation REAL NOT NULL,
            period_days INTEGER NOT NULL,
            calculated_date DATE NOT NULL,
            PRIMARY KEY (scheme_code_1, scheme_code_2, period_days),
            FOREIGN KEY (scheme_code_1) REFERENCES mutual_funds(scheme_code) ON DELETE CASCADE,
            FOREIGN KEY (scheme_code_2) REFERENCES mutual_funds(scheme_code) ON DELETE CASCADE
        )
        """
    
    def _get_sqlite_schema(self):
        """SQLite schema"""
        return """
        CREATE TABLE IF NOT EXISTS mutual_funds (
            scheme_code TEXT PRIMARY KEY,
            scheme_name TEXT NOT NULL,
            fund_house TEXT,
            category TEXT,
            sub_category TEXT,
            launch_date DATE,
            nav_date DATE,
            current_nav REAL,
            aum REAL,
            expense_ratio REAL,
            min_investment REAL,
            exit_load TEXT,
            returns_1m REAL,
            returns_3m REAL,
            returns_6m REAL,
            returns_1y REAL,
            returns_3y REAL,
            returns_5y REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS nav_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scheme_code TEXT NOT NULL,
            date DATE NOT NULL,
            nav REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (scheme_code) REFERENCES mutual_funds(scheme_code),
            UNIQUE(scheme_code, date)
        );
        
        CREATE INDEX IF NOT EXISTS idx_nav_history_scheme_date 
        ON nav_history(scheme_code, date);
        
        CREATE INDEX IF NOT EXISTS idx_mutual_funds_category 
        ON mutual_funds(category);
        
        CREATE INDEX IF NOT EXISTS idx_mutual_funds_name 
        ON mutual_funds(scheme_name);
        
        CREATE TABLE IF NOT EXISTS fund_statistics (
            scheme_code TEXT PRIMARY KEY,
            mean_daily_return REAL,
            std_daily_return REAL,
            annual_volatility REAL,
            sharpe_ratio REAL,
            max_drawdown REAL,
            avg_30d_volume REAL,
            beta REAL,
            alpha REAL,
            calculated_date DATE,
            FOREIGN KEY (scheme_code) REFERENCES mutual_funds(scheme_code)
        );
        
        CREATE TABLE IF NOT EXISTS fund_correlations (
            scheme_code_1 TEXT NOT NULL,
            scheme_code_2 TEXT NOT NULL,
            correlation REAL NOT NULL,
            period_days INTEGER NOT NULL,
            calculated_date DATE NOT NULL,
            PRIMARY KEY (scheme_code_1, scheme_code_2, period_days),
            FOREIGN KEY (scheme_code_1) REFERENCES mutual_funds(scheme_code),
            FOREIGN KEY (scheme_code_2) REFERENCES mutual_funds(scheme_code)
        )
        """
    
    # ==========================================
    # MUTUAL FUND METADATA OPERATIONS
    # ==========================================
    
    def insert_mutual_fund(self, fund_data):
        """Insert or update mutual fund metadata"""
        engine = self.get_engine()
        
        # Prepare data
        df = pd.DataFrame([fund_data])
        
        # Use different methods for different databases
        with engine.connect() as conn:
            if self.db_type == 'postgresql':
                # PostgreSQL upsert
                from sqlalchemy.dialects.postgresql import insert
                
                stmt = insert(text('mutual_funds')).values(**fund_data)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['scheme_code'],
                    set_=fund_data
                )
                conn.execute(stmt)
            else:
                # SQLite replace
                df.to_sql('mutual_funds', conn, if_exists='append', index=False, method='multi')
            
            conn.commit()
    
    def get_all_funds(self):
        """Get all mutual funds"""
        engine = self.get_engine()
        query = 'SELECT * FROM mutual_funds ORDER BY scheme_name'
        return pd.read_sql_query(query, engine)
    
    def search_funds(self, search_term='', category=None, fund_house=None, limit=100):
        """Search mutual funds with filters"""
        engine = self.get_engine()
        
        conditions = []
        params = {}
        
        if search_term:
            if self.db_type == 'postgresql':
                conditions.append("(scheme_name ILIKE :search OR scheme_code ILIKE :search)")
            else:
                conditions.append("(scheme_name LIKE :search OR scheme_code LIKE :search)")
            params['search'] = f'%{search_term}%'
        
        if category:
            conditions.append("category = :category")
            params['category'] = category
        
        if fund_house:
            conditions.append("fund_house = :fund_house")
            params['fund_house'] = fund_house
        
        where_clause = ' AND '.join(conditions) if conditions else '1=1'
        query = f"""
            SELECT * FROM mutual_funds 
            WHERE {where_clause}
            ORDER BY scheme_name 
            LIMIT :limit
        """
        params['limit'] = limit
        
        return pd.read_sql_query(text(query), engine, params=params)
    
    def get_fund_details(self, scheme_code):
        """Get details of a specific fund"""
        engine = self.get_engine()
        query = text('SELECT * FROM mutual_funds WHERE scheme_code = :code')
        result = pd.read_sql_query(query, engine, params={'code': scheme_code})
        return result.iloc[0] if not result.empty else None
    
    def get_categories(self):
        """Get all unique categories"""
        engine = self.get_engine()
        query = 'SELECT DISTINCT category FROM mutual_funds WHERE category IS NOT NULL ORDER BY category'
        result = pd.read_sql_query(query, engine)
        return result['category'].tolist()
    
    def get_fund_houses(self):
        """Get all unique fund houses"""
        engine = self.get_engine()
        query = 'SELECT DISTINCT fund_house FROM mutual_funds WHERE fund_house IS NOT NULL ORDER BY fund_house'
        result = pd.read_sql_query(query, engine)
        return result['fund_house'].tolist()
    
    # ==========================================
    # NAV HISTORY OPERATIONS
    # ==========================================
    
    def insert_nav_bulk(self, nav_df, scheme_code):
        """Insert bulk NAV data from DataFrame"""
        engine = self.get_engine()
        
        # Prepare data
        nav_df = nav_df.copy()
        nav_df['scheme_code'] = scheme_code
        nav_df = nav_df.reset_index()
        nav_df.columns = ['date', 'nav', 'scheme_code']
        nav_df = nav_df[['scheme_code', 'date', 'nav']]
        
        # Convert date to string
        nav_df['date'] = pd.to_datetime(nav_df['date']).dt.strftime('%Y-%m-%d')
        
        # Insert in chunks to avoid memory issues
        chunk_size = 1000
        with engine.connect() as conn:
            for i in range(0, len(nav_df), chunk_size):
                chunk = nav_df.iloc[i:i+chunk_size]
                
                if self.db_type == 'postgresql':
                    # Use INSERT ... ON CONFLICT for PostgreSQL
                    chunk.to_sql('nav_history', conn, if_exists='append', 
                                index=False, method='multi')
                else:
                    # SQLite
                    chunk.to_sql('nav_history', conn, if_exists='append', index=False)
            
            conn.commit()
    
    def get_nav_history(self, scheme_code, start_date=None, end_date=None):
        """Get NAV history for a fund"""
        engine = self.get_engine()
        
        conditions = ['scheme_code = :code']
        params = {'code': scheme_code}
        
        if start_date:
            conditions.append('date >= :start_date')
            params['start_date'] = start_date
        
        if end_date:
            conditions.append('date <= :end_date')
            params['end_date'] = end_date
        
        where_clause = ' AND '.join(conditions)
        query = text(f"""
            SELECT date, nav FROM nav_history 
            WHERE {where_clause}
            ORDER BY date
        """)
        
        df = pd.read_sql_query(query, engine, params=params, parse_dates=['date'])
        
        if not df.empty:
            df = df.set_index('date')
        
        return df
    
    def get_latest_nav_date(self, scheme_code):
        """Get the latest NAV date for a fund"""
        engine = self.get_engine()
        query = text('SELECT MAX(date) as latest_date FROM nav_history WHERE scheme_code = :code')
        result = pd.read_sql_query(query, engine, params={'code': scheme_code})
        return result['latest_date'].iloc[0] if not result.empty else None
    
    # ==========================================
    # STATISTICS OPERATIONS
    # ==========================================
    
    def calculate_and_store_statistics(self, scheme_code, period_days=365):
        """Calculate and store fund statistics"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        nav_df = self.get_nav_history(
            scheme_code, 
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if nav_df.empty or len(nav_df) < 30:
            return None
        
        # Calculate returns
        returns = nav_df['nav'].pct_change().dropna()
        
        if len(returns) == 0:
            return None
        
        # Calculate statistics
        mean_return = returns.mean()
        std_return = returns.std()
        annual_vol = std_return * np.sqrt(252)
        sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        max_dd = ((cumulative.cummax() - cumulative) / cumulative.cummax()).max()
        
        # Store statistics
        engine = self.get_engine()
        
        stats_data = {
            'scheme_code': scheme_code,
            'mean_daily_return': float(mean_return),
            'std_daily_return': float(std_return),
            'annual_volatility': float(annual_vol),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_dd),
            'calculated_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        df = pd.DataFrame([stats_data])
        
        with engine.connect() as conn:
            # Replace existing record
            conn.execute(text('DELETE FROM fund_statistics WHERE scheme_code = :code'), 
                        {'code': scheme_code})
            df.to_sql('fund_statistics', conn, if_exists='append', index=False)
            conn.commit()
        
        return stats_data
    
    def get_fund_statistics(self, scheme_code):
        """Get stored statistics for a fund"""
        engine = self.get_engine()
        query = text('SELECT * FROM fund_statistics WHERE scheme_code = :code')
        result = pd.read_sql_query(query, engine, params={'code': scheme_code})
        return result.iloc[0].to_dict() if not result.empty else None
    
    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            self.engine = None