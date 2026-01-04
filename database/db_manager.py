import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

class DatabaseManager:
    def __init__(self, db_path='mutual_funds.db'):
        """Initialize database connection"""
        self.db_path = db_path
        self.conn = None
        self.init_database()
    
    def get_connection(self):
        """Get database connection"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return self.conn
    
    def init_database(self):
        """Initialize database with schema"""
        conn = self.get_connection()
        
        # Read and execute schema
        schema_path = Path(__file__).parent / 'schema.sql'
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                schema = f.read()
                conn.executescript(schema)
        else:
            # Inline schema if file doesn't exist
            self._create_schema()
        
        conn.commit()
    
    def _create_schema(self):
        """Create database schema inline"""
        conn = self.get_connection()
        
        # Mutual Funds Table
        conn.execute('''
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
            )
        ''')
        
        # NAV History Table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS nav_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scheme_code TEXT NOT NULL,
                date DATE NOT NULL,
                nav REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (scheme_code) REFERENCES mutual_funds(scheme_code),
                UNIQUE(scheme_code, date)
            )
        ''')
        
        # Fund Statistics
        conn.execute('''
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
            )
        ''')
        
        # Correlations
        conn.execute('''
            CREATE TABLE IF NOT EXISTS fund_correlations (
                scheme_code_1 TEXT NOT NULL,
                scheme_code_2 TEXT NOT NULL,
                correlation REAL NOT NULL,
                period_days INTEGER NOT NULL,
                calculated_date DATE NOT NULL,
                PRIMARY KEY (scheme_code_1, scheme_code_2, period_days)
            )
        ''')
        
        # Create indexes
        conn.execute('CREATE INDEX IF NOT EXISTS idx_nav_history_scheme_date ON nav_history(scheme_code, date)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_mutual_funds_category ON mutual_funds(category)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_mutual_funds_name ON mutual_funds(scheme_name)')
        
        conn.commit()
    
    # ==========================================
    # MUTUAL FUND METADATA OPERATIONS
    # ==========================================
    
    def insert_mutual_fund(self, fund_data):
        """Insert or update mutual fund metadata"""
        conn = self.get_connection()
        
        columns = ', '.join(fund_data.keys())
        placeholders = ', '.join(['?' for _ in fund_data])
        
        query = f'''
            INSERT OR REPLACE INTO mutual_funds ({columns})
            VALUES ({placeholders})
        '''
        
        conn.execute(query, list(fund_data.values()))
        conn.commit()
    
    def get_all_funds(self):
        """Get all mutual funds"""
        conn = self.get_connection()
        query = 'SELECT * FROM mutual_funds ORDER BY scheme_name'
        return pd.read_sql_query(query, conn)
    
    def search_funds(self, search_term='', category=None, fund_house=None):
        """Search mutual funds with filters"""
        conn = self.get_connection()
        
        query = 'SELECT * FROM mutual_funds WHERE 1=1'
        params = []
        
        if search_term:
            query += ' AND (scheme_name LIKE ? OR scheme_code LIKE ?)'
            params.extend([f'%{search_term}%', f'%{search_term}%'])
        
        if category:
            query += ' AND category = ?'
            params.append(category)
        
        if fund_house:
            query += ' AND fund_house = ?'
            params.append(fund_house)
        
        query += ' ORDER BY scheme_name'
        
        return pd.read_sql_query(query, conn, params=params)
    
    def get_fund_details(self, scheme_code):
        """Get details of a specific fund"""
        conn = self.get_connection()
        query = 'SELECT * FROM mutual_funds WHERE scheme_code = ?'
        result = pd.read_sql_query(query, conn, params=[scheme_code])
        return result.iloc[0] if not result.empty else None
    
    def get_categories(self):
        """Get all unique categories"""
        conn = self.get_connection()
        query = 'SELECT DISTINCT category FROM mutual_funds WHERE category IS NOT NULL ORDER BY category'
        result = pd.read_sql_query(query, conn)
        return result['category'].tolist()
    
    def get_fund_houses(self):
        """Get all unique fund houses"""
        conn = self.get_connection()
        query = 'SELECT DISTINCT fund_house FROM mutual_funds WHERE fund_house IS NOT NULL ORDER BY fund_house'
        result = pd.read_sql_query(query, conn)
        return result['fund_house'].tolist()
    
    # ==========================================
    # NAV HISTORY OPERATIONS
    # ==========================================
    
    def insert_nav_data(self, scheme_code, date, nav):
        """Insert NAV data for a specific date"""
        conn = self.get_connection()
        
        query = '''
            INSERT OR REPLACE INTO nav_history (scheme_code, date, nav)
            VALUES (?, ?, ?)
        '''
        
        conn.execute(query, (scheme_code, date, nav))
        conn.commit()
    
    def insert_nav_bulk(self, nav_df, scheme_code):
        """Insert bulk NAV data from DataFrame"""
        conn = self.get_connection()
        
        # Prepare data
        nav_df = nav_df.copy()
        nav_df['scheme_code'] = scheme_code
        nav_df = nav_df.reset_index()
        nav_df.columns = ['date', 'nav', 'scheme_code']
        nav_df = nav_df[['scheme_code', 'date', 'nav']]
        
        # Convert date to string
        nav_df['date'] = pd.to_datetime(nav_df['date']).dt.strftime('%Y-%m-%d')
        
        # Insert
        nav_df.to_sql('nav_history', conn, if_exists='append', index=False)
        conn.commit()
    
    def get_nav_history(self, scheme_code, start_date=None, end_date=None):
        """Get NAV history for a fund"""
        conn = self.get_connection()
        
        query = 'SELECT date, nav FROM nav_history WHERE scheme_code = ?'
        params = [scheme_code]
        
        if start_date:
            query += ' AND date >= ?'
            params.append(start_date)
        
        if end_date:
            query += ' AND date <= ?'
            params.append(end_date)
        
        query += ' ORDER BY date'
        
        df = pd.read_sql_query(query, conn, params=params, parse_dates=['date'])
        
        if not df.empty:
            df = df.set_index('date')
        
        return df
    
    def get_latest_nav_date(self, scheme_code):
        """Get the latest NAV date for a fund"""
        conn = self.get_connection()
        query = 'SELECT MAX(date) as latest_date FROM nav_history WHERE scheme_code = ?'
        result = pd.read_sql_query(query, conn, params=[scheme_code])
        return result['latest_date'].iloc[0] if not result.empty else None
    
    # ==========================================
    # STATISTICS OPERATIONS
    # ==========================================
    
    def calculate_and_store_statistics(self, scheme_code, period_days=365):
        """Calculate and store fund statistics"""
        # Get NAV data
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
        conn = self.get_connection()
        
        query = '''
            INSERT OR REPLACE INTO fund_statistics 
            (scheme_code, mean_daily_return, std_daily_return, annual_volatility, 
             sharpe_ratio, max_drawdown, calculated_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        '''
        
        conn.execute(query, (
            scheme_code,
            float(mean_return),
            float(std_return),
            float(annual_vol),
            float(sharpe),
            float(max_dd),
            datetime.now().strftime('%Y-%m-%d')
        ))
        conn.commit()
        
        return {
            'mean_daily_return': mean_return,
            'std_daily_return': std_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd
        }
    
    def get_fund_statistics(self, scheme_code):
        """Get stored statistics for a fund"""
        conn = self.get_connection()
        query = 'SELECT * FROM fund_statistics WHERE scheme_code = ?'
        result = pd.read_sql_query(query, conn, params=[scheme_code])
        return result.iloc[0].to_dict() if not result.empty else None
    
    # ==========================================
    # CORRELATION OPERATIONS
    # ==========================================
    
    def calculate_and_store_correlation(self, scheme_code_1, scheme_code_2, period_days=365):
        """Calculate and store correlation between two funds"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        # Get NAV data for both funds
        nav1 = self.get_nav_history(scheme_code_1, start_date.strftime('%Y-%m-%d'))
        nav2 = self.get_nav_history(scheme_code_2, start_date.strftime('%Y-%m-%d'))
        
        if nav1.empty or nav2.empty:
            return None
        
        # Merge on date
        merged = pd.merge(nav1, nav2, left_index=True, right_index=True, suffixes=('_1', '_2'))
        
        if len(merged) < 30:
            return None
        
        # Calculate returns
        returns1 = merged['nav_1'].pct_change().dropna()
        returns2 = merged['nav_2'].pct_change().dropna()
        
        # Calculate correlation
        correlation = returns1.corr(returns2)
        
        # Store correlation
        conn = self.get_connection()
        
        query = '''
            INSERT OR REPLACE INTO fund_correlations 
            (scheme_code_1, scheme_code_2, correlation, period_days, calculated_date)
            VALUES (?, ?, ?, ?, ?)
        '''
        
        conn.execute(query, (
            scheme_code_1,
            scheme_code_2,
            float(correlation),
            period_days,
            datetime.now().strftime('%Y-%m-%d')
        ))
        conn.commit()
        
        return float(correlation)
    
    def get_correlation(self, scheme_code_1, scheme_code_2, period_days=365):
        """Get stored correlation between two funds"""
        conn = self.get_connection()
        query = '''
            SELECT correlation FROM fund_correlations 
            WHERE scheme_code_1 = ? AND scheme_code_2 = ? AND period_days = ?
        '''
        result = pd.read_sql_query(query, conn, params=[scheme_code_1, scheme_code_2, period_days])
        return result['correlation'].iloc[0] if not result.empty else None
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
