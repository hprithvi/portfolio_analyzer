# database/multi_asset_db_manager.py

from database.db_manager import DatabaseManager
import pandas as pd
from datetime import datetime

class MultiAssetDBManager(DatabaseManager):
    """Extended database manager for multi-asset support"""
    
    def __init__(self, db_url=None, db_type='postgresql'):
        super().__init__(db_url, db_type)
    
    # ==========================================
    # ASSET OPERATIONS
    # ==========================================
    
    def insert_asset(self, isin, asset_type, symbol, name, exchange=None):
        """Insert or update asset in master table"""
        engine = self.get_engine()
        
        asset_data = {
            'isin': isin,
            'asset_type': asset_type,
            'symbol': symbol,
            'name': name,
            'exchange': exchange,
            'updated_at': datetime.now()
        }
        
        df = pd.DataFrame([asset_data])
        df.to_sql('assets', engine, if_exists='append', index=False, method='multi')
    
    def get_assets_by_type(self, asset_type, limit=1000):
        """Get all assets of a specific type"""
        engine = self.get_engine()
        query = f"""
            SELECT * FROM assets 
            WHERE asset_type = '{asset_type}' AND is_active = TRUE
            ORDER BY name
            LIMIT {limit}
        """
        return pd.read_sql_query(query, engine)
    
    # ==========================================
    # MUTUAL FUND OPERATIONS
    # ==========================================
    
    def insert_mutual_fund(self, fund_data):
        """Insert mutual fund specific data"""
        engine = self.get_engine()
        
        # First insert into assets table
        if 'isin' in fund_data:
            self.insert_asset(
                isin=fund_data['isin'],
                asset_type='mutual_fund',
                symbol=fund_data.get('scheme_code'),
                name=fund_data.get('scheme_name'),
                exchange='AMFI'
            )
        
        # Then insert into mutual_funds table
        df = pd.DataFrame([fund_data])
        df.to_sql('mutual_funds', engine, if_exists='append', index=False)
    
    # ==========================================
    # STOCK OPERATIONS
    # ==========================================
    
    def insert_stock(self, stock_data):
        """Insert stock specific data"""
        engine = self.get_engine()
        
        # Insert into assets table
        self.insert_asset(
            isin=stock_data['isin'],
            asset_type='stock',
            symbol=stock_data['symbol'],
            name=stock_data['company_name'],
            exchange=stock_data.get('exchange', 'NSE')
        )
        
        # Insert into stocks table
        df = pd.DataFrame([stock_data])
        df.to_sql('stocks', engine, if_exists='append', index=False)
    
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
    
    def insert_price_history_bulk(self, price_df, isin):
        """Insert bulk price history for any asset"""
        engine = self.get_engine()
        
        price_df = price_df.copy()
        price_df['isin'] = isin
        
        # Ensure required columns
        required_cols = ['isin', 'date', 'close_price']
        if not all(col in price_df.columns for col in required_cols):
            raise ValueError("Missing required columns")
        
        price_df.to_sql('price_history', engine, if_exists='append', index=False)
    
    def get_price_history(self, isin, start_date=None, end_date=None):
        """Get price history for any asset by ISIN"""
        engine = self.get_engine()
        
        query = "SELECT * FROM price_history WHERE isin = %(isin)s"
        params = {'isin': isin}
        
        if start_date:
            query += " AND date >= %(start_date)s"
            params['start_date'] = start_date
        
        if end_date:
            query += " AND date <= %(end_date)s"
            params['end_date'] = end_date
        
        query += " ORDER BY date"
        
        return pd.read_sql_query(query, engine, params=params, parse_dates=['date'])
    
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
