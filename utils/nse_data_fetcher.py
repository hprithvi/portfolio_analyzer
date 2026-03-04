# utils/nse_data_fetcher.py

from nsepy import get_history
from datetime import datetime, timedelta
import pandas as pd

class NSEDataFetcher:
    """Use NSEPy library for reliable NSE data fetching"""
    
    def fetch_stock_data(self, symbol, start_date, end_date):
        """Fetch data for a single stock"""
        try:
            df = get_history(
                symbol=symbol,
                start=start_date,
                end=end_date
            )
            
            if df.empty:
                return None
            
            # Standardize columns
            df = df.reset_index()
            df = df.rename(columns={
                'Date': 'date',
                'Symbol': 'symbol',
                'Open': 'open_price',
                'High': 'high_price',
                'Low': 'low_price',
                'Close': 'close_price',
                'Last': 'last_price',
                'Volume': 'volume',
                'Turnover': 'value',
                'Trades': 'trades',
                'Deliverable Volume': 'deliverable_volume',
                '%Deliverble': 'deliverable_pct'
            })
            
            return df
        
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def fetch_all_symbols(self):
        """Get list of all NSE symbols"""
        try:
            from nsepy import get_symbol_list
            symbols = get_symbol_list()
            return symbols
        except Exception as e:
            print(f"Error fetching symbol list: {e}")
            return []