import requests
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from io import StringIO
import time
from bs4 import BeautifulSoup
import zipfile
import io

class MultiAssetDataFetcher:
    """Fetch data for multiple asset classes"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    # ==========================================
    # MUTUAL FUNDS
    # ==========================================
    
    def fetch_all_mf_schemes(self):
        """Get list of all mutual fund schemes with ISINs"""
        try:
            url = "https://api.mfapi.in/mf"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                schemes = response.json()
                df = pd.DataFrame(schemes)
                
                # Fetch ISIN for each scheme (from individual API calls)
                print("Fetching ISIN data for schemes...")
                isin_data = []
                
                for idx, scheme in df.iterrows():
                    if idx % 100 == 0:
                        print(f"Processed {idx}/{len(df)} schemes")
                    
                    scheme_code = scheme['schemeCode']
                    details = self.fetch_mf_scheme_details(scheme_code)
                    
                    if details and 'meta' in details:
                        meta = details['meta']
                        isin_data.append({
                            'scheme_code': scheme_code,
                            'scheme_name': scheme['schemeName'],
                            'isin': meta.get('scheme_isin', None),
                            'amc': meta.get('fund_house', ''),
                            'category': meta.get('scheme_type', ''),
                            'sub_category': meta.get('scheme_category', '')
                        })
                    
                    time.sleep(0.2)  # Rate limiting
                
                return pd.DataFrame(isin_data)
            
            return pd.DataFrame()
        
        except Exception as e:
            print(f"Error fetching MF schemes: {e}")
            return pd.DataFrame()
    
    def fetch_mf_scheme_details(self, scheme_code):
        """Get details for a specific MF scheme"""
        try:
            url = f"https://api.mfapi.in/mf/{scheme_code}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            return None
        
        except Exception as e:
            print(f"Error fetching scheme {scheme_code}: {e}")
            return None
    
    def fetch_mf_nav_history(self, scheme_code, start_date=None):
        """Get NAV history for a mutual fund"""
        data = self.fetch_mf_scheme_details(scheme_code)
        
        if not data or 'data' not in data:
            return None
        
        df = pd.DataFrame(data['data'])
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
        df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
        df = df.dropna()
        df = df.set_index('date').sort_index()
        
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df.index >= start_dt]
        
        # Rename to match common schema
        df.columns = ['close_price']
        df['open_price'] = df['close_price']
        df['high_price'] = df['close_price']
        df['low_price'] = df['close_price']
        df['volume'] = 0
        
        return df
    
    # ==========================================
    # STOCKS
    # ==========================================
    
    def fetch_nse_equity_list(self):
        """Fetch list of all NSE equity symbols with ISINs"""
        try:
            url = "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data['data'])
                return df
            
            # Fallback: Download equity list CSV
            csv_url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
            df = pd.read_csv(csv_url)
            return df
        
        except Exception as e:
            print(f"Error fetching NSE equity list: {e}")
            return pd.DataFrame()
    
    def fetch_nse_bhav_copy(self, date=None):
        """Fetch NSE bhav copy (end of day prices) for a specific date"""
        if date is None:
            date = datetime.now()
        
        date_str = date.strftime('%d%m%Y')
        month = date.strftime('%b').upper()
        year = date.strftime('%Y')
        
        try:
            # NSE bhav copy URL
            url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{date_str}.csv"
            
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text))
                
                # Clean and standardize
                df.columns = df.columns.str.strip()
                df['DATE1'] = pd.to_datetime(df['DATE1'], format='%d-%b-%Y')
                
                # Rename columns to match schema
                df = df.rename(columns={
                    'SYMBOL': 'symbol',
                    'ISIN': 'isin',
                    'DATE1': 'date',
                    'OPEN': 'open_price',
                    'HIGH': 'high_price',
                    'LOW': 'low_price',
                    'CLOSE': 'close_price',
                    'TOTTRDQTY': 'volume',
                    'TOTTRDVAL': 'value',
                    'TOTALTRADES': 'trades'
                })
                
                return df[['isin', 'symbol', 'date', 'open_price', 'high_price', 
                          'low_price', 'close_price', 'volume', 'value', 'trades']]
            
            return pd.DataFrame()
        
        except Exception as e:
            print(f"Error fetching NSE bhav copy for {date_str}: {e}")
            return pd.DataFrame()
    
    def fetch_stock_data_yfinance(self, symbol, start_date, end_date=None):
        """Fetch stock data using Yahoo Finance (backup method)"""
        try:
            if end_date is None:
                end_date = datetime.now()
            
            # Add .NS for NSE symbols
            yf_symbol = f"{symbol}.NS"
            
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                # Try BSE
                yf_symbol = f"{symbol}.BO"
                ticker = yf.Ticker(yf_symbol)
                df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                return None
            
            # Standardize columns
            df = df.reset_index()
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open_price',
                'High': 'high_price',
                'Low': 'low_price',
                'Close': 'close_price',
                'Volume': 'volume'
            })
            
            df['value'] = df['close_price'] * df['volume']
            df['trades'] = 0
            
            return df[['date', 'open_price', 'high_price', 'low_price', 
                      'close_price', 'volume', 'value', 'trades']]
        
        except Exception as e:
            print(f"Error fetching {symbol} from Yahoo Finance: {e}")
            return None
    
    # ==========================================
    # BONDS
    # ==========================================
    
    def fetch_nse_bonds_list(self):
        """Fetch list of corporate bonds from NSE"""
        try:
            url = "https://www.nseindia.com/api/corporates-债券-data?index=debt"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return pd.DataFrame(data)
            
            return pd.DataFrame()
        
        except Exception as e:
            print(f"Error fetching NSE bonds: {e}")
            return pd.DataFrame()
    
    def fetch_gsec_prices(self):
        """Fetch Government Securities prices from RBI"""
        try:
            # RBI reference rate for G-Secs
            url = "https://www.rbi.org.in/Scripts/BS_ViewMktRates.aspx"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                # Parse HTML to extract G-Sec data
                soup = BeautifulSoup(response.content, 'html.parser')
                # Extract table data (implementation depends on page structure)
                # This is a placeholder - actual implementation needed
                return pd.DataFrame()
            
            return pd.DataFrame()
        
        except Exception as e:
            print(f"Error fetching G-Sec data: {e}")
            return pd.DataFrame()
    
    # ==========================================
    # ISIN LOOKUP
    # ==========================================
    
    def lookup_isin_nse(self, symbol):
        """Get ISIN for a given NSE symbol"""
        try:
            url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('info', {}).get('isin', None)
            
            return None
        
        except Exception as e:
            print(f"Error looking up ISIN for {symbol}: {e}")
            return None
    
    # ==========================================
    # BULK DOWNLOAD
    # ==========================================
    
    def download_nse_historical(self, start_date, end_date):
        """Download historical NSE data for date range"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        all_data = []
        
        for date in date_range:
            # Skip weekends
            if date.weekday() >= 5:
                continue
            
            print(f"Downloading data for {date.strftime('%Y-%m-%d')}...")
            df = self.fetch_nse_bhav_copy(date)
            
            if not df.empty:
                all_data.append(df)
            
            time.sleep(1)  # Rate limiting
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        
        return pd.DataFrame()
