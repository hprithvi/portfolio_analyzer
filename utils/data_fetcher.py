import requests
import pandas as pd
from datetime import datetime, timedelta
import time

class MutualFundDataFetcher:
    """Fetch mutual fund data from various sources"""
    
    def __init__(self):
        self.base_url = "https://api.mfapi.in"
    
    def get_all_schemes(self):
        """Get list of all mutual fund schemes"""
        try:
            url = f"{self.base_url}/mf"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data)
                return df
            else:
                print(f"Error: Status code {response.status_code}")
                return pd.DataFrame()
        
        except Exception as e:
            print(f"Error fetching schemes: {e}")
            return pd.DataFrame()
    
    def get_scheme_details(self, scheme_code):
        """Get details and NAV history for a specific scheme"""
        try:
            url = f"{self.base_url}/mf/{scheme_code}"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return data
            else:
                print(f"Error fetching {scheme_code}: Status {response.status_code}")
                return None
        
        except Exception as e:
            print(f"Error fetching scheme {scheme_code}: {e}")
            return None
    
    def get_nav_history(self, scheme_code, start_date=None):
        """Get NAV history as DataFrame"""
        data = self.get_scheme_details(scheme_code)
        
        if not data or 'data' not in data:
            return None, None
        
        # Parse NAV history
        nav_data = data['data']
        df = pd.DataFrame(nav_data)
        
        if df.empty:
            return None, None
        
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
        df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
        df = df.dropna()
        df = df.set_index('date').sort_index()
        
        # Filter by start date
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df.index >= start_dt]
        
        # Extract metadata
        metadata = {
            'scheme_code': scheme_code,
            'scheme_name': data.get('meta', {}).get('scheme_name', ''),
            'fund_house': data.get('meta', {}).get('fund_house', ''),
            'scheme_type': data.get('meta', {}).get('scheme_type', ''),
            'scheme_category': data.get('meta', {}).get('scheme_category', ''),
        }
        
        return df[['nav']], metadata
    
    def calculate_returns(self, nav_df):
        """Calculate returns from NAV data"""
        if nav_df is None or nav_df.empty:
            return {}
        
        nav_series = nav_df['nav']
        
        returns = {}
        
        # Current NAV
        returns['current_nav'] = float(nav_series.iloc[-1])
        returns['nav_date'] = nav_series.index[-1].strftime('%Y-%m-%d')
        
        # Calculate period returns
        periods = {
            '1m': 30,
            '3m': 90,
            '6m': 180,
            '1y': 365,
            '3y': 1095,
            '5y': 1825
        }
        
        for period_name, days in periods.items():
            try:
                if len(nav_series) > days:
                    start_nav = nav_series.iloc[-days]
                    end_nav = nav_series.iloc[-1]
                    period_return = ((end_nav / start_nav) - 1) * 100
                    returns[f'returns_{period_name}'] = round(period_return, 2)
                else:
                    returns[f'returns_{period_name}'] = None
            except:
                returns[f'returns_{period_name}'] = None
        
        return returns
