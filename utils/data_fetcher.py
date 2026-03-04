import requests
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from io import BytesIO, StringIO
import time
from bs4 import BeautifulSoup
import zipfile
import io

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Referer': 'https://www.nseindia.com/',
    'Connection': 'keep-alive',
})

class MultiAssetDataFetcher:
    """Fetch data for multiple asset classes"""
    
    def __init__(self):
        self.session = requests.Session()
        # NSE requires proper headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Get cookies by visiting the main page first
        try:
            self.session.get('https://www.nseindia.com', timeout=10)
            time.sleep(1)
        except:
            pass
    
    # ==========================================
    # MUTUAL FUNDS
    # ==========================================
    
    def fetch_all_mf_schemes(self):
        """Get list of all mutual fund schemes with ISINs from list endpoint directly"""
        try:
            url = "https://api.mfapi.in/mf"
            response = requests.get(url, timeout=30)

            if response.status_code != 200:
                return pd.DataFrame()

            schemes = response.json()
            df = pd.DataFrame(schemes)

            # The list endpoint already returns isinGrowth and isinDivReinvestment
            # Use isinGrowth as primary ISIN; fall back to isinDivReinvestment
            if 'isinGrowth' in df.columns:
                df['isin'] = df['isinGrowth'].where(
                    df['isinGrowth'].notna() & (df['isinGrowth'] != ''),
                    df.get('isinDivReinvestment')
                )
            else:
                # Fallback if API format changes
                df['isin'] = None

            df = df.rename(columns={
                'schemeCode': 'scheme_code',
                'schemeName': 'scheme_name'
            })

            # Filter out rows with no ISIN
            df = df[df['isin'].notna() & (df['isin'] != '')]

            # Deduplicate by ISIN (keep first occurrence)
            df = df.drop_duplicates(subset='isin', keep='first')

            print(f"Found {len(df)} schemes with valid ISINs")
            return df

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
    
    def fetch_mf_nav_history(self, scheme_code, start_date=None, isin=None):
        """Get NAV history for a mutual fund.

        Returns:
            tuple: (DataFrame with price_history columns including isin, meta dict)
                   Returns (None, {}) on failure.
        """
        data = self.fetch_mf_scheme_details(scheme_code)

        if not data or 'data' not in data:
            return None, {}

        meta = data.get('meta', {})

        df = pd.DataFrame(data['data'])
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
        df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
        df = df.dropna()
        df = df.sort_values('date')

        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df['date'] >= start_dt]

        # Build price_history compatible DataFrame
        result = pd.DataFrame({
            'date': df['date'].values,
            'close_price': df['nav'].values,
            'open_price': df['nav'].values,
            'high_price': df['nav'].values,
            'low_price': df['nav'].values,
            'volume': 0
        })

        if isin:
            result['isin'] = isin

        return result, meta
    
    # ==========================================
    # STOCKS
    # ==========================================
        
    def fetch_nse_bhav_copy(self, date=None):
        """
        Fetch NSE bhav copy (end of day prices) for a specific date
        Uses the new NSE archives URL format
        """
        if date is None:
            date = datetime.now() - timedelta(days=1)  # Previous day

        # Skip weekends
        while date.weekday() >= 5:  # 5=Saturday, 6=Sunday
            date = date - timedelta(days=1)

        # Compute date formats after weekend adjustment
        date_formats = {
            'ddmmyyyy': date.strftime('%d%m%Y'),
            'ddMMyyyy': date.strftime('%d%b%Y').upper(),
            'yyyy': date.strftime('%Y'),
            'MMM': date.strftime('%b').upper(),
            'MM': date.strftime('%m'),
            'dd': date.strftime('%d')
        }

        try:
            url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{date_formats['ddmmyyyy']}.csv"

            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                # Check if it's a zip file
                # if url.endswith('.zip'):
                #     try:
                #         with zipfile.ZipFile(BytesIO(response.content)) as z:
                #             files = z.namelist()
                #             print(f"  ✅ ZIP FILE FOUND! Contains: {files}")
                            
                #             # Try to read first file
                #             with z.open(files[0]) as f:
                #                 first_line = f.readline().decode('utf-8')
                #                 #print(f"  First line: {first_line[:100]}")
                #             print(f"  🎉 SUCCESS - This URL works!\n")
                #             # break
                #     except zipfile.BadZipFile:
                #         print(f"  ❌ Not a valid zip file\n")
                
                # Check if it's CSV
                if 'text/csv' in response.headers.get('Content-Type', '') or url.endswith('.csv'):
                    try:
                        content_df = pd.read_csv(StringIO(response.text))
                    except Exception as e:
                        print(f"Error parsing CSV: {e}")
                        return None

                    if len(content_df) > 0:
                        return content_df

                # Check if it's JSON (API response)
                elif 'application/json' in response.headers.get('Content-Type', ''):
                    data = response.json()
                    content_df = pd.DataFrame(data)
                    if len(content_df) > 0:
                        return content_df
                
            else:
                print('Non 200 response')
                return None
        except Exception as e:
            print('Exception Message')
            print(str(e))
            return None
        
        # else:
        #         print(f"  ⚠️  Unexpected content type\n")
        # else:
        #     print(f"  ❌ Failed\n")
    
    # except Exception as e:
    #     print(f"  ❌ Error: {str(e)[:100]}\n")
    
    def download_nse_historical(self, start_date, end_date, max_retries=1):
        """
        Download historical NSE data for date range
        Includes retry logic and proper error handling
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        all_data = []
        failed_dates = []
        
        total_days = len([d for d in date_range if d.weekday() < 5])
        processed = 0
        
        print(f"\n📅 Downloading data for {total_days} trading days...")
        print(f"   From: {start_date.strftime('%Y-%m-%d')}")
        print(f"   To: {end_date.strftime('%Y-%m-%d')}\n")
        
        for date in date_range:
            # Skip weekends
            if date.weekday() >= 5:
                continue
            
            processed += 1
            success = False
            
            for attempt in range(max_retries):
                try:
                    print(f"[{processed}/{total_days}] {date.strftime('%Y-%m-%d')} ", end='')
                    
                    df = self.fetch_nse_bhav_copy(date)
                    
                    if not df.empty:
                        all_data.append(df)
                        print(f"✅ {len(df)} records")
                        success = True
                        break
                    else:
                        if attempt < max_retries - 1:
                            print(f"⚠️  Retry {attempt + 1}/{max_retries}...", end=' ')
                            time.sleep(2)
                        else:
                            print("❌ No data")
                
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"⚠️  Error, retry {attempt + 1}/{max_retries}...", end=' ')
                        time.sleep(2)
                    else:
                        print(f"❌ Failed: {str(e)[:100]}")
            
            if not success:
                failed_dates.append(date.strftime('%Y-%m-%d'))
            
            # Rate limiting
            time.sleep(1)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"\n✅ Successfully downloaded {len(combined_df)} total records")
            
            if failed_dates:
                print(f"⚠️  Failed dates ({len(failed_dates)}): {', '.join(failed_dates[:10])}")
                if len(failed_dates) > 10:
                    print(f"   ... and {len(failed_dates) - 10} more")
            
            return combined_df
        
        print("\n❌ No data downloaded")
        return pd.DataFrame()
    
    def fetch_nse_equity_list(self):
        """
        Fetch list of all NSE equity symbols with ISINs
        Uses the equity master list
        """
        try:
            # Method 1: Try to get from equity list
            url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
            
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text))
                
                # Clean column names
                df.columns = df.columns.str.strip()
                
                # Standardize column names
                # Note: after strip(), leading-space variants collapse,
                # so only stripped names are needed.
                # PAID UP VALUE and FACE VALUE are different fields;
                # only FACE VALUE maps to face_value.
                column_mapping = {
                    'SYMBOL': 'symbol',
                    'NAME OF COMPANY': 'company_name',
                    'SERIES': 'series',
                    'DATE OF LISTING': 'listing_date',
                    'PAID UP VALUE': 'paid_up_value',
                    'MARKET LOT': 'market_lot',
                    'ISIN NUMBER': 'isin',
                    'FACE VALUE': 'face_value'
                }

                df = df.rename(columns=column_mapping)

                # Drop any remaining duplicate columns
                df = df.loc[:, ~df.columns.duplicated()]
                
                # Clean up data
                if 'isin' in df.columns:
                    df['isin'] = df['isin'].str.strip()
                if 'symbol' in df.columns:
                    df['symbol'] = df['symbol'].str.strip()
                
                print(f"✅ Found {len(df)} NSE equity symbols")
                return df
            
            # Method 2: Use API endpoint
            print("⚠️  Trying alternative method...")
            
            url_api = "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
            response = self.session.get(url_api, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data.get('data', []))
                print(f"✅ Found {len(df)} F&O securities")
                return df
            
            return pd.DataFrame()
        
        except Exception as e:
            print(f"❌ Error fetching NSE equity list: {e}")
            return pd.DataFrame()
    
    def lookup_isin_nse(self, symbol):
        """
        Get ISIN for a given NSE symbol using quote API
        """
        try:
            url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                isin = data.get('info', {}).get('isin')
                if isin:
                    print(f"✅ Found ISIN for {symbol}: {isin}")
                    return isin
            
            return None
        
        except Exception as e:
            print(f"❌ Error looking up ISIN for {symbol}: {e}")
            return None