# test_nse_urls.py

import requests
from datetime import datetime, timedelta
import time
from io import BytesIO
import zipfile
import pandas as pd

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Referer': 'https://www.nseindia.com/',
    'Connection': 'keep-alive',
})

# Visit homepage to get cookies
print("Getting cookies from NSE homepage...")
try:
    session.get('https://www.nseindia.com', timeout=10)
    time.sleep(2)
    print("✅ Cookies obtained\n")
except Exception as e:
    print(f"⚠️  Warning: {e}\n")

# Test date (use a date you know has data, e.g., Jan 9, 2025)
test_date = datetime(2025, 1, 9)  # Change to a recent weekday
print(f"Testing for date: {test_date.strftime('%Y-%m-%d')}\n")

# Various date formats
date_formats = {
    'ddmmyyyy': test_date.strftime('%d%m%Y'),
    'ddMMyyyy': test_date.strftime('%d%b%Y').upper(),
    'yyyy': test_date.strftime('%Y'),
    'MMM': test_date.strftime('%b').upper(),
    'MM': test_date.strftime('%m'),
    'dd': test_date.strftime('%d'),
}

print(f"Date formats: {date_formats}\n")

# Method 1: Archives - Historical Equity Data
urls_to_test = [
    # New format (2024+)
    f"https://nsearchives.nseindia.com/content/historical/EQUITIES/{date_formats['yyyy']}/{date_formats['MMM']}/cm{date_formats['ddMMyyyy']}bhav.csv.zip",
    
    # Alternate format
    f"https://archives.nseindia.com/content/historical/EQUITIES/{date_formats['yyyy']}/{date_formats['MMM']}/cm{date_formats['ddMMyyyy']}bhav.csv.zip",
    
    # Old format without zip
    f"https://nsearchives.nseindia.com/content/historical/EQUITIES/{date_formats['yyyy']}/{date_formats['MMM']}/cm{date_formats['ddMMyyyy']}.csv",
    
    # Products folder format
    f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{date_formats['ddmmyyyy']}.csv",
    
    # Content folder
    f"https://www.nseindia.com/content/historical/EQUITIES/{date_formats['yyyy']}/{date_formats['MMM']}/cm{date_formats['ddMMyyyy']}bhav.csv.zip",
    
    # API method
    f"https://www.nseindia.com/api/reports?archives=%5B%7B%22name%22%3A%22CM%20-%20Bhavcopy%22%2C%22type%22%3A%22archives%22%2C%22category%22%3A%22capital-market%22%2C%22section%22%3A%22equities%22%7D%5D&date={date_formats['ddMMyyyy']}&type=equities&mode=single",
]

for i, url in enumerate(urls_to_test, 1):
    print(f"Method {i}: Testing URL...")
    print(f"  {url}")
    
    try:
        response = session.get(url, timeout=15, allow_redirects=True)
        print(f"  Status: {response.status_code}")
        print(f"  Content-Type: {response.headers.get('Content-Type', 'N/A')}")
        print(f"  Content-Length: {len(response.content)} bytes")
        
        if response.status_code == 200:
            # Check if it's a zip file
            if url.endswith('.zip'):
                try:
                    with zipfile.ZipFile(BytesIO(response.content)) as z:
                        files = z.namelist()
                        print(f"  ✅ ZIP FILE FOUND! Contains: {files}")
                        
                        # Try to read first file
                        with z.open(files[0]) as f:
                            first_line = f.readline().decode('utf-8')
                            print(f"  First line: {first_line[:100]}")
                        print(f"  🎉 SUCCESS - This URL works!\n")
                        break
                except zipfile.BadZipFile:
                    print(f"  ❌ Not a valid zip file\n")
            
            # Check if it's CSV
            elif 'text/csv' in response.headers.get('Content-Type', '') or url.endswith('.csv'):
                content = response.text
                #content_df = pd.read_csv(content)
                if len(content) > 100:
                    print(f"  ✅ CSV FILE FOUND!")
                    print(f"  First line: {content[:100]}")
                    #content_df.head()
                    print(f"  🎉 SUCCESS - This URL works!\n")
                    break
            
            # Check if it's JSON (API response)
            elif 'application/json' in response.headers.get('Content-Type', ''):
                import json
                data = response.json()
                print(f"  ✅ JSON RESPONSE!")
                print(f"  Keys: {list(data.keys())}")
                if data:
                    print(f"  🎉 SUCCESS - This URL works!\n")
                    print(f"  Full response: {json.dumps(data, indent=2)[:500]}")
                    break
            else:
                print(f"  ⚠️  Unexpected content type\n")
        else:
            print(f"  ❌ Failed\n")
    
    except Exception as e:
        print(f"  ❌ Error: {str(e)[:100]}\n")
    
    time.sleep(1)

print("\n" + "="*70)
print("Testing complete")