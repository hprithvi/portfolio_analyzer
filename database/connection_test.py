# database/connection_test.py

import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from database.multi_asset_db_manager import MultiAssetDBManager
import os

def test_supabase_connection():
    """Test connection to Supabase"""
    
    # Method 1: Using environment variable
    db_url = os.getenv('DATABASE_URL')
    
    # Method 2: Direct URL
    db_url = "postgresql://postgres:B-Ym/3VK%L4/.yh@db.uasonlaqjmnvhpdgputp.supabase.co:5432/postgres"
    
    try:
        db = MultiAssetDBManager(db_url=db_url, db_type='postgresql')
        
        # Test connection
        success, message = db.test_connection()
        
        if success:
            print("✅ Successfully connected to Supabase!")
            
            # Test table creation
            engine = db.get_engine()
            # with engine.connect() as conn:
            #     result = conn.execute("""
            #         SELECT table_name 
            #         FROM information_schema.tables 
            #         WHERE table_schema = 'public'
            #     """)
            #     tables = [row[0] for row in result]
            #     print(f"📊 Existing tables: {tables}")
            
            return True
        else:
            print(f"❌ Connection failed: {message}")
            return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_supabase_connection()