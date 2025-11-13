import gspread
import pandas as pd
from google.oauth2.service_account import Credentials
from typing import Dict, List, Any
from datetime import datetime
import pytz
import streamlit as st
import os

# Setting up Google Sheets
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# Use Streamlit secrets for credentials
try:
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=SCOPES
    )
    print("✅ Loaded credentials from Streamlit secrets")
except Exception as e:
    print(f"❌ Error loading credentials: {e}")
    print("Make sure 'gcp_service_account' is configured in Streamlit secrets!")
    raise

gc = gspread.authorize(creds)

# Constants
CHAT_HISTORY_TAB = "chat_history"


# ============== DATA LOADING FUNCTIONS ==============

def load_sheet_data(sheet_name: str, worksheet_name: str) -> pd.DataFrame:
    """Load data from a Google Sheet worksheet into a pandas DataFrame"""
    try:
        sh = gc.open(sheet_name)
        ws = sh.worksheet(worksheet_name)
        data = ws.get_all_records()
        df = pd.DataFrame(data)
        
        # Normalize column names: strip whitespace and lowercase
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        print(f"✅ Loaded {len(df)} rows from {worksheet_name}")
        print(f"📊 Columns: {df.columns.tolist()}")
        
        # Show sample data
        if len(df) > 0:
            print(f"📝 Sample data:\n{df.head(2)}")
        
        return df
    except Exception as e:
        print(f"❌ Error loading sheet {worksheet_name}: {e}")
        return pd.DataFrame()


def find_flights(
    df: pd.DataFrame, 
    origin: str, 
    destination: str, 
    prefer_date: str = None, 
    budget_usd: float = None, 
    max_results: int = 3
) -> List[Dict[str, Any]]:
    """Find flights matching the criteria"""
    
    if df.empty:
        print("❌ Flights DataFrame is empty")
        return []
    
    # Clean and strip input
    origin = str(origin).strip().lower() if origin else ""
    destination = str(destination).strip().lower() if destination else ""
    
    print(f"\n🔍 FLIGHT SEARCH:")
    print(f"   Looking for: '{origin}' → '{destination}'")
    print(f"   DataFrame shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    
    if not origin or not destination:
        print("❌ Empty origin or destination")
        return []
    
    # Try to identify origin and destination columns
    possible_origin_cols = ['origin', 'from', 'departure', 'departure_city', 'origin_city']
    possible_dest_cols = ['destination', 'to', 'arrival', 'arrival_city', 'destination_city']
    
    origin_col = None
    dest_col = None
    
    for col in possible_origin_cols:
        if col in df.columns:
            origin_col = col
            break
    
    for col in possible_dest_cols:
        if col in df.columns:
            dest_col = col
            break
    
    if not origin_col or not dest_col:
        print(f"❌ Missing required columns. Available: {df.columns.tolist()}")
        print(f"   Need origin column (tried: {possible_origin_cols})")
        print(f"   Need destination column (tried: {possible_dest_cols})")
        return []
    
    print(f"✅ Using columns: origin='{origin_col}', destination='{dest_col}'")
    
    # Convert to string and lowercase for comparison
    df[origin_col] = df[origin_col].astype(str).str.strip().str.lower()
    df[dest_col] = df[dest_col].astype(str).str.strip().str.lower()
    
    # Print sample data
    if len(df) > 0:
        print(f"📝 Sample origins: {df[origin_col].head(3).tolist()}")
        print(f"📝 Sample destinations: {df[dest_col].head(3).tolist()}")
    
    # Filter by origin and destination - use flexible matching
    # Try exact match first, then partial match
    q = df[
        (df[origin_col] == origin) & 
        (df[dest_col] == destination)
    ]
    
    # If no exact match, try partial match
    if len(q) == 0:
        print("   No exact match, trying partial match...")
        q = df[
            (df[origin_col].str.contains(origin, case=False, na=False, regex=False)) & 
            (df[dest_col].str.contains(destination, case=False, na=False, regex=False))
        ]
    
    print(f"   Found {len(q)} flights after filtering")
    
    # Apply budget filter if specified
    if budget_usd and len(q) > 0:
        price_cols = ['price_in_dollars', 'price', 'cost', 'price_usd', 'fare']
        price_col = None
        
        for col in price_cols:
            if col in df.columns:
                price_col = col
                break
        
        if price_col:
            q[price_col] = pd.to_numeric(q[price_col], errors='coerce')
            q = q[q[price_col] <= float(budget_usd)]
            print(f"   {len(q)} flights within budget ${budget_usd}")
            
            # Sort by price
            q = q.sort_values(price_col)
    
    results = q.head(max_results).to_dict(orient='records')
    print(f"✅ Returning {len(results)} flight results")
    
    if len(results) > 0:
        print(f"📋 Results: {results}")
    
    return results


def find_hotels(
    df: pd.DataFrame, 
    city: str, 
    budget_per_night: float = None, 
    max_results: int = 3
) -> List[Dict[str, Any]]:
    """Find hotels matching the criteria"""
    
    if df.empty:
        print("❌ Hotels DataFrame is empty")
        return []
    
    # Clean and strip input
    city = str(city).strip().lower() if city else ""
    
    print(f"\n🏨 HOTEL SEARCH:")
    print(f"   Looking for city: '{city}'")
    print(f"   DataFrame shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    
    if not city:
        print("❌ Empty city")
        return []
    
    # Try to identify city column
    possible_city_cols = ['city', 'location', 'destination', 'place']
    city_col = None
    
    for col in possible_city_cols:
        if col in df.columns:
            city_col = col
            break
    
    if not city_col:
        print(f"❌ Missing city column. Available: {df.columns.tolist()}")
        print(f"   Tried: {possible_city_cols}")
        return []
    
    print(f"✅ Using column: city='{city_col}'")
    
    # Convert to string and lowercase
    df[city_col] = df[city_col].astype(str).str.strip().str.lower()
    
    # Print sample data
    if len(df) > 0:
        print(f"📝 Sample cities: {df[city_col].head(5).tolist()}")
    
    # Filter by city - try exact match first, then partial
    q = df[df[city_col] == city]
    
    if len(q) == 0:
        print("   No exact match, trying partial match...")
        q = df[df[city_col].str.contains(city, case=False, na=False, regex=False)]
    
    print(f"   Found {len(q)} hotels after filtering")
    
    # Apply budget filter if specified
    if budget_per_night and len(q) > 0:
        price_cols = ['price_per_night_in_dollars', 'price_per_night', 'price', 'nightly_rate', 'rate']
        price_col = None
        
        for col in price_cols:
            if col in df.columns:
                price_col = col
                break
        
        if price_col:
            q[price_col] = pd.to_numeric(q[price_col], errors='coerce')
            q = q[q[price_col] <= float(budget_per_night)]
            print(f"   {len(q)} hotels within budget ${budget_per_night}/night")
            
            # Sort by price
            q = q.sort_values(price_col)
    
    results = q.head(max_results).to_dict(orient='records')
    print(f"✅ Returning {len(results)} hotel results")
    
    if len(results) > 0:
        print(f"📋 Results: {results}")
    
    return results


def save_message_to_sheet(
    sheet_name: str,
    role: str, 
    content: str, 
    session_id: str = "default"
) -> bool:
    """
    Save a single message to Google Sheets with Indian Standard Time
    """
    try:
        spreadsheet = gc.open(sheet_name)
        
        # Try to get existing worksheet, create if doesn't exist
        try:
            worksheet = spreadsheet.worksheet(CHAT_HISTORY_TAB)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(
                title=CHAT_HISTORY_TAB,
                rows=1000,
                cols=5
            )
            # Add headers
            worksheet.append_row([
                "Timestamp (IST)",
                "Session ID",
                "Role",
                "Content",
                "Message ID"
            ])
        
        # Get Indian Standard Time
        ist = pytz.timezone('Asia/Kolkata')
        timestamp = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
        message_id = f"{session_id}_{timestamp.replace(' ', '_').replace(':', '-')}"
        
        worksheet.append_row([
            timestamp,
            session_id,
            role,
            content,
            message_id
        ])
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving to Google Sheets: {e}")
        return False


def load_chat_history_from_sheet(
    sheet_name: str,
    session_id: str = "default"
) -> List[Dict[str, str]]:
    """Load chat history from Google Sheets for a specific session"""
    try:
        spreadsheet = gc.open(sheet_name)
        
        try:
            worksheet = spreadsheet.worksheet(CHAT_HISTORY_TAB)
        except gspread.exceptions.WorksheetNotFound:
            return []
        
        # Get all records
        records = worksheet.get_all_records()
        
        # Filter by session_id and sort by timestamp
        session_messages = [
            msg for msg in records 
            if msg.get("Session ID") == session_id
        ]
        
        # Sort by timestamp
        session_messages.sort(key=lambda x: x.get("Timestamp (IST)", ""))
        
        return session_messages
        
    except Exception as e:
        print(f"⚠️ Error loading from Google Sheets: {e}")
        return []


def clear_sheet_history(
    sheet_name: str,
    session_id: str = "default"
) -> bool:
    """Clear chat history from Google Sheets for a specific session"""
    try:
        spreadsheet = gc.open(sheet_name)
        worksheet = spreadsheet.worksheet(CHAT_HISTORY_TAB)
        
        # Get all records
        records = worksheet.get_all_records()
        
        # Find rows to delete (in reverse order to maintain indices)
        rows_to_delete = []
        for i, record in enumerate(records, start=2):  # start=2 because row 1 is header
            if record.get("Session ID") == session_id:
                rows_to_delete.append(i)
        
        # Delete rows in reverse order
        for row_num in reversed(rows_to_delete):
            worksheet.delete_rows(row_num)
        
        return True
        
    except Exception as e:
        print(f"⚠️ Error clearing Google Sheets: {e}")
        return False


def get_all_sessions(sheet_name: str) -> List[str]:
    """Get all unique session IDs from the chat history"""
    try:
        spreadsheet = gc.open(sheet_name)
        
        try:
            worksheet = spreadsheet.worksheet(CHAT_HISTORY_TAB)
        except gspread.exceptions.WorksheetNotFound:
            return []
        
        # Get all records
        records = worksheet.get_all_records()
        
        # Extract unique session IDs
        sessions = list(set(msg.get("Session ID") for msg in records))
        sessions.sort()
        
        return sessions
        
    except Exception as e:
        print(f"⚠️ Error getting sessions: {e}")
        return []
