from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from io import StringIO
import json, logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from scipy.stats import linregress
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from scipy.fftpack import fft
# from scipy.stats import jarque_bera, adfuller
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

import sys

# Bersihkan handler lama
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Konfigurasi ulang logging agar log muncul ke stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logging.info("✅ Logging initialized di FastAPI")

app = FastAPI(title="Data Analytics", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logging.info("✅ Logging initialized di FastAPI")
# In-memory storage
data_store: Dict[str, pd.DataFrame] = {}
analysis_cache: Dict[str, Any] = {}

# Pydantic models
class DataUploadResponse(BaseModel):
    session_id: str
    columns: List[str]
    shape: tuple
    sample_data: List[Dict]

class AggregationRequest(BaseModel):
    session_id: str
    datetime_col: str
    value_col: str
    agg_method: str
    freq: str
    category_col: Optional[str] = None  # Tambahan untuk multi time series

class AnalysisRequest(BaseModel):
    session_id: str
    module: str
    parameters: Dict

logging.info("✅ Logging initialized di FastAPI")

def clean_float(val):
    if isinstance(val, float):
        if np.isnan(val) or np.isinf(val):
            return 0.0
    return float(val)

def sanitize_for_json(obj):
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]
    else:
        return obj

# Helper function for datetime parsing
def parse_datetime_flexible(date_series):
    """
    Flexibly parse datetime with multiple format attempts
    """
    # Common datetime formats to try
    formats = [
        '%Y-%m-%d',
        '%d/%m/%Y',
        '%m/%d/%Y', 
        '%Y-%m-%d %H:%M:%S',
        '%d/%m/%Y %H:%M',
        '%m/%d/%Y %H:%M',
        '%Y%m%d',
        '%d-%m-%Y',
        '%m-%d-%Y'
    ]
    
    # Try pandas default first
    try:
        return pd.to_datetime(date_series, infer_datetime_format=True)
    except:
        pass
    
    # Try each format
    for fmt in formats:
        try:
            return pd.to_datetime(date_series, format=fmt)
        except:
            continue
    
    # Last resort - coerce errors
    try:
        return pd.to_datetime(date_series, errors='coerce')
    except:
        raise ValueError("Unable to parse datetime column")

# Generate realistic manufacturing data
def generate_manufacturing_data():
    logging.info("✅ generate manufacturing data")
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', '2024-12-31', freq='H')
    
    # Create realistic manufacturing patterns
    base_production = 850
    data = []
    
    for i, date in enumerate(dates):
        # Working hours effect (8 AM - 6 PM higher production)
        hour_multiplier = 1.3 if 8 <= date.hour <= 18 else 0.7
        
        # Day of week effect (weekends lower)
        day_multiplier = 0.4 if date.weekday() >= 5 else 1.0
        
        # Seasonal effect (summer months slightly higher)
        seasonal_multiplier = 1.2 if date.month in [6, 7, 8] else 1.0
        
        # Random maintenance downtime (5% chance)
        maintenance = 0.1 if np.random.random() < 0.05 else 1.0
        
        production = base_production * hour_multiplier * day_multiplier * seasonal_multiplier * maintenance
        production += np.random.normal(0, 50)  # Add noise
        production = max(0, production)  # No negative production
        
        # Quality metrics
        defect_rate = max(0, min(15, np.random.normal(2.5, 1.5)))
        temperature = np.random.normal(75, 5)
        
        data.append({
            'timestamp': date.strftime('%d/%m/%Y %H:%M'),  # Different format
            'production_units': round(production, 1),
            'defect_rate_percent': round(defect_rate, 2),
            'machine_temp_celsius': round(temperature, 1),
            'shift': 'Day' if 6 <= date.hour < 14 else 'Evening' if 14 <= date.hour < 22 else 'Night',
            'machine_id': f'M{(i % 5) + 1:03d}'
        })
    
    return pd.DataFrame(data)

# Generate realistic financial trading data
def generate_trading_data():
    logging.info("✅ generate trading data")
    np.random.seed(123)
    
    # Generate minute-by-minute trading data for 6 months
    start_date = datetime(2024, 1, 1, 9, 30)  # Market open
    data = []
    
    base_price = 100
    current_price = base_price
    
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    
    for days in range(180):  # 6 months
        trading_date = start_date + timedelta(days=days)
        
        # Skip weekends
        if trading_date.weekday() >= 5:
            continue
            
        # Market hours: 9:30 AM - 4:00 PM
        for minutes in range(0, 390, 5):  # Every 5 minutes
            timestamp = trading_date + timedelta(minutes=minutes)
            
            for symbol in symbols:
                # Realistic price movement with volatility
                price_change = np.random.normal(0, 0.02) * current_price
                current_price = max(10, current_price + price_change)
                
                # Volume patterns (higher at open/close)
                hour = timestamp.hour
                if hour in [9, 15]:  # Open and close hours
                    base_volume = np.random.exponential(50000)
                else:
                    base_volume = np.random.exponential(20000)
                
                volume = int(base_volume)
                
                data.append({
                    'trade_date': timestamp.strftime('%Y%m%d'),  # YYYYMMDD format
                    'trade_time': timestamp.strftime('%H:%M:%S'),
                    'symbol': symbol,
                    'price': round(current_price, 2),
                    'volume': volume,
                    'bid_ask_spread': round(np.random.uniform(0.01, 0.05), 3),
                    'market_cap_category': np.random.choice(['Large', 'Mid', 'Small'], p=[0.6, 0.3, 0.1])
                })
                
                # Reset price for next symbol
                current_price = base_price + np.random.normal(0, 20)
    
    return pd.DataFrame(data)

# Generate realistic healthcare patient data
def generate_healthcare_data():
    logging.info("✅ generate healthcare data")
    np.random.seed(456)
    
    # Generate patient admission data over 2 years
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
    data = []
    
    departments = ['Emergency', 'Cardiology', 'Orthopedics', 'Pediatrics', 'ICU']
    age_groups = ['0-18', '19-35', '36-55', '56-75', '75+']
    
    for date in dates:
        # More admissions on weekends and holidays
        base_admissions = 45 if date.weekday() < 5 else 65
        
        # Seasonal patterns (flu season in winter)
        seasonal_factor = 1.4 if date.month in [12, 1, 2, 3] else 1.0
        
        daily_admissions = int(base_admissions * seasonal_factor * np.random.uniform(0.7, 1.3))
        
        for _ in range(daily_admissions):
            # Realistic admission patterns
            admission_hour = max(0, min(23, int(np.random.exponential(12))))  # More admissions during day
            
            department = np.random.choice(departments, p=[0.35, 0.15, 0.15, 0.2, 0.15])
            age_group = np.random.choice(age_groups, p=[0.15, 0.25, 0.25, 0.25, 0.1])
            
            # Length of stay depends on department
            if department == 'Emergency':
                los = max(1, int(np.random.exponential(2)))
            elif department == 'ICU':
                los = max(1, int(np.random.exponential(7)))
            else:
                los = max(1, int(np.random.exponential(4)))
            
            # Cost depends on department and length of stay
            base_cost = {'Emergency': 2500, 'Cardiology': 8500, 'Orthopedics': 12000, 
                        'Pediatrics': 4500, 'ICU': 15000}[department]
            total_cost = base_cost + (los * 1200) + np.random.normal(0, 1000)
            total_cost = max(1000, total_cost)
            
            admission_datetime = date + timedelta(hours=admission_hour, 
                                                minutes=np.random.randint(0, 60))
            
            data.append({
                'admission_date': admission_datetime.strftime('%m-%d-%Y'),  # MM-DD-YYYY format
                'admission_time': admission_datetime.strftime('%H:%M'),
                'department': department,
                'patient_age_group': age_group,
                'length_of_stay_days': los,
                'total_cost_usd': round(total_cost, 2),
                'discharge_status': np.random.choice(['Home', 'Transfer', 'AMA'], p=[0.85, 0.12, 0.03]),
                'insurance_type': np.random.choice(['Private', 'Medicare', 'Medicaid', 'Uninsured'], 
                                                 p=[0.45, 0.25, 0.2, 0.1])
            })
    
    return pd.DataFrame(data)

# Generate realistic airline flight data with upward trend
def generate_flight_data():
    logging.info("✅ generate flight data")
    np.random.seed(789)
    
    # Generate daily flight data over 3 years with clear upward trend
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    data = []
    base_passengers = 15000  # Starting base passengers per day
    
    airlines = ['Garuda Indonesia']
    aircraft_types = ['Boeing 737']
    routes = ['CGK-DPS', 'CGK-SBY']

    logging.info("creating data")
    
    for i, date in enumerate(dates):
        # logging.info(date)
        # Strong upward trend over 3 years (recovery from pandemic + growth)
        trend_multiplier = 1 + (i / len(dates)) * 0.8  # 80% growth over period
        
        # Weekly seasonality (weekends higher for leisure travel)
        if date.weekday() in [4, 5, 6]:  # Fri, Sat, Sun
            weekly_multiplier = 1.4
        else:
            weekly_multiplier = 1.0
        
        # Strong seasonal patterns
        month = date.month
        if month in [6, 7, 12]:  # School holidays & year-end
            seasonal_multiplier = 1.6
        elif month in [1, 2]:  # Chinese New Year period
            seasonal_multiplier = 1.3
        elif month in [4, 5]:  # Eid period
            seasonal_multiplier = 1.2
        elif month in [3, 9, 10]:  # Moderate seasons
            seasonal_multiplier = 1.1
        else:  # Low season
            seasonal_multiplier = 0.8
        
        # Holiday boost (simulate specific holiday periods)
        holiday_boost = 1.0
        if (date.month == 12 and date.day >= 20) or (date.month == 1 and date.day <= 5):
            holiday_boost = 1.8  # Year-end holidays
        elif date.month == 7 and 15 <= date.day <= 31:
            holiday_boost = 1.5  # School holidays
        
        # Calculate daily passengers
        daily_passengers = base_passengers * trend_multiplier * weekly_multiplier * seasonal_multiplier * holiday_boost
        daily_passengers += np.random.normal(0, daily_passengers * 0.1)  # Add 10% noise
        daily_passengers = max(5000, int(daily_passengers))  # Minimum 5000 passengers
        
        # Generate individual flights for the day
        estimated_flights = max(30, int(daily_passengers / 180))  # ~180 passengers per flight average

        
        for flight_num in range(estimated_flights):
            departure_hour = np.random.choice(range(5, 23), p=[0.02, 0.03, 0.08, 0.12, 0.15, 0.12, 
                                                   0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 
                                                   0.02, 0.02, 0.02, 0.02, 0.02, 0.02])
            departure_minute = np.random.randint(0, 60, size=1)[0]
            
            airline = np.random.choice(airlines, p=[1])
            aircraft = np.random.choice(aircraft_types, p=[1])
            route = np.random.choice(routes, p=[0.5,0.5])
            
            # Passengers per flight varies by aircraft type
            if aircraft == 'Boeing 777':
                max_capacity = 350
            elif aircraft == 'Airbus A330':
                max_capacity = 280
            elif aircraft == 'Boeing 737':
                max_capacity = 189
            else:  # Airbus A320
                max_capacity = 180
            
            # Load factor varies by season and day
            base_load_factor = 0.75
            if date.weekday() >= 5:  # Weekend
                load_factor = min(0.95, base_load_factor + 0.15)
            else:
                load_factor = base_load_factor
            
            load_factor *= seasonal_multiplier / 1.2  # Adjust for seasonality
            load_factor = min(0.98, max(0.4, load_factor))  # Keep realistic bounds
            
            passengers = int(max_capacity * load_factor)
            revenue_per_passenger = np.random.normal(750000, 150000)  # IDR
            revenue_per_passenger = max(300000, revenue_per_passenger)
            
            fuel_cost = np.random.normal(25000000, 5000000)  # IDR per flight
            fuel_cost = max(15000000, fuel_cost)
            
            departure_time = datetime.combine(date.date(), datetime.min.time()).replace(
                hour=departure_hour,
                minute=departure_minute
            )
            
            data.append({
                'flight_date': date.strftime('%Y-%m-%d'),
                'departure_time': departure_time.strftime('%H:%M'),
                'airline': airline,
                'aircraft_type': aircraft,
                'route': route,
                'passengers_count': passengers,
                'max_capacity': max_capacity,
                'load_factor_percent': round(load_factor * 100, 1),
                'revenue_idr': int(passengers * revenue_per_passenger),
                'fuel_cost_idr': int(fuel_cost),
                'profit_idr': int(passengers * revenue_per_passenger - fuel_cost),
                'flight_duration_hours': round(np.random.uniform(1.0, 2.5), 1),
                'delay_minutes': max(0, int(np.random.exponential(15)))
            })

    logging.info("created data")
    
    return pd.DataFrame(data)

# Generate school book demand data with very strong seasonality
def generate_school_book_data():
    logging.info("✅ generate school book data")
    np.random.seed(321)
    
    # Generate daily book sales data over 3 years
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    data = []
    base_sales = 500  # Base daily sales
    
    book_categories = ['Elementary Textbooks', 'Middle School Books', 'High School Books', 
                      'University Books', 'Activity Books', 'Reference Books']
    subjects = ['Mathematics', 'Science', 'Indonesian Language', 'English', 'Social Studies', 
               'Religion', 'Arts', 'Physical Education']
    publishers = ['Erlangga', 'Tiga Serangkai', 'Yudhistira', 'Esis', 'Grafindo', 'Bumi Aksara']
    
    for date in enumerate(dates):
        date = date[1]  # Get actual date from enumerate
        month = date.month
        
        # VERY STRONG seasonal patterns for school books
        if month == 6:  # June - New school year preparation (STRONGEST PEAK)
            seasonal_multiplier = 8.0  # 800% increase!
        elif month == 7:  # July - School year starts (MAJOR PEAK)
            seasonal_multiplier = 12.0  # 1200% increase!
        elif month == 12:  # December - Second semester prep
            seasonal_multiplier = 4.0  # 400% increase
        elif month == 1:  # January - Second semester starts
            seasonal_multiplier = 5.0  # 500% increase
        elif month in [5, 11]:  # Pre-season months
            seasonal_multiplier = 2.0  # 200% increase
        elif month in [2, 8]:  # Post-peak months
            seasonal_multiplier = 1.5  # 150% increase
        elif month in [3, 4, 9, 10]:  # Regular school months
            seasonal_multiplier = 0.8  # 80% of base
        else:
            seasonal_multiplier = 0.3  # Very low during holidays
        
        # Weekly patterns (weekends lower except during peak season)
        if date.weekday() >= 5:  # Weekend
            if seasonal_multiplier > 3:  # During peak season, weekends also busy
                weekly_multiplier = 0.8
            else:
                weekly_multiplier = 0.3
        else:
            weekly_multiplier = 1.0
        
        # Year-over-year growth (education sector growing)
        year_multiplier = 1 + (date.year - 2022) * 0.12  # 12% annual growth
        
        # Calculate daily sales
        daily_sales = base_sales * seasonal_multiplier * weekly_multiplier * year_multiplier
        daily_sales += np.random.normal(0, daily_sales * 0.15)  # Add noise
        daily_sales = max(10, int(daily_sales))  # Minimum 10 books per day
        
        # Generate individual sales records
        for sale_id in range(min(daily_sales, 1000)):  # Cap at 1000 records per day for performance
            category = np.random.choice(book_categories, p=[0.3, 0.25, 0.25, 0.1, 0.05, 0.05])
            subject = np.random.choice(subjects, p=[0.25, 0.2, 0.15, 0.1, 0.1, 0.05, 0.1, 0.05])
            publisher = np.random.choice(publishers, p=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
            
            # Price varies by category and publisher
            if category == 'University Books':
                base_price = 180000
            elif category in ['High School Books', 'Reference Books']:
                base_price = 95000
            elif category == 'Middle School Books':
                base_price = 75000
            else:
                base_price = 45000
            
            # Publisher premium
            publisher_multiplier = {'Erlangga': 1.2, 'Tiga Serangkai': 1.0, 'Yudhistira': 1.1,
                                  'Esis': 1.0, 'Grafindo': 0.9, 'Bumi Aksara': 0.95}[publisher]
            
            price = int(base_price * publisher_multiplier * np.random.uniform(0.8, 1.3))
            quantity = np.random.choice([1, 2, 3, 5, 10], p=[0.4, 0.3, 0.15, 0.1, 0.05])
            
            # During peak season, bulk purchases more common
            if seasonal_multiplier > 3:
                quantity *= np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            
            # Customer type affects purchase patterns
            customer_type = np.random.choice(['Individual Parent', 'School Bulk', 'Teacher', 'Student'], 
                                           p=[0.5, 0.2, 0.2, 0.1])
            
            if customer_type == 'School Bulk':
                quantity *= np.random.randint(5, 25)
                price *= 0.85  # Bulk discount
            
            sale_time = date.replace(hour=np.random.randint(8, 20), 
                                   minute=np.random.randint(0, 60))
            
            data.append({
                'sale_date': date.strftime('%d/%m/%Y'),  # DD/MM/YYYY format
                'sale_time': sale_time.strftime('%H:%M'),
                'book_category': category,
                'subject': subject,
                'publisher': publisher,
                'unit_price_idr': price,
                'quantity_sold': quantity,
                'total_revenue_idr': price * quantity,
                'customer_type': customer_type,
                'school_grade_level': np.random.choice(['K-6', '7-9', '10-12', 'University'], 
                                                     p=[0.35, 0.25, 0.25, 0.15]),
                'is_new_curriculum': np.random.choice([True, False], p=[0.7, 0.3])
            })
    
    return pd.DataFrame(data)

# Enhanced sample datasets with more realistic data
SAMPLE_DATASETS = {
    "ecommerce_sales": {
        "name": "E-commerce Sales Data",
        "description": "Daily sales data with customer segments and seasonal patterns",
        "data": pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=365, freq='D'),
            'sales': np.random.normal(5000, 1500, 365).clip(0),
            'orders': np.random.poisson(50, 365),
            'customers': np.random.poisson(45, 365),
            'category': np.random.choice(['Electronics', 'Clothing', 'Books'], 365)
        })
    },
    "customer_transactions": {
        "name": "Customer Transaction History", 
        "description": "Individual customer purchase records with transaction details",
        "data": pd.DataFrame({
            'customer_id': np.repeat(range(1, 101), 10),
            'transaction_date': pd.date_range('2023-01-01', periods=1000, freq='D')[:1000],
            'amount': np.random.exponential(100, 1000),
            'product_category': np.random.choice(['A', 'B', 'C'], 1000),
            'quantity': np.random.poisson(2, 1000) + 1
        })
    },
    "manufacturing_production": {
        "name": "Manufacturing Production Data",
        "description": "Hourly production data with quality metrics and machine performance (DD/MM/YYYY HH:MM format)",
        "data": generate_manufacturing_data()
    },
    "financial_trading": {
        "name": "Financial Trading Data", 
        "description": "High-frequency trading data with multiple symbols and market metrics (YYYYMMDD format)",
        "data": generate_trading_data()
    },
    "healthcare_admissions": {
        "name": "Healthcare Patient Admissions",
        "description": "Hospital patient admission records with department and cost data (MM-DD-YYYY format)", 
        "data": generate_healthcare_data()
    },
    "airline_flights": {
        "name": "Airline Flight Operations", 
        "description": "Daily flight data showing strong upward trend in passenger traffic with seasonal patterns (YYYY-MM-DD format)",
        "data": generate_flight_data()
    },
    "school_book_sales": {
        "name": "School Book Sales Demand",
        "description": "Educational book sales with extremely strong seasonality around school seasons (DD/MM/YYYY format)", 
        "data": generate_school_book_data()
    }
}

logging.info("✅ Logging initialized di FastAPI")
@app.get("/")
def read_root():
    return {"message": "Data Analytics", "version": "1.0.0"}

@app.get("/sample-datasets")
def get_sample_datasets():
    return {
        key: {
            "name": value["name"],
            "description": value["description"],
            "shape": value["data"].shape,
            "columns": value["data"].columns.tolist()
        }
        for key, value in SAMPLE_DATASETS.items()
    }
logging.info("✅ Logging initialized di FastAPI")
@app.post("/load-sample/{dataset_key}")
def load_sample_dataset(dataset_key: str):

    logging.info('dataset_key')
    logging.info(dataset_key)
    logging.info('SAMPLE_DATASETS')
    logging.info(list(SAMPLE_DATASETS.keys()))

    if dataset_key not in SAMPLE_DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    session_id = f"sample_{dataset_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    data_store[session_id] = SAMPLE_DATASETS[dataset_key]["data"].copy()

    logging.info('data_store session created')
    logging.info(session_id)
    
    df = data_store[session_id]

    logging.info('df shape')
    logging.info(df.shape)

    return DataUploadResponse(
        session_id=session_id,
        columns=df.columns.tolist(),
        shape=df.shape,
        sample_data=df.to_dict('records')  # Show more sample data
    )

@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    if not file.filename.endswith(('.csv', '.xlsx')):
        raise HTTPException(status_code=400, detail="Only CSV and Excel files supported")
    
    try:
        content = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(StringIO(content.decode('utf-8')))
        else:
            df = pd.read_excel(content)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Limit size for MVP
        if len(df) > 10000:
            df = df.head(10000)
        
        session_id = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        data_store[session_id] = df
        
        return DataUploadResponse(
            session_id=session_id,
            columns=df.columns.tolist(),
            shape=df.shape,
            sample_data=df.head().to_dict('records')
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

@app.post("/aggregate-data")
def aggregate_data(request: AggregationRequest):
    if request.session_id not in data_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    df = data_store[request.session_id].copy()
    
    try:
        # Enhanced datetime parsing
        df[request.datetime_col] = parse_datetime_flexible(df[request.datetime_col])
        
        # Handle cases where datetime parsing failed
        if df[request.datetime_col].isna().any():
            failed_count = df[request.datetime_col].isna().sum()
            logging.warning(f"Failed to parse {failed_count} datetime values")
            df = df.dropna(subset=[request.datetime_col])
        
        if df.empty:
            raise HTTPException(status_code=400, detail="No valid datetime values found")
        
        # Set datetime as index and resample
        df_agg = df.set_index(request.datetime_col)
        
        # Multi time series support with category
        if request.category_col and request.category_col in df.columns:
            # Group by category and resample
            if request.agg_method == 'sum':
                result = df_agg.groupby(request.category_col)[request.value_col].resample(request.freq).sum()
            elif request.agg_method == 'mean':
                result = df_agg.groupby(request.category_col)[request.value_col].resample(request.freq).mean()
            elif request.agg_method == 'count':
                result = df_agg.groupby(request.category_col)[request.value_col].resample(request.freq).count()
            elif request.agg_method == 'median':
                result = df_agg.groupby(request.category_col)[request.value_col].resample(request.freq).median()
            else:
                raise ValueError("Unsupported aggregation method")
            
            # Reset index and create proper dataframe
            result_df = result.reset_index()
            result_df.columns = ['category', 'date', 'value']
            agg_df = result_df
        else:
            # Single time series (original logic)
            if request.agg_method == 'sum':
                result = df_agg[request.value_col].resample(request.freq).sum()
            elif request.agg_method == 'mean':
                result = df_agg[request.value_col].resample(request.freq).mean()
            elif request.agg_method == 'count':
                result = df_agg[request.value_col].resample(request.freq).count()
            elif request.agg_method == 'median':
                result = df_agg[request.value_col].resample(request.freq).median()
            else:
                raise ValueError("Unsupported aggregation method")
            
            # Create new dataframe
            agg_df = pd.DataFrame({
                'date': result.index,
                'value': result.values
            }).reset_index(drop=True)
        
        # Store aggregated data
        agg_session_id = f"{request.session_id}_agg"
        data_store[agg_session_id] = agg_df
        
        return {
            "session_id": agg_session_id,
            "shape": agg_df.shape,
            "data": agg_df.to_dict('records')[:50],  # Return first 50 rows
            "has_category": request.category_col is not None
        }
    
    except Exception as e:
        logging.error(f"Aggregation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Aggregation error: {str(e)}")

@app.post("/analyze")
def analyze_data(request: AnalysisRequest):
    if request.session_id not in data_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    df = data_store[request.session_id].copy()
    
    try:
        if request.module == "time_series":
            return analyze_time_series(df, request.parameters)
        elif request.module == "customer":
            return analyze_customer(df, request.parameters)
        else:
            raise HTTPException(status_code=400, detail="Unsupported module")
    
    except Exception as e:
        logging.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Analysis error: {str(e)}")

@app.post("/compare-models")
def compare_models_endpoint(request: AnalysisRequest):
    """Compare multiple forecasting models"""
    if request.session_id not in data_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    df = data_store[request.session_id].copy()
    
    try:
        if request.module == "time_series":
            return compare_forecasting_models(df, request.parameters)
        else:
            raise HTTPException(status_code=400, detail="Model comparison only available for time series")
    
    except Exception as e:
        logging.error(f"Model comparison error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Model comparison error: {str(e)}")
    

def compare_forecasting_models(df: pd.DataFrame, params: Dict) -> Dict:
    """Compare multiple forecasting models and return performance metrics"""
    try:
        date_col = params.get('date_col', 'date')
        value_col = params.get('value_col', 'value')
        comparison_models = params.get('comparison_models', ['linear_regression', 'random_forest', 'ets', 'naive'])
        
        # Parse and prepare data
        df[date_col] = parse_datetime_flexible(df[date_col])
        df = df.dropna(subset=[date_col])
        df = df.sort_values(date_col)
        
        if len(df) < 10:
            raise ValueError("Insufficient data for model comparison (minimum 10 points required)")
        
        comparison_results = {}
        
        # Test each model
        for model_type in comparison_models:
            try:
                logging.info(f"Testing model: {model_type}")
                
                # Get forecast for this model
                forecast_data, forecast_insight, forecast_error = create_forecast(
                    df, date_col, value_col, model_type=model_type
                )
                
                if forecast_data and forecast_error is None:
                    metrics = forecast_data['metrics']
                    
                    comparison_results[model_type] = {
                        'model_name': forecast_data['model_name'],
                        'mae': metrics['mae'],
                        'rmse': metrics['rmse'],
                        'r2': metrics.get('r2', None),
                        'mape': metrics.get('mape', None),
                        'directional_accuracy': metrics.get('directional_accuracy', None),
                        'insight': forecast_insight,
                        'status': 'success'
                    }
                else:
                    comparison_results[model_type] = {
                        'model_name': model_type.replace('_', ' ').title(),
                        'status': 'failed',
                        'error': forecast_error or 'Unknown error'
                    }
                    
            except Exception as e:
                logging.error(f"Error testing model {model_type}: {str(e)}")
                comparison_results[model_type] = {
                    'model_name': model_type.replace('_', ' ').title(),
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Rank models by performance (lower MAE is better)
        successful_models = {k: v for k, v in comparison_results.items() if v['status'] == 'success'}
        
        if successful_models:
            # Sort by MAE (ascending - lower is better)
            ranked_models = sorted(successful_models.items(), key=lambda x: x[1]['mae'])
            
            # Best model
            best_model = ranked_models[0]
            
            # Create comparison table data
            comparison_table = []
            for rank, (model_key, model_data) in enumerate(ranked_models, 1):
                comparison_table.append({
                    'rank': rank,
                    'model': model_data['model_name'],
                    'model_key': model_key,
                    'mae': model_data['mae'],
                    'rmse': model_data['rmse'],
                    'r2': model_data.get('r2'),
                    'mape': model_data.get('mape'),
                    'directional_accuracy': model_data.get('directional_accuracy'),
                    'status': 'success'
                })
            
            # Add failed models
            for model_key, model_data in comparison_results.items():
                if model_data['status'] == 'failed':
                    comparison_table.append({
                        'rank': None,
                        'model': model_data['model_name'],
                        'model_key': model_key,
                        'mae': None,
                        'rmse': None,
                        'r2': None,
                        'mape': None,
                        'directional_accuracy': None,
                        'status': 'failed',
                        'error': model_data.get('error', 'Unknown error')
                    })
            
            return {
                'comparison_results': comparison_results,
                'comparison_table': comparison_table,
                'best_model': {
                    'model_key': best_model[0],
                    'model_name': best_model[1]['model_name'],
                    'mae': best_model[1]['mae'],
                    'rmse': best_model[1]['rmse'],
                    'r2': best_model[1].get('r2'),
                    'mape': best_model[1].get('mape')
                },
                'total_models_tested': len(comparison_models),
                'successful_models': len(successful_models),
                'failed_models': len(comparison_models) - len(successful_models)
            }
        else:
            raise ValueError("All models failed to run successfully")
            
    except Exception as e:
        logging.error(f"Model comparison error: {str(e)}")
        raise

def create_forecast(data, date_column, value_column, forecast_periods=None, model_type='linear_regression'):
    """Enhanced forecast with multiple model options and seasonality handling"""
    if len(data) < 10:
        return None, None, "Insufficient data for forecasting (minimum 10 points required)"
    
    # Use last 20% for testing
    test_size = max(1, int(len(data) * 0.2))
    train_data = data.iloc[:-test_size].copy()
    test_data = data.iloc[-test_size:].copy()
    
    if len(train_data) < 5:
        return None, None, "Insufficient training data for forecasting"
    
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import numpy as np
        
        # Feature engineering
        def create_features(df, date_col, value_col):
            """Create time-based and seasonal features"""
            df = df.copy()
            df['days_numeric'] = (df[date_col] - df[date_col].min()).dt.days
            df['month'] = df[date_col].dt.month
            df['day_of_week'] = df[date_col].dt.dayofweek
            df['quarter'] = df[date_col].dt.quarter
            df['day_of_year'] = df[date_col].dt.dayofyear
            
            # Cyclical encoding for better seasonality capture
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
            df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
            
            # Lag features
            if len(df) > 7:
                df['lag_7'] = df[value_col].shift(7).fillna(df[value_col].mean())
            if len(df) > 30:
                df['lag_30'] = df[value_col].shift(30).fillna(df[value_col].mean())
                df['rolling_7'] = df[value_col].rolling(7, min_periods=1).mean()
                df['rolling_30'] = df[value_col].rolling(30, min_periods=1).mean()
            
            return df
        
        # Create features
        train_features = create_features(train_data, date_column, value_column)
        test_features = create_features(test_data, date_column, value_column)
        
        # Select feature columns (exclude original date and value columns)
        feature_cols = [
            col for col in train_features.columns
            if col not in [date_column, value_column]
            and pd.api.types.is_numeric_dtype(train_features[col])
            and not train_features[col].isna().all()
        ]
        
        X_train = train_features[feature_cols].fillna(0)
        y_train = train_features[value_column].values
        X_test = test_features[feature_cols].fillna(0)
        y_actual = test_features[value_column].values
        
        # Initialize model based on selection
        if model_type == 'linear_regression':
            model = LinearRegression()
            model_name = "Linear Regression"
            
        elif model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            model_name = "Random Forest"
            
        elif model_type == 'catboost':
            try:
                from catboost import CatBoostRegressor
                model = CatBoostRegressor(
                    iterations=100,
                    depth=6,
                    learning_rate=0.1,
                    random_seed=42,
                    verbose=False
                )
                model_name = "CatBoost"
            except ImportError:
                # Fallback to Random Forest if CatBoost not available
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model_name = "Random Forest (CatBoost not available)"
                
        elif model_type == 'arima':
            return create_arima_forecast(train_data, test_data, date_column, value_column)
            
        elif model_type == 'ets':
            return create_ets_forecast(train_data, test_data, date_column, value_column)
            
        elif model_type == 'theta':
            return create_theta_forecast(train_data, test_data, date_column, value_column)
            
        elif model_type == 'moving_average':
            return create_moving_average_forecast(train_data, test_data, date_column, value_column)
            
        elif model_type == 'naive':
            return create_naive_forecast(train_data, test_data, date_column, value_column)
            
        elif model_type == 'naive_drift':
            return create_naive_drift_forecast(train_data, test_data, date_column, value_column)
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate residuals for prediction intervals
        train_pred = model.predict(X_train)
        residuals = y_train - train_pred
        mse = np.mean(residuals**2)
        std_error = np.sqrt(mse)
        
        # Enhanced prediction intervals
        prediction_variance = std_error * np.sqrt(1 + 1/len(train_data))
        ci_50_lower = y_pred - 0.674 * prediction_variance
        ci_50_upper = y_pred + 0.674 * prediction_variance
        ci_80_lower = y_pred - 1.282 * prediction_variance
        ci_80_upper = y_pred + 1.282 * prediction_variance
        ci_95_lower = y_pred - 1.96 * prediction_variance
        ci_95_upper = y_pred + 1.96 * prediction_variance
        
        # Calculate comprehensive metrics
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        r2 = r2_score(y_actual, y_pred)
        mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100 if np.all(y_actual != 0) else float('inf')
        
        # Additional metrics
        mse_score = mean_squared_error(y_actual, y_pred)
        
        # Directional accuracy (for trend prediction)
        if len(y_actual) > 1:
            actual_direction = np.diff(y_actual) > 0
            pred_direction = np.diff(y_pred) > 0
            directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        else:
            directional_accuracy = None
        
        # Feature importance for tree-based models
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(feature_cols, model.feature_importances_))
            # Sort by importance
            feature_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        forecast_data = {
            'model_name': model_name,
            'train_dates': train_data[date_column].dt.strftime('%Y-%m-%d').tolist(),
            'train_values': train_data[value_column].tolist(),
            'test_dates': test_data[date_column].dt.strftime('%Y-%m-%d').tolist(),
            'test_actual': y_actual.tolist(),
            'test_predicted': y_pred.tolist(),
            'ci_50_lower': ci_50_lower.tolist(),
            'ci_50_upper': ci_50_upper.tolist(),
            'ci_80_lower': ci_80_lower.tolist(),
            'ci_80_upper': ci_80_upper.tolist(),
            'ci_95_lower': ci_95_lower.tolist(),
            'ci_95_upper': ci_95_upper.tolist(),
            'split_date': train_data[date_column].iloc[-1].strftime('%Y-%m-%d'),
            'metrics': {
                'mae': float(mae),
                'rmse': float(rmse),
                'mse': float(mse_score),
                'r2': float(r2),
                'mape': float(mape) if mape != float('inf') else None,
                'directional_accuracy': float(directional_accuracy) if directional_accuracy is not None else None
            },
            'feature_importance': feature_importance,
            'model_type': model_type
        }
        
        # Model-specific insights
        if model_type == 'random_forest' and feature_importance:
            top_features = list(feature_importance.keys())[:3]
            forecast_insight = f"Random Forest - Top features: {', '.join(top_features[:3])}"
        elif model_type == 'catboost' and feature_importance:
            top_features = list(feature_importance.keys())[:3]
            forecast_insight = f"CatBoost - Top features: {', '.join(top_features[:3])}"
        else:
            forecast_insight = f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}"
        
        if mape != float('inf'):
            forecast_insight += f", MAPE: {mape:.1f}%"
        
        if r2 >= 0:
            forecast_insight += f", R²: {r2:.3f}"
        
        return forecast_data, forecast_insight, None
        
    except Exception as e:
        logging.error(f"Forecasting error with {model_type}: {str(e)}")
        return None, None, f"Forecasting error: {str(e)}"


# Statistical forecasting methods
def create_arima_forecast(train_data, test_data, date_column, value_column):
    """Simple ARIMA-like forecast using statsmodels or basic implementation"""
    try:
        # Simple auto-regressive approach as ARIMA fallback
        values = train_data[value_column].values
        
        # Simple AR(1) model
        if len(values) > 1:
            # Calculate autocorrelation
            lag1_corr = np.corrcoef(values[:-1], values[1:])[0, 1] if not np.isnan(np.corrcoef(values[:-1], values[1:])[0, 1]) else 0
            
            # Simple prediction using last value + trend
            trend = np.mean(np.diff(values)) if len(values) > 1 else 0
            last_value = values[-1]
            
            predictions = []
            current_val = last_value
            
            for _ in range(len(test_data)):
                next_val = lag1_corr * current_val + (1 - lag1_corr) * np.mean(values) + trend
                predictions.append(next_val)
                current_val = next_val
            
            y_pred = np.array(predictions)
            y_actual = test_data[value_column].values
            
            # Calculate metrics
            mae = mean_absolute_error(y_actual, y_pred)
            rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
            r2 = r2_score(y_actual, y_pred)
            
            # Simple confidence intervals
            residual_std = np.std(values - np.mean(values))
            ci_50_lower = y_pred - 0.674 * residual_std
            ci_50_upper = y_pred + 0.674 * residual_std
            ci_80_lower = y_pred - 1.282 * residual_std
            ci_80_upper = y_pred + 1.282 * residual_std
            ci_95_lower = y_pred - 1.96 * residual_std
            ci_95_upper = y_pred + 1.96 * residual_std
            
            forecast_data = {
                'model_name': 'ARIMA (Simple AR)',
                'train_dates': train_data[date_column].dt.strftime('%Y-%m-%d').tolist(),
                'train_values': train_data[value_column].tolist(),
                'test_dates': test_data[date_column].dt.strftime('%Y-%m-%d').tolist(),
                'test_actual': y_actual.tolist(),
                'test_predicted': y_pred.tolist(),
                'ci_50_lower': ci_50_lower.tolist(),
                'ci_50_upper': ci_50_upper.tolist(),
                'ci_80_lower': ci_80_lower.tolist(),
                'ci_80_upper': ci_80_upper.tolist(),
                'ci_95_lower': ci_95_lower.tolist(),
                'ci_95_upper': ci_95_upper.tolist(),
                'split_date': train_data[date_column].iloc[-1].strftime('%Y-%m-%d'),
                'metrics': {
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'r2': float(r2),
                    'mape': float(np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100) if np.all(y_actual != 0) else None
                },
                'model_type': 'arima'
            }
            
            return forecast_data, f"ARIMA - MAE: {mae:.2f}, RMSE: {rmse:.2f}", None
        
    except Exception as e:
        return None, None, f"ARIMA forecasting error: {str(e)}"

def create_ets_forecast(train_data, test_data, date_column, value_column):
    """Exponential Smoothing forecast"""
    try:
        values = train_data[value_column].values
        
        # Simple exponential smoothing with trend and seasonality
        alpha = 0.3  # Smoothing parameter for level
        beta = 0.1   # Smoothing parameter for trend
        gamma = 0.1  # Smoothing parameter for seasonality (if applicable)
        
        # Initialize
        level = values[0]
        trend = np.mean(np.diff(values[:min(12, len(values))]))
        
        # Simple seasonal pattern (if enough data)
        seasonal_period = min(12, len(values) // 2)
        if seasonal_period > 1 and len(values) >= seasonal_period * 2:
            seasonal = np.mean(values[:seasonal_period])
        else:
            seasonal = 0
        
        predictions = []
        
        for i in range(len(test_data)):
            forecast = level + trend + seasonal
            predictions.append(forecast)
            
            # Update (using actual values if available for in-sample fit)
            if i < len(values) - len(train_data):
                actual = values[len(train_data) + i]
                level = alpha * actual + (1 - alpha) * (level + trend)
                trend = beta * (level - level) + (1 - beta) * trend
        
        y_pred = np.array(predictions)
        y_actual = test_data[value_column].values
        
        # Calculate metrics
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        r2 = r2_score(y_actual, y_pred)
        
        # Confidence intervals
        residual_std = np.std(values)
        ci_50_lower = y_pred - 0.674 * residual_std
        ci_50_upper = y_pred + 0.674 * residual_std
        ci_80_lower = y_pred - 1.282 * residual_std
        ci_80_upper = y_pred + 1.282 * residual_std
        ci_95_lower = y_pred - 1.96 * residual_std
        ci_95_upper = y_pred + 1.96 * residual_std
        
        forecast_data = {
            'model_name': 'Exponential Smoothing (ETS)',
            'train_dates': train_data[date_column].dt.strftime('%Y-%m-%d').tolist(),
            'train_values': train_data[value_column].tolist(),
            'test_dates': test_data[date_column].dt.strftime('%Y-%m-%d').tolist(),
            'test_actual': y_actual.tolist(),
            'test_predicted': y_pred.tolist(),
            'ci_50_lower': ci_50_lower.tolist(),
            'ci_50_upper': ci_50_upper.tolist(),
            'ci_80_lower': ci_80_lower.tolist(),
            'ci_80_upper': ci_80_upper.tolist(),
            'ci_95_lower': ci_95_lower.tolist(),
            'ci_95_upper': ci_95_upper.tolist(),
            'split_date': train_data[date_column].iloc[-1].strftime('%Y-%m-%d'),
            'metrics': {
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
                'mape': float(np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100) if np.all(y_actual != 0) else None
            },
            'model_type': 'ets'
        }
        
        return forecast_data, f"ETS - MAE: {mae:.2f}, RMSE: {rmse:.2f}", None
        
    except Exception as e:
        return None, None, f"ETS forecasting error: {str(e)}"

def create_theta_forecast(train_data, test_data, date_column, value_column):
    """Theta method forecast (simplified version)"""
    try:
        values = train_data[value_column].values
        
        # Simple theta method implementation
        # Theta = 2 gives more weight to long-term trend
        theta = 2.0
        
        # Deseasonalize if possible
        detrended = values.copy()
        
        # Simple linear trend
        time_index = np.arange(len(values))
        trend_coef = np.polyfit(time_index, values, 1)
        linear_trend = np.polyval(trend_coef, time_index)
        
        # Apply theta transformation
        theta_line = theta * linear_trend + (1 - theta) * values
        
        # Forecast using extrapolated trend
        forecast_periods = len(test_data)
        future_time = np.arange(len(values), len(values) + forecast_periods)
        forecast_trend = np.polyval(trend_coef, future_time)
        
        # Simple forecasting
        y_pred = forecast_trend
        y_actual = test_data[value_column].values
        
        # Calculate metrics
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        r2 = r2_score(y_actual, y_pred)
        
        # Confidence intervals
        residual_std = np.std(values - linear_trend[:len(values)])
        ci_50_lower = y_pred - 0.674 * residual_std
        ci_50_upper = y_pred + 0.674 * residual_std
        ci_80_lower = y_pred - 1.282 * residual_std
        ci_80_upper = y_pred + 1.282 * residual_std
        ci_95_lower = y_pred - 1.96 * residual_std
        ci_95_upper = y_pred + 1.96 * residual_std
        
        forecast_data = {
            'model_name': 'Theta Method',
            'train_dates': train_data[date_column].dt.strftime('%Y-%m-%d').tolist(),
            'train_values': train_data[value_column].tolist(),
            'test_dates': test_data[date_column].dt.strftime('%Y-%m-%d').tolist(),
            'test_actual': y_actual.tolist(),
            'test_predicted': y_pred.tolist(),
            'ci_50_lower': ci_50_lower.tolist(),
            'ci_50_upper': ci_50_upper.tolist(),
            'ci_80_lower': ci_80_lower.tolist(),
            'ci_80_upper': ci_80_upper.tolist(),
            'ci_95_lower': ci_95_lower.tolist(),
            'ci_95_upper': ci_95_upper.tolist(),
            'split_date': train_data[date_column].iloc[-1].strftime('%Y-%m-%d'),
            'metrics': {
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
                'mape': float(np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100) if np.all(y_actual != 0) else None
            },
            'model_type': 'theta'
        }
        
        return forecast_data, f"Theta - MAE: {mae:.2f}, RMSE: {rmse:.2f}", None
        
    except Exception as e:
        return None, None, f"Theta forecasting error: {str(e)}"

def create_moving_average_forecast(train_data, test_data, date_column, value_column):
    """Moving average forecast with seasonal adjustment"""
    try:
        values = train_data[value_column].values
        
        # Adaptive window size based on data length
        window_size = min(max(3, len(values) // 10), 30)
        
        # Calculate moving average
        ma = pd.Series(values).rolling(window=window_size, min_periods=1).mean()
        
        # Calculate seasonal component if enough data
        seasonal_period = min(12, len(values) // 3)
        seasonal_component = 0
        
        if seasonal_period > 2 and len(values) >= seasonal_period * 2:
            seasonal_values = []
            for i in range(seasonal_period):
                season_vals = values[i::seasonal_period]
                if len(season_vals) > 0:
                    seasonal_values.append(np.mean(season_vals) - np.mean(values))
                else:
                    seasonal_values.append(0)
            
            # Forecast with seasonality
            predictions = []
            last_ma = ma.iloc[-1]
            
            for i in range(len(test_data)):
                seasonal_idx = (len(values) + i) % seasonal_period
                seasonal_adj = seasonal_values[seasonal_idx] if seasonal_idx < len(seasonal_values) else 0
                pred = last_ma + seasonal_adj
                predictions.append(pred)
        else:
            # Simple moving average forecast
            last_ma = ma.iloc[-1]
            predictions = [last_ma] * len(test_data)
        
        y_pred = np.array(predictions)
        y_actual = test_data[value_column].values
        
        # Calculate metrics
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        r2 = r2_score(y_actual, y_pred)
        
        # Confidence intervals based on historical volatility
        residual_std = pd.Series(values).rolling(window=window_size).std().iloc[-1]
        if pd.isna(residual_std):
            residual_std = np.std(values)
            
        ci_50_lower = y_pred - 0.674 * residual_std
        ci_50_upper = y_pred + 0.674 * residual_std
        ci_80_lower = y_pred - 1.282 * residual_std
        ci_80_upper = y_pred + 1.282 * residual_std
        ci_95_lower = y_pred - 1.96 * residual_std
        ci_95_upper = y_pred + 1.96 * residual_std
        
        forecast_data = {
            'model_name': f'Moving Average (window={window_size})',
            'train_dates': train_data[date_column].dt.strftime('%Y-%m-%d').tolist(),
            'train_values': train_data[value_column].tolist(),
            'test_dates': test_data[date_column].dt.strftime('%Y-%m-%d').tolist(),
            'test_actual': y_actual.tolist(),
            'test_predicted': y_pred.tolist(),
            'ci_50_lower': ci_50_lower.tolist(),
            'ci_50_upper': ci_50_upper.tolist(),
            'ci_80_lower': ci_80_lower.tolist(),
            'ci_80_upper': ci_80_upper.tolist(),
            'ci_95_lower': ci_95_lower.tolist(),
            'ci_95_upper': ci_95_upper.tolist(),
            'split_date': train_data[date_column].iloc[-1].strftime('%Y-%m-%d'),
            'metrics': {
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
                'mape': float(np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100) if np.all(y_actual != 0) else None
            },
            'model_type': 'moving_average'
        }
        
        return forecast_data, f"Moving Average - MAE: {mae:.2f}, RMSE: {rmse:.2f}", None
        
    except Exception as e:
        return None, None, f"Moving Average forecasting error: {str(e)}"

def create_naive_forecast(train_data, test_data, date_column, value_column):
    """Naive forecast (last value repeated)"""
    try:
        last_value = train_data[value_column].iloc[-1]
        y_pred = np.array([last_value] * len(test_data))
        y_actual = test_data[value_column].values
        
        # Calculate metrics
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        r2 = r2_score(y_actual, y_pred)
        
        # Confidence intervals based on historical volatility
        historical_volatility = np.std(train_data[value_column].values)
        
        ci_50_lower = y_pred - 0.674 * historical_volatility
        ci_50_upper = y_pred + 0.674 * historical_volatility
        ci_80_lower = y_pred - 1.282 * historical_volatility
        ci_80_upper = y_pred + 1.282 * historical_volatility
        ci_95_lower = y_pred - 1.96 * historical_volatility
        ci_95_upper = y_pred + 1.96 * historical_volatility
        
        forecast_data = {
            'model_name': 'Naive (Last Value)',
            'train_dates': train_data[date_column].dt.strftime('%Y-%m-%d').tolist(),
            'train_values': train_data[value_column].tolist(),
            'test_dates': test_data[date_column].dt.strftime('%Y-%m-%d').tolist(),
            'test_actual': y_actual.tolist(),
            'test_predicted': y_pred.tolist(),
            'ci_50_lower': ci_50_lower.tolist(),
            'ci_50_upper': ci_50_upper.tolist(),
            'ci_80_lower': ci_80_lower.tolist(),
            'ci_80_upper': ci_80_upper.tolist(),
            'ci_95_lower': ci_95_lower.tolist(),
            'ci_95_upper': ci_95_upper.tolist(),
            'split_date': train_data[date_column].iloc[-1].strftime('%Y-%m-%d'),
            'metrics': {
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
                'mape': float(np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100) if np.all(y_actual != 0) else None
            },
            'model_type': 'naive'
        }
        
        return forecast_data, f"Naive - MAE: {mae:.2f}, RMSE: {rmse:.2f}", None
        
    except Exception as e:
        return None, None, f"Naive forecasting error: {str(e)}"

def create_naive_drift_forecast(train_data, test_data, date_column, value_column):
    """Naive drift forecast (last value + average historical change)"""
    try:
        values = train_data[value_column].values
        last_value = values[-1]
        
        # Calculate average drift (change per period)
        if len(values) > 1:
            drift = np.mean(np.diff(values))
        else:
            drift = 0
        
        # Generate predictions with drift
        predictions = []
        for i in range(len(test_data)):
            pred = last_value + drift * (i + 1)
            predictions.append(pred)
        
        y_pred = np.array(predictions)
        y_actual = test_data[value_column].values
        
        # Calculate metrics
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        r2 = r2_score(y_actual, y_pred)
        
        # Confidence intervals with increasing uncertainty
        base_volatility = np.std(np.diff(values)) if len(values) > 1 else np.std(values)
        
        ci_50_lower = []
        ci_50_upper = []
        ci_80_lower = []
        ci_80_upper = []
        ci_95_lower = []
        ci_95_upper = []
        
        for i in range(len(test_data)):
            # Uncertainty increases with forecast horizon
            uncertainty = base_volatility * np.sqrt(i + 1)
            
            ci_50_lower.append(y_pred[i] - 0.674 * uncertainty)
            ci_50_upper.append(y_pred[i] + 0.674 * uncertainty)
            ci_80_lower.append(y_pred[i] - 1.282 * uncertainty)
            ci_80_upper.append(y_pred[i] + 1.282 * uncertainty)
            ci_95_lower.append(y_pred[i] - 1.96 * uncertainty)
            ci_95_upper.append(y_pred[i] + 1.96 * uncertainty)
        
        forecast_data = {
            'model_name': 'Naive with Drift',
            'train_dates': train_data[date_column].dt.strftime('%Y-%m-%d').tolist(),
            'train_values': train_data[value_column].tolist(),
            'test_dates': test_data[date_column].dt.strftime('%Y-%m-%d').tolist(),
            'test_actual': y_actual.tolist(),
            'test_predicted': y_pred.tolist(),
            'ci_50_lower': ci_50_lower,
            'ci_50_upper': ci_50_upper,
            'ci_80_lower': ci_80_lower,
            'ci_80_upper': ci_80_upper,
            'ci_95_lower': ci_95_lower,
            'ci_95_upper': ci_95_upper,
            'split_date': train_data[date_column].iloc[-1].strftime('%Y-%m-%d'),
            'metrics': {
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
                'mape': float(np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100) if np.all(y_actual != 0) else None,
                'drift': float(drift)
            },
            'model_type': 'naive_drift'
        }
        
        return forecast_data, f"Naive Drift - MAE: {mae:.2f}, RMSE: {rmse:.2f}, Drift: {drift:.4f}", None
        
    except Exception as e:
        return None, None, f"Naive Drift forecasting error: {str(e)}"


def analyze_time_series(df: pd.DataFrame, params: Dict) -> Dict:
    try:
        """Enhanced time series analysis with forecasting and detailed seasonality insights"""
        logging.info('start analyze time series')
        date_col = params.get('date_col', 'date')
        value_col = params.get('value_col', 'value')
        category_col = params.get('category_col', None)
        model_type = params.get('model_type', 'linear_regression')  # Add model type parameter
        
        # Enhanced datetime parsing
        df[date_col] = parse_datetime_flexible(df[date_col])
        df = df.dropna(subset=[date_col])
        df = df.sort_values(date_col)
        
        logging.info(df)

        if df.empty:
            raise ValueError("No valid datetime data found")
        
        # Prepare data for decomposition
        df_for_decomp = df.copy()
        df_for_decomp['days_from_start'] = (df_for_decomp[date_col] - df_for_decomp[date_col].min()).dt.days
        
        # Enhanced seasonality analysis
        def analyze_detailed_seasonality(data, date_column, value_column):
            """Analyze seasonality patterns with detailed insights"""
            seasonality_insights = []
            
            # Monthly analysis
            data['month'] = data[date_column].dt.month
            data['day_of_week'] = data[date_column].dt.day_of_week
            data['day_of_year'] = data[date_column].dt.day_of_year
            data['week_of_year'] = data[date_column].dt.isocalendar().week
            
            # Monthly seasonality
            monthly_avg = data.groupby('month')[value_column].mean()
            if len(monthly_avg) >= 12:
                peak_month = monthly_avg.idxmax()
                low_month = monthly_avg.idxmin()
                month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                seasonality_insights.append(f"Monthly pattern: Peak in {month_names[peak_month]}, lowest in {month_names[low_month]}")
                
                # Calculate monthly seasonality strength
                monthly_strength = monthly_avg.std() / monthly_avg.mean() if monthly_avg.mean() != 0 else 0
                if monthly_strength > 0.2:
                    seasonality_insights.append(f"Strong monthly seasonality detected (strength: {monthly_strength:.3f})")
                else:
                    seasonality_insights.append(f"Weak monthly seasonality (strength: {monthly_strength:.3f})")
            
            # Weekly seasonality
            weekly_avg = data.groupby('day_of_week')[value_column].mean()
            if len(weekly_avg) >= 7:
                peak_day = weekly_avg.idxmax()
                low_day = weekly_avg.idxmin()
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                seasonality_insights.append(f"Weekly pattern: Peak on {day_names[peak_day]}, lowest on {day_names[low_day]}")
                
                weekly_strength = weekly_avg.std() / weekly_avg.mean() if weekly_avg.mean() != 0 else 0
                if weekly_strength > 0.1:
                    seasonality_insights.append(f"Notable weekly seasonality (strength: {weekly_strength:.3f})")
            
            # Determine primary seasonality period
            if len(data) >= 365:
                seasonality_insights.append("Primary seasonality period: Annual (yearly patterns)")
            elif len(data) >= 30:
                seasonality_insights.append("Primary seasonality period: Monthly patterns")
            elif len(data) >= 7:
                seasonality_insights.append("Primary seasonality period: Weekly patterns")
            else:
                seasonality_insights.append("Insufficient data for reliable seasonality analysis")
            
            return seasonality_insights, monthly_strength if 'monthly_strength' in locals() else 0, weekly_strength if 'weekly_strength' in locals() else 0
        
        
        # Predictability analysis
        def analyze_predictability(data, value_column):
            """Analyze how predictable the data is with detailed explanations"""
            values = data[value_column].values
            
            # Calculate coefficient of variation
            cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else float('inf')
            
            # Random walk test (Ljung-Box test on first differences)
            if len(values) > 10:
                diffs = np.diff(values)
                try:
                    # Simple autocorrelation test
                    autocorr_1 = np.corrcoef(diffs[:-1], diffs[1:])[0, 1] if len(diffs) > 1 else 0
                    if np.isnan(autocorr_1):
                        autocorr_1 = 0
                    
                    # Enhanced predictability categories with detailed explanations
                    if cv < 0.1:
                        predictability = "Very High - Data is very stable and predictable"
                        cv_explanation = "Coefficient of Variation < 0.1 indicates extremely low volatility relative to the mean. The data varies less than 10% around its average value."
                    elif cv < 0.3:
                        predictability = "High - Data shows consistent patterns"
                        cv_explanation = f"Coefficient of Variation = {cv:.3f} indicates low to moderate volatility. The data varies {cv*100:.1f}% around its average value."
                    elif cv < 0.6:
                        predictability = "Moderate - Some variability but patterns are detectable"
                        cv_explanation = f"Coefficient of Variation = {cv:.3f} indicates moderate volatility. The data varies {cv*100:.1f}% around its average value."
                    elif cv < 1.0:
                        predictability = "Low - High variability makes forecasting challenging"
                        cv_explanation = f"Coefficient of Variation = {cv:.3f} indicates high volatility. The data varies {cv*100:.1f}% around its average value."
                    else:
                        predictability = "Very Low - Data is highly volatile and unpredictable"
                        cv_explanation = f"Coefficient of Variation = {cv:.3f} indicates very high volatility. The standard deviation is larger than the mean, suggesting extreme variability."
                    
                    # Enhanced random walk characteristics with explanations
                    if abs(autocorr_1) < 0.1:
                        random_walk_insight = "Data shows random walk characteristics - changes are largely unpredictable"
                        rw_explanation = f"Autocorrelation = {autocorr_1:.3f} (close to 0) suggests that past changes don't predict future changes. Each period's change is essentially random."
                    elif autocorr_1 > 0.3:
                        random_walk_insight = "Data shows momentum - recent changes tend to continue"
                        rw_explanation = f"Positive autocorrelation = {autocorr_1:.3f} indicates momentum effects. When the value increases, it tends to keep increasing (and vice versa)."
                    elif autocorr_1 < -0.3:
                        random_walk_insight = "Data shows mean reversion - tends to return to average after changes"
                        rw_explanation = f"Negative autocorrelation = {autocorr_1:.3f} indicates mean reversion. After moving away from the average, the data tends to move back toward it."
                    else:
                        random_walk_insight = "Data shows weak autocorrelation patterns"
                        rw_explanation = f"Autocorrelation = {autocorr_1:.3f} indicates weak but detectable patterns in how changes follow each other."
                    
                    return predictability, random_walk_insight, float(cv), cv_explanation, rw_explanation
                    
                except:
                    return "Cannot determine - insufficient variation in data", "Analysis inconclusive", float(cv), "Unable to calculate due to insufficient data variation", "Unable to analyze autocorrelation patterns"
            else:
                return "Cannot determine - insufficient data points", "Need more data for analysis", float(cv), "Need at least 10 data points for reliable analysis", "Need more data points to detect patterns"



        # Multi time series support with category
        if category_col and category_col in df.columns:
            logging.info('analyze with category col')
            # Analyze by category
            categories = df[category_col].unique()
            category_stats = {}
            category_decompositions = {}
            category_forecasts = {}
            category_seasonality_insights = {}
            
            for cat in categories:
                cat_data = df[df[category_col] == cat].copy()
                if len(cat_data) > 1:
                    # Calculate trend slope using linear regression
                    cat_data['days_from_start'] = (cat_data[date_col] - cat_data[date_col].min()).dt.days
                    if len(cat_data) > 1 and cat_data['days_from_start'].std() > 0:
                        slope = np.polyfit(cat_data['days_from_start'], cat_data[value_col], 1)[0]
                    else:
                        slope = 0
                    
                    # Predictability analysis
                    predictability, random_walk_insight, cv, cv_explanation, rw_explanation = analyze_predictability(cat_data, value_col)
                    
                    category_stats[cat] = {
                        "mean": clean_float(cat_data[value_col].mean()),
                        "std": clean_float(cat_data[value_col].std()),
                        "min": clean_float(cat_data[value_col].min()),
                        "max": clean_float(cat_data[value_col].max()),
                        "trend_slope": clean_float(slope),
                        "trend": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                        "predictability": predictability,
                        "random_walk_insight": random_walk_insight,
                        "coefficient_variation": clean_float(cv)
                    }
                    
                    # Seasonality analysis per category
                    if len(cat_data) >= 7:
                        seasonality_insights, monthly_strength, weekly_strength = analyze_detailed_seasonality(cat_data, date_col, value_col)
                        category_seasonality_insights[cat] = seasonality_insights
                    else:
                        category_seasonality_insights[cat] = ["Insufficient data for seasonality analysis"]
                    
                    # Forecasting per category with selected model
                    if len(cat_data) >= 10:
                        forecast_data, forecast_insight, forecast_error = create_forecast(cat_data, date_col, value_col, model_type=model_type)
                        if forecast_data:
                            category_forecasts[cat] = forecast_data
                    
                    # Decomposition for each category
                    if len(cat_data) >= 10:
                        cat_data_sorted = cat_data.sort_values(date_col).reset_index(drop=True)
                        
                        # Simple trend calculation using linear regression
                        x_vals = np.arange(len(cat_data_sorted))
                        trend_line = np.polyval(np.polyfit(x_vals, cat_data_sorted[value_col], 1), x_vals)
                        
                        # Simple seasonal calculation using moving average
                        window_size = min(12, len(cat_data_sorted) // 2)
                        if window_size >= 3:
                            seasonal = cat_data_sorted[value_col] - pd.Series(trend_line)
                            seasonal_smooth = seasonal.rolling(window=window_size, center=True).mean().fillna(0)
                        else:
                            seasonal_smooth = np.zeros(len(cat_data_sorted))
                        
                        category_decompositions[cat] = {
                            'dates': cat_data_sorted[date_col].dt.strftime('%Y-%m-%d').tolist(),
                            'original': cat_data_sorted[value_col].tolist(),
                            'trend': trend_line.tolist(),
                            'seasonal': seasonal_smooth.tolist()
                        }
            
            # Overall statistics
            df['days_from_start'] = (df[date_col] - df[date_col].min()).dt.days
            if len(df) > 1 and df['days_from_start'].std() > 0:
                overall_slope = np.polyfit(df['days_from_start'], df[value_col], 1)[0]
            else:
                overall_slope = 0
            
            # Overall predictability
            overall_predictability, overall_random_walk, overall_cv, overall_cv_explanation, overall_rw_explanation = analyze_predictability(df, value_col)
            
            # Overall seasonality insights
            overall_seasonality_insights, monthly_strength, weekly_strength = analyze_detailed_seasonality(df, date_col, value_col)
            
            stats = {
                "mean": clean_float(df[value_col].mean()),
                "std": clean_float(df[value_col].std()),
                "min": clean_float(df[value_col].min()),
                "max": clean_float(df[value_col].max()),
                "trend_slope": clean_float(overall_slope),
                "trend": "increasing" if overall_slope > 0 else "decreasing" if overall_slope < 0 else "stable",
                "categories": category_stats,
                "predictability": overall_predictability,
                "random_walk_insight": overall_random_walk,
                "coefficient_variation": clean_float(overall_cv)
            }
            
            # Create multi-line plot with enhanced styling
            fig = px.line(df, x=date_col, y=value_col, color=category_col, 
                        title="Multi Time Series Analysis")
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#2E4057'),
                title_font_size=16,
                xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
                yaxis=dict(gridcolor='rgba(128,128,128,0.2)')
            )
            
            # Add overall decomposition
            overall_decomposition = None
            overall_forecast = None
            if len(df) >= 10:
                df_sorted = df.sort_values(date_col).reset_index(drop=True)
                x_vals = np.arange(len(df_sorted))
                trend_line = np.polyval(np.polyfit(x_vals, df_sorted[value_col], 1), x_vals)
                
                window_size = min(12, len(df_sorted) // 2)
                if window_size >= 3:
                    seasonal = df_sorted[value_col] - pd.Series(trend_line)
                    seasonal_smooth = seasonal.rolling(window=window_size, center=True).mean().fillna(0)
                else:
                    seasonal_smooth = np.zeros(len(df_sorted))
                
                overall_decomposition = {
                    'dates': df_sorted[date_col].dt.strftime('%Y-%m-%d').tolist(),
                    'original': df_sorted[value_col].tolist(),
                    'trend': trend_line.tolist(),
                    'seasonal': seasonal_smooth.tolist()
                }
                
                # Overall forecast with selected model
                forecast_data, forecast_insight, forecast_error = create_forecast(df_sorted, date_col, value_col, model_type=model_type)
                if forecast_data:
                    overall_forecast = forecast_data
                
        else:
            logging.info('analyze without category col')
            # Single time series analysis
            # Calculate trend slope using linear regression
            df['days_from_start'] = (df[date_col] - df[date_col].min()).dt.days
            if len(df) > 1 and df['days_from_start'].std() > 0:
                slope = np.polyfit(df['days_from_start'], df[value_col], 1)[0]
            else:
                slope = 0
            
            # Predictability analysis
            predictability, random_walk_insight, cv, cv_explanation, rw_explanation = analyze_predictability(df, value_col)
            
            # Seasonality insights
            seasonality_insights, monthly_strength, weekly_strength = analyze_detailed_seasonality(df, date_col, value_col)
            
            # Basic statistics
            stats = {
                "mean": float(df[value_col].mean()),
                "std": float(df[value_col].std()),
                "min": float(df[value_col].min()),
                "max": float(df[value_col].max()),
                "trend_slope": float(slope),
                "trend": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                "predictability": predictability,
                "random_walk_insight": random_walk_insight,
                "coefficient_variation": float(cv)
            }
            logging.info('analyze without category col finish')
            
            # Create time series plot with enhanced styling
            fig = px.line(df, x=date_col, y=value_col, title="Time Series Analysis")
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#2E4057'),
                title_font_size=16,
                xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
                yaxis=dict(gridcolor='rgba(128,128,128,0.2)')
            )
            fig.update_traces(line=dict(color='#3B82F6', width=2))
            
            # Single series decomposition and forecast
            category_decompositions = {}
            category_forecasts = {}
            category_seasonality_insights = {}
            overall_decomposition = None
            overall_forecast = None
            
            if len(df) >= 10:
                df_sorted = df.sort_values(date_col).reset_index(drop=True)
                x_vals = np.arange(len(df_sorted))
                trend_line = np.polyval(np.polyfit(x_vals, df_sorted[value_col], 1), x_vals)
                
                window_size = min(12, len(df_sorted) // 2)
                if window_size >= 3:
                    seasonal = df_sorted[value_col] - pd.Series(trend_line)
                    seasonal_smooth = seasonal.rolling(window=window_size, center=True).mean().fillna(0)
                else:
                    seasonal_smooth = np.zeros(len(df_sorted))
                
                overall_decomposition = {
                    'dates': df_sorted[date_col].dt.strftime('%Y-%m-%d').tolist(),
                    'original': df_sorted[value_col].tolist(),
                    'trend': trend_line.tolist(),
                    'seasonal': seasonal_smooth.tolist()
                }
                
                # Create forecast with selected model
                forecast_data, forecast_insight, forecast_error = create_forecast(df_sorted, date_col, value_col, model_type=model_type)
                if forecast_data:
                    overall_forecast = forecast_data
        
        plot_json = fig.to_json()
        
        # Enhanced seasonality analysis
        logging.info('Enhanced seasonality analysis')
        df['month'] = df[date_col].dt.month
        df['day_of_week'] = df[date_col].dt.day_name()
        df['hour'] = df[date_col].dt.hour
        
        monthly_avg = df.groupby('month')[value_col].mean()
        seasonality_strength = float(monthly_avg.std() / monthly_avg.mean()) if monthly_avg.mean() != 0 else 0
        
        # Weekly seasonality
        logging.info('Weekly seasonality')
        weekly_avg = df.groupby('day_of_week')[value_col].mean()
        weekly_seasonality = float(weekly_avg.std() / weekly_avg.mean()) if weekly_avg.mean() != 0 else 0
        
        # Volatility analysis
        logging.info('Volatility analysis')
        volatility = float(df[value_col].std() / df[value_col].mean()) if df[value_col].mean() != 0 else 0
        
        # Growth rate calculation (comparing first and last periods)
        if len(df) > 1:
            first_value = df[value_col].iloc[0]
            last_value = df[value_col].iloc[-1]
            if first_value != 0:
                growth_rate = ((last_value - first_value) / first_value) * 100
            else:
                growth_rate = 0
        else:
            growth_rate = 0
        
        logging.info('creating insight')
        
        # Enhanced insights with seasonality and predictability information
        insights = [
            f"Average value: {stats['mean']:.2f}",
            f"Overall trend: {stats['trend']} (slope: {stats['trend_slope']:.4f} per day)",
            f"Predictability: {stats['predictability']}",
            f"Random walk analysis: {stats['random_walk_insight']}",
            f"Data span: {df[date_col].min().strftime('%Y-%m-%d')} to {df[date_col].max().strftime('%Y-%m-%d')}",
            f"Total growth: {growth_rate:.1f}%",
            f"Volatility: {'High' if volatility > 0.5 else 'Moderate' if volatility > 0.2 else 'Low'} (CV: {volatility:.3f})"
        ]
        
        # Add seasonality insights to main insights
        if category_col and category_col in df.columns:
            insights.extend(overall_seasonality_insights[:2])  # Add first 2 seasonality insights
        else:
            insights.extend(seasonality_insights[:2])  # Add first 2 seasonality insights
        
        logging.info(f"stats: {stats}")
        logging.info(f"plot_json type: {type(plot_json)}")
        logging.info(f"insights: {insights}")

        response_data = {
            "module": "time_series",
            "statistics": stats,
            "plot": plot_json,
            "insights": insights,
            "seasonality_strength": seasonality_strength,
            "weekly_seasonality": weekly_seasonality,
            "volatility": volatility,
            "growth_rate": growth_rate,
            "has_categories": category_col is not None and category_col in df.columns,
            "category_decompositions": category_decompositions,
            "category_forecasts": category_forecasts,
            "category_seasonality_insights": category_seasonality_insights,
            "overall_decomposition": overall_decomposition,
            "overall_forecast": overall_forecast,
            "categories_list": list(categories) if category_col and category_col in df.columns else [],
            "seasonality_insights": overall_seasonality_insights if category_col and category_col in df.columns else seasonality_insights,
            "selected_model": model_type  # Add selected model info
        }
    
        return sanitize_for_json(response_data)
    except Exception as e:
        logging.error(f"🔥 Internal error in time series analysis: {e}")
        raise

def analyze_customer(df: pd.DataFrame, params: Dict) -> Dict:
    """Enhanced customer analysis (RFM-like)"""
    customer_col = params.get('customer_col', 'customer_id')
    amount_col = params.get('amount_col', 'amount')
    date_col = params.get('date_col', 'transaction_date')
    
    # Enhanced datetime parsing
    df[date_col] = parse_datetime_flexible(df[date_col])
    df = df.dropna(subset=[date_col])
    reference_date = df[date_col].max()
    
    # Calculate RFM metrics
    rfm = df.groupby(customer_col).agg({
        date_col: lambda x: (reference_date - x.max()).days,  # Recency
        amount_col: ['count', 'sum']  # Frequency, Monetary
    }).reset_index()
    
    rfm.columns = [customer_col, 'recency', 'frequency', 'monetary']
    
    # Simple segmentation
    rfm['recency_score'] = pd.qcut(rfm['recency'], 3, labels=['3', '2', '1'])
    rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 3, labels=['1', '2', '3'])
    rfm['monetary_score'] = pd.qcut(rfm['monetary'], 3, labels=['1', '2', '3'])
    
    # Create segments
    def segment_customers(row):
        if row['frequency_score'] == '3' and row['monetary_score'] == '3':
            return 'Champions'
        elif row['recency_score'] == '3' and row['frequency_score'] == '3':
            return 'Loyal Customers'
        elif row['recency_score'] == '1':
            return 'At Risk'
        else:
            return 'Others'
    
    rfm['segment'] = rfm.apply(segment_customers, axis=1)
    
    # Statistics
    segment_counts = rfm['segment'].value_counts().to_dict()
    
    # Create visualization
    fig = px.scatter(rfm, x='frequency', y='monetary', color='segment', 
                     title="Customer Segmentation", hover_data=['recency'])
    plot_json = fig.to_json()
    
    insights = [
        f"Total customers analyzed: {len(rfm)}",
        f"Champions: {segment_counts.get('Champions', 0)} customers",
        f"At Risk customers: {segment_counts.get('At Risk', 0)} customers"
    ]
    
    return {
        "module": "customer",
        "segments": segment_counts,
        "plot": plot_json,
        "insights": insights,
        "rfm_summary": {
            "avg_recency": float(rfm['recency'].mean()),
            "avg_frequency": float(rfm['frequency'].mean()),
            "avg_monetary": float(rfm['monetary'].mean())
        }
    }

def analyze_manufacturing(df: pd.DataFrame, params: Dict) -> Dict:
    """Manufacturing-specific analysis"""
    timestamp_col = params.get('timestamp_col', 'timestamp')
    production_col = params.get('production_col', 'production_units')
    defect_col = params.get('defect_col', 'defect_rate_percent')
    
    # Parse datetime
    df[timestamp_col] = parse_datetime_flexible(df[timestamp_col])
    df = df.dropna(subset=[timestamp_col])
    
    # Calculate KPIs
    total_production = df[production_col].sum()
    avg_defect_rate = df[defect_col].mean()
    oee = ((total_production / len(df)) / 850) * 100  # Assuming 850 is target
    
    # Shift analysis
    if 'shift' in df.columns:
        shift_performance = df.groupby('shift').agg({
            production_col: 'mean',
            defect_col: 'mean'
        }).round(2)
    
    # Create production trend plot
    fig = px.line(df.head(1000), x=timestamp_col, y=production_col, 
                  title="Production Trend (First 1000 records)")
    plot_json = fig.to_json()
    
    insights = [
        f"Total production: {total_production:,.0f} units",
        f"Average defect rate: {avg_defect_rate:.2f}%",
        f"Estimated OEE: {oee:.1f}%",
        f"Best performing shift: {shift_performance[production_col].idxmax() if 'shift' in df.columns else 'N/A'}"
    ]
    
    return {
        "module": "manufacturing",
        "kpis": {
            "total_production": float(total_production),
            "avg_defect_rate": float(avg_defect_rate),
            "oee": float(oee)
        },
        "plot": plot_json,
        "insights": insights
    }

def analyze_healthcare(df: pd.DataFrame, params: Dict) -> Dict:
    """Healthcare-specific analysis"""
    date_col = params.get('date_col', 'admission_date')
    cost_col = params.get('cost_col', 'total_cost_usd')
    los_col = params.get('los_col', 'length_of_stay_days')
    
    # Parse datetime
    df[date_col] = parse_datetime_flexible(df[date_col])
    df = df.dropna(subset=[date_col])
    
    # Calculate healthcare KPIs
    avg_los = df[los_col].mean()
    avg_cost = df[cost_col].mean()
    total_admissions = len(df)
    
    # Department analysis
    if 'department' in df.columns:
        dept_stats = df.groupby('department').agg({
            cost_col: 'mean',
            los_col: 'mean'
        }).round(2)
        
        busiest_dept = df['department'].value_counts().index[0]
    
    # Create cost distribution plot
    fig = px.histogram(df, x=cost_col, nbins=50, 
                       title="Distribution of Patient Costs")
    plot_json = fig.to_json()
    
    insights = [
        f"Total admissions: {total_admissions:,}",
        f"Average length of stay: {avg_los:.1f} days", 
        f"Average cost per patient: ${avg_cost:,.2f}",
        f"Busiest department: {busiest_dept if 'department' in df.columns else 'N/A'}"
    ]
    
    return {
        "module": "healthcare",
        "kpis": {
            "total_admissions": total_admissions,
            "avg_los": float(avg_los),
            "avg_cost": float(avg_cost)
        },
        "plot": plot_json,
        "insights": insights
    }

def analyze_airline(df: pd.DataFrame, params: Dict) -> Dict:
    """Airline-specific analysis with trend and seasonality"""
    date_col = params.get('date_col', 'flight_date')
    passengers_col = params.get('passengers_col', 'passengers_count')
    revenue_col = params.get('revenue_col', 'revenue_idr')
    
    # Parse datetime
    df[date_col] = parse_datetime_flexible(df[date_col])
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)
    
    # Daily aggregation for trend analysis
    daily_stats = df.groupby(date_col).agg({
        passengers_col: 'sum',
        revenue_col: 'sum',
        'load_factor_percent': 'mean'
    }).reset_index()
    
    # Calculate growth trend
    first_month = daily_stats.head(30)[passengers_col].mean()
    last_month = daily_stats.tail(30)[passengers_col].mean()
    growth_rate = ((last_month - first_month) / first_month) * 100 if first_month > 0 else 0
    
    # Seasonality analysis
    df['month'] = df[date_col].dt.month
    monthly_passengers = df.groupby('month')[passengers_col].sum()
    peak_month = monthly_passengers.idxmax()
    seasonality_strength = float(monthly_passengers.std() / monthly_passengers.mean())
    
    # KPIs
    total_passengers = df[passengers_col].sum()
    total_revenue = df[revenue_col].sum()
    avg_load_factor = df['load_factor_percent'].mean()
    total_flights = len(df)
    
    # Create trend plot
    fig = px.line(daily_stats, x=date_col, y=passengers_col, 
                  title="Daily Passenger Traffic Trend")
    plot_json = fig.to_json()
    
    # Route analysis
    if 'route' in df.columns:
        top_route = df.groupby('route')[passengers_col].sum().idxmax()
    
    insights = [
        f"Total passengers transported: {total_passengers:,}",
        f"Strong upward trend: {growth_rate:.1f}% growth",
        f"Peak travel month: {['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][peak_month]}",
        f"Average load factor: {avg_load_factor:.1f}%",
        f"Most popular route: {top_route if 'route' in df.columns else 'N/A'}"
    ]
    
    return {
        "module": "airline",
        "kpis": {
            "total_passengers": int(total_passengers),
            "total_revenue": int(total_revenue),
            "avg_load_factor": float(avg_load_factor),
            "total_flights": total_flights,
            "growth_rate": float(growth_rate)
        },
        "plot": plot_json,
        "insights": insights,
        "seasonality_strength": seasonality_strength
    }

def analyze_education(df: pd.DataFrame, params: Dict) -> Dict:
    """Education/School book sales analysis with extreme seasonality"""
    date_col = params.get('date_col', 'sale_date')
    revenue_col = params.get('revenue_col', 'total_revenue_idr')
    quantity_col = params.get('quantity_col', 'quantity_sold')
    
    # Parse datetime
    df[date_col] = parse_datetime_flexible(df[date_col])
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)
    
    # Daily aggregation
    daily_stats = df.groupby(date_col).agg({
        quantity_col: 'sum',
        revenue_col: 'sum'
    }).reset_index()
    
    # Extreme seasonality analysis
    df['month'] = df[date_col].dt.month
    monthly_sales = df.groupby('month')[quantity_col].sum()
    peak_month = monthly_sales.idxmax()
    lowest_month = monthly_sales.idxmin()
    
    # Calculate peak vs low season ratio
    peak_ratio = monthly_sales.max() / monthly_sales.min() if monthly_sales.min() > 0 else 0
    seasonality_strength = float(monthly_sales.std() / monthly_sales.mean())
    
    # School season identification
    school_prep_months = [6, 7]  # June-July peak
    semester_months = [1, 12]    # Dec-Jan secondary peak
    
    school_season_sales = df[df['month'].isin(school_prep_months + semester_months)][quantity_col].sum()
    total_sales = df[quantity_col].sum()
    school_season_percentage = (school_season_sales / total_sales) * 100 if total_sales > 0 else 0
    
    # Category analysis
    if 'book_category' in df.columns:
        top_category = df.groupby('book_category')[quantity_col].sum().idxmax()
        category_distribution = df.groupby('book_category')[quantity_col].sum().to_dict()
    
    # Create seasonality plot
    fig = px.line(daily_stats.head(1000), x=date_col, y=quantity_col, 
                  title="Book Sales Seasonality (First 1000 days)")
    plot_json = fig.to_json()
    
    # KPIs
    total_books_sold = df[quantity_col].sum()
    total_revenue = df[revenue_col].sum()
    avg_daily_sales = daily_stats[quantity_col].mean()
    
    insights = [
        f"EXTREME seasonality detected: {peak_ratio:.1f}x difference between peak and low months",
        f"Peak month: {['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][peak_month]} (School preparation)",
        f"School seasons account for {school_season_percentage:.1f}% of annual sales",
        f"Total books sold: {total_books_sold:,}",
        f"Most popular category: {top_category if 'book_category' in df.columns else 'N/A'}"
    ]
    
    return {
        "module": "education",
        "kpis": {
            "total_books_sold": int(total_books_sold),
            "total_revenue": int(total_revenue),
            "avg_daily_sales": float(avg_daily_sales),
            "peak_ratio": float(peak_ratio),
            "school_season_percentage": float(school_season_percentage)
        },
        "plot": plot_json,
        "insights": insights,
        "seasonality_strength": seasonality_strength,
        "category_distribution": category_distribution if 'book_category' in df.columns else {}
    }

@app.get("/session/{session_id}/data")
def get_session_data(session_id: str, limit: int = 100):
    if session_id not in data_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    df = data_store[session_id]
    return {
        "columns": df.columns.tolist(),
        "shape": df.shape,
        "data": df.head(limit).to_dict('records')
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)