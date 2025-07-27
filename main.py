from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from io import StringIO
import json, logging, sys
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import modules
from data_generators import DataGenerators
from forecasting_models import ForecastingModels
from analysis_modules import AnalysisModules
from utils import parse_datetime_flexible, clean_float, sanitize_for_json

# Configure logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

app = FastAPI(title="Data Analytics", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    category_col: Optional[str] = None

class AnalysisRequest(BaseModel):
    session_id: str
    module: str
    parameters: Dict

# Sample datasets
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
        "data": DataGenerators.generate_manufacturing_data()
    },
    "financial_trading": {
        "name": "Financial Trading Data", 
        "description": "High-frequency trading data with multiple symbols and market metrics (YYYYMMDD format)",
        "data": DataGenerators.generate_trading_data()
    },
    "healthcare_admissions": {
        "name": "Healthcare Patient Admissions",
        "description": "Hospital patient admission records with department and cost data (MM-DD-YYYY format)", 
        "data": DataGenerators.generate_healthcare_data()
    },
    "airline_flights": {
        "name": "Airline Flight Operations", 
        "description": "Daily flight data showing strong upward trend in passenger traffic with seasonal patterns (YYYY-MM-DD format)",
        "data": DataGenerators.generate_flight_data()
    },
    # "school_book_sales": {
    #     "name": "School Book Sales Demand",
    #     "description": "Educational book sales with extremely strong seasonality around school seasons (DD/MM/YYYY format)", 
    #     "data": DataGenerators.generate_school_book_data()
    # }
}

# Routes
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

@app.post("/load-sample/{dataset_key}")
def load_sample_dataset(dataset_key: str):
    if dataset_key not in SAMPLE_DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    session_id = f"sample_{dataset_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    data_store[session_id] = SAMPLE_DATASETS[dataset_key]["data"].copy()
    df = data_store[session_id]

    return DataUploadResponse(
        session_id=session_id,
        columns=df.columns.tolist(),
        shape=df.shape,
        sample_data=df.to_dict('records')
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
        
        if len(df) > 10000:
            df = df.head(10000)
        
        session_id = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        data_store[session_id] = df
        
        return DataUploadResponse(
            session_id=session_id,
            columns=df.columns.tolist(),
            shape=df.shape,
            sample_data=df.to_dict('records')
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

@app.post("/aggregate-data")
def aggregate_data(request: AggregationRequest):
    if request.session_id not in data_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    df = data_store[request.session_id].copy()
    
    try:
        df[request.datetime_col] = parse_datetime_flexible(df[request.datetime_col])
        
        if df[request.datetime_col].isna().any():
            failed_count = df[request.datetime_col].isna().sum()
            logging.warning(f"Failed to parse {failed_count} datetime values")
            df = df.dropna(subset=[request.datetime_col])
        
        if df.empty:
            raise HTTPException(status_code=400, detail="No valid datetime values found")
        
        df_agg = df.set_index(request.datetime_col)
        
        # Aggregation logic
        agg_methods = {
            'sum': lambda x: x.sum(),
            'mean': lambda x: x.mean(),
            'count': lambda x: x.count(),
            'median': lambda x: x.median()
        }
        
        if request.agg_method not in agg_methods:
            raise ValueError("Unsupported aggregation method")
        
        agg_func = agg_methods[request.agg_method]
        
        if request.category_col and request.category_col in df.columns:
            result = df_agg.groupby(request.category_col)[request.value_col].resample(request.freq).apply(agg_func)
            result_df = result.reset_index()
            result_df.columns = ['category', 'date', 'value']
            agg_df = result_df
        else:
            result = agg_func(df_agg[request.value_col].resample(request.freq))
            agg_df = pd.DataFrame({
                'date': result.index,
                'value': result.values
            }).reset_index(drop=True)
        
        agg_session_id = f"{request.session_id}_agg"
        data_store[agg_session_id] = agg_df

        logging.info('in agg')
        logging.info(agg_df)
        
        return {
            "session_id": agg_session_id,
            "shape": agg_df.shape,
            "data": agg_df.to_dict('records'),
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
            logging.info('analyze time series')
            logging.info(df.shape)
            return AnalysisModules.analyze_time_series(df, request.parameters)
        elif request.module == "customer":
            return AnalysisModules.analyze_customer(df, request.parameters)
        else:
            raise HTTPException(status_code=400, detail="Unsupported module")
    
    except Exception as e:
        logging.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Analysis error: {str(e)}")

@app.post("/compare-models")
def compare_models_endpoint(request: AnalysisRequest):
    if request.session_id not in data_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    df = data_store[request.session_id].copy()
    
    try:
        if request.module == "time_series":
            return ForecastingModels.compare_forecasting_models(df, request.parameters)
        else:
            raise HTTPException(status_code=400, detail="Model comparison only available for time series")
    
    except Exception as e:
        logging.error(f"Model comparison error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Model comparison error: {str(e)}")

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