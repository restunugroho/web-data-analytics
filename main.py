from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import pandas as pd
import numpy as np
import re
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

class ForecastRequest(BaseModel):
    session_id: str
    date_col: str
    value_col: str
    model_type: str = 'linear_regression'
    category_col: Optional[str] = None

class ModelComparisonRequest(BaseModel):
    session_id: str
    date_col: str
    value_col: str
    comparison_models: Optional[List[str]] = ['linear_regression', 'random_forest', 'ets', 'naive']
    category_col: Optional[str] = None

class AggregationRequest(BaseModel):
    session_id: str
    datetime_col: str
    value_col: str
    agg_method: str
    freq: str  # Now supports custom frequencies like "2H", "30min", "1D"
    category_col: Optional[str] = None
    
    @validator('freq')
    def validate_frequency(cls, v):
        # Support both simple format (D, H, min) and custom format (2H, 30min, 5D)
        freq_pattern = re.match(r'^(\d+)([a-zA-Z]+)$', v)
        if freq_pattern:
            multiplier = int(freq_pattern.group(1))
            unit = freq_pattern.group(2)
            valid_units = ['min', 'H', 'D', 'W', 'M', 'Q']
            if unit not in valid_units or multiplier < 1 or multiplier > 1000:
                raise ValueError(f'Invalid frequency: {v}')
        else:
            # Simple format validation
            valid_simple = ['min', 'H', 'D', 'W', 'M', 'Q']
            if v not in valid_simple:
                raise ValueError(f'Frequency must be one of: {", ".join(valid_simple)} or custom format like "2H", "30min"')
        return v
    
    @validator('agg_method')
    def validate_aggregation_method(cls, v):
        valid_methods = ['sum', 'mean', 'count', 'median']
        if v not in valid_methods:
            raise ValueError(f'Aggregation method must be one of: {", ".join(valid_methods)}')
        return v

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
}

# Basic Routes
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

# Data Management Routes
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
        # Parse datetime column
        df[request.datetime_col] = parse_datetime_flexible(df[request.datetime_col])
        
        if df[request.datetime_col].isna().any():
            failed_count = df[request.datetime_col].isna().sum()
            logging.warning(f"Failed to parse {failed_count} datetime values")
            df = df.dropna(subset=[request.datetime_col])
        
        if df.empty:
            raise HTTPException(status_code=400, detail="No valid datetime values found")
        
        df_agg = df.set_index(request.datetime_col)
        
        # Validate and parse frequency - support custom format like "2H", "30min"
        freq_pattern = re.match(r'^(\d+)([a-zA-Z]+)$', request.freq)
        if freq_pattern:
            freq_multiplier = int(freq_pattern.group(1))
            freq_unit = freq_pattern.group(2)
            valid_freq_units = ['min', 'H', 'D', 'W', 'M', 'Q']
            if freq_unit not in valid_freq_units or freq_multiplier < 1:
                raise ValueError(f"Invalid frequency format: {request.freq}")
        else:
            # Fallback for simple formats like "D", "H" (backwards compatibility)
            valid_frequencies = ['min', 'H', 'D', 'W', 'M', 'Q']
            if request.freq not in valid_frequencies:
                raise ValueError(f"Unsupported frequency: {request.freq}")
        
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
        
        # Remove NaN values
        agg_df = agg_df.dropna()
        
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

# Analysis Routes
@app.post("/analyze/time-series")
def analyze_time_series(request: AnalysisRequest):
    """Analyze time series data with statistical insights and patterns"""
    if request.session_id not in data_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    df = data_store[request.session_id].copy()
    
    try:
        logging.info('Analyzing time series')
        logging.info(f"Data shape: {df.shape}")
        return AnalysisModules.analyze_time_series(df, request.parameters)
    
    except Exception as e:
        logging.error(f"Time series analysis error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Time series analysis error: {str(e)}")

@app.post("/analyze/customer")
def analyze_customer(request: AnalysisRequest):
    """Perform customer analysis (RFM segmentation)"""
    if request.session_id not in data_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    df = data_store[request.session_id].copy()
    
    try:
        logging.info('Analyzing customer data')
        logging.info(f"Data shape: {df.shape}")
        return AnalysisModules.analyze_customer(df, request.parameters)
    
    except Exception as e:
        logging.error(f"Customer analysis error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Customer analysis error: {str(e)}")

# Forecasting Routes
@app.post("/forecast/single-model")
def forecast_single_model(request: ForecastRequest):
    """Generate forecast using a single specified model"""
    if request.session_id not in data_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    df = data_store[request.session_id].copy()
    
    try:
        logging.info(f'Creating forecast with {request.model_type} model')
        logging.info(f"Data shape: {df.shape}")
        
        # Parse datetime column
        df[request.date_col] = parse_datetime_flexible(df[request.date_col])
        df = df.dropna(subset=[request.date_col]).sort_values(request.date_col)
        
        if df.empty:
            raise ValueError("No valid datetime data found")
        
        if len(df) < 10:
            raise ValueError("Insufficient data for forecasting (minimum 10 points required)")
        
        # Handle category-based forecasting
        if request.category_col and request.category_col in df.columns:
            categories = df[request.category_col].unique()
            category_forecasts = {}
            
            for cat in categories:
                cat_data = df[df[request.category_col] == cat].copy()
                if len(cat_data) >= 10:
                    forecast_data, forecast_insight, forecast_error = ForecastingModels.create_forecast(
                        cat_data, request.date_col, request.value_col, model_type=request.model_type
                    )
                    if forecast_data and forecast_error is None:
                        category_forecasts[cat] = {
                            'forecast': forecast_data,
                            'insight': forecast_insight,
                            'status': 'success'
                        }
                    else:
                        category_forecasts[cat] = {
                            'status': 'failed',
                            'error': forecast_error or 'Forecast generation failed'
                        }
                else:
                    category_forecasts[cat] = {
                        'status': 'failed',
                        'error': f'Insufficient data for category {cat} (need at least 10 points)'
                    }
            
            return {
                "forecast_type": "multi-category",
                "model_type": request.model_type,
                "categories": list(categories),
                "category_forecasts": category_forecasts,
                "total_categories": len(categories),
                "successful_forecasts": len([f for f in category_forecasts.values() if f['status'] == 'success'])
            }
        
        else:
            # Single series forecasting
            forecast_data, forecast_insight, forecast_error = ForecastingModels.create_forecast(
                df, request.date_col, request.value_col, model_type=request.model_type
            )
            
            if forecast_error:
                raise ValueError(forecast_error)
            
            return {
                "forecast_type": "single-series",
                "model_type": request.model_type,
                "forecast": forecast_data,
                "insight": forecast_insight,
                "status": "success"
            }
    
    except Exception as e:
        logging.error(f"Single model forecast error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Forecast error: {str(e)}")

@app.post("/forecast/compare-models")
def compare_forecasting_models(request: ModelComparisonRequest):
    """Compare multiple forecasting models and rank them by performance"""
    if request.session_id not in data_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    df = data_store[request.session_id].copy()
    
    try:
        logging.info('Comparing forecasting models')
        logging.info(f"Models to compare: {request.comparison_models}")
        logging.info(f"Data shape: {df.shape}")
        
        # Prepare parameters for comparison
        comparison_params = {
            'date_col': request.date_col,
            'value_col': request.value_col,
            'comparison_models': request.comparison_models,
            'category_col': request.category_col
        }
        
        # Handle category-based comparison
        if request.category_col and request.category_col in df.columns:
            categories = df[request.category_col].unique()
            category_comparisons = {}
            
            for cat in categories:
                cat_data = df[df[request.category_col] == cat].copy()
                if len(cat_data) >= 10:
                    try:
                        comparison_result = ForecastingModels.compare_forecasting_models(
                            cat_data, comparison_params
                        )
                        category_comparisons[cat] = {
                            'comparison': comparison_result,
                            'status': 'success'
                        }
                    except Exception as e:
                        category_comparisons[cat] = {
                            'status': 'failed',
                            'error': str(e)
                        }
                else:
                    category_comparisons[cat] = {
                        'status': 'failed',
                        'error': f'Insufficient data for category {cat} (need at least 10 points)'
                    }
            
            return {
                "comparison_type": "multi-category",
                "categories": list(categories),
                "category_comparisons": category_comparisons,
                "models_tested": request.comparison_models,
                "total_categories": len(categories),
                "successful_comparisons": len([c for c in category_comparisons.values() if c['status'] == 'success'])
            }
        
        else:
            # Single series comparison
            comparison_result = ForecastingModels.compare_forecasting_models(df, comparison_params)
            
            return {
                "comparison_type": "single-series",
                "comparison": comparison_result,
                "models_tested": request.comparison_models,
                "status": "success"
            }
    
    except Exception as e:
        logging.error(f"Model comparison error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Model comparison error: {str(e)}")

@app.get("/forecast/available-models")
def get_available_models():
    """Get list of available forecasting models"""
    return {
        "models": [
            {
                "key": "linear_regression",
                "name": "Linear Regression",
                "description": "Simple linear trend model with seasonal features",
                "complexity": "Low",
                "best_for": "Data with clear trends and minimal noise"
            },
            {
                "key": "random_forest",
                "name": "Random Forest",
                "description": "Ensemble model that captures complex patterns",
                "complexity": "Medium",
                "best_for": "Data with complex seasonality and non-linear patterns"
            },
            {
                "key": "naive",
                "name": "Naive Forecast",
                "description": "Uses last observed value as forecast",
                "complexity": "Very Low",
                "best_for": "Baseline comparison and stable data"
            },
            {
                "key": "moving_average",
                "name": "Moving Average",
                "description": "Average of recent observations",
                "complexity": "Low",
                "best_for": "Smoothing noisy data and identifying trends"
            },
            {
                "key": "ets",
                "name": "Exponential Smoothing",
                "description": "Weighted average giving more weight to recent observations",
                "complexity": "Medium",
                "best_for": "Data with trend and seasonal patterns"
            }
        ]
    }

# Legacy route for backward compatibility (can be removed later)
@app.post("/analyze")
def analyze_data_legacy(request: AnalysisRequest):
    """Legacy analyze endpoint - redirects to appropriate specialized endpoint"""
    if request.module == "time_series":
        return analyze_time_series(request)
    elif request.module == "customer":
        return analyze_customer(request)
    else:
        raise HTTPException(status_code=400, detail="Unsupported module. Use /analyze/time-series or /analyze/customer")

# Legacy route for backward compatibility (can be removed later)
@app.post("/compare-models")
def compare_models_legacy(request: AnalysisRequest):
    """Legacy compare models endpoint - redirects to forecast comparison"""
    if request.module == "time_series":
        # Convert AnalysisRequest to ModelComparisonRequest
        comparison_request = ModelComparisonRequest(
            session_id=request.session_id,
            date_col=request.parameters.get('date_col', 'date'),
            value_col=request.parameters.get('value_col', 'value'),
            comparison_models=request.parameters.get('comparison_models', ['linear_regression', 'random_forest', 'ets', 'naive']),
            category_col=request.parameters.get('category_col')
        )
        return compare_forecasting_models(comparison_request)
    else:
        raise HTTPException(status_code=400, detail="Model comparison only available for time series. Use /forecast/compare-models")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)