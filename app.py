import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from datetime import datetime
import io
import logging, sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.express as px

# Hapus handler lama (bawaan Streamlit)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Atur agar logging masuk ke stdout (ditangkap oleh run.sh)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

import streamlit as st

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Data Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Enhanced colors and styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}
.main-header h1 {
    color: white;
    text-align: center;
    margin: 0;
    font-weight: 600;
}
.module-card {
    border: 2px solid #e3f2fd;
    border-radius: 15px;
    padding: 1.5rem;
    margin: 0.5rem 0;
    background: linear-gradient(135deg, #f8f9ff 0%, #e8f4fd 100%);
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
.success-box {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    border: 2px solid #28a745;
    border-radius: 10px;
    padding: 15px;
    margin: 15px 0;
    box-shadow: 0 2px 8px rgba(40,167,69,0.2);
    color: #155724;
    font-weight: 500;
}
.info-box {
    background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
    border: 2px solid #17a2b8;
    border-radius: 10px;
    padding: 15px;
    margin: 15px 0;
    box-shadow: 0 2px 8px rgba(23,162,184,0.2);
    color: #0c5460;
    font-weight: 500;
}
.warning-box {
    background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    border: 2px solid #ffc107;
    border-radius: 10px;
    padding: 15px;
    margin: 15px 0;
    box-shadow: 0 2px 8px rgba(255,193,7,0.2);
    color: #856404;
    font-weight: 500;
}
.metric-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    border: 2px solid #dee2e6;
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.filter-container {
    background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%);
    border: 2px solid #28a745;
    border-radius: 12px;
    padding: 1rem;
    margin: 1rem 0;
}
.decomposition-container {
    background: linear-gradient(135deg, #fff8e1 0%, #fffde7 100%);
    border: 2px solid #ff9800;
    border-radius: 12px;
    padding: 1rem;
    margin: 1rem 0;
}
/* Tab styling */
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 16px;
    font-weight: 600;
    color: #2E4057;
}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
    border-radius: 8px 8px 0 0;
}
/* Metric styling */
.stMetric {
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    border: 1px solid #dee2e6;
    border-radius: 10px;
    padding: 1rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
.stMetric > label {
    color: #495057 !important;
    font-weight: 600 !important;
}
.stMetric > div {
    color: #212529 !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Header
st.markdown("""
<div class="main-header">
    <h1>📊 Data Analytics</h1>
    <p style="text-align: center; color: white; margin: 0;">
        Time Series & Customer Analytics - No Setup Required
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Module Selection
st.sidebar.title("🎯 Select Analysis Module")

module = st.sidebar.selectbox(
    "Choose your analysis type:",
    ["time_series", "customer"],
    format_func=lambda x: {
        "time_series": "📈 Time Series Analysis",
        "customer": "👥 Customer Analytics"
    }[x]
)

# Module descriptions
module_descriptions = {
    "time_series": {
        "title": "📈 Time Series Analysis",
        "description": "Analyze trends, seasonality, and patterns in time-based data",
        "use_cases": ["Sales forecasting", "Website traffic analysis", "Performance monitoring"],
        "required_columns": ["Date/Time column", "Numeric value column"]
    },
    "customer": {
        "title": "👥 Customer Analytics", 
        "description": "Segment customers using RFM analysis and behavioral patterns",
        "use_cases": ["Customer segmentation", "Retention analysis", "Value-based targeting"],
        "required_columns": ["Customer ID", "Transaction amount", "Transaction date"]
    }
}

# Display module info
with st.sidebar.expander("ℹ️ Module Information", expanded=True):
    info = module_descriptions[module]
    st.write(f"**{info['title']}**")
    st.write(info['description'])
    st.write("**Use Cases:**")
    for use_case in info['use_cases']:
        st.write(f"• {use_case}")
    st.write("**Required Columns:**")
    for col in info['required_columns']:
        st.write(f"• {col}")

# Main content area
tab1, tab2, tab3 = st.tabs(["📤 Data Input", "🔧 Data Processing", "📊 Analysis Results"])

# Tab 1: Data Input
with tab1:
    st.header("📤 Data Input")
    
    data_source = st.radio(
        "Choose data source:",
        ["sample", "upload"],
        format_func=lambda x: {
            "sample": "🎯 Use Sample Dataset",
            "upload": "📁 Upload Your Data"
        }[x]
    )
    
    if data_source == "sample":
        st.subheader("🎯 Sample Datasets")
        
        try:
            response = requests.get(f"{API_BASE_URL}/sample-datasets")
            if response.status_code == 200:
                datasets = response.json()
                
                dataset_key = st.selectbox(
                    "Select a sample dataset:",
                    list(datasets.keys()),
                    format_func=lambda x: datasets[x]['name']
                )
                
                if dataset_key:
                    # Display dataset info
                    st.markdown(f"""
                    <div class="info-box">
                        <strong>{datasets[dataset_key]['name']}</strong><br>
                        📝 {datasets[dataset_key]['description']}<br>
                        📏 Shape: {datasets[dataset_key]['shape']}<br>
                        📋 Columns: {', '.join(datasets[dataset_key]['columns'])}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("🚀 Load Sample Dataset", type="primary"):
                        with st.spinner("Loading dataset..."):
                            response = requests.post(f"{API_BASE_URL}/load-sample/{dataset_key}")

                            # logging.info('response')
                            # logging.info(response)

                            if response.status_code == 200:
                                data = response.json()

                                # logging.info('data')
                                # logging.info(data)

                                st.session_state.current_session_id = data['session_id']
                                st.session_state.current_data = pd.DataFrame(data['sample_data'])
                                st.success("✅ Dataset loaded successfully!")
                                st.rerun()
            else:
                st.error("❌ Failed to fetch sample datasets")
        except Exception as e:
            st.error(f"❌ Error connecting to API: {str(e)}")
    
    else:  # upload
        st.subheader("📁 Upload Your Data")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx'],
            help="Maximum file size: 5MB. Supports CSV and Excel formats."
        )

        if uploaded_file is not None:
            if st.button("📤 Upload and Process", type="primary"):
                with st.spinner("Uploading and processing..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    response = requests.post(f"{API_BASE_URL}/upload-data", files=files)
                    # logging.info('response')
                    # logging.info(response)
                    
                    if response.status_code == 200:
                        data = response.json()
                        # logging.info('data')
                        # logging.info(data)
                        st.session_state.current_session_id = data['session_id']
                        st.session_state.current_data = pd.DataFrame(data['sample_data'])
                        st.success("✅ File uploaded successfully!")
                        st.rerun()
                    else:
                        st.error(f"❌ Upload failed: {response.text}")
    
    # Display current data
    if st.session_state.current_data is not None:
        st.subheader("🔍 Current Data Preview")
        st.dataframe(st.session_state.current_data, use_container_width=True)
        
        st.markdown(f"""
        <div class="success-box">
            📊 <strong>Data Shape:</strong> {st.session_state.current_data.shape[0]} rows × {st.session_state.current_data.shape[1]} columns<br>
            🆔 <strong>Session ID:</strong> {st.session_state.current_session_id}
        </div>
        """, unsafe_allow_html=True)

# Tab 2: Data Processing
with tab2:
    st.header("🔧 Data Processing")
    
    if st.session_state.current_data is None:
        st.warning("⚠️ Please load data first in the Data Input tab")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Column Selection")
            columns = st.session_state.current_data.columns.tolist()
            
            if module == "time_series":
                datetime_col = st.selectbox("📅 Select Date/Time Column:", columns)
                value_col = st.selectbox("📈 Select Value Column:", 
                                       [col for col in columns if col != datetime_col])
                
                # Enhanced Category column selection with better UX
                st.markdown("### 🏷️ Category Analysis (Optional)")
                enable_category = st.checkbox("Enable multi-category analysis", 
                                            help="Analyze multiple time series by category/group")
                
                category_col = None
                if enable_category:
                    category_col = st.selectbox("Select Category Column:", 
                                              [col for col in columns if col not in [datetime_col, value_col]],
                                              help="Select a column to group your time series by categories")
                    
                    # Show category preview
                    if category_col:
                        unique_categories = st.session_state.current_data[category_col].unique()
                        st.info(f"📋 Found {len(unique_categories)} categories: {', '.join(map(str, unique_categories[:5]))}" + 
                               (f" and {len(unique_categories)-5} more..." if len(unique_categories) > 5 else ""))
                
                # Check if aggregation is needed
                st.subheader("🔄 Data Aggregation")
                needs_agg = st.checkbox("Need to aggregate raw data?", 
                                      help="Check if your data has multiple records per time period")
                
                if needs_agg:
                    agg_method = st.selectbox("Aggregation Method:", 
                                            ["sum", "mean", "count", "median"])
                    freq = st.selectbox("Frequency:", 
                                      ["D", "W", "M", "Q"],
                                      format_func=lambda x: {
                                          "D": "Daily", "W": "Weekly", 
                                          "M": "Monthly", "Q": "Quarterly"
                                      }[x])
                    
                    if st.button("🔄 Aggregate Data", type="secondary"):
                        with st.spinner("Aggregating data..."):
                            payload = {
                                "session_id": st.session_state.current_session_id,
                                "datetime_col": datetime_col,
                                "value_col": value_col,
                                "agg_method": agg_method,
                                "freq": freq,
                                "category_col": category_col
                            }
                            response = requests.post(f"{API_BASE_URL}/aggregate-data", json=payload)
                            
                            if response.status_code == 200:
                                agg_data = response.json()
                                st.session_state.current_session_id = agg_data['session_id']
                                st.session_state.current_data = pd.DataFrame(agg_data['data'])
                                st.success("✅ Data aggregated successfully!")
                                if agg_data.get('has_category', False):
                                    category_col = 'category'
                                else:
                                    category_col = None
                                st.rerun()
                            else:
                                st.error(f"❌ Aggregation failed: {response.text}")
            
            elif module == "customer":
                customer_col = st.selectbox("👤 Select Customer ID Column:", columns)
                amount_col = st.selectbox("💰 Select Amount Column:", 
                                        [col for col in columns if col != customer_col])
                date_col = st.selectbox("📅 Select Date Column:", 
                                      [col for col in columns if col not in [customer_col, amount_col]])
        
        with col2:
            st.subheader("📋 Processing Summary")
            
            if module == "time_series":
                processing_info = f"""
                **Time Series Configuration:**
                - 📅 Date Column: `{datetime_col if 'datetime_col' in locals() else 'Not selected'}`
                - 📈 Value Column: `{value_col if 'value_col' in locals() else 'Not selected'}`
                - 🏷️ Category Column: `{category_col if 'category_col' in locals() and category_col else 'None (Single series)'}`
                """
                if 'needs_agg' in locals() and needs_agg:
                    processing_info += f"""
                    - 🔄 Aggregation: `{agg_method if 'agg_method' in locals() else 'Not selected'}`
                    - ⏰ Frequency: `{freq if 'freq' in locals() else 'Not selected'}`
                    """
                
            else:  # customer
                processing_info = f"""
                **Customer Analytics Configuration:**
                - 👤 Customer Column: `{customer_col if 'customer_col' in locals() else 'Not selected'}`
                - 💰 Amount Column: `{amount_col if 'amount_col' in locals() else 'Not selected'}`
                - 📅 Date Column: `{date_col if 'date_col' in locals() else 'Not selected'}`
                """
            
            st.markdown(processing_info)
            
            # Ready to analyze button
            if module == "time_series" and 'datetime_col' in locals() and 'value_col' in locals():
                analysis_ready = True
                analysis_params = {
                    'date_col': 'date' if needs_agg and 'needs_agg' in locals() else datetime_col,
                    'value_col': 'value' if needs_agg and 'needs_agg' in locals() else value_col
                }
                # Add category_col to parameters
                if category_col:
                    analysis_params['category_col'] = 'category' if needs_agg and 'needs_agg' in locals() else category_col
                    
            elif module == "customer" and 'customer_col' in locals() and 'amount_col' in locals() and 'date_col' in locals():
                analysis_ready = True
                analysis_params = {
                    'customer_col': customer_col,
                    'amount_col': amount_col,
                    'date_col': date_col
                }
            else:
                analysis_ready = False
                analysis_params = {}
            
            if analysis_ready:
                st.markdown('<div class="success-box">✅ Ready for analysis!</div>', unsafe_allow_html=True)
                
                # Model Selection for Time Series
                if module == "time_series":
                    st.markdown("### 🔮 Analysis Options")
                    
                    # TAMBAHKAN: Forecast toggle option
                    enable_forecast = st.checkbox(
                        "🔮 Enable Forecasting Analysis", 
                        value=True,
                        help="Enable this to include forecast analysis with your time series. Disable for faster analysis."
                    )
                    
                    if enable_forecast:
                        st.markdown("### 📊 Forecasting Model Selection")
                        
                        # Model categories
                        model_category = st.radio(
                            "Choose model category:",
                            ["Machine Learning", "Statistical Methods"],
                            help="Machine Learning models can capture complex patterns, while Statistical methods are more interpretable"
                        )
                        
                        if model_category == "Machine Learning":
                            model_options = {
                                'linear_regression': '📈 Linear Regression (with features)',
                                'random_forest': '🌳 Random Forest',
                                'catboost': '🚀 CatBoost (Gradient Boosting)'
                            }
                            default_model = 'random_forest'
                        else:
                            model_options = {
                                'arima': '📊 ARIMA (Auto-Regressive)',
                                'ets': '📈 ETS (Exponential Smoothing)',
                                'theta': '🎯 Theta Method',
                                'moving_average': '📉 Moving Average',
                                'naive': '➡️ Naive (Last Value)',
                                'naive_drift': '📈 Naive with Drift'
                            }
                            default_model = 'ets'
                        
                        model_type = st.selectbox(
                            "Select forecasting model:",
                            list(model_options.keys()),
                            format_func=lambda x: model_options[x],
                            index=list(model_options.keys()).index(default_model)
                        )
                        
                        # Model descriptions 
                        model_descriptions = {
                            'linear_regression': {
                                'description': 'Linear regression with time-based and seasonal features',
                                'pros': ['Fast and interpretable', 'Good for linear trends', 'Handles seasonality with feature engineering'],
                                'cons': ['Limited for complex non-linear patterns', 'Assumes linear relationships'],
                                'best_for': 'Data with clear linear trends and seasonal patterns'
                            },
                            'random_forest': {
                                'description': 'Ensemble of decision trees with time and seasonal features',
                                'pros': ['Captures complex patterns', 'Handles seasonality well', 'Feature importance available'],
                                'cons': ['Less interpretable', 'Can overfit with small datasets'],
                                'best_for': 'Complex data with non-linear patterns and strong seasonality'
                            },
                            'catboost': {
                                'description': 'Advanced gradient boosting optimized for categorical features',
                                'pros': ['Excellent for complex patterns', 'Handles seasonality automatically', 'Very accurate'],
                                'cons': ['Longer training time', 'Requires more data', 'Less interpretable'],
                                'best_for': 'Large datasets with complex seasonal and trend patterns'
                            },
                            'arima': {
                                'description': 'Auto-Regressive Integrated Moving Average (simplified implementation)',
                                'pros': ['Good for trend patterns', 'Statistically sound', 'Works with limited data'],
                                'cons': ['Limited seasonality handling', 'Assumes stationarity'],
                                'best_for': 'Data with clear trends but limited seasonal patterns'
                            },
                            'ets': {
                                'description': 'Exponential Smoothing with Trend and Seasonality',
                                'pros': ['Excellent for seasonal data', 'Adaptive to changes', 'Interpretable'],
                                'cons': ['Limited for complex patterns', 'Sensitive to outliers'],
                                'best_for': 'Data with strong seasonal patterns and moderate trends'
                            },
                            'theta': {
                                'description': 'Theta method combining trend extrapolation with seasonality',
                                'pros': ['Good balance of simplicity and accuracy', 'Handles trends well'],
                                'cons': ['Limited seasonal handling', 'Simple approach'],
                                'best_for': 'Data with strong trends but moderate seasonality'
                            },
                            'moving_average': {
                                'description': 'Weighted average of recent values with seasonal adjustment',
                                'pros': ['Simple and robust', 'Good for stable patterns', 'Handles seasonal data'],
                                'cons': ['Slow to adapt to changes', 'Limited trend handling'],
                                'best_for': 'Stable data with consistent seasonal patterns'
                            },
                            'naive': {
                                'description': 'Uses the last observed value as forecast (baseline model)',
                                'pros': ['Very simple', 'Good baseline', 'Fast computation'],
                                'cons': ['No trend or seasonality', 'Poor for changing data'],
                                'best_for': 'Baseline comparison or very stable data'
                            },
                            'naive_drift': {
                                'description': 'Naive method with linear drift based on historical average change',
                                'pros': ['Simple with trend', 'Good baseline', 'Fast computation'],
                                'cons': ['Linear assumption only', 'No seasonality'],
                                'best_for': 'Data with consistent linear trends'
                            }
                        }
                        
                        # Show model description
                        if model_type in model_descriptions:
                            desc = model_descriptions[model_type]
                            
                            with st.expander(f"ℹ️ About {model_options[model_type]}", expanded=False):
                                st.write(f"**Description:** {desc['description']}")
                                
                                col_pros, col_cons = st.columns(2)
                                with col_pros:
                                    st.write("**✅ Pros:**")
                                    for pro in desc['pros']:
                                        st.write(f"• {pro}")
                                
                                with col_cons:
                                    st.write("**⚠️ Cons:**")
                                    for con in desc['cons']:
                                        st.write(f"• {con}")
                                
                                st.info(f"**🎯 Best for:** {desc['best_for']}")
                        
                        # Add model type to parameters
                        analysis_params['model_type'] = model_type
                    else:
                        # Set default values when forecast is disabled
                        analysis_params['model_type'] = None
                        model_options = {}
                    
                    # Add forecast flag to parameters
                    analysis_params['enable_forecast'] = enable_forecast

                if enable_forecast:
                    # Quick Model Comparison Feature
                    st.markdown("---")
                    st.markdown("### ⚡ Quick Model Comparison")
                    
                    compare_models = st.checkbox(
                        "🔄 Compare multiple models", 
                        value=False,
                        help="Run analysis with multiple models and compare their performance (takes longer)"
                    )
                    
                    if compare_models:
                        st.info("🔄 **Model Comparison Mode**: This will test multiple models and show you a comparison table.")
                        
                        if model_category == "Machine Learning":
                            comparison_models = ['linear_regression', 'random_forest', 'catboost']
                        else:
                            comparison_models = ['ets', 'arima', 'theta', 'moving_average', 'naive_drift']
                        
                        # Show which models will be compared
                        model_names = [model_options[m] for m in comparison_models]
                        st.write(f"**Models to compare:** {', '.join(model_names)}")
                        
                        # Update analysis params for comparison
                        analysis_params['compare_models'] = True
                        analysis_params['comparison_models'] = comparison_models
                    else:
                        analysis_params['compare_models'] = False
                
               # Enhanced analysis button with model info
                if enable_forecast and analysis_params.get('compare_models', False):
                    button_text = "🚀 Run Analysis & Compare Models"
                elif enable_forecast:
                    button_text = f"🚀 Run Analysis with Forecast ({model_options[model_type]})"
                else:
                    button_text = "🚀 Run Analysis (No Forecast)"


                if module == "time_series":
                    if enable_forecast:
                        button_text += f" ({model_options[model_type]})"
                
                if st.button(button_text, type="primary", use_container_width=True):
                    if analysis_params.get('compare_models', False):
                        # Model comparison mode
                        with st.spinner("🔄 Comparing multiple models... This will take a few moments."):
                            payload = {
                                "session_id": st.session_state.current_session_id,
                                "module": module,
                                "parameters": analysis_params
                            }
                            
                            # First run model comparison
                            comparison_response = requests.post(f"{API_BASE_URL}/compare-models", json=payload)
                            
                            if comparison_response.status_code == 200:
                                comparison_results = comparison_response.json()
                                st.session_state.model_comparison = comparison_results
                                
                                # Show comparison results immediately
                                st.success("✅ Model comparison completed!")
                                
                                # Display comparison table
                                st.subheader("📊 Model Performance Comparison")
                                
                                comparison_table = comparison_results['comparison_table']
                                successful_models = [m for m in comparison_table if m['status'] == 'success']
                                
                                if successful_models:
                                    # Create DataFrame for display
                                    import pandas as pd
                                    df_comparison = pd.DataFrame(successful_models)
                                    
                                    # Format the dataframe for better display
                                    display_df = df_comparison[['rank', 'model', 'mae', 'rmse', 'r2', 'mape', 'directional_accuracy']].copy()
                                    
                                    # Round numeric columns
                                    numeric_cols = ['mae', 'rmse', 'r2', 'mape', 'directional_accuracy']
                                    for col in numeric_cols:
                                        if col in display_df.columns:
                                            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if x is not None else "N/A")
                                    
                                    # Rename columns for display
                                    display_df.columns = ['Rank', 'Model', 'MAE', 'RMSE', 'R²', 'MAPE (%)', 'Direction Acc (%)']
                                    
                                    st.dataframe(display_df, use_container_width=True)
                                    
                                    # Highlight best model
                                    best_model = comparison_results['best_model']
                                    st.markdown(f"""
                                    <div class="success-box">
                                        🏆 <strong>Best Model:</strong> {best_model['model_name']}<br>
                                        📊 <strong>MAE:</strong> {best_model['mae']:.3f} | 
                                        <strong>RMSE:</strong> {best_model['rmse']:.3f}
                                        {f" | <strong>R²:</strong> {best_model['r2']:.3f}" if best_model.get('r2') is not None else ""}
                                        {f" | <strong>MAPE:</strong> {best_model['mape']:.1f}%" if best_model.get('mape') is not None else ""}
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Show failed models if any
                                    failed_models = [m for m in comparison_table if m['status'] == 'failed']
                                    if failed_models:
                                        st.warning(f"⚠️ {len(failed_models)} model(s) failed to run:")
                                        for failed in failed_models:
                                            st.error(f"❌ {failed['model']}: {failed.get('error', 'Unknown error')}")
                                    
                                    # Option to run detailed analysis with best model
                                    st.markdown("---")
                                    col_auto, col_manual = st.columns(2)
                                    
                                    with col_auto:
                                        if st.button(f"🚀 Run Full Analysis with Best Model ({best_model['model_name']})", type="primary"):
                                            # Update parameters with best model
                                            analysis_params['model_type'] = best_model['model_key']
                                            analysis_params['compare_models'] = False
                                            
                                            payload_full = {
                                                "session_id": st.session_state.current_session_id,
                                                "module": module,
                                                "parameters": analysis_params
                                            }
                                            
                                            with st.spinner("🔄 Running full analysis with best model..."):
                                                response = requests.post(f"{API_BASE_URL}/analyze", json=payload_full)
                                                
                                                if response.status_code == 200:
                                                    st.session_state.analysis_results = response.json()
                                                    st.success(f"✅ Full analysis completed with {best_model['model_name']}!")
                                                    st.balloons()
                                                    st.info("📊 **Analysis complete!** Check the **Analysis Results** tab.")
                                                    time.sleep(1)
                                                    st.rerun()
                                                else:
                                                    st.error(f"❌ Full analysis failed: {response.text}")
                                    
                                    with col_manual:
                                        # Manual model selection
                                        manual_model = st.selectbox(
                                            "Or choose a specific model:",
                                            options=[m['model_key'] for m in successful_models],
                                            format_func=lambda x: next(m['model'] for m in successful_models if m['model_key'] == x),
                                            key="manual_model_select"
                                        )
                                        
                                        if st.button(f"🎯 Run with Selected Model", type="secondary"):
                                            analysis_params['model_type'] = manual_model
                                            analysis_params['compare_models'] = False
                                            
                                            payload_manual = {
                                                "session_id": st.session_state.current_session_id,
                                                "module": module,
                                                "parameters": analysis_params
                                            }
                                            
                                            with st.spinner("🔄 Running analysis with selected model..."):
                                                response = requests.post(f"{API_BASE_URL}/analyze", json=payload_manual)
                                                
                                                if response.status_code == 200:
                                                    st.session_state.analysis_results = response.json()
                                                    selected_model_name = next(m['model'] for m in successful_models if m['model_key'] == manual_model)
                                                    st.success(f"✅ Analysis completed with {selected_model_name}!")
                                                    st.balloons()
                                                    st.info("📊 **Analysis complete!** Check the **Analysis Results** tab.")
                                                    time.sleep(1)
                                                    st.rerun()
                                                else:
                                                    st.error(f"❌ Analysis failed: {response.text}")
                                
                                else:
                                    st.error("❌ All models failed to run. Please check your data quality.")
                            else:
                                st.error(f"❌ Model comparison failed: {comparison_response.text}")
                    
                    else:
                        # Single model mode (original logic)
                        with st.spinner("🔄 Running comprehensive analysis... This may take a moment."):
                            payload = {
                                "session_id": st.session_state.current_session_id,
                                "module": module,
                                "parameters": analysis_params
                            }
                            response = requests.post(f"{API_BASE_URL}/analyze", json=payload)
                            logging.info(payload)

                            logging.info('response analysis')
                            logging.info(response)
                            
                            if response.status_code == 200:
                                st.session_state.analysis_results = response.json()
                                
                                # Enhanced success message with model info
                                success_msg = "✅ Analysis completed successfully!"
                                if module == "time_series":
                                    success_msg += f" Using {model_options[model_type]} for forecasting."
                                
                                st.success(success_msg)
                                st.balloons()  # Add celebration effect
                                
                                # Auto-switch to results tab message
                                st.info("📊 **Analysis complete!** Please check the **Analysis Results** tab to view your insights, charts, and forecasts.")
                                
                                # Add a small delay and rerun to refresh the UI
                                import time
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"❌ Analysis failed: {response.text}")
            
            else:
                st.warning("⚠️ Please configure all required columns")

# Tab 3: Analysis Results - Enhanced with filters and decomposition
with tab3:
    st.header("📊 Analysis Results")
    
    if st.session_state.analysis_results is None:
        st.info("🔍 Run analysis in the Data Processing tab to see results here")
    else:
        results = st.session_state.analysis_results
        
        # Enhanced insights section with seasonality details
        st.subheader("💡 Key Insights", help="AI-generated insights based on your data analysis")
        
        insight_container = st.container()
        with insight_container:
            for i, insight in enumerate(results['insights']):
                st.markdown(f"""
                <div class="info-box">
                    <strong>📈 Insight {i+1}:</strong> {insight}
                </div>
                """, unsafe_allow_html=True)
        
        # Add detailed seasonality insights if available
        if results['module'] == 'time_series' and results.get('seasonality_insights'):
            with st.expander("🌊 Detailed Seasonality Analysis", expanded=False):
                seasonality_insights = results['seasonality_insights']
                for insight in seasonality_insights:
                    st.markdown(f"• {insight}")
        
        # Enhanced visualization section for time series
        if results['module'] == 'time_series':
            st.subheader("📈 Interactive Visualization")
            
            # Category dropdown filter (if categories exist)
            if results.get('has_categories', False):
                st.markdown("""
                <div class="filter-container">
                    <h4 style="margin-top: 0; color: #155724;">🏷️ Category Filters</h4>
                </div>
                """, unsafe_allow_html=True)
                
                categories_list = results.get('categories_list', [])
                
                col_filter1, col_filter2 = st.columns([3, 1])
                with col_filter1:
                    # Enhanced dropdown multiselect for categories
                    selected_categories = st.multiselect(
                        "Select categories to display:",
                        options=categories_list,
                        default=categories_list,
                        help="Choose which categories to show in the chart. You can select multiple categories."
                    )
                
                with col_filter2:
                    # Quick selection buttons
                    if st.button("Select All", use_container_width=True):
                        selected_categories = categories_list
                        st.rerun()
                    
                    if st.button("Clear All", use_container_width=True):
                        selected_categories = []
                        st.rerun()
            else:
                selected_categories = []
            
            # Enhanced chart options
            st.markdown("""
            <div class="decomposition-container">
                <h4 style="margin-top: 0; color: #e65100;">📊 Chart Options & Analysis Tools</h4>
            </div>
            """, unsafe_allow_html=True)
            
            col_options1, col_options2, col_options3 = st.columns(3)
            
            with col_options1:
                show_decomposition = st.checkbox(
                    "📈 Show Decomposition Charts", 
                    value=False,
                    help="Display separate charts for trend and seasonal components"
                )
                
                show_trend_overlay = st.checkbox(
                    "📉 Show Trend Line", 
                    value=False,
                    help="Add trend line overlay to main chart"
                )
            
            with col_options2:
                show_seasonal_overlay = st.checkbox(
                    "🌊 Show Seasonal Pattern", 
                    value=False,
                    help="Add seasonal pattern overlay to main chart"
                )
                
                # Enhanced forecasting options - HANYA jika ada forecast data
                if results.get('overall_forecast') or results.get('category_forecasts'):
                    show_forecast = st.checkbox(
                        "🔮 Show Forecast", 
                        value=False,
                        help="Display forecast for the last 20% of data with prediction intervals"
                    )
                else:
                    show_forecast = False
                    st.markdown("*🔮 Forecast: Not available (disabled in analysis)*")
            
            with col_options3:
                if show_forecast and (results.get('overall_forecast') or results.get('category_forecasts')):
                    show_confidence_intervals = st.checkbox(
                        "📊 Show Confidence Intervals", 
                        value=True,
                        help="Display 50% and 80% confidence intervals for forecasts"
                    )
                    
                    forecast_transparency = st.slider(
                        "Forecast Transparency",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.7,
                        step=0.1,
                        help="Adjust transparency of forecast visualization"
                    )
                else:
                    show_confidence_intervals = False
                    forecast_transparency = 0.7
            
            # Main chart with enhanced features
            if 'plot' in results:
                fig_dict = json.loads(results['plot'])
                fig = go.Figure(fig_dict)
                
                # Apply category filter
                if results.get('has_categories', False) and selected_categories:
                    filtered_data = []
                    for trace in fig.data:
                        if hasattr(trace, 'name') and trace.name in selected_categories:
                            filtered_data.append(trace)
                        elif not hasattr(trace, 'name'):
                            filtered_data.append(trace)
                    fig.data = filtered_data
                
                # Add trend overlay
                if show_trend_overlay:
                    decomp_data = results.get('overall_decomposition') or results.get('category_decompositions', {})
                    
                    if results.get('overall_decomposition'):
                        fig.add_trace(go.Scatter(
                            x=pd.to_datetime(decomp_data['dates']),
                            y=decomp_data['trend'],
                            mode='lines',
                            name='Trend',
                            line=dict(color='rgba(128,128,128,0.8)', width=2, dash='dash'),
                            hovertemplate='Trend: %{y:.2f}<extra></extra>'
                        ))
                    elif selected_categories and results.get('category_decompositions'):
                        for cat in selected_categories:
                            if cat in results['category_decompositions']:
                                cat_decomp = results['category_decompositions'][cat]
                                fig.add_trace(go.Scatter(
                                    x=pd.to_datetime(cat_decomp['dates']),
                                    y=cat_decomp['trend'],
                                    mode='lines',
                                    name=f'{cat} - Trend',
                                    line=dict(color='rgba(128,128,128,0.6)', width=1, dash='dash'),
                                    hovertemplate=f'{cat} Trend: %{{y:.2f}}<extra></extra>'
                                ))
                
                # Add seasonal overlay
                if show_seasonal_overlay:
                    decomp_data = results.get('overall_decomposition') or results.get('category_decompositions', {})
                    
                    if results.get('overall_decomposition'):
                        seasonal_offset = np.array(decomp_data['seasonal']) + np.mean(decomp_data['original'])
                        fig.add_trace(go.Scatter(
                            x=pd.to_datetime(decomp_data['dates']),
                            y=seasonal_offset,
                            mode='lines',
                            name='Seasonal Pattern',
                            line=dict(color='rgba(169,169,169,0.7)', width=1, dash='dot'),
                            hovertemplate='Seasonal: %{y:.2f}<extra></extra>'
                        ))
                    elif selected_categories and results.get('category_decompositions'):
                        for cat in selected_categories:
                            if cat in results['category_decompositions']:
                                cat_decomp = results['category_decompositions'][cat]
                                seasonal_offset = np.array(cat_decomp['seasonal']) + np.mean(cat_decomp['original'])
                                fig.add_trace(go.Scatter(
                                    x=pd.to_datetime(cat_decomp['dates']),
                                    y=seasonal_offset,
                                    mode='lines',
                                    name=f'{cat} - Seasonal',
                                    line=dict(color='rgba(169,169,169,0.5)', width=1, dash='dot'),
                                    hovertemplate=f'{cat} Seasonal: %{{y:.2f}}<extra></extra>'
                                ))
                
                # Enhanced forecasting visualization
                if show_forecast:
                    forecast_data = results.get('overall_forecast') or results.get('category_forecasts', {})
                    
                    if results.get('overall_forecast'):
                        # Single series forecast with enhanced metrics display
                        forecast = forecast_data
                        
                        # Model information header
                        st.markdown(f"""
                        <div class="info-box">
                            <h4 style="margin-top: 0;">🔮 Forecast Results - {forecast.get('model_name', 'Unknown Model')}</h4>
                            <p><strong>Model Type:</strong> {forecast.get('model_type', 'unknown').title().replace('_', ' ')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # FIXED: Add vertical line to separate train/test - ensure proper date conversion
                        try:
                            split_date = pd.to_datetime(forecast['split_date'])
                            fig.add_vline(
                                x=split_date,
                                line_dash="solid",
                                line_color="rgba(255,0,0,0.8)",
                                line_width=3,
                                annotation_text="📊 Train/Test Split",
                                annotation_position="top",
                                annotation=dict(
                                    bgcolor="rgba(255,255,255,0.8)",
                                    bordercolor="rgba(255,0,0,0.8)",
                                    borderwidth=1
                                )
                            )
                        except Exception as e:
                            st.warning(f"Could not add split line: {e}")
                        
                        # FIXED: Add forecast line with proper data conversion
                        try:
                            test_dates = pd.to_datetime(forecast['test_dates'])
                            test_predicted = forecast['test_predicted']
                            
                            fig.add_trace(go.Scatter(
                                x=test_dates,
                                y=test_predicted,
                                mode='lines+markers',
                                name='🔮 Forecast',
                                line=dict(color='rgba(255,165,0,{})'.format(forecast_transparency), width=4, dash='dash'),
                                marker=dict(size=6, color='orange'),
                                hovertemplate='Forecast: %{y:.2f}<br>Date: %{x}<extra></extra>'
                            ))
                            
                            # FIXED: Add actual test values for comparison
                            test_actual = forecast['test_actual']
                            fig.add_trace(go.Scatter(
                                x=test_dates,
                                y=test_actual,
                                mode='lines+markers',
                                name='📈 Actual (Test)',
                                line=dict(color='rgba(0,128,0,0.8)', width=3),
                                marker=dict(size=6, color='green'),
                                hovertemplate='Actual: %{y:.2f}<br>Date: %{x}<extra></extra>'
                            ))
                            
                        except Exception as e:
                            st.error(f"Error adding forecast traces: {e}")
                        
                        # FIXED: Add confidence intervals with proper error handling
                        if show_confidence_intervals:
                            try:
                                # 95% CI (outermost)
                                fig.add_trace(go.Scatter(
                                    x=test_dates,
                                    y=forecast['ci_95_upper'],
                                    mode='lines',
                                    line=dict(width=0),
                                    showlegend=False,
                                    hoverinfo='skip'
                                ))
                                fig.add_trace(go.Scatter(
                                    x=test_dates,
                                    y=forecast['ci_95_lower'],
                                    mode='lines',
                                    line=dict(width=0),
                                    fill='tonexty',
                                    fillcolor=f'rgba(255,165,0,{forecast_transparency*0.1})',
                                    name='95% Confidence',
                                    hovertemplate='95% CI: %{y:.2f}<extra></extra>'
                                ))
                                
                                # 80% CI
                                fig.add_trace(go.Scatter(
                                    x=test_dates,
                                    y=forecast['ci_80_upper'],
                                    mode='lines',
                                    line=dict(width=0),
                                    showlegend=False,
                                    hoverinfo='skip'
                                ))
                                fig.add_trace(go.Scatter(
                                    x=test_dates,
                                    y=forecast['ci_80_lower'],
                                    mode='lines',
                                    line=dict(width=0),
                                    fill='tonexty',
                                    fillcolor=f'rgba(255,165,0,{forecast_transparency*0.2})',
                                    name='80% Confidence',
                                    hovertemplate='80% CI: %{y:.2f}<extra></extra>'
                                ))
                                
                                # 50% CI (innermost)
                                fig.add_trace(go.Scatter(
                                    x=test_dates,
                                    y=forecast['ci_50_upper'],
                                    mode='lines',
                                    line=dict(width=0),
                                    showlegend=False,
                                    hoverinfo='skip'
                                ))
                                fig.add_trace(go.Scatter(
                                    x=test_dates,
                                    y=forecast['ci_50_lower'],
                                    mode='lines',
                                    line=dict(width=0),
                                    fill='tonexty',
                                    fillcolor=f'rgba(255,165,0,{forecast_transparency*0.4})',
                                    name='50% Confidence',
                                    hovertemplate='50% CI: %{y:.2f}<extra></extra>'
                                ))
                                
                            except Exception as e:
                                st.warning(f"Could not add confidence intervals: {e}")
                        
                        # FIXED: Enhanced forecast accuracy metrics display
                        st.subheader("📊 Forecast Performance Metrics")
                        
                        metrics = forecast['metrics']
                        
                        # Main metrics row
                        col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                        
                        with col_metric1:
                            mae_value = metrics['mae']
                            st.metric(
                                "🎯 MAE", 
                                f"{mae_value:.2f}", 
                                help="Mean Absolute Error - average prediction error"
                            )
                        
                        with col_metric2:
                            rmse_value = metrics['rmse']
                            st.metric(
                                "📊 RMSE", 
                                f"{rmse_value:.2f}", 
                                help="Root Mean Square Error - penalizes larger errors more"
                            )
                        
                        with col_metric3:
                            if metrics.get('mape'):
                                mape_value = metrics['mape']
                                # Color coding for MAPE
                                if mape_value < 10:
                                    mape_color = "🟢"
                                elif mape_value < 20:
                                    mape_color = "🟡"
                                else:
                                    mape_color = "🔴"
                                
                                st.metric(
                                    f"{mape_color} MAPE", 
                                    f"{mape_value:.1f}%", 
                                    help="Mean Absolute Percentage Error"
                                )
                            else:
                                st.metric("📈 MAPE", "N/A", help="Not available for data with zero values")
                        
                        with col_metric4:
                            if metrics.get('r2') is not None:
                                r2_value = metrics['r2']
                                # Color coding for R²
                                if r2_value > 0.8:
                                    r2_color = "🟢"
                                elif r2_value > 0.6:
                                    r2_color = "🟡"
                                elif r2_value > 0:
                                    r2_color = "🟠"
                                else:
                                    r2_color = "🔴"
                                
                                st.metric(
                                    f"{r2_color} R²", 
                                    f"{r2_value:.3f}", 
                                    help="R-squared - explained variance (higher is better)"
                                )
                            else:
                                st.metric("📈 R²", "N/A", help="R-squared not available")
                        
                        # Additional metrics row
                        if metrics.get('directional_accuracy') or metrics.get('mse'):
                            col_metric5, col_metric6, col_metric7, col_metric8 = st.columns(4)
                            
                            with col_metric5:
                                if metrics.get('mse'):
                                    st.metric(
                                        "📏 MSE", 
                                        f"{metrics['mse']:.2f}", 
                                        help="Mean Squared Error"
                                    )
                            
                            with col_metric6:
                                if metrics.get('directional_accuracy'):
                                    dir_acc = metrics['directional_accuracy']
                                    dir_color = "🟢" if dir_acc > 70 else "🟡" if dir_acc > 50 else "🔴"
                                    st.metric(
                                        f"{dir_color} Direction Accuracy", 
                                        f"{dir_acc:.1f}%", 
                                        help="Percentage of correct trend direction predictions"
                                    )
                            
                            with col_metric7:
                                # Model-specific metrics
                                if metrics.get('drift'):
                                    st.metric(
                                        "📈 Drift", 
                                        f"{metrics['drift']:.4f}", 
                                        help="Average change per period (for drift models)"
                                    )
                            
                            with col_metric8:
                                # Forecast horizon
                                st.metric(
                                    "⏱️ Test Periods", 
                                    f"{len(test_dates)}", 
                                    help="Number of periods used for testing"
                                )
                        
                        # Feature importance for ML models
                        if forecast.get('feature_importance'):
                            st.subheader("🔍 Feature Importance")
                            
                            importance = forecast['feature_importance']
                            
                            # Create horizontal bar chart for feature importance
                            features = list(importance.keys())[:10]  # Top 10 features
                            values = [importance[f] for f in features]
                            
                            fig_importance = go.Figure(go.Bar(
                                x=values,
                                y=features,
                                orientation='h',
                                marker_color='rgba(55, 128, 191, 0.7)',
                                hovertemplate='%{y}: %{x:.3f}<extra></extra>'
                            ))
                            
                            fig_importance.update_layout(
                                title="Top 10 Most Important Features",
                                xaxis_title="Importance",
                                yaxis_title="Features",
                                height=400,
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#2E4057')
                            )
                            
                            st.plotly_chart(fig_importance, use_container_width=True)
                            
                            # Feature explanation
                            with st.expander("📚 Feature Explanations", expanded=False):
                                st.markdown("""
                                **Feature Types:**
                                - `days_numeric`: Linear time trend
                                - `month_sin/cos`: Monthly seasonality (cyclical)
                                - `dow_sin/cos`: Day-of-week seasonality (cyclical)
                                - `doy_sin/cos`: Day-of-year seasonality (cyclical)
                                - `lag_X`: Values from X periods ago
                                - `rolling_X`: Moving average over X periods
                                - `month/quarter`: Direct seasonal indicators
                                """)
                        
                        # Model comparison suggestion
                        st.markdown("---")
                        st.subheader("💡 Model Performance Insights")
                        
                        # Performance assessment
                        performance_insights = []
                        
                        if metrics.get('mape'):
                            mape = metrics['mape']
                            if mape < 10:
                                performance_insights.append("🟢 **Excellent MAPE** (<10%) - Very accurate forecasts")
                            elif mape < 20:
                                performance_insights.append("🟡 **Good MAPE** (10-20%) - Reasonably accurate forecasts")
                            else:
                                performance_insights.append("🔴 **High MAPE** (>20%) - Consider trying different models")
                        
                        if metrics.get('r2') is not None:
                            r2 = metrics['r2']
                            if r2 > 0.8:
                                performance_insights.append("🟢 **High R²** (>0.8) - Model explains data very well")
                            elif r2 > 0.6:
                                performance_insights.append("🟡 **Moderate R²** (0.6-0.8) - Good model fit")
                            elif r2 > 0:
                                performance_insights.append("🟠 **Low R²** (0-0.6) - Model has limited explanatory power")
                            else:
                                performance_insights.append("🔴 **Negative R²** - Model performs worse than simple average")
                        
                        if metrics.get('directional_accuracy'):
                            dir_acc = metrics['directional_accuracy']
                            if dir_acc > 70:
                                performance_insights.append("🟢 **High Directional Accuracy** (>70%) - Good trend prediction")
                            elif dir_acc > 50:
                                performance_insights.append("🟡 **Moderate Directional Accuracy** (50-70%) - Some trend prediction ability")
                            else:
                                performance_insights.append("🔴 **Low Directional Accuracy** (<50%) - Poor trend prediction")
                        
                        # Display insights
                        for insight in performance_insights:
                            st.markdown(insight)
                        
                        # Model recommendations
                        current_model = forecast.get('model_type', 'unknown')
                        st.markdown("**🚀 Recommendations:**")
                        
                        # Dynamic model recommendations based on performance
                        if show_forecast and results.get('overall_forecast'):
                            forecast = results['overall_forecast']
                            metrics = forecast['metrics']
                            
                            st.markdown("---")
                            st.subheader("💡 Performance-Based Insights")
                            
                            # Performance assessment berdasarkan metrics aktual
                            performance_insights = []
                            
                            # MAE-based assessment
                            mae = metrics['mae']
                            data_mean = results['statistics']['mean']
                            mae_ratio = mae / abs(data_mean) if abs(data_mean) > 0 else float('inf')
                            
                            if mae_ratio < 0.05:
                                performance_insights.append("🟢 **Excellent MAE** (<5% of data mean) - Very accurate forecasts")
                            elif mae_ratio < 0.15:
                                performance_insights.append("🟡 **Good MAE** (5-15% of data mean) - Reasonably accurate forecasts")
                            else:
                                performance_insights.append("🔴 **High MAE** (>15% of data mean) - Consider trying different models")
                            
                            # MAPE-based assessment
                            if metrics.get('mape'):
                                mape = metrics['mape']
                                if mape < 10:
                                    performance_insights.append("🟢 **Excellent MAPE** (<10%) - Very accurate forecasts")
                                elif mape < 20:
                                    performance_insights.append("🟡 **Good MAPE** (10-20%) - Reasonably accurate forecasts")
                                else:
                                    performance_insights.append("🔴 **High MAPE** (>20%) - Consider trying different models")
                            
                            # R²-based assessment
                            if metrics.get('r2') is not None:
                                r2 = metrics['r2']
                                if r2 > 0.8:
                                    performance_insights.append("🟢 **High R²** (>0.8) - Model explains data very well")
                                elif r2 > 0.6:
                                    performance_insights.append("🟡 **Moderate R²** (0.6-0.8) - Good model fit")
                                elif r2 > 0:
                                    performance_insights.append("🟠 **Low R²** (0-0.6) - Model has limited explanatory power")
                                else:
                                    performance_insights.append("🔴 **Negative R²** - Model performs worse than simple average")
                            
                            # Directional accuracy assessment
                            if metrics.get('directional_accuracy'):
                                dir_acc = metrics['directional_accuracy']
                                if dir_acc > 70:
                                    performance_insights.append("🟢 **High Directional Accuracy** (>70%) - Good trend prediction")
                                elif dir_acc > 50:
                                    performance_insights.append("🟡 **Moderate Directional Accuracy** (50-70%) - Some trend prediction ability")
                                else:
                                    performance_insights.append("🔴 **Low Directional Accuracy** (<50%) - Poor trend prediction")
                            
                            # Display insights
                            for insight in performance_insights:
                                st.markdown(insight)
                            
                            # Dynamic model recommendations based on actual performance
                            current_model = forecast.get('model_type', 'unknown')
                            current_model_name = forecast.get('model_name', 'Current Model')
                            
                            st.markdown("**🚀 Performance-Based Recommendations:**")
                            
                            # Recommendation logic based on actual metrics
                            recommendations = []
                            
                            # Poor performance indicators
                            poor_performance = (
                                mae_ratio > 0.15 or 
                                (metrics.get('mape', 0) > 20) or 
                                (metrics.get('r2', 0) < 0.3) or
                                (metrics.get('directional_accuracy', 0) < 50)
                            )
                            
                            # Good performance indicators  
                            good_performance = (
                                mae_ratio < 0.1 and 
                                (metrics.get('mape', 100) < 15) and 
                                (metrics.get('r2', 0) > 0.6) and
                                (metrics.get('directional_accuracy', 0) > 60)
                            )
                            
                            if good_performance:
                                recommendations.append(f"✅ **{current_model_name}** shows excellent performance for your data")
                                recommendations.append("🎯 Consider using this model for production forecasting")
                                
                                # Suggest model comparison only if not already done
                                if not st.session_state.get('model_comparison'):
                                    recommendations.append("🔄 Optional: Run model comparison to confirm this is the best choice")
                            
                            elif poor_performance:
                                recommendations.append(f"⚠️ **{current_model_name}** shows suboptimal performance")
                                
                                # Specific recommendations based on model type and data characteristics
                                data_volatility = results['statistics'].get('coefficient_variation', 0)
                                has_strong_trend = abs(results['statistics'].get('trend_slope', 0)) > 0.01
                                has_seasonality = results.get('seasonality_strength', 0) > 0.2
                                
                                if current_model in ['naive', 'naive_drift']:
                                    if has_seasonality:
                                        recommendations.append("🌊 Try **ETS** or **Moving Average** for better seasonal handling")
                                    elif has_strong_trend:
                                        recommendations.append("📈 Try **Random Forest** or **Linear Regression** for trend capture")
                                    else:
                                        recommendations.append("🎯 Try **ETS** or **ARIMA** for better pattern recognition")
                                
                                elif current_model == 'linear_regression':
                                    if data_volatility > 0.3:
                                        recommendations.append("🌳 Try **Random Forest** or **CatBoost** for non-linear patterns")
                                    elif has_seasonality:
                                        recommendations.append("🌊 Try **ETS** for better seasonal modeling")
                                    else:
                                        recommendations.append("📊 Try **ARIMA** or **Theta** methods")
                                
                                elif current_model in ['random_forest', 'catboost']:
                                    if metrics.get('r2', 0) < 0.2:
                                        recommendations.append("📊 Try **ETS** or **ARIMA** - simpler models might work better")
                                        recommendations.append("🔍 Check data quality - complex models struggling suggests data issues")
                                    else:
                                        recommendations.append("⚙️ Try tuning model parameters or different ML approaches")
                                
                                elif current_model in ['ets', 'arima']:
                                    if data_volatility > 0.5:
                                        recommendations.append("🌳 Try **Random Forest** for handling high volatility")
                                    elif not has_seasonality:
                                        recommendations.append("📈 Try **Linear Regression** or **Theta** method")
                                    else:
                                        recommendations.append("🎯 Try **CatBoost** for complex seasonal patterns")
                                
                                else:
                                    recommendations.append("🔄 **Run model comparison** to find the best performing model")
                            
                            else:
                                # Moderate performance
                                recommendations.append(f"🟡 **{current_model_name}** shows moderate performance")
                                recommendations.append("🔄 **Run model comparison** to potentially find better alternatives")
                                
                                # Suggest specific alternatives based on data characteristics
                                seasonality_strength = results.get('seasonality_strength', 0)
                                if seasonality_strength > 0.3:
                                    recommendations.append("🌊 Strong seasonality detected - try **ETS** if not already used")
                                
                                trend_slope = abs(results['statistics'].get('trend_slope', 0))
                                if trend_slope > 0.01:
                                    recommendations.append("📈 Strong trend detected - try **Random Forest** or **CatBoost**")
                            
                            # Display recommendations
                            for rec in recommendations:
                                if rec.startswith("✅"):
                                    st.success(rec)
                                elif rec.startswith("⚠️") or rec.startswith("🔴"):
                                    st.warning(rec)  
                                elif rec.startswith("🔄"):
                                    st.info(rec)
                                else:
                                    st.info(rec)

                if show_forecast and st.checkbox("🔧 Show Forecast Debug Info", value=False):
                    if results.get('overall_forecast'):
                        st.json(results['overall_forecast'])
                    else:
                        st.warning("No forecast data available")
                        
                    # Check if forecast is in results
                    st.write("Available keys in results:", list(results.keys()))
                    
                    # Show forecast data structure
                    if 'overall_forecast' in results:
                        st.write("Forecast data keys:", list(results['overall_forecast'].keys()))

                # Enhanced styling
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#2E4057', size=12),
                    title_font_size=18,
                    title_font_color='#2E4057',
                    xaxis=dict(
                        gridcolor='rgba(128,128,128,0.2)',
                        title_font_color='#495057',
                        tickfont_color='#495057'
                    ),
                    yaxis=dict(
                        gridcolor='rgba(128,128,128,0.2)',
                        title_font_color='#495057',
                        tickfont_color='#495057'
                    ),
                    legend=dict(
                        bgcolor='rgba(255,255,255,0.9)',
                        bordercolor='rgba(128,128,128,0.5)',
                        borderwidth=1,
                        font=dict(size=11)
                    ),
                    height=600,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Decomposition charts
            if show_decomposition:
                st.subheader("🔍 Time Series Decomposition")
                
                decomp_data = results.get('overall_decomposition')
                category_decomps = results.get('category_decompositions', {})
                
                if decomp_data or category_decomps:
                    if results.get('has_categories', False) and selected_categories:
                        # Multi-category decomposition
                        for cat in selected_categories:
                            if cat in category_decomps:
                                st.markdown(f"**📊 Decomposition for {cat}:**")
                                
                                cat_decomp = category_decomps[cat]
                                dates = pd.to_datetime(cat_decomp['dates'])
                                
                                # Create enhanced subplots
                                from plotly.subplots import make_subplots
                                
                                fig_decomp = make_subplots(
                                    rows=3, cols=1,
                                    subplot_titles=('📈 Original Data', '📉 Trend Component', '🌊 Seasonal Component'),
                                    vertical_spacing=0.08,
                                    row_heights=[0.4, 0.3, 0.3]
                                )
                                
                                # Original data
                                fig_decomp.add_trace(
                                    go.Scatter(x=dates, y=cat_decomp['original'], 
                                             name='Original', line=dict(color='#3B82F6', width=2),
                                             hovertemplate='Original: %{y:.2f}<extra></extra>'),
                                    row=1, col=1
                                )
                                
                                # Trend
                                fig_decomp.add_trace(
                                    go.Scatter(x=dates, y=cat_decomp['trend'], 
                                             name='Trend', line=dict(color='#EF4444', width=2),
                                             hovertemplate='Trend: %{y:.2f}<extra></extra>'),
                                    row=2, col=1
                                )
                                
                                # Seasonal
                                fig_decomp.add_trace(
                                    go.Scatter(x=dates, y=cat_decomp['seasonal'], 
                                             name='Seasonal', line=dict(color='#10B981', width=2),
                                             hovertemplate='Seasonal: %{y:.2f}<extra></extra>'),
                                    row=3, col=1
                                )
                                
                                fig_decomp.update_layout(
                                    height=650,
                                    showlegend=False,
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='#2E4057'),
                                    title_font_color='#2E4057'
                                )
                                
                                fig_decomp.update_xaxes(gridcolor='rgba(128,128,128,0.2)')
                                fig_decomp.update_yaxes(gridcolor='rgba(128,128,128,0.2)')
                                
                                st.plotly_chart(fig_decomp, use_container_width=True)
                                st.markdown("---")
                    
                    elif decomp_data:
                        # Single series decomposition
                        dates = pd.to_datetime(decomp_data['dates'])
                        
                        from plotly.subplots import make_subplots
                        
                        fig_decomp = make_subplots(
                            rows=3, cols=1,
                            subplot_titles=('📈 Original Data', '📉 Trend Component', '🌊 Seasonal Component'),
                            vertical_spacing=0.08,
                            row_heights=[0.4, 0.3, 0.3]
                        )
                        
                        # Original data
                        fig_decomp.add_trace(
                            go.Scatter(x=dates, y=decomp_data['original'], 
                                     name='Original', line=dict(color='#3B82F6', width=2),
                                     hovertemplate='Original: %{y:.2f}<extra></extra>'),
                            row=1, col=1
                        )
                        
                        # Trend
                        fig_decomp.add_trace(
                            go.Scatter(x=dates, y=decomp_data['trend'], 
                                     name='Trend', line=dict(color='#EF4444', width=2),
                                     hovertemplate='Trend: %{y:.2f}<extra></extra>'),
                            row=2, col=1
                        )
                        
                        # Seasonal
                        fig_decomp.add_trace(
                            go.Scatter(x=dates, y=decomp_data['seasonal'], 
                                     name='Seasonal', line=dict(color='#10B981', width=2),
                                     hovertemplate='Seasonal: %{y:.2f}<extra></extra>'),
                            row=3, col=1
                        )
                        
                        fig_decomp.update_layout(
                            height=650,
                            showlegend=False,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#2E4057'),
                            title_font_color='#2E4057'
                        )
                        
                        fig_decomp.update_xaxes(gridcolor='rgba(128,128,128,0.2)')
                        fig_decomp.update_yaxes(gridcolor='rgba(128,128,128,0.2)')
                        
                        st.plotly_chart(fig_decomp, use_container_width=True)
                else:
                    st.info("📊 Decomposition requires at least 10 data points")
        
        else:
            # Non-time series visualization (customer analytics, etc.)
            st.subheader("📈 Visualization")
            if 'plot' in results:
                fig_dict = json.loads(results['plot'])
                fig = go.Figure(fig_dict)
                
                # Enhanced styling for customer analytics
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#2E4057', size=12),
                    title_font_size=18,
                    title_font_color='#2E4057',
                    xaxis=dict(
                        gridcolor='rgba(128,128,128,0.2)',
                        title_font_color='#495057',
                        tickfont_color='#495057'
                    ),
                    yaxis=dict(
                        gridcolor='rgba(128,128,128,0.2)',
                        title_font_color='#495057',
                        tickfont_color='#495057'
                    ),
                    legend=dict(
                        bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='rgba(128,128,128,0.5)',
                        borderwidth=1
                    ),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
            
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #2E4057; margin-top: 0;">📊 Key Statistics</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if results['module'] == 'time_series':
                stats = results['statistics']
                
                # Enhanced metrics with better colors
                col1_1, col1_2 = st.columns(2)
                
                with col1_1:
                    st.metric(
                        "📈 Average Value", 
                        f"{stats['mean']:.2f}",
                        help="The arithmetic mean of all values in your time series"
                    )
                    
                    st.metric(
                        "📏 Range", 
                        f"{stats['max'] - stats['min']:.2f}",
                        help="The difference between highest and lowest values"
                    )
                
                with col1_2:
                    st.metric(
                        "📊 Std Deviation", 
                        f"{stats['std']:.2f}",
                        help="Measures how spread out your data points are"
                    )
                    
                    # Enhanced trend display
                    trend_value = stats.get('trend_slope', 0)
                    trend_direction = stats['trend'].title()
                    trend_color = "🔺" if trend_value > 0 else "🔻" if trend_value < 0 else "➡️"
                    
                    st.metric(
                        f"{trend_color} Trend", 
                        f"{trend_direction}",
                        delta=f"{trend_value:+.4f}/day",
                        help="Overall direction and slope of your data trend"
                    )
                
                # Enhanced predictability metrics
                predictability = stats.get('predictability', 'Unknown')
                cv = stats.get('coefficient_variation', 0)
                cv_explanation = stats.get('cv_explanation', 'Coefficient of Variation measures relative variability')
                rw_explanation = stats.get('rw_explanation', 'Random walk analysis examines autocorrelation patterns')

                # Color coding based on predictability
                if 'Very High' in predictability:
                    pred_color = "#4caf50"
                    pred_icon = "🎯"
                elif 'High' in predictability:
                    pred_color = "#8bc34a"
                    pred_icon = "🎯"
                elif 'Moderate' in predictability:
                    pred_color = "#ff9800"
                    pred_icon = "🎲"
                elif 'Low' in predictability:
                    pred_color = "#ff5722"
                    pred_icon = "⚡"
                else:
                    pred_color = "#9e9e9e"
                    pred_icon = "❓"

                
                # Enhanced predictability display with expandable explanations
                with st.expander("🎯 Predictability Analysis - Click for Details", expanded=True):
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {pred_color}15 0%, {pred_color}25 100%); 
                                border: 2px solid {pred_color}; padding: 15px; margin: 10px 0; border-radius: 10px;">
                        <strong>{pred_icon} Predictability Level:</strong> {predictability}<br>
                        <strong>📊 Coefficient of Variation:</strong> {cv:.3f}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed explanations
                    st.markdown("**📚 What is Coefficient of Variation (CV)?**")
                    st.info(f"""
                    {cv_explanation}
                    
                    **Interpretation Guide:**
                    - CV < 0.1: Very stable data (excellent for forecasting)
                    - CV 0.1-0.3: Moderately stable (good for forecasting)  
                    - CV 0.3-0.6: Moderate volatility (fair for forecasting)
                    - CV 0.6-1.0: High volatility (challenging to forecast)
                    - CV > 1.0: Very high volatility (very difficult to forecast)
                    """)

                # Random walk analysis with enhanced explanation
                random_walk_insight = stats.get('random_walk_insight', 'Unknown')

                with st.expander("🔄 Random Walk Analysis - Click for Details", expanded=False):
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #e1f5fe 0%, #b3e5fc 100%); 
                                border: 2px solid #03a9f4; padding: 15px; margin: 10px 0; border-radius: 10px;">
                        <strong>🔄 Pattern Analysis:</strong> {random_walk_insight}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("**📚 What is Random Walk Analysis?**")
                    st.info(f"""
                    {rw_explanation}
                    
                    **Pattern Types:**
                    - **Random Walk**: Changes are unpredictable (like stock prices)
                    - **Momentum**: Trends tend to continue (like growing businesses)
                    - **Mean Reversion**: Values return to average (like seasonal patterns)
                    - **Weak Patterns**: Some predictable elements but mostly random
                    
                    **Autocorrelation Values:**
                    - Close to 0: Random walk behavior
                    - Positive (>0.3): Momentum/trending behavior  
                    - Negative (<-0.3): Mean reversion behavior
                    """)
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {pred_color}15 0%, {pred_color}25 100%); 
                            border: 2px solid {pred_color}; padding: 15px; margin: 10px 0; border-radius: 10px;">
                    <strong>{pred_icon} Predictability:</strong> {predictability}<br>
                    <small>Coefficient of Variation: {cv:.3f}</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Random walk analysis
                random_walk_insight = stats.get('random_walk_insight', 'Unknown')
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #e1f5fe 0%, #b3e5fc 100%); 
                            border: 2px solid #03a9f4; padding: 15px; margin: 10px 0; border-radius: 10px;">
                    <strong>🔄 Random Walk Analysis:</strong><br>
                    <small>{random_walk_insight}</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Category breakdown if available
                if results.get('has_categories', False) and 'categories' in stats:
                    st.markdown("**🏷️ Category Performance:**")
                    
                    for cat, cat_stats in stats['categories'].items():
                        cat_trend_color = "🔺" if cat_stats['trend_slope'] > 0 else "🔻" if cat_stats['trend_slope'] < 0 else "➡️"
                        
                        # Predictability color for category
                        cat_pred = cat_stats.get('predictability', 'Unknown')
                        if 'Very High' in cat_pred or 'High' in cat_pred:
                            cat_bg_color = "#e8f5e8"
                        elif 'Moderate' in cat_pred:
                            cat_bg_color = "#fff3e0"
                        else:
                            cat_bg_color = "#ffebee"
                        
                        st.markdown(f"""
                        <div style="background: {cat_bg_color}; 
                                    border-left: 4px solid #007bff; padding: 10px; margin: 5px 0; border-radius: 5px;">
                            <strong>{cat_trend_color} {cat}:</strong> {cat_stats['trend'].title()}<br>
                            <small>Mean: {cat_stats['mean']:.2f} | Slope: {cat_stats['trend_slope']:+.4f}/day</small><br>
                            <small>Predictability: {cat_pred.split(' - ')[0]}</small>
                        </div>
                        """, unsafe_allow_html=True)
                
            elif results['module'] == 'customer':
                segments = results['segments']
                
                # Enhanced customer segment display
                segment_colors = {
                    'Champions': '#28a745',
                    'Loyal Customers': '#17a2b8', 
                    'Potential Loyalists': '#6f42c1',
                    'New Customers': '#20c997',
                    'Promising': '#fd7e14',
                    'Need Attention': '#ffc107',
                    'About to Sleep': '#dc3545',
                    'At Risk': '#e83e8c',
                    'Cannot Lose Them': '#dc3545',
                    'Hibernating': '#6c757d',
                    'Lost': '#343a40',
                    'Others': '#6c757d'
                }
                
                st.write("**👥 Customer Segments:**")
                for segment, count in segments.items():
                    color = segment_colors.get(segment, '#6c757d')
                    
                    # Calculate percentage
                    total_customers = sum(segments.values())
                    percentage = (count / total_customers * 100) if total_customers > 0 else 0
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {color}15 0%, {color}25 100%); 
                                border-left: 4px solid {color}; padding: 12px; margin: 8px 0; border-radius: 8px;">
                        <strong style="color: {color};">{segment}:</strong> {count} customers ({percentage:.1f}%)<br>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #2E4057; margin-top: 0;">🔍 Advanced Analysis</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if results['module'] == 'time_series':
                # Enhanced seasonality and volatility metrics
                seasonality_strength = results.get('seasonality_strength', 0)
                weekly_seasonality = results.get('weekly_seasonality', 0)
                volatility = results.get('volatility', 0)
                growth_rate = results.get('growth_rate', 0)
                
                # Seasonality analysis with visual indicators
                seasonality_level = "High" if seasonality_strength > 0.3 else "Moderate" if seasonality_strength > 0.15 else "Low"
                seasonality_icon = "🔥" if seasonality_strength > 0.3 else "🔶" if seasonality_strength > 0.15 else "🔵"
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
                            border: 2px solid #2196f3; padding: 15px; margin: 10px 0; border-radius: 10px;">
                    <strong>{seasonality_icon} Monthly Seasonality:</strong> {seasonality_level}<br>
                    <small>Strength: {seasonality_strength:.3f} | Weekly: {weekly_seasonality:.3f}</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Volatility analysis
                volatility_level = "High" if volatility > 0.5 else "Moderate" if volatility > 0.2 else "Low"
                volatility_icon = "⚡" if volatility > 0.5 else "🌊" if volatility > 0.2 else "📈"
                volatility_color = "#ff5722" if volatility > 0.5 else "#ff9800" if volatility > 0.2 else "#4caf50"
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {volatility_color}15 0%, {volatility_color}25 100%); 
                            border: 2px solid {volatility_color}; padding: 15px; margin: 10px 0; border-radius: 10px;">
                    <strong>{volatility_icon} Volatility:</strong> {volatility_level}<br>
                    <small>Coefficient: {volatility:.3f}</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Growth analysis
                growth_icon = "📈" if growth_rate > 0 else "📉" if growth_rate < 0 else "➡️"
                growth_color = "#4caf50" if growth_rate > 0 else "#f44336" if growth_rate < 0 else "#9e9e9e"
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {growth_color}15 0%, {growth_color}25 100%); 
                            border: 2px solid {growth_color}; padding: 15px; margin: 10px 0; border-radius: 10px;">
                    <strong>{growth_icon} Total Growth:</strong> {growth_rate:+.1f}%<br>
                    <small>From first to last data point</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Forecast accuracy (if available)
                if results.get('overall_forecast') and results['overall_forecast'].get('metrics'):
                    metrics = results['overall_forecast']['metrics']
                    mae = metrics['mae']
                    rmse = metrics['rmse']
                    mape = metrics.get('mape')
                    
                    # Accuracy assessment
                    accuracy_color = "#4caf50" if mae < stats['std'] * 0.5 else "#ff9800" if mae < stats['std'] else "#f44336"
                    accuracy_icon = "🎯" if mae < stats['std'] * 0.5 else "🎲" if mae < stats['std'] else "⚠️"
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {accuracy_color}15 0%, {accuracy_color}25 100%); 
                                border: 2px solid {accuracy_color}; padding: 15px; margin: 10px 0; border-radius: 10px;">
                        <strong>{accuracy_icon} Forecast Accuracy:</strong><br>
                        <small>MAE: {mae:.2f} | RMSE: {rmse:.2f}</small><br>
                        {f'<small>MAPE: {mape:.1f}%</small>' if mape else ''}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Trend slope detailed explanation
                trend_slope = results['statistics'].get('trend_slope', 0)
                slope_magnitude = abs(trend_slope)
                
                if slope_magnitude > 1:
                    slope_desc = "Very steep"
                elif slope_magnitude > 0.1:
                    slope_desc = "Moderate"
                elif slope_magnitude > 0.01:
                    slope_desc = "Gentle"
                else:
                    slope_desc = "Very gentle"
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); 
                            border: 2px solid #9c27b0; padding: 15px; margin: 10px 0; border-radius: 10px;">
                    <strong>📏 Trend Analysis:</strong><br>
                    Slope: {trend_slope:.6f} units/day<br>
                    <small>Steepness: {slope_desc}</small>
                </div>
                """, unsafe_allow_html=True)
                
            elif results['module'] == 'customer':
                rfm = results['rfm_summary']
                
                # Enhanced RFM display with color coding
                # Recency color (lower is better)
                recency_color = "#4caf50" if rfm['avg_recency'] < 30 else "#ff9800" if rfm['avg_recency'] < 90 else "#f44336"
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {recency_color}15 0%, {recency_color}25 100%); 
                            border: 2px solid {recency_color}; padding: 15px; margin: 10px 0; border-radius: 10px;">
                    <strong>📅 Average Recency:</strong> {rfm['avg_recency']:.1f} days<br>
                    <small>How recently customers purchased</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Frequency (higher is better)
                frequency_color = "#4caf50" if rfm['avg_frequency'] > 5 else "#ff9800" if rfm['avg_frequency'] > 2 else "#f44336"
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {frequency_color}15 0%, {frequency_color}25 100%); 
                            border: 2px solid {frequency_color}; padding: 15px; margin: 10px 0; border-radius: 10px;">
                    <strong>🔄 Average Frequency:</strong> {rfm['avg_frequency']:.1f} transactions<br>
                    <small>How often customers purchase</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Monetary value
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); 
                            border: 2px solid #4caf50; padding: 15px; margin: 10px 0; border-radius: 10px;">
                    <strong>💰 Average Monetary:</strong> ${rfm['avg_monetary']:.2f}<br>
                    <small>Average customer lifetime value</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Customer health indicators
                segments = results['segments']
                total_customers = sum(segments.values())
                
                champions_pct = (segments.get('Champions', 0) / total_customers) * 100 if total_customers > 0 else 0
                at_risk_pct = (segments.get('At Risk', 0) / total_customers) * 100 if total_customers > 0 else 0
                loyal_pct = (segments.get('Loyal Customers', 0) / total_customers) * 100 if total_customers > 0 else 0
                
                health_score = champions_pct + loyal_pct
                health_color = "#4caf50" if health_score > 40 else "#ff9800" if health_score > 20 else "#f44336"
                health_icon = "💚" if health_score > 40 else "💛" if health_score > 20 else "❤️"
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {health_color}15 0%, {health_color}25 100%); 
                            border: 2px solid {health_color}; padding: 15px; margin: 10px 0; border-radius: 10px;">
                    <strong>{health_icon} Customer Health:</strong> {health_score:.1f}%<br>
                    <small>Champions + Loyal customers</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Enhanced export section
        st.markdown("---")
        st.subheader("📥 Export & Actions")
        
        col_export1, col_export2, col_export3 = st.columns(3)
        
        with col_export1:
            if st.button("📄 Generate PDF Report", use_container_width=True):
                st.markdown("""
                <div class="warning-box">
                    🚧 PDF export feature coming soon!<br>
                    <small>Will include charts, insights, and detailed analysis</small>
                </div>
                """, unsafe_allow_html=True)
        
        with col_export2:
            if st.button("📊 Export to Excel", use_container_width=True):
                st.markdown("""
                <div class="warning-box">
                    🚧 Excel export feature coming soon!<br>
                    <small>Will include raw data and processed results</small>
                </div>
                """, unsafe_allow_html=True)
        
        with col_export3:
            if st.button("🔄 Run New Analysis", use_container_width=True):
                st.session_state.analysis_results = None
                st.session_state.current_data = None
                st.session_state.current_session_id = None
                st.success("🔄 Session cleared! You can now load new data.")
                st.rerun()

    # Model Comparison Results (if available)
    if hasattr(st.session_state, 'model_comparison') and st.session_state.model_comparison:
        st.subheader("🏆 Model Comparison Results")
        
        comparison_data = st.session_state.model_comparison
        comparison_table = comparison_data['comparison_table']
        successful_models = [m for m in comparison_table if m['status'] == 'success']
        
        if successful_models:
            # Create performance comparison chart
            import pandas as pd
            df_comp = pd.DataFrame(successful_models)
            
            # Create comparison visualization
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # MAE Comparison
                fig_mae = px.bar(
                    df_comp, 
                    x='model', 
                    y='mae',
                    title='MAE Comparison (Lower is Better)',
                    color='mae',
                    color_continuous_scale='RdYlGn_r'  # Red for high, Green for low
                )
                fig_mae.update_layout(
                    xaxis_tickangle=45,
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#2E4057')
                )
                st.plotly_chart(fig_mae, use_container_width=True)
            
            with col_chart2:
                # R² Comparison (if available)
                models_with_r2 = df_comp[df_comp['r2'].notna()]
                if not models_with_r2.empty:
                    fig_r2 = px.bar(
                        models_with_r2, 
                        x='model', 
                        y='r2',
                        title='R² Comparison (Higher is Better)',
                        color='r2',
                        color_continuous_scale='RdYlGn'  # Red for low, Green for high
                    )
                    fig_r2.update_layout(
                        xaxis_tickangle=45,
                        height=400,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#2E4057')
                    )
                    st.plotly_chart(fig_r2, use_container_width=True)
                else:
                    st.info("R² comparison not available for these models")
            
            # Detailed comparison table
            with st.expander("📊 Detailed Comparison Table", expanded=False):
                display_df = df_comp[['rank', 'model', 'mae', 'rmse', 'r2', 'mape', 'directional_accuracy']].copy()
                
                # Format numeric columns
                numeric_cols = ['mae', 'rmse', 'r2', 'mape', 'directional_accuracy']
                for col in numeric_cols:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if x is not None else "N/A")
                
                display_df.columns = ['Rank', 'Model', 'MAE', 'RMSE', 'R²', 'MAPE (%)', 'Direction Acc (%)']
                st.dataframe(display_df, use_container_width=True)
            
            # Best model summary
            best_model = comparison_data['best_model']
            st.markdown(f"""
            <div class="success-box">
                🏆 <strong>Champion Model:</strong> {best_model['model_name']}<br>
                📈 <strong>Performance:</strong> MAE {best_model['mae']:.3f}, RMSE {best_model['rmse']:.3f}
                {f", R² {best_model['r2']:.3f}" if best_model.get('r2') is not None else ""}
                {f", MAPE {best_model['mape']:.1f}%" if best_model.get('mape') is not None else ""}
            </div>
            """, unsafe_allow_html=True)
            
            # Model selection insights
            st.markdown("### 💡 Model Selection Insights")
            
            # Analyze results and provide recommendations
            mae_values = [m['mae'] for m in successful_models if m['mae'] is not None]
            if len(mae_values) > 1:
                mae_range = max(mae_values) - min(mae_values)
                mae_cv = np.std(mae_values) / np.mean(mae_values) if np.mean(mae_values) > 0 else 0
                
                if mae_cv < 0.1:
                    st.info("📊 **Model Performance**: All models show similar accuracy - choose based on interpretability needs")
                elif mae_cv < 0.3:
                    st.success("📊 **Model Performance**: Clear performance differences - best model significantly outperforms others")
                else:
                    st.warning("📊 **Model Performance**: Large performance variations - data may be challenging to predict")
            
            # Clear comparison results button
            if st.button("🗑️ Clear Comparison Results", type="secondary"):
                if hasattr(st.session_state, 'model_comparison'):
                    delattr(st.session_state, 'model_comparison')
                st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    📊 Data Analytics | Built with FastAPI & Streamlit | 
    <a href="#" style="color: #667eea;">Documentation</a> | 
    <a href="#" style="color: #667eea;">API Reference</a>
</div>
""", unsafe_allow_html=True)