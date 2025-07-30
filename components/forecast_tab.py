import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from datetime import datetime, timedelta
from fe_utils.api_client import APIClient

def _show_api_status():
    """Show API connection status"""
    
    with st.expander("🔌 API Connection Status", expanded=False):
        try:
            # Simple test to check if APIClient is working
            # You might want to add a health check endpoint
            st.success("✅ APIClient imported successfully")
            st.info("📡 Ready to call forecast endpoints:")
            st.code("""
            - /forecast/single-model
            - /forecast/compare-models
            """)
            
        except Exception as e:
            st.error(f"❌ API Client Error: {str(e)}")
            st.info("💡 Please check backend server is running and accessible")

# Enhanced version of render_forecast_tab with better error handling
def render_forecast_tab():
    """Render the forecast configuration and results tab"""
    st.header("🔮 Forecasting")
    
    # Check if data is loaded
    if st.session_state.current_data is None:
        st.warning("⚠️ Please load data first in the Data Input tab")
        _render_forecast_info()
        return
    
    # Check if analysis results exist
    if st.session_state.analysis_results is None:
        st.info("ℹ️ Run analysis first to enable forecasting")
        _render_forecast_info()
        return
    
    results = st.session_state.analysis_results
    
    # Check for time series capability
    has_time_data = _check_time_series_capability(results)
    
    if not has_time_data:
        st.warning("⚠️ Forecasting requires time series data with date/time information")
        _render_forecast_info()
        return
    
    # Show API connection status
    _show_api_status()
    
    # Check if we already have forecast results
    if st.session_state.get('forecast_results'):
        _render_forecast_results_with_config(st.session_state.forecast_results)
    else:
        # Main forecast interface
        _render_forecast_interface(results)


def _render_forecast_info():
    """Render forecast information when no analysis exists"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 30px; border-radius: 20px; margin: 20px 0; text-align: center;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
        <h2 style="margin: 0; color: white;">🔮 Forecasting Capabilities</h2>
        <p style="margin: 15px 0 0 0; color: rgba(255,255,255,0.9); font-size: 18px;">
            Generate accurate predictions for your time series data
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 20px; border-radius: 15px; 
                    border-left: 5px solid #3b82f6; margin: 10px 0;">
            <h4 style="margin: 0 0 10px 0; color: #1e293b;">🤖 ML Models</h4>
            <ul style="margin: 0; padding-left: 20px; color: #475569;">
                <li>Random Forest</li>
                <li>Linear Regression</li>
                <li>CatBoost</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 20px; border-radius: 15px; 
                    border-left: 5px solid #10b981; margin: 10px 0;">
            <h4 style="margin: 0 0 10px 0; color: #1e293b;">📊 Statistical Models</h4>
            <ul style="margin: 0; padding-left: 20px; color: #475569;">
                <li>ETS (Exponential Smoothing)</li>
                <li>ARIMA</li>
                <li>Theta Method</li>
                <li>Moving Average</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: white; padding: 20px; border-radius: 15px; 
                    border-left: 5px solid #f59e0b; margin: 10px 0;">
            <h4 style="margin: 0 0 10px 0; color: #1e293b;">✨ Features</h4>
            <ul style="margin: 0; padding-left: 20px; color: #475569;">
                <li>Model Comparison</li>
                <li>Confidence Intervals</li>
                <li>Performance Metrics</li>
                <li>Feature Importance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### 🚀 Getting Started")
    
    steps = [
        ("1️⃣", "Load Data", "Upload your time series data in the Data Input tab"),
        ("2️⃣", "Process Data", "Complete preprocessing in the Data Processing tab"),
        ("3️⃣", "Run Analysis", "Analyze your data in the Analysis tab"),
        ("4️⃣", "Forecast", "Return here to generate predictions")
    ]
    
    cols = st.columns(4)
    for i, (icon, title, desc) in enumerate(steps):
        with cols[i]:
            st.markdown(f"""
            <div style="background: white; padding: 15px; border-radius: 10px; 
                        text-align: center; border: 2px solid #e2e8f0; margin: 5px 0;">
                <div style="font-size: 24px; margin-bottom: 5px;">{icon}</div>
                <h5 style="margin: 5px 0; color: #1e293b;">{title}</h5>
                <p style="margin: 0; font-size: 12px; color: #64748b;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

def _render_forecast_interface(analysis_results):
    """Render the main forecast interface"""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #ff6b35, #f7931e); 
                padding: 20px; border-radius: 15px; margin: 20px 0; text-align: center;">
        <h3 style="margin: 0; color: white;">⚙️ Forecast Configuration</h3>
        <p style="margin: 5px 0 0 0; color: rgba(255,255,255,0.9);">
            Configure your forecasting parameters
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Simple configuration interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Forecast mode selection with clear explanation
        st.markdown("#### 🎯 Forecasting Mode")
        forecast_mode = st.radio(
            "Choose your approach:",
            ["🚀 Quick Forecast", "🏆 Compare Models"],
            help="""
            **Quick Forecast**: Fast prediction with a single reliable model
            **Compare Models**: Test multiple models to find the absolute best one
            """,
            horizontal=True
        )
        
        # Model selection for quick forecast
        if forecast_mode == "🚀 Quick Forecast":
            st.markdown("#### 🤖 Select Model")
            model_choice = st.selectbox(
                "Choose forecasting model:",
                [
                    ("Random Forest", "random_forest", "🌳 Great for complex patterns and feature relationships"),
                    ("ETS (Exponential Smoothing)", "ets", "📊 Excellent for seasonal data and trends"),
                    ("Linear Regression", "linear_regression", "📈 Simple and interpretable for linear trends"),
                    ("Moving Average", "moving_average", "🔄 Good baseline for stable data")
                ],
                format_func=lambda x: f"{x[0]} - {x[2]}",
                help="Each model has different strengths. Random Forest and ETS are generally good starting points."
            )
            selected_model = model_choice[1]
            model_name = model_choice[0]
        else:
            selected_model = "comparison"
            model_name = "Best Model (Auto-selected)"
        
        # Forecast horizon
        st.markdown("#### 📅 Forecast Horizon")
        forecast_periods = st.number_input(
            "How many periods to forecast?",
            min_value=1,
            max_value=50,
            value=12,
            help="Number of time periods to predict into the future (e.g., 12 months, 30 days)"
        )
    
    with col2:
        # Configuration summary
        st.markdown("#### 📋 Summary")
        st.markdown(f"""
        <div style="background: #f8fafc; padding: 20px; border-radius: 10px; 
                    border-left: 4px solid #3b82f6;">
            <p style="margin: 0 0 10px 0;"><strong>Mode:</strong> {forecast_mode}</p>
            <p style="margin: 0 0 10px 0;"><strong>Model:</strong> {model_name}</p>
            <p style="margin: 0 0 10px 0;"><strong>Periods:</strong> {forecast_periods}</p>
            <div style="margin-top: 15px; padding: 10px; background: #dbeafe; 
                        border-radius: 5px; text-align: center;">
                ✅ <strong>Ready to Forecast!</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Generate forecast button
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        if forecast_mode == "🚀 Quick Forecast":
            button_text = f"🔮 Generate Forecast with {model_name}"
            button_help = f"Create forecast predictions using {model_name}"
        else:
            button_text = "🏆 Compare Models & Generate Best Forecast"
            button_help = "Test multiple models and automatically select the best performer"
        
        if st.button(button_text, type="primary", use_container_width=True, help=button_help):
            _execute_forecast(forecast_mode, selected_model, model_name, forecast_periods, analysis_results)
    
    # Reset button
    with col_btn3:
        if st.button("🔄 Reset", type="secondary", use_container_width=True):
            if 'forecast_results' in st.session_state:
                del st.session_state.forecast_results
            st.success("✅ Reset complete!")
            st.rerun()



def _check_forecast_data_availability(analysis_results):
    """Check if forecast data is available in analysis results"""
    
    has_forecast = False
    forecast_info = []
    
    # Check for single forecast
    if analysis_results.get('overall_forecast'):
        has_forecast = True
        forecast_info.append("✅ Single model forecast available")
        
        # Check data completeness
        forecast = analysis_results['overall_forecast']
        if forecast.get('test_dates') and forecast.get('test_actual') and forecast.get('test_predicted'):
            forecast_info.append("✅ Test data complete")
        else:
            forecast_info.append("⚠️ Test data incomplete")
        
        if forecast.get('metrics'):
            forecast_info.append("✅ Performance metrics available")
        else:
            forecast_info.append("⚠️ Performance metrics missing")
    
    # Check for category forecasts
    if analysis_results.get('category_forecasts'):
        has_forecast = True
        categories = list(analysis_results['category_forecasts'].keys())
        forecast_info.append(f"✅ Category forecasts available: {', '.join(categories[:3])}{'...' if len(categories) > 3 else ''}")
    
    # Check for model comparison
    if analysis_results.get('model_comparison'):
        has_forecast = True
        forecast_info.append("✅ Model comparison results available")
    
    return has_forecast, forecast_info

def render_forecast_tab():
    """Render the forecast configuration and results tab"""
    st.header("🔮 Forecasting")
    
    # Check if data is loaded
    if st.session_state.current_data is None:
        st.warning("⚠️ Please load data first in the Data Input tab")
        _render_forecast_info()
        return
    
    # Check if analysis results exist
    if st.session_state.analysis_results is None:
        st.info("ℹ️ Run analysis first to enable forecasting")
        _render_forecast_info()
        return
    
    results = st.session_state.analysis_results
    
    # Check if forecast data is available
    has_forecast_data, forecast_info = _check_forecast_data_availability(results)
    
    # Show forecast data status
    with st.expander("📊 Forecast Data Status", expanded=not has_forecast_data):
        for info in forecast_info:
            if "✅" in info:
                st.success(info)
            elif "⚠️" in info:
                st.warning(info)
            else:
                st.info(info)
        
        if not has_forecast_data:
            st.error("❌ No forecast data found in analysis results")
            st.info("💡 Please ensure your backend analysis includes forecasting step")
    
    # More flexible check for time series capability
    has_time_data = False
    
    # Check various indicators that this could be time series data
    if results.get('module') == 'time_series':
        has_time_data = True
    elif 'df' in results and results['df'] is not None:
        df = results['df']
        # Check if there's a date column or datetime index
        if 'date' in df.columns or hasattr(df.index, 'dtype') and 'datetime' in str(df.index.dtype):
            has_time_data = True
    elif 'plot' in results:
        # If there's a plot, assume it might be time series
        has_time_data = True
    
    if not has_time_data:
        st.warning("⚠️ Forecasting requires time series data with date/time information")
        _render_forecast_info()
        return
    
    # Check if we already have forecast results in session state
    if st.session_state.get('forecast_results'):
        _render_forecast_results_with_config(st.session_state.forecast_results)
    elif has_forecast_data:
        # Use existing forecast data from analysis
        st.session_state.forecast_results = {
            'overall_forecast': results.get('overall_forecast'),
            'category_forecasts': results.get('category_forecasts'),
            'plot': results.get('plot', '{}'),
            'statistics': results.get('statistics', {}),
            'df': results.get('df'),
            'model_comparison': results.get('model_comparison')
        }
        _render_forecast_results_with_config(st.session_state.forecast_results)
    else:
        # Show interface to generate new forecast
        _render_forecast_interface(results)

def _render_forecast_interface(analysis_results):
    """Render the main forecast interface"""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #ff6b35, #f7931e); 
                padding: 20px; border-radius: 15px; margin: 20px 0; text-align: center;">
        <h3 style="margin: 0; color: white;">⚙️ Generate New Forecast</h3>
        <p style="margin: 5px 0 0 0; color: rgba(255,255,255,0.9);">
            Configure parameters to create a new forecast
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if backend supports forecasting
    st.info("🔄 **Note**: This will trigger a new forecast generation via backend API")
    
    # Simple configuration interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Forecast mode selection
        st.markdown("#### 🎯 Forecasting Mode")
        forecast_mode = st.radio(
            "Choose your approach:",
            ["🚀 Quick Forecast", "🏆 Compare Models"],
            help="""
            **Quick Forecast**: Fast prediction with a single reliable model
            **Compare Models**: Test multiple models to find the absolute best one
            """,
            horizontal=True
        )
        
        # Model selection for quick forecast
        if forecast_mode == "🚀 Quick Forecast":
            st.markdown("#### 🤖 Select Model")
            model_choice = st.selectbox(
                "Choose forecasting model:",
                [
                    ("Random Forest", "random_forest", "🌳 Great for complex patterns and feature relationships"),
                    ("ETS (Exponential Smoothing)", "ets", "📊 Excellent for seasonal data and trends"),
                    ("Linear Regression", "linear_regression", "📈 Simple and interpretable for linear trends"),
                    ("Moving Average", "moving_average", "🔄 Good baseline for stable data")
                ],
                format_func=lambda x: f"{x[0]} - {x[2]}",
                help="Each model has different strengths. Random Forest and ETS are generally good starting points."
            )
            selected_model = model_choice[1]
            model_name = model_choice[0]
        else:
            selected_model = "comparison"
            model_name = "Best Model (Auto-selected)"
        
        # Forecast horizon
        st.markdown("#### 📅 Forecast Horizon")
        forecast_periods = st.number_input(
            "How many periods to forecast?",
            min_value=1,
            max_value=50,
            value=12,
            help="Number of time periods to predict into the future (e.g., 12 months, 30 days)"
        )
    
    with col2:
        # Configuration summary
        st.markdown("#### 📋 Summary")
        st.markdown(f"""
        <div style="background: #f8fafc; padding: 20px; border-radius: 10px; 
                    border-left: 4px solid #3b82f6;">
            <p style="margin: 0 0 10px 0;"><strong>Mode:</strong> {forecast_mode}</p>
            <p style="margin: 0 0 10px 0;"><strong>Model:</strong> {model_name}</p>
            <p style="margin: 0 0 10px 0;"><strong>Periods:</strong> {forecast_periods}</p>
            <div style="margin-top: 15px; padding: 10px; background: #dbeafe; 
                        border-radius: 5px; text-align: center;">
                ⚠️ <strong>Requires Backend API</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Generate forecast button
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        if forecast_mode == "🚀 Quick Forecast":
            button_text = f"🔮 Generate Forecast with {model_name}"
            button_help = f"Create forecast predictions using {model_name} via backend API"
        else:
            button_text = "🏆 Compare Models & Generate Best Forecast"
            button_help = "Test multiple models via backend API and automatically select the best performer"
        
        if st.button(button_text, type="primary", use_container_width=True, help=button_help):
            _execute_forecast(forecast_mode, selected_model, model_name, forecast_periods, analysis_results)
    
    # Reset button
    with col_btn3:
        if st.button("🔄 Reset", type="secondary", use_container_width=True):
            if 'forecast_results' in st.session_state:
                del st.session_state.forecast_results
            st.success("✅ Reset complete!")
            st.rerun()

def _execute_forecast(forecast_mode, selected_model, model_name, forecast_periods, analysis_results):
    """Execute forecast generation using real API client"""
    
    progress_container = st.container()
    
    with progress_container:
        if forecast_mode == "🚀 Quick Forecast":
            with st.spinner(f"🔮 Generating forecast with {model_name}..."):
                try:
                    # Prepare payload for single model forecast
                    payload = {
                        'data': analysis_results.get('df', pd.DataFrame()).to_dict('records') if analysis_results.get('df') is not None else [],
                        'model_type': selected_model,
                        'model_name': model_name,
                        'forecast_periods': forecast_periods,
                        'existing_analysis': analysis_results
                    }
                    
                    # Call real API using APIClient
                    forecast_results = APIClient.forecast_single_model(payload)
                    
                    if forecast_results:
                        # Store results in session state
                        st.session_state.forecast_results = forecast_results
                        st.success(f"✅ Forecast generated successfully with {model_name}!")
                        st.balloons()
                    else:
                        st.error(f"❌ Failed to generate forecast with {model_name}")
                        st.info("💡 Please check backend API connection and data format")
                        return
                        
                except Exception as e:
                    st.error(f"❌ Error generating forecast: {str(e)}")
                    st.info("💡 Please check your internet connection and backend API status")
                    return
                    
        else:  # Model comparison mode
            with st.spinner("🏆 Comparing models and generating forecast... This may take a few moments."):
                try:
                    # Prepare payload for model comparison
                    payload = {
                        'data': analysis_results.get('df', pd.DataFrame()).to_dict('records') if analysis_results.get('df') is not None else [],
                        'forecast_periods': forecast_periods,
                        'existing_analysis': analysis_results,
                        'models_to_compare': ['random_forest', 'ets', 'linear_regression', 'moving_average']  # Default models to compare
                    }
                    
                    # Call real API using APIClient
                    forecast_results = APIClient.compare_forecast_models(payload)
                    
                    if forecast_results:
                        # Store results in session state
                        st.session_state.forecast_results = forecast_results
                        
                        # Get best model name from results
                        best_model_name = "Best Model"
                        if forecast_results.get('overall_forecast'):
                            best_model_name = forecast_results['overall_forecast'].get('model_name', 'Best Model')
                        elif forecast_results.get('best_model'):
                            best_model_name = forecast_results['best_model'].get('model_name', 'Best Model')
                        
                        st.success(f"✅ Model comparison complete! {best_model_name} selected as best performer.")
                        st.balloons()
                    else:
                        st.error("❌ Failed to compare models and generate forecast")
                        st.info("💡 Please check backend API connection and data format")
                        return
                        
                except Exception as e:
                    st.error(f"❌ Error comparing models: {str(e)}")
                    st.info("💡 Please check your internet connection and backend API status")
                    return
    
    # Auto-refresh to show results
    import time
    time.sleep(1)
    st.rerun()

def _prepare_forecast_payload(analysis_results, model_type=None, forecast_periods=12):
    """Prepare payload for forecast API calls"""
    
    # Base payload structure
    payload = {
        'forecast_periods': forecast_periods
    }
    
    # Add data
    if analysis_results.get('df') is not None:
        df = analysis_results['df']
        
        # Convert DataFrame to records format
        payload['data'] = df.to_dict('records')
        
        # Add data metadata
        if 'date' in df.columns:
            payload['date_column'] = 'date'
        if 'value' in df.columns:
            payload['target_column'] = 'value'
        else:
            # Find first numeric column as target
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                payload['target_column'] = numeric_cols[0]
    
    # Add model type for single model forecast
    if model_type:
        payload['model_type'] = model_type
    
    # Add existing analysis context
    if analysis_results.get('statistics'):
        payload['data_statistics'] = analysis_results['statistics']
    
    if analysis_results.get('seasonality_info'):
        payload['seasonality_info'] = analysis_results['seasonality_info']
    
    # Add preprocessing info if available
    if analysis_results.get('preprocessing_info'):
        payload['preprocessing_info'] = analysis_results['preprocessing_info']
    
    return payload

def _validate_forecast_response(response):
    """Validate forecast API response"""
    
    if not response:
        return False, "Empty response from API"
    
    # Check for overall_forecast or category_forecasts
    if not (response.get('overall_forecast') or response.get('category_forecasts')):
        return False, "No forecast data in response"
    
    # Validate overall_forecast structure if present
    if response.get('overall_forecast'):
        forecast = response['overall_forecast']
        
        required_fields = ['test_dates', 'test_actual', 'test_predicted', 'metrics']
        missing_fields = [field for field in required_fields if not forecast.get(field)]
        
        if missing_fields:
            return False, f"Missing required fields in forecast: {missing_fields}"
        
        # Check if arrays have same length
        test_dates = forecast.get('test_dates', [])
        test_actual = forecast.get('test_actual', [])
        test_predicted = forecast.get('test_predicted', [])
        
        if len(test_dates) != len(test_actual) or len(test_dates) != len(test_predicted):
            return False, "Inconsistent data lengths in forecast arrays"
        
        # Check metrics
        metrics = forecast.get('metrics', {})
        if not any(metrics.get(metric) is not None for metric in ['mae', 'rmse', 'mape', 'r2']):
            return False, "No valid performance metrics in forecast"
    
    return True, "Valid forecast response"


def _generate_mock_forecast_results(model_type, model_name, forecast_periods, analysis_results):
    """Generate mock forecast results for demonstration"""
    
    # Get sample data from analysis results
    df = analysis_results.get('df')
    if df is not None and len(df) > 0:
        # Use actual data for realistic mock
        if 'date' in df.columns:
            last_date = pd.to_datetime(df['date'].iloc[-1])
            if 'value' in df.columns:
                last_values = df['value'].tail(20).values
            else:
                # Use first numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    last_values = df[numeric_cols[0]].tail(20).values
                else:
                    last_values = np.random.randn(20)
        else:
            last_date = pd.Timestamp.now()
            last_values = np.random.randn(20)
    else:
        last_date = pd.Timestamp.now()
        last_values = np.random.randn(20)
    
    # Generate mock test data (last 20% for testing)
    test_size = min(10, len(last_values) // 2)
    train_values = last_values[:-test_size] if test_size > 0 else last_values
    test_actual = last_values[-test_size:] if test_size > 0 else last_values[-5:]
    
    # Generate test dates
    test_dates = pd.date_range(
        start=last_date - pd.Timedelta(days=test_size-1),
        end=last_date,
        freq='D'
    )
    
    # Generate mock predictions with some realistic error
    base_trend = np.mean(np.diff(test_actual)) if len(test_actual) > 1 else 0
    noise_level = np.std(test_actual) * 0.1 if len(test_actual) > 1 else 1
    
    test_predicted = []
    for i, actual in enumerate(test_actual):
        # Add some realistic prediction error
        error = np.random.normal(0, noise_level)
        predicted = actual + error
        test_predicted.append(predicted)
    
    test_predicted = np.array(test_predicted)
    
    # Calculate performance metrics
    mae = np.mean(np.abs(test_actual - test_predicted))
    rmse = np.sqrt(np.mean((test_actual - test_predicted) ** 2))
    
    # Calculate MAPE (handle division by zero)
    mape = None
    if not np.any(test_actual == 0):
        mape = np.mean(np.abs((test_actual - test_predicted) / test_actual)) * 100
    
    # Calculate R²
    ss_res = np.sum((test_actual - test_predicted) ** 2)
    ss_tot = np.sum((test_actual - np.mean(test_actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Calculate directional accuracy
    if len(test_actual) > 1:
        actual_diff = np.diff(test_actual)
        pred_diff = np.diff(test_predicted)
        correct_direction = np.sum(np.sign(actual_diff) == np.sign(pred_diff))
        directional_accuracy = (correct_direction / len(actual_diff)) * 100
    else:
        directional_accuracy = 50.0
    
    # Generate confidence intervals
    prediction_std = np.std(test_actual - test_predicted)
    ci_95_upper = test_predicted + 1.96 * prediction_std
    ci_95_lower = test_predicted - 1.96 * prediction_std
    ci_80_upper = test_predicted + 1.28 * prediction_std
    ci_80_lower = test_predicted - 1.28 * prediction_std
    ci_50_upper = test_predicted + 0.67 * prediction_std
    ci_50_lower = test_predicted - 0.67 * prediction_std
    
    # Mock forecast structure
    forecast_results = {
        'overall_forecast': {
            'model_name': model_name,
            'model_type': model_type,
            'test_dates': test_dates.tolist(),
            'test_actual': test_actual.tolist(),
            'test_predicted': test_predicted.tolist(),
            'split_date': test_dates[0].isoformat(),
            'ci_95_upper': ci_95_upper.tolist(),
            'ci_95_lower': ci_95_lower.tolist(),
            'ci_80_upper': ci_80_upper.tolist(),
            'ci_80_lower': ci_80_lower.tolist(),
            'ci_50_upper': ci_50_upper.tolist(),
            'ci_50_lower': ci_50_lower.tolist(),
            'metrics': {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'r2': r2,
                'directional_accuracy': directional_accuracy,
                'mse': rmse ** 2,
                'drift': np.mean(np.diff(test_actual)) if len(test_actual) > 1 else 0
            }
        },
        'plot': analysis_results.get('plot', '{}'),  # Use existing plot if available
        'statistics': analysis_results.get('statistics', {}),
        'df': df
    }
    
    return forecast_results

def _render_forecast_results_with_config(forecast_results):
    """Render forecast results with display configuration"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                padding: 20px; border-radius: 15px; margin: 20px 0; text-align: center;">
        <h3 style="margin: 0; color: white;">🎉 Forecast Complete!</h3>
        <p style="margin: 5px 0 0 0; color: rgba(255,255,255,0.9);">
            Your predictions are ready. Configure display options below.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display settings at the top
    forecast_settings = _render_forecast_display_settings()
    
    # Render the forecast results directly here (no import needed)
    if forecast_results.get('overall_forecast'):
        _render_single_series_forecast_results(forecast_results, forecast_settings)
    
    # Add action buttons at the bottom
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Generate New Forecast", type="secondary", use_container_width=True):
            del st.session_state.forecast_results
            st.rerun()
    
    with col2:
        if st.button("📊 View Analysis", type="secondary", use_container_width=True):
            st.info("Navigate to Analysis tab to view detailed analysis")
    
    with col3:
        if st.button("💾 Export Results", type="secondary", use_container_width=True):
            st.info("Export functionality would be implemented here")

def _render_forecast_display_settings():
    """Render forecast display settings"""
    with st.expander("⚙️ Display Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_confidence_intervals = st.checkbox(
                "📊 Confidence Intervals", 
                value=True,
                key="show_confidence_intervals",
                help="Show uncertainty bands around predictions"
            )
            
            show_actual_vs_predicted = st.checkbox(
                "📈 Actual vs Predicted", 
                value=True,
                key="show_actual_vs_predicted",
                help="Compare predicted values with actual test data"
            )
        
        with col2:
            show_train_test_split = st.checkbox(
                "📊 Train/Test Split Line", 
                value=True,
                key="show_train_test_split",
                help="Show where training data ends and testing begins"
            )
            
            forecast_transparency = st.slider(
                "Forecast Opacity",
                min_value=0.3,
                max_value=1.0,
                value=0.8,
                step=0.1,
                key="forecast_transparency",
                help="Adjust transparency of forecast elements"
            )
        
        with col3:
            chart_height = st.selectbox(
                "Chart Height",
                options=[400, 500, 600, 700],
                index=1,
                key="chart_height",
                help="Adjust the height of the forecast chart"
            )
            
            show_performance_details = st.checkbox(
                "📋 Detailed Metrics", 
                value=False,
                key="show_performance_details",
                help="Show additional performance metrics and analysis"
            )
    
    return {
        'show_confidence_intervals': show_confidence_intervals,
        'show_actual_vs_predicted': show_actual_vs_predicted,
        'forecast_transparency': forecast_transparency,
        'show_train_test_split': show_train_test_split,
        'chart_height': chart_height,
        'show_performance_details': show_performance_details
    }

def _render_single_series_forecast_results(forecast_results, settings):
    """Render single series forecast results"""
    forecast = forecast_results['overall_forecast']
    
    # Model info
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 15px; margin: 20px 0; 
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);">
        <h4 style="margin: 0; color: white; text-align: center;">
            🔮 {forecast.get('model_name', 'Forecast Model')}
        </h4>
        <p style="margin: 5px 0 0 0; color: rgba(255,255,255,0.9); text-align: center;">
            Model Type: {forecast.get('model_type', 'unknown').title().replace('_', ' ')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main forecast chart
    fig = _create_enhanced_forecast_chart(forecast_results, forecast, settings)
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    _render_forecast_metrics(forecast, settings)
    
    # Additional analysis sections
    if settings['show_performance_details']:
        _render_detailed_forecast_analysis(forecast, forecast_results)

def _create_enhanced_forecast_chart(forecast_results, forecast, settings):
    """Create enhanced forecast chart"""
    # Start with the original plot if available
    fig = go.Figure()
    
    # Try to get base chart from analysis results
    if 'plot' in forecast_results:
        try:
            fig_dict = json.loads(forecast_results['plot'])
            fig = go.Figure(fig_dict)
        except:
            pass
    
    # Enhanced styling
    fig.update_layout(
        height=settings['chart_height'],
        plot_bgcolor='rgba(248,249,250,0.8)',
        paper_bgcolor='rgba(255,255,255,0.95)',
        font=dict(family="Inter", color='#2d3748', size=12),
        title_font=dict(size=20, color='#2d3748'),
        margin=dict(l=60, r=60, t=80, b=60),
        xaxis=dict(
            gridcolor='rgba(203,213,224,0.5)',
            gridwidth=1,
            title_font=dict(color='#4a5568', size=14),
            tickfont=dict(color='#718096', size=11)
        ),
        yaxis=dict(
            gridcolor='rgba(203,213,224,0.5)',
            gridwidth=1,
            title_font=dict(color='#4a5568', size=14),
            tickfont=dict(color='#718096', size=11)
        ),
        legend=dict(
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='rgba(203,213,224,0.8)',
            font=dict(size=11)
        ),
        hovermode='x unified'
    )
    
    # Add train/test split line with improved logic
    if settings['show_train_test_split']:
        _add_train_test_split_line(fig, forecast, forecast_results)
    
    # Add forecast and actual test data
    if settings['show_actual_vs_predicted']:
        _add_forecast_comparison(fig, forecast, settings)
    
    # Add confidence intervals
    if settings['show_confidence_intervals']:
        _add_confidence_intervals(fig, forecast, settings)
    
    return fig

def _add_train_test_split_line(fig, forecast, forecast_results):
    """Add train/test split line to chart with improved logic"""
    try:
        split_date = None
        
        # Method 1: Direct from forecast split_date
        if 'split_date' in forecast and forecast['split_date']:
            try:
                split_date_raw = forecast['split_date']
                if isinstance(split_date_raw, str):
                    split_date = pd.to_datetime(split_date_raw, errors='coerce')
                elif hasattr(split_date_raw, 'to_pydatetime'):
                    split_date = split_date_raw.to_pydatetime()
                else:
                    split_date = pd.to_datetime(str(split_date_raw), errors='coerce')
            except:
                split_date = None
        
        # Method 2: From test_dates (first test date)
        if split_date is None and 'test_dates' in forecast and forecast['test_dates']:
            try:
                test_dates_raw = forecast['test_dates']
                if isinstance(test_dates_raw, (list, np.ndarray)) and len(test_dates_raw) > 0:
                    first_test = test_dates_raw[0]
                    if isinstance(first_test, str):
                        split_date = pd.to_datetime(first_test, errors='coerce')
                    else:
                        split_date = pd.to_datetime(str(first_test), errors='coerce')
            except:
                split_date = None
        
        # Method 3: Calculate from forecast_results data
        if split_date is None and 'df' in forecast_results and forecast_results['df'] is not None:
            try:
                df = forecast_results['df']
                if len(df) > 0:
                    split_idx = int(len(df) * 0.8)
                    if split_idx < len(df):
                        if 'date' in df.columns:
                            split_date = pd.to_datetime(df['date'].iloc[split_idx], errors='coerce')
                        elif hasattr(df.index, 'dtype') and 'datetime' in str(df.index.dtype):
                            split_date = df.index[split_idx]
            except:
                split_date = None
        
        # Method 4: Fallback - estimate from existing chart data
        if split_date is None:
            try:
                # Check if there's existing data in the chart
                if len(fig.data) > 0 and hasattr(fig.data[0], 'x') and fig.data[0].x:
                    x_data = fig.data[0].x
                    if len(x_data) > 1:
                        # Use 80% point as split
                        split_idx = int(len(x_data) * 0.8)
                        if split_idx < len(x_data):
                            split_date = pd.to_datetime(x_data[split_idx], errors='coerce')
            except:
                split_date = None
        
        # Method 5: Final fallback - use test dates minus some buffer
        if split_date is None and 'test_dates' in forecast and forecast['test_dates']:
            try:
                test_dates = pd.to_datetime(forecast['test_dates'], errors='coerce')
                if len(test_dates) > 0 and not pd.isna(test_dates[0]):
                    # Estimate split as a few days before first test date
                    buffer_days = max(1, len(test_dates) // 4)  # Dynamic buffer
                    split_date = test_dates[0] - pd.Timedelta(days=buffer_days)
            except:
                split_date = None
        
        # Add the line if we successfully got a split date
        if split_date is not None and not pd.isna(split_date):
            fig.add_vline(
                x=split_date,
                line_dash="dash",
                line_color="rgba(239,68,68,0.8)",
                line_width=3,
                annotation_text="📊 Train/Test Split",
                annotation_position="top",
                annotation=dict(
                    bgcolor="rgba(255,255,255,0.95)",
                    bordercolor="rgba(239,68,68,0.8)",
                    borderwidth=1,
                    font=dict(size=11, color="#2d3748")
                )
            )
        else:
            # If all methods fail, add a note to the chart
            fig.add_annotation(
                x=0.02, y=0.98,
                xref="paper", yref="paper",
                text="ℹ️ Train/Test split point not determinable",
                showarrow=False,
                font=dict(size=10, color="#64748b"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(203,213,224,0.5)",
                borderwidth=1
            )
    
    except Exception as e:
        # Silently handle any remaining errors
        pass

def _add_forecast_comparison(fig, forecast, settings):
    """Add forecast vs actual comparison to chart"""
    try:
        test_dates = pd.to_datetime(forecast['test_dates'])
        test_predicted = forecast['test_predicted']
        test_actual = forecast['test_actual']
        
        # Add forecast line
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=test_predicted,
            mode='lines+markers',
            name='🔮 Forecast',
            line=dict(
                color=f'rgba(255,140,0,{settings["forecast_transparency"]})', 
                width=4, 
                dash='dash'
            ),
            marker=dict(size=8, color='rgba(255,140,0,0.9)',
                       line=dict(width=2, color='white')),
            hovertemplate='<b>Forecast</b><br>%{y:.2f}<br>%{x}<extra></extra>'
        ))
        
        # Add actual test values
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=test_actual,
            mode='lines+markers',
            name='📈 Actual (Test)',
            line=dict(color='rgba(34,197,94,0.9)', width=3),
            marker=dict(size=8, color='rgba(34,197,94,0.9)',
                       line=dict(width=2, color='white')),
            hovertemplate='<b>Actual</b><br>%{y:.2f}<br>%{x}<extra></extra>'
        ))
    except Exception as e:
        st.warning(f"Could not add forecast comparison: {str(e)}")

def _add_confidence_intervals(fig, forecast, settings):
    """Add confidence intervals to chart"""
    try:
        test_dates = pd.to_datetime(forecast['test_dates'])
        transparency = settings['forecast_transparency']
        
        # Colors for different confidence levels
        ci_colors = {
            '95': f'rgba(255,140,0,{transparency*0.15})',
            '80': f'rgba(255,140,0,{transparency*0.25})',
            '50': f'rgba(255,140,0,{transparency*0.4})'
        }
        
        # Add confidence intervals (95%, 80%, 50%)
        ci_levels = ['95', '80', '50']
        
        for level in ci_levels:
            upper_key = f'ci_{level}_upper'
            lower_key = f'ci_{level}_lower'
            
            if upper_key in forecast and lower_key in forecast:
                # Add upper bound (invisible)
                fig.add_trace(go.Scatter(
                    x=test_dates,
                    y=forecast[upper_key],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Add lower bound with fill
                fig.add_trace(go.Scatter(
                    x=test_dates,
                    y=forecast[lower_key],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor=ci_colors[level],
                    name=f'{level}% Confidence',
                    hovertemplate=f'<b>{level}% CI</b><br>%{{y:.2f}}<extra></extra>'
                ))
    except Exception as e:
        # Silently handle missing confidence interval data
        pass

def _render_forecast_metrics(forecast, settings):
    """Render forecast performance metrics"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #e2e8f0 0%, #f8fafc 100%); 
                padding: 20px; border-radius: 15px; margin: 20px 0;
                border-left: 5px solid #3b82f6;">
        <h4 style="margin: 0 0 15px 0; color: #1e293b;">📊 Performance Metrics</h4>
    """, unsafe_allow_html=True)
    
    metrics = forecast.get('metrics', {})
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        mae_val = metrics.get('mae', 0)
        st.metric(
            "🎯 MAE", 
            f"{mae_val:.2f}",
            help="""
            **MAE - Mean Absolute Error**
            
            Average absolute difference between actual and predicted values.
            
            **Lower is better** 📉
            - Value 0 = Perfect predictions
            - In same units as original data
            
            Easy to interpret and not affected by extreme outliers.
            """
        )
    
    with col2:
        rmse_val = metrics.get('rmse', 0)
        st.metric(
            "📊 RMSE", 
            f"{rmse_val:.2f}",
            help="""
            **RMSE - Root Mean Square Error**
            
            Square root of average squared prediction errors.
            
            **Lower is better** 📉
            - More sensitive to large errors than MAE
            - In same units as original data
            
            Gives larger penalty to big prediction errors.
            """
        )
    
    with col3:
        if metrics.get('mape'):
            mape_value = metrics['mape']
            mape_delta = "🟢" if mape_value < 10 else "🟡" if mape_value < 20 else "🔴"
            st.metric(
                f"📈 MAPE", 
                f"{mape_value:.1f}%",
                delta=mape_delta,
                help="""
                **MAPE - Mean Absolute Percentage Error**
                
                Average percentage error of predictions.
                
                **Lower is better** 📉
                
                **Industry Standards:**
                - < 10% = EXCELLENT 🟢
                - 10-20% = GOOD 🟡  
                - 20-50% = FAIR 🟠
                - > 50% = POOR 🔴
                
                Easy to understand as percentage, scale-independent.
                """
            )
        else:
            st.metric("📈 MAPE", "N/A", help="MAPE not available due to zero values in data")
    
    with col4:
        if metrics.get('r2') is not None:
            r2_value = metrics['r2']
            r2_delta = "🟢" if r2_value > 0.8 else "🟡" if r2_value > 0.6 else "🔴"
            st.metric(
                f"📊 R²", 
                f"{r2_value:.3f}",
                delta=r2_delta,
                help="""
                **R² - Coefficient of Determination**
                
                Percentage of data variation explained by the model.
                
                **Higher is better** 📈
                - Range: -∞ to 1
                - 1.0 = Perfect model
                - 0.0 = No better than simple average
                
                **Quality Standards:**
                - > 0.8 = EXCELLENT 🟢
                - 0.6-0.8 = GOOD 🟡
                - 0.3-0.6 = FAIR 🟠  
                - < 0.3 = POOR 🔴
                """
            )
        else:
            st.metric("📊 R²", "N/A", help="R² not available for this model")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Additional metrics if detailed view enabled
    if settings['show_performance_details']:
        _render_detailed_forecast_metrics(metrics, forecast)

def _render_detailed_forecast_metrics(metrics, forecast):
    """Render detailed performance metrics"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%); 
                padding: 15px; border-radius: 10px; margin: 15px 0;
                border-left: 4px solid #8b5cf6;">
        <h5 style="margin: 0 0 10px 0; color: #374151;">📋 Additional Performance Metrics</h5>
    </div>
    """, unsafe_allow_html=True)
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        if metrics.get('directional_accuracy'):
            dir_acc = metrics['directional_accuracy']
            st.metric(
                "🎯 Direction Accuracy", 
                f"{dir_acc:.1f}%",
                help="""
                **Direction Accuracy**
                
                Percentage of correct trend direction predictions (up/down).
                
                **Higher is better** 📈
                - > 70% = EXCELLENT 🟢
                - 50-70% = GOOD 🟡
                - < 50% = POOR 🔴
                
                Important for trading/investment decisions.
                """
            )
        else:
            # Try to calculate directional accuracy if we have test data
            try:
                if 'test_actual' in forecast and 'test_predicted' in forecast:
                    actual = np.array(forecast['test_actual'])
                    predicted = np.array(forecast['test_predicted'])
                    if len(actual) > 1 and len(predicted) > 1:
                        actual_diff = np.diff(actual)
                        pred_diff = np.diff(predicted)
                        correct_direction = np.sum(np.sign(actual_diff) == np.sign(pred_diff))
                        dir_acc = (correct_direction / len(actual_diff)) * 100
                        st.metric("🎯 Direction Accuracy", f"{dir_acc:.1f}%")
                    else:
                        st.metric("🎯 Direction Accuracy", "N/A", help="Insufficient data")
                else:
                    st.metric("🎯 Direction Accuracy", "N/A")
            except:
                st.metric("🎯 Direction Accuracy", "N/A")
    
    with col6:
        if metrics.get('drift'):
            drift_val = metrics['drift']
            st.metric(
                "📈 Drift", 
                f"{drift_val:.4f}",
                help="""
                **Drift - Average Change per Period**
                
                - Positive (+) = Upward trend
                - Negative (-) = Downward trend  
                - Near 0 = Stable
                
                In same units as original data.
                """
            )
        else:
            # Calculate drift if we have test data
            try:
                if 'test_actual' in forecast:
                    actual = np.array(forecast['test_actual'])
                    if len(actual) > 1:
                        drift_val = np.mean(np.diff(actual))
                        st.metric("📈 Drift", f"{drift_val:.4f}")
                    else:
                        st.metric("📈 Drift", "N/A")
                else:
                    st.metric("📈 Drift", "N/A")
            except:
                st.metric("📈 Drift", "N/A")
    
    with col7:
        test_periods = len(forecast.get('test_dates', []))
        if test_periods > 0:
            st.metric(
                "⏱️ Test Periods", 
                f"{test_periods}",
                help="""
                **Test Periods**
                
                Number of time periods used to test model accuracy.
                
                - More periods = More reliable evaluation
                - Minimum 10-20 periods recommended
                - Typically 20% of total data
                """
            )
        else:
            # Try alternative ways to get test period count
            if 'test_actual' in forecast:
                test_periods = len(forecast['test_actual'])
                st.metric("⏱️ Test Periods", f"{test_periods}")
            else:
                st.metric("⏱️ Test Periods", "N/A")
    
    with col8:
        if metrics.get('mse'):
            mse_val = metrics['mse']
            st.metric(
                "📏 MSE", 
                f"{mse_val:.3f}",
                help="""
                **MSE - Mean Squared Error**
                
                Average of squared prediction errors.
                
                **Lower is better** 📉
                - In squared units of original data
                - More sensitive to outliers
                - Relationship: RMSE = √MSE
                """
            )
        elif metrics.get('rmse'):
            # Calculate MSE from RMSE if available
            rmse_val = metrics['rmse']
            mse_val = rmse_val ** 2
            st.metric("📏 MSE", f"{mse_val:.3f}", help="Calculated from RMSE")
        else:
            st.metric("📏 MSE", "N/A")
    
    st.markdown("</div>", unsafe_allow_html=True)

def _render_detailed_forecast_analysis(forecast, forecast_results):
    """Render detailed forecast analysis"""
    st.markdown("#### 🔍 Advanced Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Model Performance Assessment")
        _render_performance_assessment(forecast)
    
    with col2:
        st.markdown("#### 🚀 Improvement Recommendations")
        _render_forecast_recommendations(forecast)
    
    # Feature importance for ML models
    if forecast.get('feature_importance'):
        st.markdown("#### 🔍 Feature Importance Analysis")
        _render_feature_importance(forecast)

def _render_performance_assessment(forecast):
    """Render performance assessment"""
    metrics = forecast.get('metrics', {})
    
    mae = metrics.get('mae', float('inf'))
    mape = metrics.get('mape', float('inf'))
    r2 = metrics.get('r2', 0)
    
    # Performance assessment logic
    if mape and mape < 10:
        st.success("🟢 **Excellent Accuracy** - Very accurate predictions")
    elif mape and mape < 20:
        st.warning("🟡 **Good Accuracy** - Reasonably accurate predictions")
    elif mape:
        st.error("🔴 **Needs Improvement** - Consider alternative models")
    else:
        st.info("ℹ️ **MAPE Assessment Unavailable** - Using other metrics")
    
    if r2 is not None:
        if r2 > 0.8:
            st.success("🟢 **Excellent Model Fit** - High explanatory power")
        elif r2 > 0.6:
            st.warning("🟡 **Good Model Fit** - Acceptable explanatory power")
        elif r2 > 0:
            st.warning("🟠 **Moderate Model Fit** - Limited explanatory power")
        else:
            st.error("🔴 **Poor Model Fit** - Model struggles with data patterns")

def _render_forecast_recommendations(forecast):
    """Render forecast recommendations"""
    current_model = forecast.get('model_type', 'unknown')
    metrics = forecast.get('metrics', {})
    
    # Performance indicators
    mape = metrics.get('mape', float('inf'))
    r2 = metrics.get('r2', 0)
    
    poor_performance = (mape and mape > 20) or (r2 is not None and r2 < 0.3)
    good_performance = (mape and mape < 15) and (r2 is not None and r2 > 0.6)
    
    if good_performance:
        st.success("✅ **Model Performance:** Excellent!")
        st.info("🎯 Current model is suitable for production forecasting")
    elif poor_performance:
        st.warning("💡 **Improvement Suggestions:**")
        if current_model in ['naive', 'naive_drift']:
            st.info("• Try ETS or Random Forest for better accuracy")
        elif current_model == 'linear_regression':
            st.info("• Try Random Forest for non-linear patterns")
        elif current_model in ['random_forest', 'catboost']:
            st.info("• Try ETS for simpler, more interpretable model")
        
        st.info("🏆 Use model comparison to automatically find the best approach")
    else:
        st.info("🟡 **Moderate Performance** - Model shows acceptable results")
        st.info("🔄 Consider running model comparison for potential improvements")

def _render_feature_importance(forecast):
    """Render feature importance for ML models"""
    importance = forecast.get('feature_importance', {})
    
    if not importance:
        st.info("Feature importance data not available for this model")
        return
    
    # Create horizontal bar chart
    features = list(importance.keys())[:10]  # Top 10 features
    values = [importance[f] for f in features]
    
    fig_importance = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker=dict(
            color=values,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Importance")
        ),
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
    ))
    
    fig_importance.update_layout(
        title="🔍 Top 10 Most Important Features",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=400,
        plot_bgcolor='rgba(248,249,250,0.8)',
        paper_bgcolor='rgba(255,255,255,0.95)',
        font=dict(family="Inter", color='#2d3748', size=12),
        margin=dict(l=150, r=60, t=60, b=60)
    )
    
    st.plotly_chart(fig_importance, use_container_width=True)
    
    st.markdown("""
    **🔬 Feature Types Explained:**
    - **days_numeric**: Linear time trend component
    - **month_sin/cos**: Monthly seasonal patterns (cyclical)
    - **dow_sin/cos**: Day-of-week patterns (cyclical) 
    - **doy_sin/cos**: Day-of-year seasonal patterns (cyclical)
    - **lag_X**: Historical values from X periods ago
    - **rolling_X**: Moving averages over X periods
    - **month/quarter**: Direct seasonal indicators
    """)

# Helper functions for multi-category and model comparison (if needed in future)
def _render_multi_category_forecast_results(forecast_results, settings):
    """Render multi-category forecast results"""
    category_forecasts = forecast_results['category_forecasts']
    
    st.markdown("#### 🏷️ Category-wise Forecasts")
    
    # Category selector
    selected_category = st.selectbox(
        "📊 Select category to analyze:",
        options=list(category_forecasts.keys()),
        key="category_selector"
    )
    
    if selected_category and selected_category in category_forecasts:
        forecast = category_forecasts[selected_category]
        
        # Category-specific info
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 15px; border-radius: 10px; margin: 15px 0;">
            <h4 style="margin: 0; color: white;">🔮 Forecast: {selected_category}</h4>
            <p style="margin: 5px 0 0 0; color: rgba(255,255,255,0.9);">
                Model: {forecast.get('model_name', 'Unknown Model')}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create forecast chart for selected category
        fig = _create_enhanced_forecast_chart(forecast_results, forecast, settings)
        fig.update_layout(title=f"Forecast for {selected_category}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics for selected category
        _render_forecast_metrics(forecast, settings)