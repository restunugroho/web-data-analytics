import streamlit as st
from components.config import setup_page_config, load_custom_css
from fe_utils.session_state import initialize_session_state
import logging, sys

# Import tab modules
from components.data_input import render_data_input_tab
from components.data_processing import render_data_processing_tab
from components.analysis_tab import render_analysis_tab
from components.analysis_results import render_analysis_results_tab
from components.forecast_tab import render_forecast_tab


# Konfigurasi logging di awal app.py
def setup_logging():
    """Setup logging configuration untuk Streamlit"""
    # Set level logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),  # Output ke stdout
            logging.FileHandler('logs/app.log', mode='a')  # Juga ke file
        ]
    )
    
    # Set Streamlit logger
    streamlit_logger = logging.getLogger('streamlit')
    streamlit_logger.setLevel(logging.INFO)
    
    # Set logger untuk komponen kustom
    app_logger = logging.getLogger(__name__)
    app_logger.setLevel(logging.INFO)
    
    return app_logger

def main():
    """Main Streamlit application"""

    # Setup logging pertama kali
    logger = setup_logging()
    logger.info("Starting Data Analytics Platform")

    setup_page_config()
    load_custom_css()
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Data Analytics Platform</h1>
        <p style="color: rgba(255,255,255,0.9); text-align: center; margin: 0;">
            Professional-grade analytics with separated analysis and forecasting capabilities
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Module selection
    module = st.sidebar.selectbox(
        "📂 Select Analysis Module:",
        ["time_series", "customer"],
        format_func=lambda x: {
            "time_series": "📈 Time Series Analysis",
            "customer": "👥 Customer Analytics"
        }[x],
        help="Choose the type of analysis you want to perform"
    )
    
    # Main navigation tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📤 Data Input", 
        "🔧 Data Processing", 
        "📊 Analysis",
        "📈 Results", 
        "🔮 Forecast",
        "ℹ️ Help"
    ])
    
    with tab1:
        render_data_input_tab()
    
    with tab2:
        render_data_processing_tab(module)
    
    with tab3:
        render_analysis_tab(module)
    
    with tab4:
        render_analysis_results_tab()
    
    with tab5:
        render_forecast_tab()
    
    with tab6:
        render_help_tab(module)
    
    # Sidebar status
    _render_sidebar_status()

def render_help_tab(module):
    """Render help and documentation tab"""
    st.header("ℹ️ Help & Documentation")
    
    # Quick start guide
    st.subheader("🚀 Quick Start Guide")
    
    st.markdown("""
    <div class="info-box">
        <h4>📋 Step-by-Step Workflow</h4>
        <ol>
            <li><strong>📤 Data Input:</strong> Load your data (sample datasets or upload your own)</li>
            <li><strong>🔧 Data Processing:</strong> Configure preprocessing (column selection, aggregation)</li>
            <li><strong>📊 Analysis:</strong> Run comprehensive data analysis</li>
            <li><strong>📈 Results:</strong> View insights, charts, and key statistics</li>
            <li><strong>🔮 Forecast:</strong> Generate predictions (Time Series only)</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Workflow explanation
    st.subheader("🔄 Analysis vs Forecasting")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📊 Analysis Tab:**
        - Comprehensive data analysis
        - Statistical insights and patterns
        - Trend and seasonality detection
        - Interactive visualizations
        - No forecasting included
        """)
    
    with col2:
        st.markdown("""
        **🔮 Forecast Tab:**
        - Requires completed analysis first
        - Multiple forecasting models
        - Model comparison capabilities
        - Confidence intervals
        - Performance metrics
        """)
    
    # Module-specific help
    if module == "time_series":
        _render_time_series_help()
    elif module == "customer":
        _render_customer_help()
    
    # API structure explanation
    st.subheader("🔧 Technical Architecture")
    
    with st.expander("📡 API Endpoints Used", expanded=False):
        st.markdown("""
        **Analysis Endpoints:**
        - `/analyze/time-series` - Time series analysis without forecasting
        - `/analyze/customer` - Customer analytics and RFM segmentation
        
        **Forecasting Endpoints:**
        - `/forecast/available-models` - Get available forecasting models
        - `/forecast/single-model` - Generate forecast with single model
        - `/forecast/compare-models` - Compare multiple models and forecast
        
        **Legacy Endpoints:** (backward compatibility)
        - `/analyze` - Routes to module-specific analysis
        - `/compare-models` - Routes to forecast comparison
        """)
    
    # Troubleshooting
    st.subheader("🔧 Troubleshooting")
    
    with st.expander("❓ Common Issues & Solutions"):
        st.markdown("""
        **🚫 "No data loaded" error:**
        - Make sure to load data in the Data Input tab first
        - Check that your file format is supported (CSV, Excel)
        
        **📅 Date parsing issues:**
        - Ensure date columns are in recognizable formats
        - Common formats: YYYY-MM-DD, DD/MM/YYYY, MM-DD-YYYY
        
        **🔮 "No forecast data" warning:**
        - Complete analysis first in the Analysis tab
        - Forecasting is only available for Time Series module
        - Use the Forecast tab for prediction generation
        
        **📊 Poor model performance:**
        - Try different models using model comparison
        - Check data quality and preprocessing settings
        - Ensure sufficient data (minimum 50+ observations recommended)
        
        **🔄 "Analysis required" message:**
        - Analysis and forecasting are now separate steps
        - Complete analysis before attempting forecasting
        """)

def _render_time_series_help():
    """Render time series specific help"""
    st.subheader("📈 Time Series Analysis Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📊 Data Requirements:**
        - Date/time column (various formats supported)
        - Numeric value column
        - Optional: Category column for multi-series analysis
        - Minimum 50+ observations recommended
        """)
        
        st.markdown("""
        **🔄 Preprocessing Options:**
        - **Aggregation:** Combine multiple records per time period
        - **Frequency:** Daily, Weekly, Monthly, Quarterly
        - **Methods:** Sum, Mean, Count, Median
        """)
    
    with col2:
        st.markdown("""
        **📊 Analysis Features:**
        - Statistical analysis and trends
        - Seasonality detection and decomposition
        - Data quality assessment
        - Interactive visualizations
        """)
        
        st.markdown("""
        **🔮 Forecasting Models:**
        - **ETS:** Best for seasonal data
        - **Random Forest:** Good for complex patterns
        - **ARIMA:** Good for trend data
        - **Linear Regression:** Simple and interpretable
        - **Model Comparison:** Automatic best model selection
        """)

def _render_customer_help():
    """Render customer analytics specific help"""
    st.subheader("👥 Customer Analytics Guide")
    
    st.markdown("""
    **📊 RFM Analysis:**
    - **Recency:** How recently customer made a purchase
    - **Frequency:** How often customer makes purchases  
    - **Monetary:** How much customer spends
    
    **👥 Customer Segments:**
    - **Champions:** High value, frequent, recent customers
    - **Loyal Customers:** Regular customers with good value
    - **At Risk:** Previously good customers showing decline
    - **New Customers:** Recent first-time buyers
    - **Potential Loyalists:** Recent customers with good potential
    
    **📈 Analysis Output:**
    - Customer segmentation visualization
    - RFM score distributions
    - Segment characteristics and insights
    - Actionable recommendations per segment
    """)
    
    st.info("💡 **Note:** Forecasting is currently not available for Customer Analytics module")

def _render_sidebar_status():
    """Render sidebar status information"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Session Status")
    
    # Data status
    if st.session_state.current_data is not None:
        df = st.session_state.current_data
        st.sidebar.success(f"✅ Data Loaded: {df.shape[0]} × {df.shape[1]}")
        
        # Show data type
        columns = df.columns.tolist()
        if set(columns) == {'category', 'date', 'value'} or set(columns) == {'date', 'value'}:
            st.sidebar.info("🔄 Data: Preprocessed")
        else:
            st.sidebar.warning("⚙️ Data: Needs preprocessing")
    else:
        st.sidebar.error("❌ No data loaded")
    
    # Analysis status
    if st.session_state.analysis_results is not None:
        st.sidebar.success("✅ Analysis Complete")
        
        # Show analysis module
        module = st.session_state.analysis_results.get('module', 'unknown')
        st.sidebar.info(f"📊 Module: {module.title().replace('_', ' ')}")
    else:
        st.sidebar.info("📊 Analysis: Not run")
    
    # Forecast status
    if st.session_state.get('forecast_results'):
        st.sidebar.success("🔮 Forecast Available")
        
        # Show forecast details if available
        forecast_results = st.session_state.forecast_results
        if forecast_results.get('overall_forecast'):
            model_name = forecast_results['overall_forecast'].get('model_name', 'Unknown')
            st.sidebar.info(f"🤖 Model: {model_name}")
    elif st.session_state.analysis_results and st.session_state.analysis_results.get('module') == 'time_series':
        st.sidebar.warning("🔮 Forecast: Ready to generate")
    else:
        st.sidebar.info("🔮 Forecast: Not available")
    
    # Model comparison status
    if st.session_state.get('model_comparison'):
        st.sidebar.success("🏆 Model Comparison Done")
    
    # Session info
    if st.session_state.current_session_id:
        with st.sidebar.expander("🔍 Session Details", expanded=False):
            st.write(f"**Session ID:** {st.session_state.current_session_id}")
            if st.session_state.current_data is not None:
                st.write(f"**Data Shape:** {st.session_state.current_data.shape}")
                st.write(f"**Columns:** {', '.join(st.session_state.current_data.columns.tolist())}")
            
            # Show available results
            if st.session_state.analysis_results:
                st.write("**Analysis:** ✅ Available")
            if st.session_state.get('forecast_results'):
                st.write("**Forecast:** ✅ Available")

if __name__ == "__main__":
    main()