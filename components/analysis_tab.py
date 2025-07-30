import streamlit as st
import pandas as pd
import time
from fe_utils.api_client import APIClient
import logging

def render_analysis_tab(module):
    """Render the simplified analysis configuration and execution tab"""
    st.header("📊 Data Analysis")
    
    if st.session_state.current_data is None:
        st.warning("⚠️ Please load data first in the Data Input tab")
        return
    
    # Check if preprocessing is needed
    preprocessing_status = _check_preprocessing_status(module)
    
    if not preprocessing_status['ready']:
        st.warning("⚠️ Please complete data preprocessing first in the Data Processing tab")
        return
    
    # Analysis configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        analysis_params = _render_analysis_configuration(module, preprocessing_status)
    
    with col2:
        _render_analysis_summary(module, analysis_params)
    
    # Analysis execution
    if analysis_params.get('ready', False):
        _render_analysis_execution(module, analysis_params)

def _check_preprocessing_status(module):
    """Check if data preprocessing is complete"""
    columns = st.session_state.current_data.columns.tolist()
    
    if module == "time_series":
        # Check if data is in standard format
        is_preprocessed = (
            set(columns) == {'category', 'date', 'value'} or 
            set(columns) == {'date', 'value'}
        )
        
        if is_preprocessed:
            return {
                'ready': True,
                'type': 'time_series',
                'has_category': 'category' in columns,
                'date_col': 'date',
                'value_col': 'value',
                'category_col': 'category' if 'category' in columns else None
            }
        else:
            # Try to get from preprocessing params
            preprocessing_params = st.session_state.get('preprocessing_params', {})
            if preprocessing_params.get('ready'):
                return {
                    'ready': True,
                    'type': 'time_series',
                    'date_col': preprocessing_params.get('date_col'),
                    'value_col': preprocessing_params.get('value_col'),
                    'category_col': preprocessing_params.get('category_col'),
                    'has_category': preprocessing_params.get('category_col') is not None
                }
    
    elif module == "customer":
        preprocessing_params = st.session_state.get('preprocessing_params', {})
        if preprocessing_params.get('ready'):
            return {
                'ready': True,
                'type': 'customer',
                'customer_col': preprocessing_params.get('customer_col'),
                'amount_col': preprocessing_params.get('amount_col'),
                'date_col': preprocessing_params.get('date_col')
            }
    
    return {'ready': False}

def _render_analysis_configuration(module, preprocessing_status):
    """Render simplified analysis configuration section"""
    st.subheader("⚙️ Analysis Configuration")
    
    if module == "time_series":
        return _render_time_series_analysis_config(preprocessing_status)
    elif module == "customer":
        return _render_customer_analysis_config(preprocessing_status)
    
    return {'ready': False}

def _render_time_series_analysis_config(preprocessing_status):
    """Render simplified time series analysis configuration"""
    st.markdown("#### 📊 Time Series Analysis Settings")
    
    # Basic configuration
    config = {
        'date_col': preprocessing_status['date_col'],
        'value_col': preprocessing_status['value_col'],
        'category_col': preprocessing_status['category_col'],
        'ready': True
    }
    
    st.info("""
    📊 **Analysis will include:**
    - ✅ Trend analysis per category (if multi-category)
    - ✅ Seasonality detection per category
    - ✅ Category comparison statistics
    - ✅ Interactive charts with category-specific overlays
    """)
    
    return config

def _render_customer_analysis_config(preprocessing_status):
    """Render simplified customer analysis configuration"""
    st.markdown("#### 👥 Customer Analytics Settings")
    
    config = {
        'customer_col': preprocessing_status['customer_col'],
        'amount_col': preprocessing_status['amount_col'],
        'date_col': preprocessing_status['date_col'],
        'ready': True
    }
    
    st.info("""
    👥 **Analysis will include:**
    - ✅ RFM (Recency, Frequency, Monetary) analysis
    - ✅ Customer segmentation
    - ✅ Key customer metrics
    - ✅ Customer behavior insights
    """)
    
    return config

def _render_analysis_summary(module, analysis_params):
    """Render analysis configuration summary"""
    st.subheader("📋 Analysis Summary")
    
    if not analysis_params.get('ready'):
        st.info("Configure analysis settings to see summary")
        return
    
    if module == "time_series":
        summary_info = f"""
        **📊 Time Series Analysis Configuration:**
        - 📅 Date Column: `{analysis_params.get('date_col', 'Not set')}`
        - 📈 Value Column: `{analysis_params.get('value_col', 'Not set')}`
        - 🏷️ Category Column: `{analysis_params.get('category_col') or 'None (Single series)'}`
        - 📊 Analysis Type: `{'Multi-category' if analysis_params.get('category_col') else 'Single-series'}`
        """
    
    elif module == "customer":
        summary_info = f"""
        **👥 Customer Analytics Configuration:**
        - 👤 Customer Column: `{analysis_params.get('customer_col', 'Not set')}`
        - 💰 Amount Column: `{analysis_params.get('amount_col', 'Not set')}`
        - 📅 Date Column: `{analysis_params.get('date_col', 'Not set')}`
        - 📊 Analysis Type: `RFM Segmentation`
        """
    
    st.markdown(summary_info)
    
    if analysis_params.get('ready'):
        st.success("✅ Ready for analysis!")

def _render_analysis_execution(module, analysis_params):
    """Render simplified analysis execution section"""
    st.markdown("---")
    st.subheader("🚀 Execute Analysis")
    
    button_text = "🚀 Run Analysis"
    button_help = "Run comprehensive data analysis with trend, seasonality, and statistics"
    
    col_btn1, col_btn2 = st.columns([2, 1])
    
    with col_btn1:
        if st.button(button_text, type="primary", use_container_width=True, help=button_help):
            _execute_analysis(module, analysis_params)
    
    with col_btn2:
        if st.button("🔄 Reset", type="secondary", use_container_width=True, help="Reset configuration"):
            # Clear analysis configuration
            if 'preprocessing_params' in st.session_state:
                del st.session_state.preprocessing_params
            st.success("Configuration reset!")
            st.rerun()

def _execute_analysis(module, analysis_params):
    """Execute the simplified analysis"""
    with st.spinner("🔄 Running analysis... This may take a moment."):
        payload = {
            "session_id": st.session_state.current_session_id,
            "module": module,
            "parameters": analysis_params
        }
        
        # Use module-specific analysis method
        if module == "time_series":
            results = APIClient.analyze_time_series(payload)
        elif module == "customer":
            results = APIClient.analyze_customer(payload)
        else:
            st.error(f"❌ Unknown module: {module}")
            return
        
        if results:
            st.session_state.analysis_results = results
            
            st.success("✅ Analysis completed successfully!")
            st.balloons()
            st.info("📊 **Analysis complete!** Check the **Analysis Results** tab to view your insights and interactive charts.")
            
            time.sleep(1)
            st.rerun()