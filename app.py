import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from datetime import datetime
import logging, sys
import time

# Import custom modules
from components.config import setup_page_config, load_custom_css
from components.data_input import render_data_input_tab
from components.data_processing import render_data_processing_tab
from components.analysis_results import render_analysis_results_tab
from components.forecast_results import render_forecast_results_tab
from fe_utils.api_client import APIClient
from fe_utils.session_state import initialize_session_state

# Setup logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def main():
    # Page setup
    setup_page_config()
    load_custom_css()
    
    # Initialize session state
    initialize_session_state()
    
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
    
    # Determine tabs based on analysis results
    if st.session_state.analysis_results and st.session_state.analysis_results.get('module') == 'time_series':
        if st.session_state.analysis_results.get('overall_forecast') or st.session_state.analysis_results.get('category_forecasts'):
            tabs = st.tabs(["📤 Data Input", "🔧 Data Processing", "📊 Analysis Results", "🔮 Forecast Results"])
            tab1, tab2, tab3, tab4 = tabs
        else:
            tabs = st.tabs(["📤 Data Input", "🔧 Data Processing", "📊 Analysis Results"])
            tab1, tab2, tab3 = tabs
            tab4 = None
    else:
        tabs = st.tabs(["📤 Data Input", "🔧 Data Processing", "📊 Analysis Results"])
        tab1, tab2, tab3 = tabs
        tab4 = None
    
    # Render tabs
    with tab1:
        render_data_input_tab()
    
    with tab2:
        render_data_processing_tab(module)
    
    with tab3:
        render_analysis_results_tab()
    
    if tab4:
        with tab4:
            render_forecast_results_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        📊 Data Analytics | Built with FastAPI & Streamlit | 
        <a href="#" style="color: #667eea;">Documentation</a> | 
        <a href="#" style="color: #667eea;">API Reference</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()