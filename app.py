import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from datetime import datetime
import io
import logging, sys

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
                
                # Tambahan: Category column untuk multi time series
                category_col = st.selectbox("🏷️ Select Category Column (Optional):", 
                                          ["None"] + [col for col in columns if col not in [datetime_col, value_col]],
                                          help="Optional: Select a category column for multi-line time series analysis")
                
                if category_col == "None":
                    category_col = None
                
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
                                "category_col": category_col  # Tambahan
                            }
                            response = requests.post(f"{API_BASE_URL}/aggregate-data", json=payload)
                            
                            if response.status_code == 200:
                                agg_data = response.json()
                                st.session_state.current_session_id = agg_data['session_id']
                                st.session_state.current_data = pd.DataFrame(agg_data['data'])
                                st.success("✅ Data aggregated successfully!")
                                # Update category_col untuk analisis berikutnya
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
                # Tambahan category_col ke parameters
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
                
                if st.button("🚀 Run Analysis", type="primary"):
                    with st.spinner("Running analysis..."):
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
                            st.success("✅ Analysis completed!")
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
        
        # Display insights with enhanced styling
        st.subheader("💡 Key Insights", help="AI-generated insights based on your data analysis")
        
        insight_container = st.container()
        with insight_container:
            for i, insight in enumerate(results['insights']):
                st.markdown(f"""
                <div class="info-box">
                    <strong>📈 Insight {i+1}:</strong> {insight}
                </div>
                """, unsafe_allow_html=True)
        
        # Enhanced visualization section for time series
        if results['module'] == 'time_series':
            st.subheader("📈 Interactive Visualization")
            
            # Category filter (if categories exist)
            if results.get('has_categories', False):
                st.markdown("""
                <div class="filter-container">
                    <h4 style="margin-top: 0; color: #155724;">🏷️ Category Filters</h4>
                </div>
                """, unsafe_allow_html=True)
                
                categories_list = results.get('categories_list', [])
                
                col_filter1, col_filter2 = st.columns([2, 1])
                with col_filter1:
                    selected_categories = st.multiselect(
                        "Select categories to display:",
                        options=categories_list,
                        default=categories_list,
                        help="Choose which categories to show in the chart"
                    )
                
                with col_filter2:
                    show_all_categories = st.checkbox(
                        "Show all categories", 
                        value=True,
                        help="Toggle to show/hide all categories at once"
                    )
                    
                    if show_all_categories:
                        selected_categories = categories_list
            else:
                selected_categories = []
            
            # Decomposition options
            st.markdown("""
            <div class="decomposition-container">
                <h4 style="margin-top: 0; color: #e65100;">📊 Time Series Decomposition</h4>
            </div>
            """, unsafe_allow_html=True)
            
            col_decomp1, col_decomp2, col_decomp3 = st.columns(3)
            
            with col_decomp1:
                show_decomposition = st.checkbox(
                    "Show Decomposition Charts", 
                    value=False,
                    help="Display separate charts for trend and seasonal components"
                )
            
            with col_decomp2:
                show_trend_overlay = st.checkbox(
                    "Show Trend Line on Main Chart", 
                    value=False,
                    help="Add gray trend line to main chart for comparison"
                )
            
            with col_decomp3:
                show_seasonal_overlay = st.checkbox(
                    "Show Seasonal Pattern on Main Chart", 
                    value=False,
                    help="Add gray seasonal line to main chart"
                )
            
            # Main chart
            if 'plot' in results:
                fig_dict = json.loads(results['plot'])
                fig = go.Figure(fig_dict)
                
                # Apply category filter
                if results.get('has_categories', False) and selected_categories:
                    # Remove traces not in selected categories
                    filtered_data = []
                    for trace in fig.data:
                        if hasattr(trace, 'name') and trace.name in selected_categories:
                            filtered_data.append(trace)
                        elif not hasattr(trace, 'name'):  # Handle cases without names
                            filtered_data.append(trace)
                    
                    fig.data = filtered_data
                
                # Add trend overlay if requested
                if show_trend_overlay:
                    decomp_data = results.get('overall_decomposition') or results.get('category_decompositions', {})
                    
                    if results.get('overall_decomposition'):
                        # Single series trend
                        fig.add_trace(go.Scatter(
                            x=pd.to_datetime(decomp_data['dates']),
                            y=decomp_data['trend'],
                            mode='lines',
                            name='Trend',
                            line=dict(color='rgba(128,128,128,0.8)', width=2, dash='dash'),
                            hovertemplate='Trend: %{y:.2f}<extra></extra>'
                        ))
                    elif selected_categories and results.get('category_decompositions'):
                        # Multi-series trend
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
                
                # Add seasonal overlay if requested
                if show_seasonal_overlay:
                    decomp_data = results.get('overall_decomposition') or results.get('category_decompositions', {})
                    
                    if results.get('overall_decomposition'):
                        # Single series seasonal (offset to show pattern)
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
                        # Multi-series seasonal
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
                        bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='rgba(128,128,128,0.5)',
                        borderwidth=1
                    ),
                    height=500
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
                                
                                # Create subplots for this category
                                from plotly.subplots import make_subplots
                                
                                fig_decomp = make_subplots(
                                    rows=3, cols=1,
                                    subplot_titles=('Original Data', 'Trend Component', 'Seasonal Component'),
                                    vertical_spacing=0.08,
                                    row_heights=[0.4, 0.3, 0.3]
                                )
                                
                                # Original data
                                fig_decomp.add_trace(
                                    go.Scatter(x=dates, y=cat_decomp['original'], 
                                             name='Original', line=dict(color='#3B82F6', width=2)),
                                    row=1, col=1
                                )
                                
                                # Trend
                                fig_decomp.add_trace(
                                    go.Scatter(x=dates, y=cat_decomp['trend'], 
                                             name='Trend', line=dict(color='#EF4444', width=2)),
                                    row=2, col=1
                                )
                                
                                # Seasonal
                                fig_decomp.add_trace(
                                    go.Scatter(x=dates, y=cat_decomp['seasonal'], 
                                             name='Seasonal', line=dict(color='#10B981', width=2)),
                                    row=3, col=1
                                )
                                
                                fig_decomp.update_layout(
                                    height=600,
                                    showlegend=False,
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='#2E4057')
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
                            subplot_titles=('Original Data', 'Trend Component', 'Seasonal Component'),
                            vertical_spacing=0.08,
                            row_heights=[0.4, 0.3, 0.3]
                        )
                        
                        # Original data
                        fig_decomp.add_trace(
                            go.Scatter(x=dates, y=decomp_data['original'], 
                                     name='Original', line=dict(color='#3B82F6', width=2)),
                            row=1, col=1
                        )
                        
                        # Trend
                        fig_decomp.add_trace(
                            go.Scatter(x=dates, y=decomp_data['trend'], 
                                     name='Trend', line=dict(color='#EF4444', width=2)),
                            row=2, col=1
                        )
                        
                        # Seasonal
                        fig_decomp.add_trace(
                            go.Scatter(x=dates, y=decomp_data['seasonal'], 
                                     name='Seasonal', line=dict(color='#10B981', width=2)),
                            row=3, col=1
                        )
                        
                        fig_decomp.update_layout(
                            height=600,
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
        
        # Enhanced detailed results section
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
                
                # Category breakdown if available
                if results.get('has_categories', False) and 'categories' in stats:
                    st.markdown("**🏷️ Category Performance:**")
                    
                    for cat, cat_stats in stats['categories'].items():
                        cat_trend_color = "🔺" if cat_stats['trend_slope'] > 0 else "🔻" if cat_stats['trend_slope'] < 0 else "➡️"
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                                    border-left: 4px solid #007bff; padding: 10px; margin: 5px 0; border-radius: 5px;">
                            <strong>{cat_trend_color} {cat}:</strong> {cat_stats['trend'].title()}<br>
                            <small>Mean: {cat_stats['mean']:.2f} | Slope: {cat_stats['trend_slope']:+.4f}/day</small>
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