import streamlit as st
import pandas as pd
import time
from fe_utils.api_client import APIClient
import logging

def render_data_processing_tab(module):
    """Render the data processing tab - focused on preprocessing only"""
    st.header("🔧 Data Preprocessing")
    
    if st.session_state.current_data is None:
        st.warning("⚠️ Please load data first in the Data Input tab")
        return
    
    # Main preprocessing section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Before Preprocessing")
        _render_current_data_info()
    
    with col2:
        st.subheader("⚙️ Preprocessing Configuration")
        preprocessing_params = _render_preprocessing_config(module)
    
    # Show aggregated data if available
    if preprocessing_params.get('aggregated_data') is not None:
        st.markdown("---")
        _render_before_after_comparison(preprocessing_params)

def _render_current_data_info():
    """Display current data information"""
    df = st.session_state.current_data
    
    # Data preview
    st.dataframe(df.head(), use_container_width=True, height=200)
    
    # Basic info
    st.markdown(f"""
    <div class="info-box">
        📊 <strong>Shape:</strong> {df.shape[0]} rows × {df.shape[1]} columns<br>
        📋 <strong>Columns:</strong> {', '.join(df.columns.tolist())}
    </div>
    """, unsafe_allow_html=True)
    
    # Data types and missing values
    with st.expander("📋 Data Quality Summary", expanded=False):
        info_data = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing = df[col].isnull().sum()
            missing_pct = (missing / len(df)) * 100
            unique_count = df[col].nunique()
            
            info_data.append({
                'Column': col,
                'Type': dtype,
                'Missing': f"{missing} ({missing_pct:.1f}%)",
                'Unique': unique_count
            })
        
        info_df = pd.DataFrame(info_data)
        st.dataframe(info_df, use_container_width=True)

def _render_preprocessing_config(module):
    """Render preprocessing configuration"""
    columns = st.session_state.current_data.columns.tolist()
    agg_counter = st.session_state.get('aggregation_counter', 0)
    
    # Check if data is already aggregated
    is_aggregated = set(columns) == {'category', 'date', 'value'} or set(columns) == {'date', 'value'}
    
    if is_aggregated:
        st.success("✅ Data is already in processed format")
        return {'ready': True, 'aggregated_data': None}
    
    # Column selection based on module
    if module == "time_series":
        return _handle_time_series_preprocessing(columns, agg_counter)
    elif module == "customer":
        return _handle_customer_preprocessing(columns, agg_counter)
    
    return {'ready': False}

def _handle_time_series_preprocessing(columns, agg_counter):
    """Handle time series preprocessing configuration"""
    # Detect likely datetime and value columns
    datetime_candidates = [col for col in columns if any(keyword in col.lower() 
                          for keyword in ['date', 'time', 'day', 'month', 'year'])]
    numeric_candidates = [col for col in columns if 
                         pd.api.types.is_numeric_dtype(st.session_state.current_data[col])]
    
    # Column selection
    datetime_col = st.selectbox(
        "📅 Select Date/Time Column:", 
        columns,
        index=columns.index(datetime_candidates[0]) if datetime_candidates else 0,
        key=f"datetime_col_{agg_counter}",
        help="Select the column containing date/time values"
    )
    
    available_value_cols = [col for col in columns if col != datetime_col]
    value_col = st.selectbox(
        "📈 Select Value Column:", 
        available_value_cols,
        index=0,
        key=f"value_col_{agg_counter}",
        help="Select the column containing the values to analyze"
    )
    
    # Category analysis (optional)
    st.markdown("#### 🏷️ Category Analysis (Optional)")
    available_category_cols = [col for col in columns if col not in [datetime_col, value_col]]
    has_category_cols = len(available_category_cols) > 0
    
    enable_category = st.checkbox(
        "Enable multi-category analysis", 
        value=False,
        help="Analyze multiple time series by category/group",
        key=f"enable_category_{agg_counter}"
    )
    
    category_col = None
    if enable_category and has_category_cols:
        category_col = st.selectbox(
            "Select Category Column:", 
            available_category_cols,
            key=f"category_col_{agg_counter}",
            help="Select a column to group your time series by categories"
        )
        
        if category_col:
            unique_categories = st.session_state.current_data[category_col].unique()
            st.info(f"📋 Found {len(unique_categories)} categories: {', '.join(map(str, unique_categories[:5]))}" + 
                   (f" and {len(unique_categories)-5} more..." if len(unique_categories) > 5 else ""))
    
    # Data aggregation settings
    st.markdown("#### 🔄 Data Aggregation")
    needs_agg = st.checkbox(
        "Aggregate raw data?", 
        help="Check if your data has multiple records per time period that need to be aggregated",
        key=f"needs_agg_{agg_counter}"
    )
    
    preprocessing_params = {
        'date_col': datetime_col,
        'value_col': value_col,
        'category_col': category_col,
        'needs_agg': needs_agg,
        'ready': True,
        'aggregated_data': None
    }
    
    if needs_agg:
        agg_method = st.selectbox(
            "Aggregation Method:", 
            ["sum", "mean", "count", "median"],
            key=f"agg_method_{agg_counter}",
            help="Choose how to aggregate multiple values in the same time period"
        )
        freq = st.selectbox(
            "Frequency:", 
            ["D", "W", "M", "Q"],
            format_func=lambda x: {
                "D": "Daily", "W": "Weekly", 
                "M": "Monthly", "Q": "Quarterly"
            }[x],
            key=f"freq_{agg_counter}",
            help="Choose the time frequency for aggregation"
        )
        
        preprocessing_params.update({
            'agg_method': agg_method,
            'freq': freq
        })
        
        # Aggregation button
        st.markdown("---")
        if st.button("🔄 Apply Aggregation", type="primary", key=f"aggregate_btn_{agg_counter}"):
            aggregated_data = _handle_data_aggregation(preprocessing_params)
            if aggregated_data is not None:
                preprocessing_params['aggregated_data'] = aggregated_data
                return preprocessing_params
    
    return preprocessing_params

def _handle_customer_preprocessing(columns, agg_counter):
    """Handle customer analytics preprocessing configuration"""
    customer_col = st.selectbox(
        "👤 Select Customer ID Column:", 
        columns,
        key=f"customer_col_{agg_counter}",
        help="Select the column containing customer identifiers"
    )
    amount_col = st.selectbox(
        "💰 Select Amount Column:", 
        [col for col in columns if col != customer_col],
        key=f"amount_col_{agg_counter}",
        help="Select the column containing transaction amounts"
    )
    date_col = st.selectbox(
        "📅 Select Date Column:", 
        [col for col in columns if col not in [customer_col, amount_col]],
        key=f"date_col_customer_{agg_counter}",
        help="Select the column containing transaction dates"
    )
    
    return {
        'customer_col': customer_col,
        'amount_col': amount_col,
        'date_col': date_col,
        'ready': True,
        'aggregated_data': None
    }

def _handle_data_aggregation(preprocessing_params):
    """Handle data aggregation and return aggregated data"""
    with st.spinner("Aggregating data..."):
        payload = {
            "session_id": st.session_state.current_session_id,
            "datetime_col": preprocessing_params['date_col'],
            "value_col": preprocessing_params['value_col'],
            "agg_method": preprocessing_params['agg_method'],
            "freq": preprocessing_params['freq'],
            "category_col": preprocessing_params['category_col']
        }
        
        agg_data = APIClient.aggregate_data(payload)
        if agg_data:
            # Update session state
            st.session_state.current_session_id = agg_data['session_id']
            new_data = pd.DataFrame(agg_data['data'])
            st.session_state.current_data = new_data
            
            # Set flags
            st.session_state.data_just_aggregated = True
            st.session_state.aggregation_counter = st.session_state.get('aggregation_counter', 0) + 1
            
            st.success("✅ Data aggregated successfully!")
            return new_data
    
    return None

def _render_before_after_comparison(preprocessing_params):
    """Render before/after comparison of data"""
    if preprocessing_params.get('aggregated_data') is None:
        return
    
    st.subheader("📊 Before vs After Preprocessing")
    
    col_before, col_after = st.columns(2)
    
    # This would show the original data vs aggregated data
    # Implementation depends on how you want to store the original data
    
    with col_before:
        st.markdown("**📥 Original Data**")
        # Show original data sample
        
    with col_after:
        st.markdown("**📤 Processed Data**")
        aggregated_data = preprocessing_params['aggregated_data']
        st.dataframe(aggregated_data.head(), use_container_width=True)
        
        st.markdown(f"""
        <div class="success-box">
            📊 <strong>New Shape:</strong> {aggregated_data.shape[0]} rows × {aggregated_data.shape[1]} columns<br>
            📋 <strong>Columns:</strong> {', '.join(aggregated_data.columns.tolist())}
        </div>
        """, unsafe_allow_html=True)

# Store preprocessing parameters in session state for other tabs
def get_preprocessing_params():
    """Get preprocessing parameters from session state"""
    return st.session_state.get('preprocessing_params', {})

def set_preprocessing_params(params):
    """Set preprocessing parameters in session state"""
    st.session_state.preprocessing_params = params