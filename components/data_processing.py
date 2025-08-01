import streamlit as st
import pandas as pd
import time
import re
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
    st.session_state.original_data = df
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

def _render_category_filters(columns, agg_counter):
    """Render category filters section"""
    st.markdown("#### 🔍 Advanced Filtering (Optional)")
    st.markdown("Apply additional filters to focus your analysis on specific data segments.")
    
    # Identify categorical columns (include all non-numeric columns)
    categorical_candidates = []
    df = st.session_state.current_data
    
    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            unique_count = df[col].nunique()
            # Include all categorical columns but give preference to reasonable ones
            if unique_count > 1:  # At least 2 unique values
                categorical_candidates.append(col)
    
    if not categorical_candidates:
        st.info("ℹ️ No categorical columns available for additional filtering.")
        return {}
    
    # Enable category filtering
    enable_filters = st.checkbox(
        "🎯 Enable advanced data filtering", 
        value=False,
        help="Apply additional filters to focus analysis on specific categories or segments",
        key=f"enable_filters_{agg_counter}"
    )
    
    if not enable_filters:
        return {}
    
    filters = {}
    used_columns = []  # Track used columns to prevent duplicates
    
    # Allow up to 3 category filters
    num_filters = st.number_input(
        "Number of filters to apply:",
        min_value=1,
        max_value=3,
        value=1,
        key=f"num_filters_{agg_counter}",
        help="Select how many category columns you want to filter by (maximum 3)"
    )
    
    # Keep track of cumulative row count for progressive filtering
    current_row_count = len(df)
    
    for i in range(num_filters):
        with st.expander(f"📂 Filter {i+1}", expanded=True):
            # Available columns (excluding already used ones)
            available_columns = [col for col in categorical_candidates if col not in used_columns]
            
            if not available_columns:
                st.warning("⚠️ No more categorical columns available for filtering.")
                break
            
            # Column selection
            filter_col = st.selectbox(
                f"Select column for filter {i+1}:",
                available_columns,
                key=f"filter_col_{i}_{agg_counter}",
                help=f"Choose the categorical column for filter {i+1}"
            )
            
            if filter_col:
                used_columns.append(filter_col)
                
                # Get unique values from current dataset (considering previous filters)
                if i == 0:
                    # First filter uses original data
                    temp_df = df.copy()
                else:
                    # Subsequent filters use data after applying previous filters (simulated)
                    temp_df = df.copy()
                    for prev_col, prev_values in list(filters.items()):
                        temp_df = temp_df[temp_df[prev_col].astype(str).isin(prev_values)]
                
                unique_values = sorted(temp_df[filter_col].dropna().unique().astype(str))
                
                # Show warning if too many unique values
                if len(unique_values) > 100:
                    st.warning(f"⚠️ Column '{filter_col}' has {len(unique_values)} unique values. Consider using prefix/suffix filtering for better performance.")
                
                # Filter method selection
                filter_method = st.radio(
                    f"How would you like to select values for {filter_col}?",
                    ["Select specific values", "Select by prefix", "Select by suffix"],
                    key=f"filter_method_{i}_{agg_counter}",
                    help="Choose how to select values: pick specific ones, or all values starting/ending with certain text"
                )
                
                selected_values = []
                
                if filter_method == "Select specific values":
                    # Limit options if too many unique values
                    display_values = unique_values[:100] if len(unique_values) > 100 else unique_values
                    if len(unique_values) > 100:
                        st.info(f"ℹ️ Showing first 100 values out of {len(unique_values)}. Use prefix/suffix for more options.")
                    
                    selected_values = st.multiselect(
                        f"Choose values to include:",
                        display_values,
                        key=f"filter_values_{i}_{agg_counter}",
                        help=f"Select specific values from {filter_col} to include in your analysis"
                    )
                
                elif filter_method == "Select by prefix":
                    prefix = st.text_input(
                        f"Enter prefix (values starting with):",
                        key=f"filter_prefix_{i}_{agg_counter}",
                        help="Enter text that values should START with (e.g., 'Product_' to select 'Product_A', 'Product_B', etc.)"
                    )
                    
                    if prefix:
                        matching_values = [val for val in unique_values if str(val).startswith(prefix)]
                        if matching_values:
                            st.success(f"✅ Found {len(matching_values)} values starting with '{prefix}': {', '.join(matching_values[:5])}" + 
                                     (f" and {len(matching_values)-5} more..." if len(matching_values) > 5 else ""))
                            selected_values = matching_values
                        else:
                            st.warning(f"⚠️ No values found starting with '{prefix}'")
                
                elif filter_method == "Select by suffix":
                    suffix = st.text_input(
                        f"Enter suffix (values ending with):",
                        key=f"filter_suffix_{i}_{agg_counter}",
                        help="Enter text that values should END with (e.g., '_2024' to select 'Sales_2024', 'Revenue_2024', etc.)"
                    )
                    
                    if suffix:
                        matching_values = [val for val in unique_values if str(val).endswith(suffix)]
                        if matching_values:
                            st.success(f"✅ Found {len(matching_values)} values ending with '{suffix}': {', '.join(matching_values[:5])}" + 
                                     (f" and {len(matching_values)-5} more..." if len(matching_values) > 5 else ""))
                            selected_values = matching_values
                        else:
                            st.warning(f"⚠️ No values found ending with '{suffix}'")
                
                if selected_values:
                    filters[filter_col] = selected_values
                    
                    # Calculate progressive filter count (simulate without heavy computation)
                    if i == 0:
                        # First filter: calculate from original data
                        filtered_count = len(temp_df[temp_df[filter_col].astype(str).isin(selected_values)])
                        base_count = len(df)
                    else:
                        # Subsequent filters: estimate based on selectivity
                        unique_in_selection = len(selected_values)
                        total_unique = len(unique_values)
                        selectivity = unique_in_selection / total_unique if total_unique > 0 else 0
                        filtered_count = int(current_row_count * selectivity)
                        base_count = current_row_count
                    
                    current_row_count = filtered_count
                    
                    st.info(f"📊 This filter will include approximately {filtered_count:,} rows out of {base_count:,} available rows ({filtered_count/base_count*100:.1f}%)")
    
    # Show combined filter preview
    if filters:
        st.markdown("---")
        st.markdown("**🔍 Combined Filter Summary:**")
        
        # For final preview, actually calculate to ensure accuracy
        filtered_df = df.copy()
        for col, values in filters.items():
            filtered_df = filtered_df[filtered_df[col].astype(str).isin(values)]
        
        final_count = filtered_df.shape[0]
        original_count = df.shape[0]
        
        if final_count > 0:
            st.success(f"✅ Combined filters will result in {final_count:,} rows ({final_count/original_count*100:.1f}% of original data)")
            
            # Show sample of filtered data
            with st.expander("👀 Preview filtered data", expanded=False):
                st.dataframe(filtered_df.head(10), use_container_width=True)
        else:
            st.error("❌ Combined filters result in 0 rows. Please adjust your filter criteria.")
            return {}
    
    return filters

def _apply_category_filters(df, filters):
    """Apply category filters to dataframe"""
    if not filters:
        return df
    
    filtered_df = df.copy()
    
    for col, values in filters.items():
        filtered_df = filtered_df[filtered_df[col].astype(str).isin(values)]
    
    return filtered_df

def _render_preprocessing_config(module):
    """Render preprocessing configuration"""
    columns = st.session_state.current_data.columns.tolist()
    agg_counter = st.session_state.get('aggregation_counter', 0)
    
    # Check if data is already aggregated
    is_aggregated = set(columns) == {'category', 'date', 'value'} or set(columns) == {'date', 'value'}
    
    if is_aggregated:
        st.success("✅ Data is already in processed format")
        return {'ready': True, 'aggregated_data': None}
    
    # Column selection based on module (moved to top)
    if module == "time_series":
        return _handle_time_series_preprocessing(columns, agg_counter)
    elif module == "customer":
        return _handle_customer_preprocessing(columns, agg_counter)
    
    return {'ready': False}

def _handle_time_series_preprocessing(columns, agg_counter):
    """Handle time series preprocessing configuration"""
    df = st.session_state.current_data
    
    # Step 1: Column Mapping (Required)
    st.markdown("#### 📊 Step 1: Column Mapping")
    st.markdown("Map your data columns to the required fields for time series analysis.")
    
    # Detect likely datetime and value columns
    datetime_candidates = [col for col in columns if any(keyword in col.lower() 
                          for keyword in ['date', 'time', 'day', 'month', 'year'])]
    numeric_candidates = [col for col in columns if 
                         pd.api.types.is_numeric_dtype(df[col])]
    
    # Column selection in organized layout
    col_map1, col_map2 = st.columns(2)
    
    with col_map1:
        datetime_col = st.selectbox(
            "📅 Date/Time Column:", 
            columns,
            index=columns.index(datetime_candidates[0]) if datetime_candidates else 0,
            key=f"datetime_col_{agg_counter}",
            help="Select the column containing date/time values"
        )
    
    with col_map2:
        available_value_cols = [col for col in columns if col != datetime_col]
        value_col = st.selectbox(
            "📈 Value Column:", 
            available_value_cols,
            index=0,
            key=f"value_col_{agg_counter}",
            help="Select the column containing the values to analyze"
        )
    
    st.markdown("---")
    
    # Step 2: Category Analysis (Optional)
    st.markdown("#### 🏷️ Step 2: Category Analysis (Optional)")
    st.markdown("Group your time series by categories for multi-series analysis.")
    
    available_category_cols = [col for col in columns if col not in [datetime_col, value_col]]
    has_category_cols = len(available_category_cols) > 0
    
    enable_category = st.checkbox(
        "📊 Enable multi-category analysis", 
        value=False,
        help="Analyze multiple time series grouped by category/segment",
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
            unique_categories = df[category_col].unique()
            with st.container():
                st.success(f"📋 Found {len(unique_categories)} categories: {', '.join(map(str, unique_categories[:5]))}" + 
                          (f" and {len(unique_categories)-5} more..." if len(unique_categories) > 5 else ""))
    
    st.markdown("---")
    
    # Step 3: Advanced Filtering (Optional) - Now comes after column mapping
    category_filters = _render_category_filters(columns, agg_counter)
    
    # Add separator if filters were applied
    if category_filters:
        st.markdown("---")
    
    # Step 4: Data Aggregation Settings
    st.markdown("#### 🔄 Step 4: Data Aggregation Settings")
    st.markdown("Configure how to aggregate your raw data by time periods.")
    
    needs_agg = st.checkbox(
        "🔧 Aggregate raw data", 
        help="Check if your data has multiple records per time period that need to be combined",
        key=f"needs_agg_{agg_counter}"
    )
    
    preprocessing_params = {
        'date_col': datetime_col,
        'value_col': value_col,
        'category_col': category_col,
        'needs_agg': needs_agg,
        'category_filters': category_filters,
        'ready': True,
        'aggregated_data': None
    }
    
    if needs_agg:
        # Show current data context if filters applied
        if category_filters:
            filtered_df = _apply_category_filters(df, category_filters)
            if filtered_df.empty:
                st.error("❌ Category filters resulted in empty dataset. Please adjust your filters.")
                return {'ready': False}
            
            with st.container():
                st.info(f"📊 Aggregation will be applied to filtered data: {filtered_df.shape[0]:,} rows")
        
        # Aggregation configuration in clean, organized sections
        with st.container():
            st.markdown("**⚙️ Aggregation Configuration**")
            
            # Method selection
            agg_method = st.selectbox(
                "Aggregation Method:", 
                ["sum", "mean", "count", "median"],
                key=f"agg_method_{agg_counter}",
                help="Choose how to combine multiple values in the same time period",
                format_func=lambda x: {
                    "sum": "Sum - Add all values together",
                    "mean": "Average - Calculate mean of values", 
                    "count": "Count - Count number of records",
                    "median": "Median - Calculate median of values"
                }[x]
            )
            
            st.markdown("**📅 Time Frequency Settings**")
            
            # Time frequency configuration
            col_freq_num, col_freq_unit = st.columns([1, 2])
            
            with col_freq_num:
                freq_multiplier = st.number_input(
                    "Every:",
                    min_value=1,
                    max_value=1000,
                    value=1,
                    step=1,
                    key=f"freq_multiplier_{agg_counter}",
                    help="Number of time units (e.g., 2 for 'every 2 hours')"
                )
            
            with col_freq_unit:
                freq_unit = st.selectbox(
                    "Time Unit:", 
                    ["min", "H", "D", "W", "M", "Q"],
                    format_func=lambda x: {
                        "min": "Minute(s)", 
                        "H": "Hour(s)",
                        "D": "Day(s)", 
                        "W": "Week(s)", 
                        "M": "Month(s)", 
                        "Q": "Quarter(s)"
                    }[x],
                    key=f"freq_unit_{agg_counter}",
                    help="Choose the time unit for aggregation"
                )
        
        # Combine multiplier and unit to create pandas frequency string
        freq = f"{freq_multiplier}{freq_unit}"
        
        # Display the resulting configuration in a highlighted summary box
        freq_display = {
            "min": "minute(s)", 
            "H": "hour(s)",
            "D": "day(s)", 
            "W": "week(s)", 
            "M": "month(s)", 
            "Q": "quarter(s)"
        }[freq_unit]
        
        with st.container():
            st.markdown(f"""
            <div style="background-color: #e8f4fd; padding: 15px; border-radius: 8px; border-left: 4px solid #1f77b4; margin: 10px 0;">
                <h4 style="margin: 0 0 10px 0; color: #1f77b4;">📋 Aggregation Summary</h4>
                <strong>Method:</strong> {agg_method.title()}<br>
                <strong>Frequency:</strong> Every {freq_multiplier} {freq_display}<br>
                <strong>Categories:</strong> {'Yes (' + category_col + ')' if category_col else 'No'}<br>
                <strong>Filters:</strong> {len(category_filters) if category_filters else 0} applied
            </div>
            """, unsafe_allow_html=True)
        
        # Show additional helpful info based on frequency selection
        if freq_unit in ["min", "H"] and freq_multiplier <= 5:
            st.info("ℹ️ High-frequency aggregation selected. Ensure your data has sufficient time granularity.")
        elif freq_unit == "M" and freq_multiplier >= 6:
            st.info("ℹ️ Long-term aggregation selected. This will create fewer data points with broader time spans.")
        
        preprocessing_params.update({
            'agg_method': agg_method,
            'freq': freq
        })
        
        # Apply aggregation button section
        st.markdown("---")
        with st.container():
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                if st.button("🚀 Apply Preprocessing", type="primary", key=f"aggregate_btn_{agg_counter}", use_container_width=True):
                    aggregated_data = _handle_data_aggregation(preprocessing_params)
                    if aggregated_data is not None:
                        preprocessing_params['aggregated_data'] = aggregated_data
                        return preprocessing_params
    
    return preprocessing_params

def _handle_customer_preprocessing(columns, agg_counter):
    """Handle customer analytics preprocessing configuration"""
    df = st.session_state.current_data
    
    # Step 1: Column Mapping (Required)
    st.markdown("#### 📊 Step 1: Column Mapping")
    st.markdown("Map your data columns to the required fields for customer analytics.")
    
    # Organized column selection layout
    col_map1, col_map2, col_map3 = st.columns(3)
    
    with col_map1:
        customer_col = st.selectbox(
            "👤 Customer ID Column:", 
            columns,
            key=f"customer_col_{agg_counter}",
            help="Select the column containing customer identifiers"
        )
    
    with col_map2:
        amount_col = st.selectbox(
            "💰 Amount Column:", 
            [col for col in columns if col != customer_col],
            key=f"amount_col_{agg_counter}",
            help="Select the column containing transaction amounts"
        )
    
    with col_map3:
        date_col = st.selectbox(
            "📅 Date Column:", 
            [col for col in columns if col not in [customer_col, amount_col]],
            key=f"date_col_customer_{agg_counter}",
            help="Select the column containing transaction dates"
        )
    
    # Column mapping summary
    with st.container():
        st.markdown(f"""
        <div style="background-color: #e8f5e8; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745; margin: 10px 0;">
            <h4 style="margin: 0 0 10px 0; color: #28a745;">📋 Column Mapping Summary</h4>
            <strong>👤 Customer ID:</strong> <code>{customer_col}</code><br>
            <strong>💰 Amount:</strong> <code>{amount_col}</code><br>
            <strong>📅 Date:</strong> <code>{date_col}</code>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Step 2: Advanced Filtering (Optional) - Now comes after column mapping
    category_filters = _render_category_filters(columns, agg_counter)
    
    return {
        'customer_col': customer_col,
        'amount_col': amount_col,
        'date_col': date_col,
        'category_filters': category_filters,
        'ready': True,
        'aggregated_data': None
    }

def _handle_data_aggregation(preprocessing_params):
    """Handle data aggregation and return aggregated data"""
    with st.spinner("🔄 Processing your data..."):
        # Apply category filters to current data if any
        df = st.session_state.current_data
        if preprocessing_params.get('category_filters'):
            df = _apply_category_filters(df, preprocessing_params['category_filters'])
            # Temporarily update current_data for aggregation
            original_data = st.session_state.current_data
            st.session_state.current_data = df
        
        payload = {
            "session_id": st.session_state.current_session_id,
            "datetime_col": preprocessing_params['date_col'],
            "value_col": preprocessing_params['value_col'],
            "agg_method": preprocessing_params['agg_method'],
            "freq": preprocessing_params['freq'],
            "category_col": preprocessing_params['category_col']
        }
        
        agg_data = APIClient.aggregate_data(payload)
        
        # Restore original data if it was temporarily changed
        if preprocessing_params.get('category_filters'):
            st.session_state.current_data = original_data
        
        if agg_data:
            # Update session state
            st.session_state.current_session_id = agg_data['session_id']
            new_data = pd.DataFrame(agg_data['data'])
            st.session_state.current_data = new_data
            
            # Set flags
            st.session_state.data_just_aggregated = True
            st.session_state.aggregation_counter = st.session_state.get('aggregation_counter', 0) + 1
            
            # Parse frequency for success message
            freq = preprocessing_params['freq']
            freq_pattern = re.match(r'^(\d+)([a-zA-Z]+)$', freq)
            
            if freq_pattern:
                freq_multiplier = int(freq_pattern.group(1))
                freq_unit = freq_pattern.group(2)
            else:
                # Simple format fallback
                freq_multiplier = 1
                freq_unit = freq
            
            # Show success message with frequency info
            if freq_multiplier == 1:
                freq_display_success = {
                    "min": "minutely", 
                    "H": "hourly",
                    "D": "daily", 
                    "W": "weekly", 
                    "M": "monthly", 
                    "Q": "quarterly"
                }.get(freq_unit, freq)
            else:
                freq_unit_display = {
                    "min": "minute", 
                    "H": "hour",
                    "D": "day", 
                    "W": "week", 
                    "M": "month", 
                    "Q": "quarter"
                }.get(freq_unit, freq_unit)
                freq_display_success = f"every {freq_multiplier} {freq_unit_display}{'s' if freq_multiplier > 1 else ''}"
            
            filter_info = ""
            if preprocessing_params.get('category_filters'):
                filter_count = len(preprocessing_params['category_filters'])
                filter_info = f" with {filter_count} category filter{'s' if filter_count > 1 else ''}"
            
            st.success(f"✅ Data processed successfully! Aggregated to {freq_display_success} frequency{filter_info}")
            return new_data
    
    return None

def _render_before_after_comparison(preprocessing_params):
    """Render before/after comparison of data"""
    if preprocessing_params.get('aggregated_data') is None:
        return
    
    st.subheader("📊 Preprocessing Results")
    
    col_before, col_after = st.columns(2)
    
    # This would show the original data vs aggregated data
    # Implementation depends on how you want to store the original data
    
    with col_before:
        st.markdown("**📥 Original Data**")
        # Show original data sample
        original_data = st.session_state.original_data
        st.dataframe(original_data.head(), use_container_width=True, height=200)

        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 3px solid #6c757d;">
            📊 <strong>Shape:</strong> {original_data.shape[0]:,} rows × {original_data.shape[1]} columns<br>
            📋 <strong>Columns:</strong> {', '.join(original_data.columns.tolist())}
        </div>
        """, unsafe_allow_html=True)
        
    with col_after:
        st.markdown("**📤 Processed Data**")
        aggregated_data = preprocessing_params['aggregated_data']
        st.dataframe(aggregated_data.head(), use_container_width=True, height=200)
        
        st.markdown(f"""
        <div style="background-color: #d4edda; padding: 10px; border-radius: 5px; border-left: 3px solid #28a745;">
            📊 <strong>Shape:</strong> {aggregated_data.shape[0]:,} rows × {aggregated_data.shape[1]} columns<br>
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