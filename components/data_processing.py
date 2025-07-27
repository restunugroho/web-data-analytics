import streamlit as st
import pandas as pd
import time
from fe_utils.api_client import APIClient
import logging

def render_data_processing_tab(module):
    """Render the data processing tab"""
    st.header("🔧 Data Processing")
    
    if st.session_state.current_data is None:
        st.warning("⚠️ Please load data first in the Data Input tab")
        return
    
    col1, col2 = st.columns(2)

    logging.info('render_data_processing_tab')
    logging.info(col1)
    logging.info(col2)
    with col1:
        analysis_params = _render_column_selection(module)
    logging.info('analysis')
    logging.info(analysis_params)
    with col2:
        _render_processing_summary(module, analysis_params)
    
    
   
    if analysis_params.get('ready', False):
        _render_analysis_section(module, analysis_params)

def _render_column_selection(module):
    """Render column selection based on module type"""
    st.subheader("📊 Column Selection")
    columns = st.session_state.current_data.columns.tolist()
    analysis_params = {}
    
    # TAMBAHAN: Check apakah baru saja selesai agregasi
    if st.session_state.get('data_just_aggregated', False):
        st.info("ℹ️ Data has been aggregated. Column structure updated.")
        # Reset flag setelah ditampilkan
        st.session_state.data_just_aggregated = False
    
    if module == "time_series":
        analysis_params = _handle_time_series_config(columns)
    elif module == "customer":
        analysis_params = _handle_customer_config(columns)
    
    return analysis_params

def _handle_time_series_config(columns):
    """Handle time series configuration"""

    logging.info('_handle_time_series_config')
    
    # TAMBAHAN: Buat unique key berdasarkan aggregation counter
    agg_counter = st.session_state.get('aggregation_counter', 0)
    
    # PERBAIKAN: Deteksi apakah data sudah teragregasi berdasarkan struktur kolom
    is_aggregated = set(columns) == {'category', 'date', 'value'} or set(columns) == {'date', 'value'}
    
    if is_aggregated:
        # Data sudah teragregasi, gunakan struktur standar
        if 'date' in columns:
            datetime_col_default = columns.index('date')
        else:
            # Cari kolom yang kemungkinan datetime
            datetime_candidates = [col for col in columns if any(keyword in col.lower() for keyword in ['date', 'time', 'day', 'month', 'year'])]
            datetime_col_default = columns.index(datetime_candidates[0]) if datetime_candidates else 0
            
        if 'value' in columns:
            value_col_default = columns.index('value')
        else:
            # Cari kolom numerik yang bukan datetime
            numeric_candidates = [col for col in columns if col != columns[datetime_col_default]]
            value_col_default = columns.index(numeric_candidates[0]) if numeric_candidates else 1
    else:
        datetime_col_default = 0
        value_col_default = 1
    
    datetime_col = st.selectbox(
        "📅 Select Date/Time Column:", 
        columns,
        index=datetime_col_default,
        key=f"datetime_col_{agg_counter}"
    )
    
    available_value_cols = [col for col in columns if col != datetime_col]
    if is_aggregated and 'value' in available_value_cols:
        value_col_default_idx = available_value_cols.index('value')
    else:
        value_col_default_idx = 0
        
    value_col = st.selectbox(
        "📈 Select Value Column:", 
        available_value_cols,
        index=value_col_default_idx,
        key=f"value_col_{agg_counter}"
    )
    
    # Category analysis
    st.markdown("### 🏷️ Category Analysis (Optional)")
    
    # PERBAIKAN: Tentukan default checkbox berdasarkan struktur data
    available_category_cols = [col for col in columns if col not in [datetime_col, value_col]]
    has_category_cols = len(available_category_cols) > 0
    
    # Default enable jika data teragregasi dan ada kolom 'category'
    default_enable_category = is_aggregated and 'category' in available_category_cols
    
    enable_category = st.checkbox(
        "Enable multi-category analysis", 
        value=default_enable_category,
        help="Analyze multiple time series by category/group",
        key=f"enable_category_{agg_counter}"
    )
    
    logging.info(datetime_col)
    logging.info(value_col)
    
    category_col = None
    if enable_category and has_category_cols:
        # PERBAIKAN: Untuk data teragregasi, default ke 'category' jika ada
        if is_aggregated and 'category' in available_category_cols:
            default_index = available_category_cols.index('category')
        elif st.session_state.get('updated_category_col') and st.session_state.updated_category_col in available_category_cols:
            default_index = available_category_cols.index(st.session_state.updated_category_col)
        else:
            default_index = 0
            
        category_col = st.selectbox(
            "Select Category Column:", 
            available_category_cols,
            index=default_index,
            help="Select a column to group your time series by categories",
            key=f"category_col_{agg_counter}"
        )
        
        if category_col:
            unique_categories = st.session_state.current_data[category_col].unique()
            st.info(f"📋 Found {len(unique_categories)} categories: {', '.join(map(str, unique_categories[:5]))}" + 
                   (f" and {len(unique_categories)-5} more..." if len(unique_categories) > 5 else ""))
    elif enable_category and not has_category_cols:
        st.warning("No available columns for category analysis.")
    
    # PERBAIKAN: Log hasil final
    logging.info(f"enable_category: {enable_category}")
    logging.info(f"category_col: {category_col}")
    
    # Data aggregation - hanya tampilkan jika data belum teragregasi
    if not is_aggregated:
        st.subheader("🔄 Data Aggregation")
        needs_agg = st.checkbox(
            "Need to aggregate raw data?", 
            help="Check if your data has multiple records per time period",
            key=f"needs_agg_{agg_counter}"
        )
    else:
        needs_agg = False
        st.subheader("✅ Data Already Aggregated")
        st.info("Data is already in aggregated format and ready for analysis.")
    
    
    analysis_params = {
        'date_col': datetime_col,
        'value_col': value_col,
        'category_col': category_col,
        'needs_agg': needs_agg,
        'ready': True
    }

    
    if needs_agg:
        agg_method = st.selectbox(
            "Aggregation Method:", 
            ["sum", "mean", "count", "median"],
            key=f"agg_method_{agg_counter}"
        )
        freq = st.selectbox(
            "Frequency:", 
            ["D", "W", "M", "Q"],
            format_func=lambda x: {
                "D": "Daily", "W": "Weekly", 
                "M": "Monthly", "Q": "Quarterly"
            }[x],
            key=f"freq_{agg_counter}"
        )
        
        analysis_params.update({
            'agg_method': agg_method,
            'freq': freq
        })
        
        if st.button("🔄 Aggregate Data", type="secondary", key=f"aggregate_btn_{agg_counter}"):
            _handle_data_aggregation(analysis_params)
    return analysis_params

def _handle_customer_config(columns):
    """Handle customer analytics configuration"""
    agg_counter = st.session_state.get('aggregation_counter', 0)
    
    customer_col = st.selectbox(
        "👤 Select Customer ID Column:", 
        columns,
        key=f"customer_col_{agg_counter}"
    )
    amount_col = st.selectbox(
        "💰 Select Amount Column:", 
        [col for col in columns if col != customer_col],
        key=f"amount_col_{agg_counter}"
    )
    date_col = st.selectbox(
        "📅 Select Date Column:", 
        [col for col in columns if col not in [customer_col, amount_col]],
        key=f"date_col_customer_{agg_counter}"
    )
    
    return {
        'customer_col': customer_col,
        'amount_col': amount_col,
        'date_col': date_col,
        'ready': True
    }

def _handle_data_aggregation(analysis_params):
    """Handle data aggregation"""
    logging.info('_handle_data_aggregation')
    with st.spinner("Aggregating data..."):
        payload = {
            "session_id": st.session_state.current_session_id,
            "datetime_col": analysis_params['date_col'],
            "value_col": analysis_params['value_col'],
            "agg_method": analysis_params['agg_method'],
            "freq": analysis_params['freq'],
            "category_col": analysis_params['category_col']
        }
        
        agg_data = APIClient.aggregate_data(payload)
        logging.info(agg_data)
        if agg_data:
            st.session_state.current_session_id = agg_data['session_id']
            st.session_state.current_data = pd.DataFrame(agg_data['data'])
            logging.info(st.session_state.current_data)
            
            # PERBAIKAN: Tidak perlu simpan updated_category_col karena sudah ditangani di logic deteksi
            
            # Set flags untuk handling setelah rerun
            st.session_state.data_just_aggregated = True
            
            # Increment counter untuk reset widget keys
            st.session_state.aggregation_counter = st.session_state.get('aggregation_counter', 0) + 1
            
            st.success("✅ Data aggregated successfully!")
            st.rerun()

def _render_processing_summary(module, analysis_params):
    """Render processing summary"""
    st.subheader("📋 Processing Summary")
    logging.info('_render_processing_summary')
    logging.info(analysis_params)
    
    if module == "time_series" and analysis_params.get('ready'):
        # PERBAIKAN: Langsung gunakan category_col dari analysis_params
        display_category_col = analysis_params.get('category_col')
            
        processing_info = f"""
        **Time Series Configuration:**
        - 📅 Date Column: `{analysis_params.get('date_col', 'Not selected')}`
        - 📈 Value Column: `{analysis_params.get('value_col', 'Not selected')}`
        - 🏷️ Category Column: `{display_category_col or 'None (Single series)'}`
        """
        if analysis_params.get('needs_agg') and analysis_params.get('agg_method'):
            processing_info += f"""
            - 🔄 Aggregation: `{analysis_params.get('agg_method', 'Not selected')}`
            - ⏰ Frequency: `{analysis_params.get('freq', 'Not selected')}`
            """
        
    elif module == "customer" and analysis_params.get('ready'):
        processing_info = f"""
        **Customer Analytics Configuration:**
        - 👤 Customer Column: `{analysis_params.get('customer_col', 'Not selected')}`
        - 💰 Amount Column: `{analysis_params.get('amount_col', 'Not selected')}`
        - 📅 Date Column: `{analysis_params.get('date_col', 'Not selected')}`
        """
    else:
        processing_info = "**Configuration:** Please select required columns"
    
    st.markdown(processing_info)
    
    if analysis_params.get('ready'):
        st.markdown('<div class="success-box">✅ Ready for analysis!</div>', unsafe_allow_html=True)

def _render_analysis_section(module, analysis_params):
    """Render analysis configuration and execution"""
    if module == "time_series":
        # PERBAIKAN: Tidak perlu update analysis_params dengan updated_category_col
        _render_time_series_analysis(analysis_params)
    else:
        _render_simple_analysis(module, analysis_params)

def _render_time_series_analysis(analysis_params):
    """Render time series analysis configuration"""
    st.markdown("### 🔮 Analysis Options")
    
    # Forecast toggle
    enable_forecast = st.checkbox(
        "🔮 Enable Forecasting Analysis", 
        value=True,
        help="Enable this to include forecast analysis with your time series. Disable for faster analysis."
    )
    
    analysis_params['enable_forecast'] = enable_forecast
    
    if enable_forecast:
        _render_model_selection(analysis_params)
        _render_model_comparison_option(analysis_params)
    
    # Analysis button
    button_text = _get_analysis_button_text(enable_forecast, analysis_params)
    
    if st.button(button_text, type="primary", use_container_width=True):
        _execute_analysis("time_series", analysis_params)

def _render_model_selection(analysis_params):
    """Render model selection interface"""
    st.markdown("### 📊 Forecasting Model Selection")
    
    model_category = st.radio(
        "Choose model category:",
        ["Machine Learning", "Statistical Methods"],
        help="Machine Learning models can capture complex patterns, while Statistical methods are more interpretable"
    )
    
    model_options, default_model = _get_model_options(model_category)
    
    model_type = st.selectbox(
        "Select forecasting model:",
        list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=list(model_options.keys()).index(default_model)
    )
    
    analysis_params['model_type'] = model_type
    analysis_params['model_category'] = model_category
    analysis_params['model_options'] = model_options
    
    # Show model description
    _show_model_description(model_type, model_options)

def _render_model_comparison_option(analysis_params):
    """Render model comparison options"""
    st.markdown("---")
    st.markdown("### ⚡ Quick Model Comparison")
    
    compare_models = st.checkbox(
        "🔄 Compare multiple models", 
        value=False,
        help="Run analysis with multiple models and compare their performance (takes longer)"
    )
    
    if compare_models:
        st.info("🔄 **Model Comparison Mode**: This will test multiple models and show you a comparison table.")
        
        model_category = analysis_params.get('model_category', 'Machine Learning')
        if model_category == "Machine Learning":
            comparison_models = ['linear_regression', 'random_forest', 'catboost']
        else:
            comparison_models = ['ets', 'arima', 'theta', 'moving_average', 'naive_drift']
        
        model_names = [analysis_params['model_options'][m] for m in comparison_models]
        st.write(f"**Models to compare:** {', '.join(model_names)}")
        
        analysis_params['compare_models'] = True
        analysis_params['comparison_models'] = comparison_models
    else:
        analysis_params['compare_models'] = False

def _render_simple_analysis(module, analysis_params):
    """Render simple analysis for non-time series modules"""
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        _execute_analysis(module, analysis_params)

def _get_model_options(model_category):
    """Get model options based on category"""
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
    
    return model_options, default_model

def _show_model_description(model_type, model_options):
    """Show detailed model description"""
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
        'ets': {
            'description': 'Exponential Smoothing with Trend and Seasonality',
            'pros': ['Excellent for seasonal data', 'Adaptive to changes', 'Interpretable'],
            'cons': ['Limited for complex patterns', 'Sensitive to outliers'],
            'best_for': 'Data with strong seasonal patterns and moderate trends'
        },
        'arima': {
            'description': 'Auto-Regressive Integrated Moving Average (simplified implementation)',
            'pros': ['Good for trend patterns', 'Statistically sound', 'Works with limited data'],
            'cons': ['Limited seasonality handling', 'Assumes stationarity'],
            'best_for': 'Data with clear trends but limited seasonal patterns'
        }
        # Add other model descriptions...
    }
    
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

def _get_analysis_button_text(enable_forecast, analysis_params):
    """Get appropriate button text for analysis"""
    if not enable_forecast:
        return "🚀 Run Analysis (No Forecast)"
    elif analysis_params.get('compare_models', False):
        return "🚀 Run Analysis & Compare Models"
    else:
        model_options = analysis_params.get('model_options', {})
        model_type = analysis_params.get('model_type', '')
        model_name = model_options.get(model_type, '')
        return f"🚀 Run Analysis with Forecast ({model_name})"

def _execute_analysis(module, analysis_params):
    """Execute the analysis"""
    if analysis_params.get('compare_models', False):
        _execute_model_comparison(module, analysis_params)
    else:
        _execute_single_analysis(module, analysis_params)

def _execute_model_comparison(module, analysis_params):
    """Execute model comparison analysis"""
    with st.spinner("🔄 Comparing multiple models... This will take a few moments."):
        payload = {
            "session_id": st.session_state.current_session_id,
            "module": module,
            "parameters": analysis_params
        }
        
        comparison_results = APIClient.compare_models(payload)
        if comparison_results:
            st.session_state.model_comparison = comparison_results
            st.success("✅ Model comparison completed!")
            
            # Show quick comparison results
            _show_quick_comparison_results(comparison_results, analysis_params)

def _execute_single_analysis(module, analysis_params):
    """Execute single model analysis"""
    with st.spinner("🔄 Running comprehensive analysis... This may take a moment."):
        payload = {
            "session_id": st.session_state.current_session_id,
            "module": module,
            "parameters": analysis_params
        }
        
        logging.info('_execute_single_analysis')
        logging.info(payload)

        results = APIClient.analyze_data(payload)
        if results:
            st.session_state.analysis_results = results
            
            success_msg = "✅ Analysis completed successfully!"
            if module == "time_series" and analysis_params.get('enable_forecast'):
                model_options = analysis_params.get('model_options', {})
                model_type = analysis_params.get('model_type', '')
                model_name = model_options.get(model_type, '')
                success_msg += f" Using {model_name} for forecasting."
            
            st.success(success_msg)
            st.balloons()
            st.info("📊 **Analysis complete!** Check the **Analysis Results** tab to view your insights and charts.")
            time.sleep(1)
            st.rerun()

def _show_quick_comparison_results(comparison_results, analysis_params):
    """Show quick comparison results and options to proceed"""
    comparison_table = comparison_results['comparison_table']
    successful_models = [m for m in comparison_table if m['status'] == 'success']
    
    if successful_models:
        # Display comparison table
        st.subheader("📊 Model Performance Comparison")
        
        import pandas as pd
        df_comparison = pd.DataFrame(successful_models)
        display_df = df_comparison[['rank', 'model', 'mae', 'rmse', 'r2', 'mape', 'directional_accuracy']].copy()
        
        # Format numeric columns
        numeric_cols = ['mae', 'rmse', 'r2', 'mape', 'directional_accuracy']
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if x is not None else "N/A")
        
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
        </div>
        """, unsafe_allow_html=True)
        
        # Buttons to proceed with full analysis
        st.markdown("---")
        col_auto, col_manual = st.columns(2)
        
        with col_auto:
            if st.button(f"🚀 Run Full Analysis with Best Model ({best_model['model_name']})", type="primary"):
                _run_full_analysis_with_model(best_model['model_key'], analysis_params)
        
        with col_manual:
            manual_model = st.selectbox(
                "Or choose a specific model:",
                options=[m['model_key'] for m in successful_models],
                format_func=lambda x: next(m['model'] for m in successful_models if m['model_key'] == x),
                key="manual_model_select"
            )
            
            if st.button("🎯 Run with Selected Model", type="secondary"):
                _run_full_analysis_with_model(manual_model, analysis_params)

def _run_full_analysis_with_model(model_key, analysis_params):
    """Run full analysis with selected model"""
    analysis_params['model_type'] = model_key
    analysis_params['compare_models'] = False
    
    payload = {
        "session_id": st.session_state.current_session_id,
        "module": "time_series",
        "parameters": analysis_params
    }

    logging.info('_run_full_analysis_with_model')
    logging.info(payload)
    
    with st.spinner("🔄 Running full analysis..."):
        results = APIClient.analyze_data(payload)
        if results:
            st.session_state.analysis_results = results
            st.success("✅ Full analysis completed!")
            st.balloons()
            st.info("📊 **Analysis complete!** Check the **Analysis Results** tab.")
            time.sleep(1)
            st.rerun()