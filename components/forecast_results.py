import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json

def render_forecast_results_tab():
    """Render the forecast results tab"""
    st.header("🔮 Forecast Results")
    
    if st.session_state.analysis_results is None:
        st.info("🔍 Run analysis in the Data Processing tab to see results here")
        return
    
    results = st.session_state.analysis_results
    
    # Check if forecast data exists
    if not (results.get('overall_forecast') or results.get('category_forecasts')):
        st.warning("⚠️ No forecast data available. Enable forecasting in the Data Processing tab.")
        return
    
    # Forecast settings section - only call once here
    settings = _render_forecast_settings()
    
    st.markdown("---")
    
    # Forecast visualization and metrics
    if results.get('overall_forecast'):
        _render_single_series_forecast(results, settings)
    elif results.get('category_forecasts'):
        _render_multi_category_forecast(results, settings)

def _render_forecast_settings():
    """Render forecast display settings"""
    st.markdown("""
    <div class="forecast-settings">
        <h3 style="margin-top: 0; color: #e65100;">⚙️ Forecast Display Settings</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col_settings1, col_settings2, col_settings3 = st.columns(3)
    
    with col_settings1:
        show_confidence_intervals = st.checkbox(
            "📊 Show Confidence Intervals", 
            value=True,
            key="show_confidence_intervals",
            help="Display 50%, 80%, and 95% confidence intervals"
        )
        
        show_actual_vs_predicted = st.checkbox(
            "📈 Show Actual vs Predicted", 
            value=True,
            key="show_actual_vs_predicted",
            help="Compare forecast with actual test values"
        )
    
    with col_settings2:
        forecast_transparency = st.slider(
            "Forecast Transparency",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            key="forecast_transparency",
            help="Adjust transparency of forecast visualization"
        )
        
        show_train_test_split = st.checkbox(
            "📊 Show Train/Test Split", 
            value=True,
            key="show_train_test_split",
            help="Display vertical line separating training and test data"
        )
    
    with col_settings3:
        chart_height = st.selectbox(
            "Chart Height",
            options=[500, 600, 700, 800],
            index=1,
            key="chart_height",
            help="Adjust chart height for better visibility"
        )
        
        show_performance_details = st.checkbox(
            "📋 Show Performance Details", 
            value=True,
            key="show_performance_details",
            help="Display detailed performance metrics and insights"
        )
    
    return {
        'show_confidence_intervals': show_confidence_intervals,
        'show_actual_vs_predicted': show_actual_vs_predicted,
        'forecast_transparency': forecast_transparency,
        'show_train_test_split': show_train_test_split,
        'chart_height': chart_height,
        'show_performance_details': show_performance_details
    }

def _render_single_series_forecast(results, settings):
    """Render single series forecast results"""
    forecast = results['overall_forecast']
    
    # Model information header
    st.markdown(f"""
    <div class="forecast-container">
        <h3 style="margin-top: 0;">🔮 Forecast Results - {forecast.get('model_name', 'Unknown Model')}</h3>
        <p><strong>Model Type:</strong> {forecast.get('model_type', 'unknown').title().replace('_', ' ')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create forecast visualization
    fig = _create_forecast_chart(results, forecast, settings)
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    _render_forecast_metrics(forecast, settings)
    
    # Feature importance for ML models
    if forecast.get('feature_importance') and settings['show_performance_details']:
        _render_feature_importance(forecast)
    
    # Performance insights
    if settings['show_performance_details']:
        _render_performance_insights(forecast, results)

def _render_multi_category_forecast(results, settings):
    """Render multi-category forecast results"""
    category_forecasts = results['category_forecasts']
    
    st.subheader("🏷️ Category-wise Forecasts")
    
    # Category selector
    selected_category = st.selectbox(
        "Select category to view:",
        options=list(category_forecasts.keys()),
        key="category_selector",
        help="Choose a category to view its forecast results"
    )
    
    if selected_category and selected_category in category_forecasts:
        forecast = category_forecasts[selected_category]
        
        # Create category-specific results display
        st.markdown(f"""
        <div class="forecast-container">
            <h4 style="margin-top: 0;">🔮 Forecast for {selected_category}</h4>
            <p><strong>Model:</strong> {forecast.get('model_name', 'Unknown Model')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create forecast chart for selected category
        fig = _create_category_forecast_chart(results, selected_category, forecast, settings)
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics for selected category
        _render_forecast_metrics(forecast, settings)

def _create_forecast_chart(results, forecast, settings):
    """Create comprehensive forecast chart"""
    # Start with the original plot
    fig_dict = json.loads(results['plot'])
    fig = go.Figure(fig_dict)
    
    # Add train/test split line
    if settings['show_train_test_split']:
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
    
    # Add forecast and actual test data
    if settings['show_actual_vs_predicted']:
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
                line=dict(color=f'rgba(255,165,0,{settings["forecast_transparency"]})', width=4, dash='dash'),
                marker=dict(size=6, color='orange'),
                hovertemplate='Forecast: %{y:.2f}<br>Date: %{x}<extra></extra>'
            ))
            
            # Add actual test values
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
    
    # Add confidence intervals
    if settings['show_confidence_intervals']:
        _add_confidence_intervals(fig, forecast, test_dates, settings)
    
    # Update layout
    fig.update_layout(
        height=settings['chart_height'],
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2E4057', size=12),
        title_font_size=18,
        title_font_color='#2E4057',
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)', title_font_color='#495057'),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', title_font_color='#495057'),
        legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='rgba(128,128,128,0.5)'),
        hovermode='x unified'
    )
    
    return fig

def _create_category_forecast_chart(results, selected_category, forecast, settings):
    """Create forecast chart for specific category"""
    # This would be similar to _create_forecast_chart but for category-specific data
    # Implementation would depend on how category data is structured in the results
    fig = go.Figure()
    
    # Add basic category data and forecast
    # Implementation details would depend on the actual data structure
    
    return fig

def _add_confidence_intervals(fig, forecast, test_dates, settings):
    """Add confidence intervals to the forecast chart"""
    try:
        transparency = settings['forecast_transparency']
        
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
            fillcolor=f'rgba(255,165,0,{transparency*0.1})',
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
            fillcolor=f'rgba(255,165,0,{transparency*0.2})',
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
            fillcolor=f'rgba(255,165,0,{transparency*0.4})',
            name='50% Confidence',
            hovertemplate='50% CI: %{y:.2f}<extra></extra>'
        ))
        
    except Exception as e:
        st.warning(f"Could not add confidence intervals: {e}")

def _render_forecast_metrics(forecast, settings):
    """Render comprehensive forecast performance metrics"""
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
            mape_color = "🟢" if mape_value < 10 else "🟡" if mape_value < 20 else "🔴"
            
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
            r2_color = "🟢" if r2_value > 0.8 else "🟡" if r2_value > 0.6 else "🟠" if r2_value > 0 else "🔴"
            
            st.metric(
                f"{r2_color} R²", 
                f"{r2_value:.3f}", 
                help="R-squared - explained variance (higher is better)"
            )
        else:
            st.metric("📈 R²", "N/A", help="R-squared not available")
    
    # Additional metrics if detailed view is enabled
    if settings['show_performance_details'] and (metrics.get('directional_accuracy') or metrics.get('mse')):
        col_metric5, col_metric6, col_metric7, col_metric8 = st.columns(4)
        
        with col_metric5:
            if metrics.get('mse'):
                st.metric("📏 MSE", f"{metrics['mse']:.2f}", help="Mean Squared Error")
        
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
            if metrics.get('drift'):
                st.metric("📈 Drift", f"{metrics['drift']:.4f}", help="Average change per period")
        
        with col_metric8:
            test_periods = len(forecast.get('test_dates', []))
            st.metric("⏱️ Test Periods", f"{test_periods}", help="Number of periods used for testing")

def _render_feature_importance(forecast):
    """Render feature importance chart for ML models"""
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

def _render_performance_insights(forecast, results):
    """Render detailed performance insights and recommendations"""
    st.markdown("---")
    st.subheader("💡 Model Performance Insights")
    
    metrics = forecast['metrics']
    stats = results['statistics']
    
    # Performance assessment
    performance_insights = []
    
    # MAE-based assessment
    mae = metrics['mae']
    data_mean = stats['mean']
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
    
    # Display insights
    for insight in performance_insights:
        if insight.startswith("🟢"):
            st.success(insight)
        elif insight.startswith("🔴"):
            st.error(insight)
        else:
            st.warning(insight)
    
    # Model-specific recommendations
    _render_model_recommendations(forecast, results, mae_ratio, performance_insights)

def _render_model_recommendations(forecast, results, mae_ratio, performance_insights):
    """Render model-specific recommendations"""
    st.markdown("**🚀 Performance-Based Recommendations:**")
    
    current_model = forecast.get('model_type', 'unknown')
    current_model_name = forecast.get('model_name', 'Current Model')
    
    # Recommendation logic based on performance
    recommendations = []
    
    # Check performance indicators
    poor_performance = (
        mae_ratio > 0.15 or 
        (forecast['metrics'].get('mape', 0) > 20) or 
        (forecast['metrics'].get('r2', 0) < 0.3)
    )
    
    good_performance = (
        mae_ratio < 0.1 and 
        (forecast['metrics'].get('mape', 100) < 15) and 
        (forecast['metrics'].get('r2', 0) > 0.6)
    )
    
    if good_performance:
        recommendations.append(f"✅ **{current_model_name}** shows excellent performance for your data")
        recommendations.append("🎯 Consider using this model for production forecasting")
    elif poor_performance:
        recommendations.append(f"⚠️ **{current_model_name}** shows suboptimal performance")
        
        # Data characteristics for recommendations
        data_volatility = results['statistics'].get('coefficient_variation', 0)
        has_strong_trend = abs(results['statistics'].get('trend_slope', 0)) > 0.01
        has_seasonality = results.get('seasonality_strength', 0) > 0.2
        
        # Model-specific recommendations
        if current_model in ['naive', 'naive_drift']:
            if has_seasonality:
                recommendations.append("🌊 Try **ETS** or **Moving Average** for better seasonal handling")
            elif has_strong_trend:
                recommendations.append("📈 Try **Random Forest** or **Linear Regression** for trend capture")
        elif current_model == 'linear_regression':
            if data_volatility > 0.3:
                recommendations.append("🌳 Try **Random Forest** or **CatBoost** for non-linear patterns")
        elif current_model in ['random_forest', 'catboost']:
            if forecast['metrics'].get('r2', 0) < 0.2:
                recommendations.append("📊 Try **ETS** or **ARIMA** - simpler models might work better")
    else:
        recommendations.append(f"🟡 **{current_model_name}** shows moderate performance")
        recommendations.append("🔄 **Run model comparison** to potentially find better alternatives")
    
    # Display recommendations
    for rec in recommendations:
        if rec.startswith("✅"):
            st.success(rec)
        elif rec.startswith("⚠️"):
            st.warning(rec)
        else:
            st.info(rec)