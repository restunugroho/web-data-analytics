import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from datetime import datetime, timedelta

def render_forecast_results_tab():
    """Render the forecast results tab with improved UI"""
    st.header("🔮 Forecast Results")
    
    if st.session_state.analysis_results is None:
        st.info("🔍 Run analysis in the Data Processing tab to see results here")
        return
    
    results = st.session_state.analysis_results
    
    # Check if forecast data exists
    if not (results.get('overall_forecast') or results.get('category_forecasts')):
        st.warning("⚠️ No forecast data available. Enable forecasting in the Data Processing tab.")
        return
    
    # Main content with better layout
    if results.get('overall_forecast'):
        _render_single_series_forecast(results)
    elif results.get('category_forecasts'):
        _render_multi_category_forecast(results)

def _render_forecast_settings():
    """Render forecast display settings in sidebar or compact layout"""
    with st.container():
        st.markdown("""
        <div style="background: linear-gradient(90deg, #ff6b35, #f7931e); 
                    padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h4 style="margin: 0; color: white; text-align: center;">
                ⚙️ Forecast Display Settings
            </h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Compact 2-column layout for settings
        col_left, col_right = st.columns(2)
        
        with col_left:
            show_confidence_intervals = st.checkbox(
                "📊 Confidence Intervals", 
                value=True,
                key="show_confidence_intervals",
                help="""
                **Confidence Intervals**
                
                Shows uncertainty zones around predictions:
                • **50%** = 50% chance actual values fall in this zone (darkest)
                • **80%** = 80% chance actual values fall in this zone (medium)
                • **95%** = 95% chance actual values fall in this zone (lightest)
                
                **Interpretation:**
                - **Narrow zones** = More certain predictions
                - **Wide zones** = Less certain predictions
                - Higher percentage = wider zones
                
                Useful for understanding prediction reliability.
                """
            )
            
            show_actual_vs_predicted = st.checkbox(
                "📈 Actual vs Predicted", 
                value=True,
                key="show_actual_vs_predicted",
                help="""
                **Actual vs Predicted Comparison**
                
                Compares two lines:
                • **Green line (Actual)** = Real values that occurred
                • **Orange line (Forecast)** = Model predictions
                
                **What to expect:**
                - Lines **close together** = Accurate model
                - Lines **far apart** = Less accurate model
                - **Similar patterns** = Model captures trends well
                
                This is the easiest way to see how well the model performs.
                """
            )
            
            show_train_test_split = st.checkbox(
                "📊 Train/Test Split Line", 
                value=True,
                key="show_train_test_split",
                help="""
                **Train/Test Split Line**
                
                Red vertical line that separates:
                • **Left of line** = Training data (used to train model)
                • **Right of line** = Test data (used to evaluate model)
                
                **Why important:**
                - Model only "sees" data on the left while learning
                - Data on the right tests prediction accuracy
                - Ensures fair and objective evaluation
                
                Typically uses 80% data for training, 20% for testing.
                """
            )
        
        with col_right:
            forecast_transparency = st.slider(
                "Forecast Opacity",
                min_value=0.3,
                max_value=1.0,
                value=0.8,
                step=0.1,
                key="forecast_transparency"
            )
            
            chart_height = st.selectbox(
                "Chart Height",
                options=[400, 500, 600, 700],
                index=1,
                key="chart_height"
            )
            
            show_performance_details = st.checkbox(
                "📋 Detailed Metrics", 
                value=False,
                key="show_performance_details",
                help="""
                **Detailed Performance Metrics**
                
                Shows additional metrics for deep analysis:
                • **MSE** = Mean Squared Error (squared errors)
                • **Direction Accuracy** = Trend direction prediction accuracy
                • **Drift** = Average change per period
                • **Test Periods** = Number of periods tested
                • **Feature Importance** = Important factors for prediction
                • **Performance Insights** = Model improvement suggestions
                
                Useful for technical understanding of model performance.
                """
            )
    
    return {
        'show_confidence_intervals': show_confidence_intervals,
        'show_actual_vs_predicted': show_actual_vs_predicted,
        'forecast_transparency': forecast_transparency,
        'show_train_test_split': show_train_test_split,
        'chart_height': chart_height,
        'show_performance_details': show_performance_details
    }

def _render_single_series_forecast(results):
    """Render single series forecast results with improved layout"""
    forecast = results['overall_forecast']
    
    # Settings at the top in a compact container
    settings = _render_forecast_settings()
    
    # Model info in a nice card
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 15px; margin: 20px 0; 
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);">
        <h3 style="margin: 0; color: white; text-align: center;">
            🔮 {forecast.get('model_name', 'Forecast Model')}
        </h3>
        <p style="margin: 5px 0 0 0; color: rgba(255,255,255,0.9); text-align: center;">
            Model Type: {forecast.get('model_type', 'unknown').title().replace('_', ' ')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main forecast chart
    fig = _create_enhanced_forecast_chart(results, forecast, settings)
    st.plotly_chart(fig, use_container_width=True, key="main_forecast_chart")
    
    # Metrics in a clean row
    _render_compact_forecast_metrics(forecast, settings)
    
    # Conditional detailed sections - auto expand if enabled
    if settings['show_performance_details']:
        
        # Feature importance for ML models
        if forecast.get('feature_importance'):
            with st.expander("🔍 Feature Importance Analysis", expanded=True):
                _render_feature_importance(forecast)
        
        # Performance insights - always expanded when detailed metrics enabled
        with st.expander("💡 Detailed Performance Analysis", expanded=True):
            _render_performance_insights(forecast, results)

def _render_multi_category_forecast(results):
    """Render multi-category forecast results"""
    category_forecasts = results['category_forecasts']
    
    # Settings first
    settings = _render_forecast_settings()
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%); 
                padding: 15px; border-radius: 10px; margin: 20px 0;">
        <h3 style="margin: 0; color: white; text-align: center;">🏷️ Category-wise Forecasts</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Category selector
    selected_category = st.selectbox(
        "📊 Select category to analyze:",
        options=list(category_forecasts.keys()),
        key="category_selector",
        help="Choose a category to view its forecast results"
    )
    
    if selected_category and selected_category in category_forecasts:
        forecast = category_forecasts[selected_category]
        
        # Create category-specific results display
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
        fig = _create_category_forecast_chart(results, selected_category, forecast, settings)
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics for selected category
        _render_compact_forecast_metrics(forecast, settings)

def _create_enhanced_forecast_chart(results, forecast, settings):
    """Create enhanced forecast chart with better styling"""
    # Start with the original plot
    try:
        fig_dict = json.loads(results['plot'])
        fig = go.Figure(fig_dict)
    except:
        fig = go.Figure()
    
    # Enhanced styling
    fig.update_layout(
        height=settings['chart_height'],
        plot_bgcolor='rgba(248,249,250,0.8)',
        paper_bgcolor='rgba(255,255,255,0.95)',
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", 
                 color='#2d3748', size=12),
        title_font=dict(size=20, color='#2d3748', family="Inter"),
        margin=dict(l=60, r=60, t=80, b=60),
        xaxis=dict(
            gridcolor='rgba(203,213,224,0.5)',
            gridwidth=1,
            title_font=dict(color='#4a5568', size=14),
            tickfont=dict(color='#718096', size=11),
            showline=True,
            linecolor='rgba(203,213,224,0.8)'
        ),
        yaxis=dict(
            gridcolor='rgba(203,213,224,0.5)',
            gridwidth=1,
            title_font=dict(color='#4a5568', size=14),
            tickfont=dict(color='#718096', size=11),
            showline=True,
            linecolor='rgba(203,213,224,0.8)'
        ),
        legend=dict(
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='rgba(203,213,224,0.8)',
            borderwidth=1,
            font=dict(size=11),
            x=0.02, y=0.98,  # Position legend at top-left
            xanchor='left', yanchor='top'
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.95)",
            font_size=12,
            font_family="Inter"
        )
    )
    
    # Add train/test split line (with proper timestamp handling)
    if settings['show_train_test_split']:
        try:
            split_date = None
            
            # Method 1: Direct from forecast (handle string conversion properly)
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
            
            # Method 3: Calculate from results data
            if split_date is None and 'df' in results and results['df'] is not None:
                try:
                    df = results['df']
                    if len(df) > 0:
                        split_idx = int(len(df) * 0.8)
                        if split_idx < len(df):
                            if 'date' in df.columns:
                                split_date = pd.to_datetime(df['date'].iloc[split_idx], errors='coerce')
                            elif hasattr(df.index, 'dtype') and 'datetime' in str(df.index.dtype):
                                split_date = df.index[split_idx]
                except:
                    split_date = None
            
            # Method 4: Fallback - create approximate split date
            if split_date is None:
                try:
                    # Try to get any date from the chart data
                    if 'test_dates' in forecast and forecast['test_dates']:
                        test_dates = pd.to_datetime(forecast['test_dates'], errors='coerce')
                        if len(test_dates) > 0 and not pd.isna(test_dates[0]):
                            # Approximate split as one period before first test
                            split_date = test_dates[0] - pd.Timedelta(days=1)
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
            
        except Exception as e:
            # Silently handle errors - don't show error to users
            pass
    
    # Add forecast and actual test data
    if settings['show_actual_vs_predicted']:
        try:
            test_dates = pd.to_datetime(forecast['test_dates'])
            test_predicted = forecast['test_predicted']
            test_actual = forecast['test_actual']
            
            # Add forecast line with enhanced styling
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
                marker=dict(
                    size=8, 
                    color='rgba(255,140,0,0.9)',
                    line=dict(width=2, color='white')
                ),
                hovertemplate='<b>Forecast</b><br>%{y:.2f}<br>%{x}<extra></extra>'
            ))
            
            # Add actual test values with enhanced styling
            fig.add_trace(go.Scatter(
                x=test_dates,
                y=test_actual,
                mode='lines+markers',
                name='📈 Actual (Test)',
                line=dict(color='rgba(34,197,94,0.9)', width=3),
                marker=dict(
                    size=8, 
                    color='rgba(34,197,94,0.9)',
                    line=dict(width=2, color='white')
                ),
                hovertemplate='<b>Actual</b><br>%{y:.2f}<br>%{x}<extra></extra>'
            ))
            
        except Exception as e:
            st.warning(f"Could not add forecast comparison: {str(e)}")
    
    # Add confidence intervals with better styling
    if settings['show_confidence_intervals']:
        _add_enhanced_confidence_intervals(fig, forecast, settings)
    
    return fig

def _create_category_forecast_chart(results, selected_category, forecast, settings):
    """Create forecast chart for specific category"""
    # Similar to main chart but for category-specific data
    fig = _create_enhanced_forecast_chart(results, forecast, settings)
    fig.update_layout(title=f"Forecast for {selected_category}")
    return fig

def _add_enhanced_confidence_intervals(fig, forecast, settings):
    """Add enhanced confidence intervals to the forecast chart"""
    try:
        test_dates = pd.to_datetime(forecast['test_dates'])
        transparency = settings['forecast_transparency']
        
        # Colors for different confidence levels
        ci_colors = {
            '95': f'rgba(255,140,0,{transparency*0.15})',
            '80': f'rgba(255,140,0,{transparency*0.25})',
            '50': f'rgba(255,140,0,{transparency*0.4})'
        }
        
        # 95% CI (outermost)
        if 'ci_95_upper' in forecast and 'ci_95_lower' in forecast:
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
                fillcolor=ci_colors['95'],
                name='95% Confidence',
                hovertemplate='<b>95% CI</b><br>%{y:.2f}<extra></extra>'
            ))
        
        # 80% CI
        if 'ci_80_upper' in forecast and 'ci_80_lower' in forecast:
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
                fillcolor=ci_colors['80'],
                name='80% Confidence',
                hovertemplate='<b>80% CI</b><br>%{y:.2f}<extra></extra>'
            ))
        
        # 50% CI (innermost)
        if 'ci_50_upper' in forecast and 'ci_50_lower' in forecast:
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
                fillcolor=ci_colors['50'],
                name='50% Confidence',
                hovertemplate='<b>50% CI</b><br>%{y:.2f}<extra></extra>'
            ))
        
    except Exception as e:
        # Silently handle missing confidence interval data
        pass

def _render_compact_forecast_metrics(forecast, settings):
    """Render compact forecast performance metrics"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #e2e8f0 0%, #f8fafc 100%); 
                padding: 20px; border-radius: 15px; margin: 20px 0;
                border-left: 5px solid #3b82f6;">
        <h4 style="margin: 0 0 15px 0; color: #1e293b;">📊 Performance Metrics</h4>
    """, unsafe_allow_html=True)
    
    metrics = forecast['metrics']
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        mae_value = metrics['mae']
        st.metric(
            "🎯 MAE", 
            f"{mae_value:.2f}",
            help="""
            **MAE - Mean Absolute Error**
            
            **Formula:** MAE = Σ|Actual - Predicted| / n
            
            **Meaning:** Average absolute difference between actual and predicted values
            
            **Interpretation:**
            • **Value 0** = Perfect predictions (no error)
            • **LOWER is BETTER** 📉
            • In same units as original data
            
            **Example:** If MAE = 5 and data is in dollars, then average prediction error is $5
            
            **Advantage:** Easy to interpret, not affected by extreme outliers
            """
        )
    
    with col2:
        rmse_value = metrics['rmse']
        st.metric(
            "📊 RMSE", 
            f"{rmse_value:.2f}",
            help="""
            **RMSE - Root Mean Square Error**
            
            **Formula:** RMSE = √(Σ(Actual - Predicted)² / n)
            
            **Meaning:** Square root of average squared prediction errors
            
            **Interpretation:**
            • **Value 0** = Perfect predictions
            • **LOWER is BETTER** 📉
            • More sensitive to large errors
            • In same units as original data
            
            **RMSE vs MAE:**
            • If RMSE >> MAE = Some very large prediction errors exist
            • If RMSE ≈ MAE = Prediction errors are relatively consistent
            
            **Advantage:** Gives larger penalty to big errors
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
                
                **Formula:** MAPE = (Σ|Actual - Predicted| / |Actual|) / n × 100%
                
                **Meaning:** Average percentage error of predictions
                
                **Interpretation:**
                • **0%** = Perfect predictions
                • **LOWER is BETTER** 📉
                • In percentage (%) format
                
                **Industry Standards:**
                • **< 10%** = EXCELLENT accuracy 🟢
                • **10-20%** = GOOD accuracy 🟡  
                • **20-50%** = FAIR accuracy 🟠
                • **> 50%** = POOR accuracy 🔴
                
                **Advantage:** Easy to understand as percentage, scale-independent
                **Limitation:** Cannot be calculated if any actual values = 0
                """
            )
        else:
            st.metric("📈 MAPE", "N/A", help="""
            **MAPE not available**
            
            MAPE cannot be calculated because there are actual values = 0 in the data.
            Division by zero would cause mathematical error.
            
            **Alternative:** Use MAE or RMSE for accuracy evaluation.
            """)
    
    with col4:
        if metrics.get('r2') is not None:
            r2_value = metrics['r2']
            r2_delta = "🟢" if r2_value > 0.8 else "🟡" if r2_value > 0.6 else "🔴"
            st.metric(
                f"📊 R²", 
                f"{r2_value:.3f}",
                delta=r2_delta,
                help="""
                **R² - Coefficient of Determination (R-squared)**
                
                **Formula:** R² = 1 - (SS_res / SS_tot)
                - SS_res = Σ(Actual - Predicted)²
                - SS_tot = Σ(Actual - Mean)²
                
                **Meaning:** Percentage of data variation explained by the model
                
                **Interpretation:**
                • **Range:** -∞ to 1
                • **HIGHER is BETTER** 📈
                • **1.0** = Model explains 100% of data variation (perfect)
                • **0.0** = Model no better than simple average
                • **Negative** = Model worse than simple average
                
                **Quality Standards:**
                • **> 0.8** = EXCELLENT model 🟢
                • **0.6-0.8** = GOOD model 🟡
                • **0.3-0.6** = FAIR model 🟠  
                • **< 0.3** = POOR model 🔴
                
                **Advantage:** Shows how well model captures data patterns
                """
            )
        else:
            st.metric("📊 R²", "N/A", help="""
            **R² not available**
            
            R-squared cannot be calculated for this model.
            Usually because model doesn't provide required residual information.
            
            **Alternative:** Use MAE, RMSE, or MAPE for evaluation.
            """)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Additional metrics only if detailed view is enabled
    if settings['show_performance_details']:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%); 
                    padding: 15px; border-radius: 10px; margin: 15px 0;
                    border-left: 4px solid #8b5cf6;">
            <h5 style="margin: 0 0 10px 0; color: #374151;">📋 Additional Performance Metrics</h5>
        </div>
        """, unsafe_allow_html=True)
        
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            if metrics.get('mse'):
                mse_val = metrics['mse']
                st.metric("📏 MSE", f"{mse_val:.3f}", help="""
                **MSE - Mean Squared Error**
                
                **Formula:** MSE = Σ(Actual - Predicted)² / n
                
                **Meaning:** Average of squared prediction errors
                
                **Interpretation:**
                • **Value 0** = Perfect predictions
                • **LOWER is BETTER** 📉
                • In squared units of original data
                • Gives large penalty for big errors
                
                **Relationship:** RMSE = √MSE
                
                **Use case:** More sensitive to outliers, good for detecting large errors
                """)
            elif metrics.get('rmse'):
                # Calculate MSE from RMSE if available
                rmse_val = metrics['rmse']
                mse_val = rmse_val ** 2
                st.metric("📏 MSE", f"{mse_val:.3f}", help="""
                **MSE - Mean Squared Error (calculated from RMSE)**
                
                **Formula:** MSE = RMSE²
                
                **Meaning:** Average of squared prediction errors
                
                **Interpretation:**
                • **LOWER is BETTER** 📉  
                • Calculated from available RMSE
                • Shows error variability
                
                **Note:** This value is automatically calculated from RMSE
                """)
            else:
                st.metric("📏 MSE", "N/A", help="Mean Squared Error not available - no RMSE or MSE data found")
        
        with col6:
            if metrics.get('directional_accuracy'):
                dir_acc = metrics['directional_accuracy']
                dir_color = "Normal" if dir_acc > 60 else "Inverse"
                st.metric(
                    "🎯 Direction Accuracy", 
                    f"{dir_acc:.1f}%",
                    delta=f"{dir_color}",
                    help="""
                    **Direction Accuracy - Trend Direction Prediction**
                    
                    **Formula:** DA = (Correct directions / Total predictions) × 100%
                    
                    **Meaning:** Percentage of correct trend direction predictions (up/down)
                    
                    **Interpretation:**
                    • **100%** = Always correct in predicting direction
                    • **HIGHER is BETTER** 📈
                    • **> 70%** = EXCELLENT 🟢
                    • **50-70%** = GOOD 🟡
                    • **< 50%** = POOR 🔴
                    
                    **Example:** 
                    - If actual value goes up, does prediction also go up?
                    - If actual value goes down, does prediction also go down?
                    
                    **Use case:** Important for trading/investment and business planning
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
                            st.metric("🎯 Direction Accuracy", f"{dir_acc:.1f}%", 
                                    help="""
                                    **Direction Accuracy (automatically calculated)**
                                    
                                    Calculated from test data:
                                    • Compares actual vs predicted direction changes
                                    • **HIGHER is BETTER** 📈
                                    
                                    **Same interpretation as standard Direction Accuracy**
                                    """)
                        else:
                            st.metric("🎯 Direction Accuracy", "N/A", help="Insufficient test data to calculate direction accuracy (need minimum 2 periods)")
                    else:
                        st.metric("🎯 Direction Accuracy", "N/A", help="Test actual and predicted data not available for direction accuracy calculation")
                except:
                    st.metric("🎯 Direction Accuracy", "N/A", help="Cannot calculate direction accuracy due to data format issues")
        
        with col7:
            if metrics.get('drift'):
                drift_val = metrics['drift']
                st.metric("📈 Drift", f"{drift_val:.4f}", help="""
                **Drift - Average Shift**
                
                **Formula:** Drift = Σ(Value[t] - Value[t-1]) / n
                
                **Meaning:** Average change in value per time period
                
                **Interpretation:**
                • **Positive (+)** = Consistent upward trend
                • **Negative (-)** = Consistent downward trend  
                • **Near 0** = No clear trend
                • In same units as original data
                
                **Example:**
                - Drift = +5.2 → Average increase of 5.2 units per period
                - Drift = -2.1 → Average decrease of 2.1 units per period
                
                **Use case:** Understanding long-term trends in data
                """)
            else:
                # Calculate drift if we have test data
                try:
                    if 'test_actual' in forecast:
                        actual = np.array(forecast['test_actual'])
                        if len(actual) > 1:
                            drift_val = np.mean(np.diff(actual))
                            st.metric("📈 Drift", f"{drift_val:.4f}", help="""
                            **Drift (calculated from test data)**
                            
                            Calculated from actual value changes:
                            • **Positive** = Upward trend
                            • **Negative** = Downward trend
                            • **Near 0** = Stable
                            
                            **Note:** Calculated only from test periods
                            """)
                        else:
                            st.metric("📈 Drift", "N/A", help="Insufficient test data to calculate drift (need minimum 2 periods)")
                    else:
                        st.metric("📈 Drift", "N/A", help="Test data not available for drift calculation")
                except:
                    st.metric("📈 Drift", "N/A", help="Cannot calculate drift due to data format issues")
        
        with col8:
            test_periods = len(forecast.get('test_dates', []))
            if test_periods > 0:
                st.metric("⏱️ Test Periods", f"{test_periods}", help="""
                **Test Periods - Testing Duration**
                
                **Meaning:** Number of time periods used to test model accuracy
                
                **Interpretation:**
                • **More periods** = More reliable evaluation
                • **Minimum 10-20 periods** for good evaluation
                • Typically 20% of total data
                
                **Example:**
                - 30 periods = 30 days/months/years (depending on data)
                - More periods = Higher confidence in model performance
                
                **Use case:** Ensures model is tested with sufficient data
                """)
            else:
                # Try alternative ways to get test period count
                if 'test_actual' in forecast:
                    test_periods = len(forecast['test_actual'])
                    st.metric("⏱️ Test Periods", f"{test_periods}", help="Number of test periods (from actual data)")
                elif 'test_predicted' in forecast:
                    test_periods = len(forecast['test_predicted'])
                    st.metric("⏱️ Test Periods", f"{test_periods}", help="Number of test periods (from predicted data)")
                else:
                    st.metric("⏱️ Test Periods", "N/A", help="Test period count information not available")
        
        # Show available metrics for debugging
        if st.checkbox("🔍 Show Available Metrics", key="debug_metrics"):
            st.write("**Available metrics keys:**", list(metrics.keys()))
            st.write("**Available forecast keys:**", list(forecast.keys()))
            
            # Show sample of key metrics
            st.write("**Metrics values:**")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    st.write(f"- {key}: {value}")
                else:
                    st.write(f"- {key}: {type(value)} - {str(value)[:50]}...")
        
        st.markdown("</div>", unsafe_allow_html=True)

def _render_feature_importance(forecast):
    """Render feature importance chart for ML models"""
    importance = forecast['feature_importance']
    
    # Create horizontal bar chart for feature importance
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
    
    # Feature explanation
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

def _render_performance_insights(forecast, results):
    """Render detailed performance insights and recommendations"""
    metrics = forecast['metrics']
    stats = results['statistics']
    
    # Performance assessment
    mae = metrics['mae']
    data_mean = stats['mean']
    mae_ratio = mae / abs(data_mean) if abs(data_mean) > 0 else float('inf')
    
    # Create performance cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Accuracy Assessment")
        
        if mae_ratio < 0.05:
            st.success("🟢 **Excellent Accuracy** - MAE < 5% of data mean")
        elif mae_ratio < 0.15:
            st.warning("🟡 **Good Accuracy** - MAE 5-15% of data mean")
        else:
            st.error("🔴 **Needs Improvement** - MAE > 15% of data mean")
        
        # MAPE assessment
        if metrics.get('mape'):
            mape = metrics['mape']
            if mape < 10:
                st.success("🟢 **Low MAPE** - Very accurate forecasts")
            elif mape < 20:
                st.warning("🟡 **Moderate MAPE** - Reasonably accurate")
            else:
                st.error("🔴 **High MAPE** - Consider model alternatives")
    
    with col2:
        st.markdown("#### 📊 Model Fit Quality")
        
        if metrics.get('r2') is not None:
            r2 = metrics['r2']
            if r2 > 0.8:
                st.success("🟢 **Excellent Fit** - R² > 0.8")
            elif r2 > 0.6:
                st.warning("🟡 **Good Fit** - R² 0.6-0.8") 
            elif r2 > 0:
                st.warning("🟠 **Moderate Fit** - R² 0-0.6")
            else:
                st.error("🔴 **Poor Fit** - Negative R²")
        
        # Directional accuracy
        if metrics.get('directional_accuracy'):
            dir_acc = metrics['directional_accuracy']
            if dir_acc > 70:
                st.success(f"🎯 **Good Direction Prediction** - {dir_acc:.1f}%")
            else:
                st.warning(f"⚠️ **Moderate Direction Accuracy** - {dir_acc:.1f}%")
    
    # Recommendations
    st.markdown("#### 🚀 Improvement Recommendations")
    _render_model_recommendations(forecast, results, mae_ratio)

def _render_model_recommendations(forecast, results, mae_ratio):
    """Render model-specific recommendations"""
    current_model = forecast.get('model_type', 'unknown')
    current_model_name = forecast.get('model_name', 'Current Model')
    
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
        st.success(f"✅ **{current_model_name}** is performing excellently!")
        st.info("🎯 This model is ready for production forecasting")
    elif poor_performance:
        st.error(f"⚠️ **{current_model_name}** shows room for improvement")
        
        # Data-driven recommendations
        data_volatility = results['statistics'].get('coefficient_variation', 0)
        has_strong_trend = abs(results['statistics'].get('trend_slope', 0)) > 0.01
        has_seasonality = results.get('seasonality_strength', 0) > 0.2
        
        recommendations = []
        if current_model in ['naive', 'naive_drift']:
            if has_seasonality:
                recommendations.append("🌊 Try **ETS** or **Moving Average** for seasonality")
            elif has_strong_trend:
                recommendations.append("📈 Try **Random Forest** for trend patterns")
        elif current_model == 'linear_regression':
            if data_volatility > 0.3:
                recommendations.append("🌳 Try **Random Forest** for non-linear patterns")
        elif current_model in ['random_forest', 'catboost']:
            if forecast['metrics'].get('r2', 0) < 0.2:
                recommendations.append("📊 Try **ETS** - simpler might be better")
        
        for rec in recommendations:
            st.info(rec)
    else:
        st.warning(f"🟡 **{current_model_name}** shows moderate performance")
        st.info("🔄 Run model comparison to explore alternatives")