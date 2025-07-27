import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
from fe_utils.session_state import clear_session

def render_analysis_results_tab():
    """Render the analysis results tab"""
    st.header("📊 Analysis Results")
    
    if st.session_state.analysis_results is None:
        st.info("🔍 Run analysis in the Data Processing tab to see results here")
        return
    
    results = st.session_state.analysis_results
    
    # Key insights section - more compact
    _render_key_insights(results)
    
    # Main visualization
    _render_main_visualization(results)
    
    # Statistics and analysis
    col1, col2 = st.columns(2)
    with col1:
        _render_key_statistics(results)
    with col2:
        _render_advanced_analysis(results)
    
    # Model comparison results if available
    if hasattr(st.session_state, 'model_comparison') and st.session_state.model_comparison:
        _render_model_comparison_results()
    
    # Export and actions
    _render_export_section()

def _render_key_insights(results):
    """Render key insights in a more compact format"""
    st.subheader("💡 Key Insights")
    
    # Create a more compact insight display
    insights = results.get('insights', [])
    
    # Show first 4 insights in a 2x2 grid
    if len(insights) >= 4:
        col1, col2 = st.columns(2)
        
        with col1:
            for i in range(0, min(4, len(insights)), 2):
                st.markdown(f"""
                <div class="insight-card">
                    <strong>📈 Insight {i+1}:</strong> {insights[i]}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            for i in range(1, min(4, len(insights)), 2):
                st.markdown(f"""
                <div class="insight-card">
                    <strong>📊 Insight {i+1}:</strong> {insights[i]}
                </div>
                """, unsafe_allow_html=True)
        
        # Show remaining insights in expandable section
        if len(insights) > 4:
            with st.expander(f"📋 View {len(insights) - 4} More Insights", expanded=False):
                for i in range(4, len(insights)):
                    st.markdown(f"""
                    <div class="insight-card">
                        <strong>📈 Insight {i+1}:</strong> {insights[i]}
                    </div>
                    """, unsafe_allow_html=True)
    else:
        # Show all insights if less than 4
        for i, insight in enumerate(insights):
            st.markdown(f"""
            <div class="insight-card">
                <strong>📈 Insight {i+1}:</strong> {insight}
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed seasonality insights for time series
    if results['module'] == 'time_series' and results.get('seasonality_insights'):
        with st.expander("🌊 Detailed Seasonality Analysis", expanded=False):
            for insight in results['seasonality_insights']:
                st.markdown(f"• {insight}")

def _render_main_visualization(results):
    """Render main visualization based on module type"""
    if results['module'] == 'time_series':
        _render_time_series_visualization(results)
    else:
        _render_other_visualization(results)

def _render_time_series_visualization(results):
    """Render time series visualization with interactive options"""
    st.subheader("📈 Interactive Visualization")
    
    # Category filters if available
    selected_categories = _render_category_filters(results)
    
    # Chart options
    chart_options = _render_chart_options(results)
    
    # Main chart
    if 'plot' in results:
        fig = _create_enhanced_chart(results, selected_categories, chart_options)
        st.plotly_chart(fig, use_container_width=True)
    
    # Decomposition charts
    if chart_options.get('show_decomposition'):
        _render_decomposition_charts(results, selected_categories)

def _render_category_filters(results):
    """Render category filter controls"""
    selected_categories = []
    
    if results.get('has_categories', False):
        st.markdown("""
        <div class="filter-container">
            <h4 style="margin-top: 0; color: #155724;">🏷️ Category Filters</h4>
        </div>
        """, unsafe_allow_html=True)
        
        categories_list = results.get('categories_list', [])
        
        col_filter1, col_filter2 = st.columns([3, 1])
        with col_filter1:
            selected_categories = st.multiselect(
                "Select categories to display:",
                options=categories_list,
                default=categories_list,
                help="Choose which categories to show in the chart."
            )
        
        with col_filter2:
            if st.button("Select All", use_container_width=True):
                selected_categories = categories_list
                st.rerun()
            
            if st.button("Clear All", use_container_width=True):
                selected_categories = []
                st.rerun()
    
    return selected_categories

def _render_chart_options(results):
    """Render chart option controls"""
    st.markdown("""
    <div class="decomposition-container">
        <h4 style="margin-top: 0; color: #e65100;">📊 Chart Options & Analysis Tools</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_decomposition = st.checkbox("📈 Show Decomposition Charts", value=False)
        show_trend_overlay = st.checkbox("📉 Show Trend Line", value=False)
    
    with col2:
        show_seasonal_overlay = st.checkbox("🌊 Show Seasonal Pattern", value=False)
        
        # Only show forecast option if forecast data exists (but don't show forecast here)
        has_forecast = results.get('overall_forecast') or results.get('category_forecasts')
        if has_forecast:
            st.markdown("*🔮 Forecast: Available in Forecast tab*")
        else:
            st.markdown("*🔮 Forecast: Not available*")
    
    with col3:
        # Additional options can go here
        pass
    
    return {
        'show_decomposition': show_decomposition,
        'show_trend_overlay': show_trend_overlay,
        'show_seasonal_overlay': show_seasonal_overlay
    }

def _create_enhanced_chart(results, selected_categories, chart_options):
    """Create enhanced chart with overlays"""
    fig_dict = json.loads(results['plot'])
    fig = go.Figure(fig_dict)
    
    # Apply category filter
    if results.get('has_categories', False) and selected_categories:
        filtered_data = []
        for trace in fig.data:
            if hasattr(trace, 'name') and trace.name in selected_categories:
                filtered_data.append(trace)
            elif not hasattr(trace, 'name'):
                filtered_data.append(trace)
        fig.data = filtered_data
    
    # Add overlays
    if chart_options.get('show_trend_overlay'):
        _add_trend_overlay(fig, results, selected_categories)
    
    if chart_options.get('show_seasonal_overlay'):
        _add_seasonal_overlay(fig, results, selected_categories)
    
    # Enhanced styling
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2E4057', size=12),
        title_font_size=18,
        title_font_color='#2E4057',
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)', title_font_color='#495057'),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', title_font_color='#495057'),
        legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='rgba(128,128,128,0.5)'),
        height=600,
        hovermode='x unified'
    )
    
    return fig

def _add_trend_overlay(fig, results, selected_categories):
    """Add trend overlay to chart"""
    decomp_data = results.get('overall_decomposition') or results.get('category_decompositions', {})
    
    if results.get('overall_decomposition'):
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(decomp_data['dates']),
            y=decomp_data['trend'],
            mode='lines',
            name='Trend',
            line=dict(color='rgba(128,128,128,0.8)', width=2, dash='dash'),
            hovertemplate='Trend: %{y:.2f}<extra></extra>'
        ))

def _add_seasonal_overlay(fig, results, selected_categories):
    """Add seasonal overlay to chart"""
    decomp_data = results.get('overall_decomposition') or results.get('category_decompositions', {})
    
    if results.get('overall_decomposition'):
        seasonal_offset = np.array(decomp_data['seasonal']) + np.mean(decomp_data['original'])
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(decomp_data['dates']),
            y=seasonal_offset,
            mode='lines',
            name='Seasonal Pattern',
            line=dict(color='rgba(169,169,169,0.7)', width=1, dash='dot'),
            hovertemplate='Seasonal: %{y:.2f}<extra></extra>'
        ))

def _render_decomposition_charts(results, selected_categories):
    """Render decomposition charts"""
    st.subheader("🔍 Time Series Decomposition")
    
    decomp_data = results.get('overall_decomposition')
    
    if decomp_data:
        dates = pd.to_datetime(decomp_data['dates'])
        
        from plotly.subplots import make_subplots
        
        fig_decomp = make_subplots(
            rows=3, cols=1,
            subplot_titles=('📈 Original Data', '📉 Trend Component', '🌊 Seasonal Component'),
            vertical_spacing=0.08,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Add traces
        fig_decomp.add_trace(
            go.Scatter(x=dates, y=decomp_data['original'], 
                     name='Original', line=dict(color='#3B82F6', width=2)),
            row=1, col=1
        )
        
        fig_decomp.add_trace(
            go.Scatter(x=dates, y=decomp_data['trend'], 
                     name='Trend', line=dict(color='#EF4444', width=2)),
            row=2, col=1
        )
        
        fig_decomp.add_trace(
            go.Scatter(x=dates, y=decomp_data['seasonal'], 
                     name='Seasonal', line=dict(color='#10B981', width=2)),
            row=3, col=1
        )
        
        fig_decomp.update_layout(
            height=650,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2E4057')
        )
        
        st.plotly_chart(fig_decomp, use_container_width=True)

def _render_other_visualization(results):
    """Render visualization for non-time series modules"""
    st.subheader("📈 Visualization")
    
    if 'plot' in results:
        fig_dict = json.loads(results['plot'])
        fig = go.Figure(fig_dict)
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2E4057', size=12),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

def _render_key_statistics(results):
    """Render key statistics section"""
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #2E4057; margin-top: 0;">📊 Key Statistics</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if results['module'] == 'time_series':
        _render_time_series_stats(results)
    elif results['module'] == 'customer':
        _render_customer_stats(results)

def _render_time_series_stats(results):
    """Render time series statistics"""
    stats = results['statistics']
    
    col1_1, col1_2 = st.columns(2)
    
    with col1_1:
        st.metric("📈 Average Value", f"{stats['mean']:.2f}")
        st.metric("📏 Range", f"{stats['max'] - stats['min']:.2f}")
    
    with col1_2:
        st.metric("📊 Std Deviation", f"{stats['std']:.2f}")
        
        trend_value = stats.get('trend_slope', 0)
        trend_direction = stats['trend'].title()
        trend_color = "🔺" if trend_value > 0 else "🔻" if trend_value < 0 else "➡️"
        
        st.metric(
            f"{trend_color} Trend", 
            f"{trend_direction}",
            delta=f"{trend_value:+.4f}/day"
        )

def _render_customer_stats(results):
    """Render customer analytics statistics"""
    segments = results['segments']
    segment_colors = {
        'Champions': '#28a745',
        'Loyal Customers': '#17a2b8',
        'Potential Loyalists': '#6f42c1',
        'New Customers': '#20c997',
        'At Risk': '#dc3545',
        'Others': '#6c757d'
    }
    
    st.write("**👥 Customer Segments:**")
    for segment, count in segments.items():
        color = segment_colors.get(segment, '#6c757d')
        total_customers = sum(segments.values())
        percentage = (count / total_customers * 100) if total_customers > 0 else 0
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {color}15 0%, {color}25 100%); 
                    border-left: 4px solid {color}; padding: 8px; margin: 4px 0; border-radius: 5px;">
            <strong style="color: {color};">{segment}:</strong> {count} customers ({percentage:.1f}%)
        </div>
        """, unsafe_allow_html=True)

def _render_advanced_analysis(results):
    """Render advanced analysis section"""
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #2E4057; margin-top: 0;">🔍 Advanced Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if results['module'] == 'time_series':
        _render_time_series_advanced(results)
    elif results['module'] == 'customer':
        _render_customer_advanced(results)

def _render_time_series_advanced(results):
    """Render advanced time series analysis"""
    stats = results['statistics']
    
    # Predictability analysis
    predictability = stats.get('predictability', 'Unknown')
    cv = stats.get('coefficient_variation', 0)
    
    pred_colors = {
        'Very High': "#4caf50", 'High': "#8bc34a", 
        'Moderate': "#ff9800", 'Low': "#ff5722"
    }
    pred_color = next((color for level, color in pred_colors.items() if level in predictability), "#9e9e9e")
    pred_icon = "🎯" if 'High' in predictability else "🎲" if 'Moderate' in predictability else "⚡"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {pred_color}15 0%, {pred_color}25 100%); 
                border: 2px solid {pred_color}; padding: 12px; margin: 8px 0; border-radius: 8px;">
        <strong>{pred_icon} Predictability:</strong> {predictability}<br>
        <small>CV: {cv:.3f}</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Seasonality and volatility
    seasonality_strength = results.get('seasonality_strength', 0)
    volatility = results.get('volatility', 0)
    
    seasonality_level = "High" if seasonality_strength > 0.3 else "Moderate" if seasonality_strength > 0.15 else "Low"
    seasonality_icon = "🔥" if seasonality_strength > 0.3 else "🔶" if seasonality_strength > 0.15 else "🔵"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
                border: 2px solid #2196f3; padding: 12px; margin: 8px 0; border-radius: 8px;">
        <strong>{seasonality_icon} Seasonality:</strong> {seasonality_level}<br>
        <small>Strength: {seasonality_strength:.3f}</small>
    </div>
    """, unsafe_allow_html=True)

def _render_customer_advanced(results):
    """Render advanced customer analytics"""
    rfm = results['rfm_summary']
    
    # RFM metrics with color coding
    recency_color = "#4caf50" if rfm['avg_recency'] < 30 else "#ff9800" if rfm['avg_recency'] < 90 else "#f44336"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {recency_color}15 0%, {recency_color}25 100%); 
                border: 2px solid {recency_color}; padding: 12px; margin: 8px 0; border-radius: 8px;">
        <strong>📅 Avg Recency:</strong> {rfm['avg_recency']:.1f} days
    </div>
    """, unsafe_allow_html=True)
    
    frequency_color = "#4caf50" if rfm['avg_frequency'] > 5 else "#ff9800" if rfm['avg_frequency'] > 2 else "#f44336"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {frequency_color}15 0%, {frequency_color}25 100%); 
                border: 2px solid {frequency_color}; padding: 12px; margin: 8px 0; border-radius: 8px;">
        <strong>🔄 Avg Frequency:</strong> {rfm['avg_frequency']:.1f} transactions
    </div>
    """, unsafe_allow_html=True)

def _render_model_comparison_results():
    """Render model comparison results if available"""
    st.subheader("🏆 Model Comparison Results")
    
    comparison_data = st.session_state.model_comparison
    comparison_table = comparison_data['comparison_table']
    successful_models = [m for m in comparison_table if m['status'] == 'success']
    
    if successful_models:
        # Create performance comparison chart
        df_comp = pd.DataFrame(successful_models)
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # MAE Comparison
            fig_mae = px.bar(
                df_comp, 
                x='model', 
                y='mae',
                title='MAE Comparison (Lower is Better)',
                color='mae',
                color_continuous_scale='RdYlGn_r'
            )
            fig_mae.update_layout(
                xaxis_tickangle=45,
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#2E4057')
            )
            st.plotly_chart(fig_mae, use_container_width=True)
        
        with col_chart2:
            # R² Comparison (if available)
            models_with_r2 = df_comp[df_comp['r2'].notna()]
            if not models_with_r2.empty:
                fig_r2 = px.bar(
                    models_with_r2, 
                    x='model', 
                    y='r2',
                    title='R² Comparison (Higher is Better)',
                    color='r2',
                    color_continuous_scale='RdYlGn'
                )
                fig_r2.update_layout(
                    xaxis_tickangle=45,
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#2E4057')
                )
                st.plotly_chart(fig_r2, use_container_width=True)
            else:
                st.info("R² comparison not available for these models")
        
        # Best model summary
        best_model = comparison_data['best_model']
        st.markdown(f"""
        <div class="success-box">
            🏆 <strong>Champion Model:</strong> {best_model['model_name']}<br>
            📈 <strong>Performance:</strong> MAE {best_model['mae']:.3f}, RMSE {best_model['rmse']:.3f}
            {f", R² {best_model['r2']:.3f}" if best_model.get('r2') is not None else ""}
        </div>
        """, unsafe_allow_html=True)
        
        # Clear comparison results button
        if st.button("🗑️ Clear Comparison Results", type="secondary"):
            if hasattr(st.session_state, 'model_comparison'):
                delattr(st.session_state, 'model_comparison')
            st.rerun()

def _render_export_section():
    """Render export and actions section"""
    st.markdown("---")
    st.subheader("📥 Export & Actions")
    
    col_export1, col_export2, col_export3 = st.columns(3)
    
    with col_export1:
        if st.button("📄 Generate PDF Report", use_container_width=True):
            st.markdown("""
            <div class="warning-box">
                🚧 PDF export feature coming soon!
            </div>
            """, unsafe_allow_html=True)
    
    with col_export2:
        if st.button("📊 Export to Excel", use_container_width=True):
            st.markdown("""
            <div class="warning-box">
                🚧 Excel export feature coming soon!
            </div>
            """, unsafe_allow_html=True)
    
    with col_export3:
        if st.button("🔄 Run New Analysis", use_container_width=True):
            clear_session()
            st.success("🔄 Session cleared! You can now load new data.")
            st.rerun()