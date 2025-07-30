import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import logging
from fe_utils.session_state import clear_session

def render_analysis_results_tab():
    """Render the simplified analysis results tab - focused on trend, seasonality, statistics and chart"""
    st.header("📈 Analysis Results")
    
    if st.session_state.analysis_results is None:
        st.info("🔍 Run analysis in the Analysis tab to see results here")
        return
    
    results = st.session_state.analysis_results
    
    # Main visualization with interactive features
    _render_main_visualization(results)
    
    # Statistics and analysis in columns
    col1, col2 = st.columns(2)
    with col1:
        _render_key_statistics(results)
    with col2:
        _render_trend_seasonality_analysis(results)

def _render_main_visualization(results):
    """Render main visualization with interactive options"""
    st.subheader("📊 Interactive Visualization")
    
    if 'time_series' in results['module']:
        _render_time_series_visualization(results)
    else:
        _render_other_visualization(results)

def _render_time_series_visualization(results):
    """Render time series visualization with interactive options"""
    # Category filters if available
    selected_categories = _render_category_filters(results)
    
    # Chart options
    chart_options = _render_chart_options(results)
    
    # Main chart
    if 'plot' in results:
        fig = _create_enhanced_chart(results, selected_categories, chart_options)
        st.plotly_chart(fig, use_container_width=True)

def _render_category_filters(results):
    """Render category filter controls"""
    selected_categories = []
    
    # Check if we have category data
    has_categories = results.get('has_categories', False)
    categories_list = results.get('categories_list', [])
    
    # Also check if the original chart has multiple traces (indicating categories)
    if 'plot' in results:
        try:
            fig_dict = json.loads(results['plot'])
            chart_traces = fig_dict.get('data', [])
            if len(chart_traces) > 1:
                has_categories = True
                # Extract category names from trace names if not already available
                if not categories_list:
                    categories_list = [trace.get('name', f'Series {i+1}') for i, trace in enumerate(chart_traces) if trace.get('name')]
        except:
            pass
    
    if has_categories and categories_list:
        st.markdown("#### 🏷️ Category Filters")
        
        col_filter1, col_filter2, col_filter3 = st.columns([4, 1, 1])
        
        with col_filter1:
            selected_categories = st.multiselect(
                "Select categories to display:",
                options=categories_list,
                default=categories_list,
                help="Choose which categories to show in the chart.",
                key="category_filter"
            )
        
        with col_filter2:
            if st.button("✅ All", use_container_width=True, help="Select all categories"):
                st.session_state.category_filter = categories_list
                st.rerun()
        
        with col_filter3:
            if st.button("❌ None", use_container_width=True, help="Clear all selections"):
                st.session_state.category_filter = []
                st.rerun()
        
        # Show selected count
        if selected_categories:
            st.info(f"📊 Showing {len(selected_categories)} of {len(categories_list)} categories")
        else:
            st.warning("⚠️ No categories selected - chart will be empty")
    
    return selected_categories

def _render_chart_options(results):
    """Render chart option controls"""
    st.markdown("#### 📊 Chart Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_trend_overlay = st.checkbox("📉 Show Trend Line", value=False)
    
    with col2:
        show_seasonal_overlay = st.checkbox("🌊 Show Seasonal Pattern", value=False)
        
        # Chart styling options
        chart_height = st.selectbox(
            "Chart Height",
            options=[400, 500, 600, 700, 800],
            index=2,
            help="Select the height of the chart in pixels"
        )
    
    with col3:
        # Info about forecasting
        st.info("🔮 **Want forecasting?**\nGo to the **Forecast** tab to generate predictions!")
    
    return {
        'show_trend_overlay': show_trend_overlay,
        'show_seasonal_overlay': show_seasonal_overlay,
        'chart_height': chart_height
    }

def _create_enhanced_chart(results, selected_categories, chart_options):
    """Create enhanced chart with overlays and category filtering"""
    fig_dict = json.loads(results['plot'])
    fig = go.Figure(fig_dict)
    
    # Apply category filter if categories are selected
    if selected_categories:
        filtered_data = []
        for trace in fig.data:
            trace_name = getattr(trace, 'name', None)
            # Keep trace if it's in selected categories or if it has no name (might be a single series)
            if trace_name is None or trace_name in selected_categories:
                filtered_data.append(trace)
        fig.data = filtered_data
    
    # Add overlays if requested
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
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.2)', 
            title_font_color='#495057',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            gridcolor='rgba(128,128,128,0.2)', 
            title_font_color='#495057',
            showgrid=True,
            zeroline=False
        ),
        legend=dict(
            bgcolor='rgba(255,255,255,0.9)', 
            bordercolor='rgba(128,128,128,0.5)',
            borderwidth=1
        ),
        height=chart_options.get('chart_height', 600),
        hovermode='x unified',
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def _add_trend_overlay(fig, results, selected_categories):
    """Add trend overlay to chart - handles both single-series and multi-category"""
    analysis_type = results.get('analysis_type', 'single-series')
    
    if analysis_type == 'single-series':
        # Single series - use overall decomposition
        decomp_data = results.get('overall_decomposition')
        if decomp_data:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(decomp_data['dates']),
                y=decomp_data['trend'],
                mode='lines',
                name='📉 Overall Trend',
                line=dict(color='rgba(255,99,71,0.8)', width=3, dash='dash'),
                hovertemplate='<b>Overall Trend</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
            ))
    
    elif analysis_type == 'multi-category':
        # Multi-category - use category decompositions
        category_decompositions = results.get('category_decompositions', {})
        colors = ['rgba(255,99,71,0.8)', 'rgba(54,162,235,0.8)', 'rgba(255,206,86,0.8)', 
                 'rgba(75,192,192,0.8)', 'rgba(153,102,255,0.8)', 'rgba(255,159,64,0.8)']
        
        for i, (category, decomp_data) in enumerate(category_decompositions.items()):
            # Only show trend for selected categories
            if not selected_categories or category in selected_categories:
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(decomp_data['dates']),
                    y=decomp_data['trend'],
                    mode='lines',
                    name=f'📉 {category} Trend',
                    line=dict(color=color, width=2, dash='dash'),
                    hovertemplate=f'<b>{category} Trend</b><br>Date: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
                ))

def _add_seasonal_overlay(fig, results, selected_categories):
    """Add seasonal overlay to chart - handles both single-series and multi-category"""
    analysis_type = results.get('analysis_type', 'single-series')
    
    if analysis_type == 'single-series':
        # Single series - use overall decomposition
        decomp_data = results.get('overall_decomposition')
        if decomp_data:
            # Offset seasonal component to make it visible
            trend_mean = np.mean(decomp_data['trend']) if decomp_data.get('trend') else 0
            seasonal_offset = np.array(decomp_data['seasonal']) + trend_mean
            
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(decomp_data['dates']),
                y=seasonal_offset,
                mode='lines',
                name='🌊 Overall Seasonal',
                line=dict(color='rgba(50,205,50,0.7)', width=2, dash='dot'),
                hovertemplate='<b>Overall Seasonal Pattern</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
            ))
    
    elif analysis_type == 'multi-category':
        # Multi-category - use category decompositions
        category_decompositions = results.get('category_decompositions', {})
        colors = ['rgba(50,205,50,0.7)', 'rgba(30,144,255,0.7)', 'rgba(255,215,0,0.7)', 
                 'rgba(255,20,147,0.7)', 'rgba(138,43,226,0.7)', 'rgba(255,140,0,0.7)']
        
        for i, (category, decomp_data) in enumerate(category_decompositions.items()):
            # Only show seasonal for selected categories
            if not selected_categories or category in selected_categories:
                if decomp_data.get('trend') and decomp_data.get('seasonal'):
                    # Offset seasonal component to make it visible
                    trend_mean = np.mean(decomp_data['trend'])
                    seasonal_offset = np.array(decomp_data['seasonal']) + trend_mean
                    
                    color = colors[i % len(colors)]
                    fig.add_trace(go.Scatter(
                        x=pd.to_datetime(decomp_data['dates']),
                        y=seasonal_offset,
                        mode='lines',
                        name=f'🌊 {category} Seasonal',
                        line=dict(color=color, width=2, dash='dot'),
                        hovertemplate=f'<b>{category} Seasonal</b><br>Date: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
                    ))

def _render_other_visualization(results):
    """Render visualization for non-time series modules"""
    if 'plot' in results:
        fig_dict = json.loads(results['plot'])
        fig = go.Figure(fig_dict)
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2E4057', size=12),
            height=500,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)

def _render_key_statistics(results):
    """Render key statistics section"""
    st.subheader("📊 Key Statistics")
    
    if 'time_series' in results['module']:
        _render_time_series_stats(results)
    elif results['module'] == 'customer':
        _render_customer_stats(results)

def _render_time_series_stats(results):
    """Render time series statistics - handles both single-series and multi-category"""
    analysis_type = results.get('analysis_type', 'single-series')
    
    if analysis_type == 'single-series':
        _render_single_series_stats(results)
    elif analysis_type == 'multi-category':
        _render_multi_category_stats(results)

def _render_single_series_stats(results):
    """Render single series statistics as cards"""
    stats = results.get('statistics', {})
    
    # Create elegant cards for single series
    st.markdown("#### 📊 Statistical Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        mean_val = stats.get('mean', 0)
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
                    border-left: 4px solid #2196f3; padding: 15px; margin: 8px 0; border-radius: 8px;">
            <div style="font-size: 14px; color: #1976d2; font-weight: bold;">📈 Average Value</div>
            <div style="font-size: 24px; font-weight: bold; color: #0d47a1;">{mean_val:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        range_val = stats.get('max', 0) - stats.get('min', 0)
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); 
                    border-left: 4px solid #9c27b0; padding: 15px; margin: 8px 0; border-radius: 8px;">
            <div style="font-size: 14px; color: #7b1fa2; font-weight: bold;">📏 Range</div>
            <div style="font-size: 24px; font-weight: bold; color: #4a148c;">{range_val:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        std_val = stats.get('std', 0)
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); 
                    border-left: 4px solid #4caf50; padding: 15px; margin: 8px 0; border-radius: 8px;">
            <div style="font-size: 14px; color: #388e3c; font-weight: bold;">📊 Std Deviation</div>
            <div style="font-size: 24px; font-weight: bold; color: #1b5e20;">{std_val:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        data_points = stats.get('data_points', 0)
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); 
                    border-left: 4px solid #ff9800; padding: 15px; margin: 8px 0; border-radius: 8px;">
            <div style="font-size: 14px; color: #f57c00; font-weight: bold;">📊 Data Points</div>
            <div style="font-size: 24px; font-weight: bold; color: #e65100;">{data_points}</div>
        </div>
        """, unsafe_allow_html=True)

def _render_multi_category_stats(results):
    """Render multi-category statistics as comparison table"""
    # Get category statistics from ['statistics']['categories']
    statistics = results.get('statistics', {})
    category_stats = statistics.get('categories', {})
    
    if not category_stats:
        st.info("📊 Category statistics not available")
        return
    
    st.markdown("#### 📊 Category Comparison")
    
    # Create comparison DataFrame
    comparison_data = []
    for category, stats in category_stats.items():
        comparison_data.append({
            'Category': category,
            'Mean': f"{stats.get('mean', 0):.2f}",
            'Std Dev': f"{stats.get('std', 0):.2f}",
            'Range': f"{(stats.get('max', 0) - stats.get('min', 0)):.2f}",
            'Data Points': stats.get('data_points', 0)
        })
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        
        # Style the table
        st.markdown("""
        <style>
        .comparison-table {
            background: linear-gradient(135deg, #fafafa 0%, #f5f5f5 100%);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="comparison-table">', unsafe_allow_html=True)
            
            # Create elegant table display
            for i, row in df_comparison.iterrows():
                colors = ['#2196f3', '#4caf50', '#ff9800', '#9c27b0', '#f44336', '#00bcd4']
                color = colors[i % len(colors)]
                
                cols = st.columns([2, 1, 1, 1, 1])
                
                with cols[0]:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {color}15 0%, {color}25 100%); 
                                border-left: 4px solid {color}; padding: 10px; border-radius: 5px; margin: 2px 0;">
                        <strong style="color: {color};">{row['Category']}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols[1]:
                    st.markdown(f"**{row['Mean']}**<br><small>Mean</small>", unsafe_allow_html=True)
                
                with cols[2]:
                    st.markdown(f"**{row['Std Dev']}**<br><small>Std Dev</small>", unsafe_allow_html=True)
                
                with cols[3]:
                    st.markdown(f"**{row['Range']}**<br><small>Range</small>", unsafe_allow_html=True)
                
                with cols[4]:
                    st.markdown(f"**{row['Data Points']}**<br><small>Points</small>", unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

def _render_customer_stats(results):
    """Render customer analytics statistics"""
    segments = results.get('segments', {})
    
    st.write("**👥 Customer Segments:**")
    for segment, count in segments.items():
        total_customers = sum(segments.values())
        percentage = (count / total_customers * 100) if total_customers > 0 else 0
        st.write(f"• **{segment}:** {count} customers ({percentage:.1f}%)")

def _render_trend_seasonality_analysis(results):
    """Render trend and seasonality analysis"""
    st.subheader("📈 Trend & Seasonality")
    
    if 'time_series' in results['module']:
        _render_time_series_trend_seasonality(results)
    elif results['module'] == 'customer':
        _render_customer_trend_analysis(results)

def _render_time_series_trend_seasonality(results):
    """Render time series trend and seasonality analysis - handles both single-series and multi-category"""
    analysis_type = results.get('analysis_type', 'single-series')
    
    if analysis_type == 'single-series':
        _render_single_series_trend_seasonality(results)
    elif analysis_type == 'multi-category':
        _render_multi_category_trend_seasonality(results)

def _render_single_series_trend_seasonality(results):
    """Render single series trend and seasonality analysis"""
    stats = results.get('statistics', {})
    
    # Trend Analysis
    st.markdown("#### 📈 Trend Analysis")
    trend_value = stats.get('trend_slope', 0)
    trend_direction = stats.get('trend', 'stable').title()
    
    if trend_value > 0:
        trend_color = "#4caf50"
        trend_icon = "🔺"
        trend_desc = "Increasing trend detected"
    elif trend_value < 0:
        trend_color = "#f44336"
        trend_icon = "🔻"
        trend_desc = "Decreasing trend detected"
    else:
        trend_color = "#ff9800"
        trend_icon = "➡️"
        trend_desc = "Stable, no clear trend"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {trend_color}15 0%, {trend_color}25 100%); 
                border: 2px solid {trend_color}; padding: 15px; margin: 10px 0; border-radius: 10px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 18px; font-weight: bold; color: {trend_color};">
                    {trend_icon} {trend_direction} Trend
                </div>
                <div style="font-size: 12px; color: #666; margin-top: 5px;">{trend_desc}</div>
                <div style="font-size: 11px; color: #888; margin-top: 3px;">Slope: {trend_value:+.4f} units/day</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Seasonality Analysis
    st.markdown("#### 🌊 Seasonality Analysis")
    seasonality_strength = results.get('seasonality_strength', 0)
    
    if seasonality_strength > 0.3:
        seasonality_level = "High"
        seasonality_color = "#ff5722"
        seasonality_icon = "🔥"
        seasonality_desc = "Strong seasonal patterns detected"
    elif seasonality_strength > 0.15:
        seasonality_level = "Moderate"
        seasonality_color = "#ff9800"
        seasonality_icon = "🔶"
        seasonality_desc = "Moderate seasonal patterns"
    else:
        seasonality_level = "Low"
        seasonality_color = "#2196f3"
        seasonality_icon = "🔵"
        seasonality_desc = "Weak or no seasonal patterns"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {seasonality_color}15 0%, {seasonality_color}25 100%); 
                border: 2px solid {seasonality_color}; padding: 15px; margin: 10px 0; border-radius: 10px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 18px; font-weight: bold; color: {seasonality_color};">
                    {seasonality_icon} {seasonality_level} Seasonality
                </div>
                <div style="font-size: 12px; color: #666; margin-top: 5px;">{seasonality_desc}</div>
                <div style="font-size: 11px; color: #888; margin-top: 3px;">Strength: {seasonality_strength:.3f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)



def _render_multi_category_trend_seasonality(results):
    """Render multi-category trend and seasonality analysis"""
    # Get category statistics from ['statistics']['categories']
    statistics = results.get('statistics', {})
    category_stats = statistics.get('categories', {})
    category_decompositions = results.get('category_decompositions', {})
    
    if not category_stats:
        st.info("📈 Category trend/seasonality data not available")
        return

    st.markdown("#### 📈 Category Trend Comparison")
    
    colors = ['#2196f3', '#4caf50', '#ff9800', '#9c27b0', '#f44336', '#00bcd4']

    # Trend comparison
    for i, (category, stats) in enumerate(category_stats.items()):
        trend_value = stats.get('trend_slope', 0)
        trend_direction = stats.get('trend', 'stable').title()
        
        if trend_value > 0:
            trend_icon = "🔺"
        elif trend_value < 0:
            trend_icon = "🔻"
        else:
            trend_icon = "➡️"
        
        color = colors[i % len(colors)]
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {color}15 0%, {color}25 100%);
        border-left: 4px solid {color}; padding: 12px; margin: 5px 0; border-radius: 8px;">
        <div style="font-weight: bold; color: {color}; font-size: 16px;">
        {trend_icon} {category}
        </div>
        <div style="font-size: 12px; color: #666; margin-top: 3px;">
        {trend_direction} trend (slope: {trend_value:+.4f})
        </div>
        </div>
        """, unsafe_allow_html=True)

    # Seasonality comparison - get from category_decompositions
    if category_decompositions:
        st.markdown("#### 🌊 Category Seasonality Comparison")
        
        for i, (category, decomp_data) in enumerate(category_decompositions.items()):
            seasonality_analysis = decomp_data.get('seasonality_analysis', {})
            
            if seasonality_analysis:
                strength = seasonality_analysis.get('strength', 0)
                period_points = seasonality_analysis.get('period_points', 'N/A')
                interpretation = seasonality_analysis.get('interpretation', 'No interpretation available')
                
                if strength > 0.3:
                    seasonality_level = "High"
                    seasonality_icon = "🔥"
                elif strength > 0.15:
                    seasonality_level = "Moderate"
                    seasonality_icon = "🔶"
                else:
                    seasonality_level = "Low"
                    seasonality_icon = "🔵"
                
                color = colors[i % len(colors)]
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {color}15 0%, {color}25 100%);
                border-left: 4px solid {color}; padding: 12px; margin: 5px 0; border-radius: 8px;">
                <div style="font-weight: bold; color: {color}; font-size: 16px;">
                {seasonality_icon} {category}
                </div>
                <div style="font-size: 12px; color: #666; margin-top: 3px;">
                {seasonality_level} seasonality (strength: {strength:.3f}, period: {period_points} points)
                </div>
                <div style="font-size: 11px; color: #888; margin-top: 2px; font-style: italic;">
                {interpretation}
                </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        # If no category_decompositions data, show message
        st.markdown("#### 🌊 Category Seasonality")
        st.info("🌊 Category-specific seasonality data not available in current analysis results")


def _render_customer_trend_analysis(results):
    """Render customer analytics trend analysis"""
    rfm = results.get('rfm_summary', {})
    
    # RFM Analysis
    st.markdown("#### 💼 RFM Analysis")
    
    avg_recency = rfm.get('avg_recency', 0)
    recency_color = "#4caf50" if avg_recency < 30 else "#ff9800" if avg_recency < 90 else "#f44336"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {recency_color}15 0%, {recency_color}25 100%); 
                border: 2px solid {recency_color}; padding: 12px; margin: 8px 0; border-radius: 8px;">
        <strong>📅 Average Recency:</strong> {avg_recency:.1f} days<br>
        <small>How recently customers made purchases</small>
    </div>
    """, unsafe_allow_html=True)
    
    avg_frequency = rfm.get('avg_frequency', 0)
    frequency_color = "#4caf50" if avg_frequency > 5 else "#ff9800" if avg_frequency > 2 else "#f44336"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {frequency_color}15 0%, {frequency_color}25 100%); 
                border: 2px solid {frequency_color}; padding: 12px; margin: 8px 0; border-radius: 8px;">
        <strong>🔄 Average Frequency:</strong> {avg_frequency:.1f} purchases<br>
        <small>Average transactions per customer</small>
    </div>
    """, unsafe_allow_html=True)
    
    avg_monetary = rfm.get('avg_monetary', 0)
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #4caf5015 0%, #4caf5025 100%); 
                border: 2px solid #4caf50; padding: 12px; margin: 8px 0; border-radius: 8px;">
        <strong>💰 Average Monetary Value:</strong> ${avg_monetary:.2f}<br>
        <small>Average spending per customer</small>
    </div>
    """, unsafe_allow_html=True)