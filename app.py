import requests
import pandas as pd
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

# Custom CSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 2rem;
}
.main-header h1 {
    color: white;
    text-align: center;
    margin: 0;
}
.module-card {
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    background: #f8f9fa;
}
.success-box {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0;
}
.info-box {
    background-color: #d1ecf1;
    border: 1px solid #bee5eb;
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0;
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


# Tab 3: Analysis Results
with tab3:
    st.header("📊 Analysis Results")
    
    if st.session_state.analysis_results is None:
        st.info("🔍 Run analysis in the Data Processing tab to see results here")
    else:
        results = st.session_state.analysis_results
        
        # Display insights with help tooltips
        st.subheader("💡 Key Insights", help="AI-generated insights based on your data analysis. These highlight the most important patterns and trends discovered.")
        
        # Create insight help messages based on module type
        if results['module'] == 'time_series':
            insight_helps = {
                "trend": "The overall direction of your data over time - whether values are generally increasing, decreasing, or staying stable",
                "seasonality": "Recurring patterns that repeat at regular intervals (daily, weekly, monthly, yearly)",
                "volatility": "How much your data values fluctuate or vary from the average",
                "growth": "The rate at which your values are increasing or decreasing over time",
                "correlation": "Statistical relationship between different variables in your dataset",
                "outliers": "Unusual data points that are significantly different from the normal pattern"
            }
        else:  # customer analytics  
            insight_helps = {
                "segments": "Groups of customers with similar purchasing behavior patterns",
                "RFM": "Recency (how recently), Frequency (how often), Monetary (how much) - key metrics for customer analysis",
                "retention": "How well you're keeping customers coming back over time",
                "lifetime value": "The total value a customer brings to your business over their relationship with you",
                "churn": "Customers who have stopped purchasing or engaging with your business",
                "acquisition": "New customers gained during a specific time period"
            }
        
        # Display insights with contextual help
        for i, insight in enumerate(results['insights']):
            # Try to match insight with help text based on keywords
            help_text = "Additional context about this insight"
            insight_lower = insight.lower()
            
            if results['module'] == 'time_series':
                if any(word in insight_lower for word in ['trend', 'increasing', 'decreasing', 'stable']):
                    help_text = insight_helps["trend"]
                elif any(word in insight_lower for word in ['seasonal', 'pattern', 'cycle']):
                    help_text = insight_helps["seasonality"]
                elif any(word in insight_lower for word in ['volatile', 'variation', 'fluctuat']):
                    help_text = insight_helps["volatility"]
                elif any(word in insight_lower for word in ['growth', 'rate']):
                    help_text = insight_helps["growth"]
                elif any(word in insight_lower for word in ['outlier', 'unusual', 'anomal']):
                    help_text = insight_helps["outliers"]
            else:  # customer
                if any(word in insight_lower for word in ['segment', 'group', 'cluster']):
                    help_text = insight_helps["segments"]
                elif any(word in insight_lower for word in ['rfm', 'recency', 'frequency', 'monetary']):
                    help_text = insight_helps["RFM"]
                elif any(word in insight_lower for word in ['retention', 'repeat', 'return']):
                    help_text = insight_helps["retention"]
                elif any(word in insight_lower for word in ['lifetime', 'value', 'ltv']):
                    help_text = insight_helps["lifetime value"]
                elif any(word in insight_lower for word in ['churn', 'lost', 'inactive']):
                    help_text = insight_helps["churn"]
            
            st.write(f"• {insight}", help=help_text)
        
        # Display visualization
        st.subheader("📈 Visualization")
        if 'plot' in results:
            fig_dict = json.loads(results['plot'])
            fig = go.Figure(fig_dict)
            st.plotly_chart(fig, use_container_width=True)
        
        # Display detailed results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Statistics", help="Key statistical measures that summarize your data's characteristics")
            if results['module'] == 'time_series':
                stats = results['statistics']
                st.metric("Average Value", f"{stats['mean']:.2f}", 
                         help="The arithmetic mean of all values in your time series")
                st.metric("Standard Deviation", f"{stats['std']:.2f}",
                         help="Measures how spread out your data points are from the average")
                st.metric("Range", f"{stats['min']:.2f} - {stats['max']:.2f}",
                         help="The difference between the highest and lowest values in your dataset")
                
                # Enhanced trend display with numerical value
                trend_value = stats.get('trend_slope', 0)
                trend_direction = stats['trend'].title()
                trend_display = f"{trend_direction} ({trend_value:+.4f})"
                
                st.metric("Trend", trend_display,
                         help="The slope of the linear trend line fitted to your data. Positive values indicate increasing trend, negative values indicate decreasing trend. The magnitude shows how steep the trend is - larger absolute values mean steeper trends. Calculated using linear regression: y = mx + b, where m is the slope shown here.")
                
            elif results['module'] == 'customer':
                segments = results['segments']
                st.write("**Customer Segments:**", help="Groups of customers categorized by their purchasing behavior")
                for segment, count in segments.items():
                    segment_helps = {
                        "Champions": "Your best customers - high recency, frequency, and monetary value",
                        "Loyal Customers": "Regular customers with good frequency and monetary value",
                        "Potential Loyalists": "Recent customers with potential to become loyal",
                        "New Customers": "Recent first-time buyers",
                        "Promising": "Recent customers with good monetary value",
                        "Need Attention": "Customers showing declining engagement",
                        "About to Sleep": "Customers at risk of churning",
                        "At Risk": "Previously good customers who haven't purchased recently",
                        "Cannot Lose Them": "High-value customers at risk of churning",
                        "Hibernating": "Inactive customers with previously good value",
                        "Lost": "Customers who have likely churned"
                    }
                    help_text = segment_helps.get(segment, "Customer segment based on RFM analysis")
                    st.metric(segment, count, help=help_text)
        
        with col2:
            st.subheader("🔍 Detailed Analysis", help="In-depth metrics and analysis results specific to your chosen module")
            if results['module'] == 'time_series':
                # Seasonality Analysis
                seasonality_strength = results.get('seasonality_strength', 0)
                weekly_seasonality = results.get('weekly_seasonality', 0)
                
                st.write(f"**📅 Monthly Seasonality:** {seasonality_strength:.3f}",
                        help="Measures how strong the monthly seasonal patterns are in your data. Values closer to 1 indicate stronger seasonality. This is calculated as the coefficient of variation of monthly averages (standard deviation / mean).")
                
                seasonality_level = "High" if seasonality_strength > 0.3 else "Moderate" if seasonality_strength > 0.15 else "Low"
                st.write(f"**📈 Seasonality Level:** {seasonality_level}",
                        help="Interpretation: High (>0.3) = Strong seasonal patterns with significant monthly variations; Moderate (0.15-0.3) = Some seasonal patterns; Low (≤0.15) = Weak or no clear seasonal patterns")
                
                st.write(f"**📊 Weekly Seasonality:** {weekly_seasonality:.3f}",
                        help="Measures weekly patterns in your data. Higher values indicate stronger day-of-week effects (e.g., higher sales on weekends).")
                
                # Volatility Analysis
                volatility = results.get('volatility', 0)
                volatility_level = "High" if volatility > 0.5 else "Moderate" if volatility > 0.2 else "Low"
                
                st.write(f"**🌊 Volatility:** {volatility:.3f} ({volatility_level})",
                        help="Measures how much your data fluctuates relative to its average (coefficient of variation). High volatility (>0.5) means unpredictable swings; Low volatility (≤0.2) means stable, predictable patterns.")
                
                # Growth Analysis
                growth_rate = results.get('growth_rate', 0)
                growth_direction = "Growth" if growth_rate > 0 else "Decline" if growth_rate < 0 else "Stable"
                
                st.write(f"**📈 Total Growth:** {growth_rate:.1f}% ({growth_direction})",
                        help="Percentage change from the first data point to the last data point in your time series. Positive values indicate overall growth, negative values indicate decline.")
                
                # Trend Analysis (dengan penjelasan slope)
                trend_slope = results['statistics'].get('trend_slope', 0)
                st.write(f"**📏 Trend Slope:** {trend_slope:.6f} units/day",
                        help="The daily rate of change calculated using linear regression. This tells you how much your values increase (positive) or decrease (negative) per day on average. Larger absolute values indicate steeper trends.")
                
                # Category Analysis (jika ada)
                if results.get('has_categories', False):
                    st.write("**🏷️ Category Breakdown:**")
                    categories = results['statistics'].get('categories', {})
                    for cat, cat_stats in categories.items():
                        st.write(f"• {cat}: Trend {cat_stats['trend']} ({cat_stats['trend_slope']:.4f}/day)")
                
            elif results['module'] == 'customer':
                rfm = results['rfm_summary']
                
                st.write("**🎯 RFM Analysis Overview:**",
                        help="RFM (Recency, Frequency, Monetary) is a proven method for analyzing customer behavior and value. It helps identify your most valuable customers and those at risk of churning.")
                
                st.write(f"**📅 Average Recency:** {rfm['avg_recency']:.1f} days",
                        help="How recently customers made their last purchase. Lower values are better - they indicate more recent engagement. Values >90 days may indicate customers at risk of churning.")
                
                st.write(f"**🔄 Average Frequency:** {rfm['avg_frequency']:.1f} transactions",
                        help="How often customers make purchases. Higher values indicate more loyal, engaged customers. This metric helps identify your most active customer segments.")
                
                st.write(f"**💰 Average Monetary Value:** ${rfm['avg_monetary']:.2f}",
                        help="Average total amount spent per customer over their entire relationship with your business. This helps identify high-value customers who contribute most to revenue.")
                
                # Customer Distribution Insights
                segments = results['segments']
                total_customers = sum(segments.values())
                
                st.write("**📊 Customer Distribution Insights:**")
                
                # Calculate key percentages
                champions_pct = (segments.get('Champions', 0) / total_customers) * 100 if total_customers > 0 else 0
                at_risk_pct = (segments.get('At Risk', 0) / total_customers) * 100 if total_customers > 0 else 0
                loyal_pct = (segments.get('Loyal Customers', 0) / total_customers) * 100 if total_customers > 0 else 0
                
                st.write(f"• **Top Customers (Champions):** {champions_pct:.1f}% of customer base",
                        help="Your best customers with high recency, frequency, and monetary scores. These are your most valuable customers who should receive premium treatment.")
                
                st.write(f"• **Loyal Customers:** {loyal_pct:.1f}% of customer base", 
                        help="Regular customers with good purchase frequency and value. Focus on maintaining their loyalty and potentially upgrading them to Champions.")
                
                st.write(f"• **At-Risk Customers:** {at_risk_pct:.1f}% of customer base",
                        help="Previously valuable customers who haven't purchased recently. These customers need immediate attention through targeted re-engagement campaigns.")
                
                # Business Health Indicators
                healthy_customers_pct = champions_pct + loyal_pct
                st.write(f"**🏥 Customer Health Score:** {healthy_customers_pct:.1f}%",
                        help="Percentage of customers in 'Champions' and 'Loyal Customers' segments. Values >40% indicate a healthy customer base; <20% suggests need for customer retention strategies.")
        
        # Export option
        st.subheader("📥 Export Results")
        if st.button("📄 Generate PDF Report"):
            st.info("🚧 PDF export feature coming soon!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    📊 Data Analytics | Built with FastAPI & Streamlit | 
    <a href="#" style="color: #667eea;">Documentation</a> | 
    <a href="#" style="color: #667eea;">API Reference</a>
</div>
""", unsafe_allow_html=True)