import streamlit as st

# Configuration
API_BASE_URL = "http://localhost:8000"

def setup_page_config():
    """Setup Streamlit page configuration"""
    st.set_page_config(
        page_title="Data Analytics",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def load_custom_css():
    """Load custom CSS styles"""
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
        padding: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(40,167,69,0.2);
        color: #155724;
        font-weight: 500;
        font-size: 0.9rem;
    }
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 2px solid #17a2b8;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(23,162,184,0.2);
        color: #0c5460;
        font-weight: 500;
        font-size: 0.9rem;
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(255,193,7,0.2);
        color: #856404;
        font-weight: 500;
        font-size: 0.9rem;
    }
    .insight-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        font-size: 0.9rem;
        line-height: 1.4;
    }
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 2px solid #dee2e6;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .forecast-container {
        background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%);
        border: 2px solid #28a745;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .forecast-settings {
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
