import streamlit as st

def initialize_session_state():
    """Initialize session state variables"""
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = None
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'model_comparison' not in st.session_state:
        st.session_state.model_comparison = None

def clear_session():
    """Clear all session state"""
    st.session_state.analysis_results = None
    st.session_state.current_data = None
    st.session_state.current_session_id = None
    if hasattr(st.session_state, 'model_comparison'):
        delattr(st.session_state, 'model_comparison')