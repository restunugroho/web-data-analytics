import streamlit as st
import pandas as pd
from fe_utils.api_client import APIClient

def render_data_input_tab():
    """Render the data input tab"""
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
        _render_sample_data_section()
    else:
        _render_upload_data_section()
    
    # Display current data
    if st.session_state.current_data is not None:
        _render_data_preview()

def _render_sample_data_section():
    """Render sample data selection section"""
    st.subheader("🎯 Sample Datasets")
    
    datasets = APIClient.get_sample_datasets()
    if datasets:
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
                📋 Columns: {', '.join(datasets[dataset_key]['columns'])}
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚀 Load Sample Dataset", type="primary"):
                with st.spinner("Loading dataset..."):
                    data = APIClient.load_sample_dataset(dataset_key)
                    if data:
                        st.session_state.current_session_id = data['session_id']
                        st.session_state.current_data = pd.DataFrame(data['sample_data'])
                        st.success("✅ Dataset loaded successfully!")
                        st.rerun()

def _render_upload_data_section():
    """Render file upload section"""
    st.subheader("📁 Upload Your Data")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx'],
        help="Maximum file size: 5MB. Supports CSV and Excel formats."
    )

    if uploaded_file is not None:
        if st.button("📤 Upload and Process", type="primary"):
            with st.spinner("Uploading and processing..."):
                data = APIClient.upload_data(uploaded_file)
                if data:
                    st.session_state.current_session_id = data['session_id']
                    st.session_state.current_data = pd.DataFrame(data['sample_data'])
                    st.success("✅ File uploaded successfully!")
                    st.rerun()

def _render_data_preview():
    """Render data preview section"""
    st.subheader("🔍 Current Data Preview")
    st.dataframe(st.session_state.current_data, use_container_width=True)
    
    st.markdown(f"""
    <div class="success-box">
        📊 <strong>Data Shape:</strong> {st.session_state.current_data.shape[0]} rows × {st.session_state.current_data.shape[1]} columns<br>
        🆔 <strong>Session ID:</strong> {st.session_state.current_session_id}
    </div>
    """, unsafe_allow_html=True)