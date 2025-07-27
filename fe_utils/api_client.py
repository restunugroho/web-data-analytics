import requests
import streamlit as st
from components.config import API_BASE_URL
import logging

class APIClient:
    @staticmethod
    def get_sample_datasets():
        """Get available sample datasets"""
        try:
            response = requests.get(f"{API_BASE_URL}/sample-datasets")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"❌ Error connecting to API: {str(e)}")
            return None
    
    @staticmethod
    def load_sample_dataset(dataset_key):
        """Load a sample dataset"""
        try:
            response = requests.post(f"{API_BASE_URL}/load-sample/{dataset_key}")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")
            return None
    
    @staticmethod
    def upload_data(uploaded_file):
        """Upload user data"""
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            response = requests.post(f"{API_BASE_URL}/upload-data", files=files)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"❌ Error uploading file: {str(e)}")
            return None
    
    @staticmethod
    def aggregate_data(payload):
        """Aggregate data"""
        try:
            response = requests.post(f"{API_BASE_URL}/aggregate-data", json=payload)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"❌ Aggregation failed: {str(e)}")
            return None
    
    @staticmethod
    def analyze_data(payload):
        """Run analysis"""
        try:
            logging.info('payload analyze')
            logging.info(payload)
            response = requests.post(f"{API_BASE_URL}/analyze", json=payload)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            return None
    
    @staticmethod
    def compare_models(payload):
        """Compare multiple models"""
        try:
            response = requests.post(f"{API_BASE_URL}/compare-models", json=payload)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"❌ Model comparison failed: {str(e)}")
            return None