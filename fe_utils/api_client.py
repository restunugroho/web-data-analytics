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
    
    # ===== ANALYSIS METHODS =====
    @staticmethod
    def analyze_time_series(payload):
        """Run time series analysis (without forecasting)"""
        try:
            logging.info('Time series analysis payload:')
            logging.info(payload)
            response = requests.post(f"{API_BASE_URL}/analyze/time-series", json=payload)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"❌ Time series analysis failed: {str(e)}")
            return None
    
    @staticmethod
    def analyze_customer(payload):
        """Run customer analytics analysis"""
        try:
            logging.info('Customer analysis payload:')
            logging.info(payload)
            response = requests.post(f"{API_BASE_URL}/analyze/customer", json=payload)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"❌ Customer analysis failed: {str(e)}")
            return None
    
    # ===== FORECASTING METHODS =====
    @staticmethod
    def get_available_forecast_models():
        """Get available forecasting models"""
        try:
            response = requests.get(f"{API_BASE_URL}/forecast/available-models")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"❌ Error getting forecast models: {str(e)}")
            return None
    
    @staticmethod
    def forecast_single_model(payload):
        """Run forecasting with a single model"""
        logging.info('forecast_single_model')
        try:
            logging.info('Single model forecast payload:')
            logging.info(payload)
            response = requests.post(f"{API_BASE_URL}/forecast/single-model", json=payload)
            logging.info(response)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"❌ Single model forecasting failed: {str(e)}")
            return None
    
    @staticmethod
    def compare_forecast_models(payload):
        """Compare multiple forecasting models"""
        try:
            logging.info('Model comparison forecast payload:')
            logging.info(payload)
            response = requests.post(f"{API_BASE_URL}/forecast/compare-models", json=payload)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"❌ Model comparison forecasting failed: {str(e)}")
            return None
    
    # ===== LEGACY METHODS (for backward compatibility) =====
    @staticmethod
    def analyze_data(payload):
        """Legacy method - route analysis based on module"""
        module = payload.get('module', 'time_series')
        if module == 'time_series':
            return APIClient.analyze_time_series(payload)
        elif module == 'customer':
            return APIClient.analyze_customer(payload)
        else:
            st.error(f"❌ Unknown module: {module}")
            return None
    
    @staticmethod
    def compare_models(payload):
        """Legacy method - use new forecast comparison"""
        return APIClient.compare_forecast_models(payload)