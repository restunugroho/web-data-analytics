import pandas as pd
import numpy as np

def parse_datetime_flexible(date_series):
    """Flexibly parse datetime with multiple format attempts"""
    formats = [
        '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S',
        '%d/%m/%Y %H:%M', '%m/%d/%Y %H:%M', '%Y%m%d', '%d-%m-%Y', '%m-%d-%Y'
    ]
    
    try:
        return pd.to_datetime(date_series, infer_datetime_format=True)
    except:
        pass
    
    for fmt in formats:
        try:
            return pd.to_datetime(date_series, format=fmt)
        except:
            continue
    
    try:
        return pd.to_datetime(date_series, errors='coerce')
    except:
        raise ValueError("Unable to parse datetime column")

def clean_float(val):
    if isinstance(val, float):
        if np.isnan(val) or np.isinf(val):
            return 0.0
    return float(val)

def sanitize_for_json(obj):
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]
    else:
        return obj