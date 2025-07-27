import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import parse_datetime_flexible
import logging

class ForecastingModels:
    @staticmethod
    def create_features(df, date_col, value_col):
        """Create time-based and seasonal features"""
        df = df.copy()
        df['days_numeric'] = (df[date_col] - df[date_col].min()).dt.days
        df['month'] = df[date_col].dt.month
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['quarter'] = df[date_col].dt.quarter
        df['day_of_year'] = df[date_col].dt.dayofyear
        
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        # Lag features
        if len(df) > 7:
            df['lag_7'] = df[value_col].shift(7).fillna(df[value_col].mean())
        if len(df) > 30:
            df['lag_30'] = df[value_col].shift(30).fillna(df[value_col].mean())
            df['rolling_7'] = df[value_col].rolling(7, min_periods=1).mean()
            df['rolling_30'] = df[value_col].rolling(30, min_periods=1).mean()
        
        return df

    @staticmethod
    def create_forecast(data, date_column, value_column, model_type='linear_regression'):
        """Enhanced forecast with multiple model options"""
        if len(data) < 10:
            return None, None, "Insufficient data for forecasting (minimum 10 points required)"
        
        test_size = max(1, int(len(data) * 0.2))
        train_data = data.iloc[:-test_size].copy()
        test_data = data.iloc[-test_size:].copy()
        
        if len(train_data) < 5:
            return None, None, "Insufficient training data for forecasting"
        
        try:
            # Create features
            train_features = ForecastingModels.create_features(train_data, date_column, value_column)
            test_features = ForecastingModels.create_features(test_data, date_column, value_column)
            
            feature_cols = [col for col in train_features.columns
                          if col not in [date_column, value_column]
                          and pd.api.types.is_numeric_dtype(train_features[col])
                          and not train_features[col].isna().all()]
            
            X_train = train_features[feature_cols].fillna(0)
            y_train = train_features[value_column].values
            X_test = test_features[feature_cols].fillna(0)
            y_actual = test_features[value_column].values
            
            # Model selection
            models = {
                'linear_regression': (LinearRegression(), "Linear Regression"),
                'random_forest': (RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42), "Random Forest"),
                'naive': (None, "Naive (Last Value)"),
                'moving_average': (None, "Moving Average"),
                'ets': (None, "Exponential Smoothing")
            }
            
            if model_type in ['naive', 'moving_average', 'ets']:
                return ForecastingModels._statistical_forecast(train_data, test_data, date_column, value_column, model_type)
            
            model, model_name = models.get(model_type, models['linear_regression'])
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_actual, y_pred)
            rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
            r2 = r2_score(y_actual, y_pred)
            mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100 if np.all(y_actual != 0) else float('inf')
            
            # Prediction intervals
            train_pred = model.predict(X_train)
            residuals = y_train - train_pred
            std_error = np.sqrt(np.mean(residuals**2))
            prediction_variance = std_error * np.sqrt(1 + 1/len(train_data))
            
            ci_levels = [0.674, 1.282, 1.96]  # 50%, 80%, 95%
            confidence_intervals = {}
            for i, level in enumerate(['50', '80', '95']):
                confidence_intervals[f'ci_{level}_lower'] = (y_pred - ci_levels[i] * prediction_variance).tolist()
                confidence_intervals[f'ci_{level}_upper'] = (y_pred + ci_levels[i] * prediction_variance).tolist()
            
            forecast_data = {
                'model_name': model_name,
                'train_dates': train_data[date_column].dt.strftime('%Y-%m-%d').tolist(),
                'train_values': train_data[value_column].tolist(),
                'test_dates': test_data[date_column].dt.strftime('%Y-%m-%d').tolist(),
                'test_actual': y_actual.tolist(),
                'test_predicted': y_pred.tolist(),
                **confidence_intervals,
                'split_date': train_data[date_column].iloc[-1].strftime('%Y-%m-%d'),
                'metrics': {
                    'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2),
                    'mape': float(mape) if mape != float('inf') else None
                },
                'model_type': model_type
            }
            
            forecast_insight = f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}"
            if mape != float('inf'):
                forecast_insight += f", MAPE: {mape:.1f}%"
            
            return forecast_data, forecast_insight, None
            
        except Exception as e:
            logging.error(f"Forecasting error with {model_type}: {str(e)}")
            return None, None, f"Forecasting error: {str(e)}"

    @staticmethod
    def _statistical_forecast(train_data, test_data, date_column, value_column, model_type):
        """Statistical forecasting methods"""
        values = train_data[value_column].values
        y_actual = test_data[value_column].values
        
        if model_type == 'naive':
            y_pred = np.array([values[-1]] * len(test_data))
            model_name = 'Naive (Last Value)'
        elif model_type == 'moving_average':
            window_size = min(max(3, len(values) // 10), 30)
            ma = pd.Series(values).rolling(window=window_size, min_periods=1).mean()
            y_pred = np.array([ma.iloc[-1]] * len(test_data))
            model_name = f'Moving Average (window={window_size})'
        elif model_type == 'ets':
            # Simple exponential smoothing
            alpha = 0.3
            level = values[0]
            trend = np.mean(np.diff(values[:min(12, len(values))]))
            predictions = []
            for _ in range(len(test_data)):
                forecast = level + trend
                predictions.append(forecast)
            y_pred = np.array(predictions)
            model_name = 'Exponential Smoothing (ETS)'
        
        # Calculate metrics
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        r2 = r2_score(y_actual, y_pred)
        
        # Simple confidence intervals
        historical_volatility = np.std(values)
        ci_levels = [0.674, 1.282, 1.96]
        confidence_intervals = {}
        for i, level in enumerate(['50', '80', '95']):
            confidence_intervals[f'ci_{level}_lower'] = (y_pred - ci_levels[i] * historical_volatility).tolist()
            confidence_intervals[f'ci_{level}_upper'] = (y_pred + ci_levels[i] * historical_volatility).tolist()
        
        forecast_data = {
            'model_name': model_name,
            'train_dates': train_data[date_column].dt.strftime('%Y-%m-%d').tolist(),
            'train_values': train_data[value_column].tolist(),
            'test_dates': test_data[date_column].dt.strftime('%Y-%m-%d').tolist(),
            'test_actual': y_actual.tolist(),
            'test_predicted': y_pred.tolist(),
            **confidence_intervals,
            'split_date': train_data[date_column].iloc[-1].strftime('%Y-%m-%d'),
            'metrics': {
                'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2),
                'mape': float(np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100) if np.all(y_actual != 0) else None
            },
            'model_type': model_type
        }
        
        return forecast_data, f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}", None

    @staticmethod
    def compare_forecasting_models(df, params):
        """Compare multiple forecasting models"""
        try:
            date_col = params.get('date_col', 'date')
            value_col = params.get('value_col', 'value')
            comparison_models = params.get('comparison_models', ['linear_regression', 'random_forest', 'ets', 'naive'])
            
            df[date_col] = parse_datetime_flexible(df[date_col])
            df = df.dropna(subset=[date_col]).sort_values(date_col)
            
            if len(df) < 10:
                raise ValueError("Insufficient data for model comparison")
            
            comparison_results = {}
            
            for model_type in comparison_models:
                try:
                    forecast_data, forecast_insight, forecast_error = ForecastingModels.create_forecast(
                        df, date_col, value_col, model_type=model_type)
                    
                    if forecast_data and forecast_error is None:
                        metrics = forecast_data['metrics']
                        comparison_results[model_type] = {
                            'model_name': forecast_data['model_name'],
                            'mae': metrics['mae'], 'rmse': metrics['rmse'],
                            'r2': metrics.get('r2'), 'mape': metrics.get('mape'),
                            'insight': forecast_insight, 'status': 'success'
                        }
                    else:
                        comparison_results[model_type] = {
                            'model_name': model_type.replace('_', ' ').title(),
                            'status': 'failed', 'error': forecast_error or 'Unknown error'
                        }
                except Exception as e:
                    comparison_results[model_type] = {
                        'model_name': model_type.replace('_', ' ').title(),
                        'status': 'failed', 'error': str(e)
                    }
            
            # Rank models
            successful_models = {k: v for k, v in comparison_results.items() if v['status'] == 'success'}
            
            if successful_models:
                ranked_models = sorted(successful_models.items(), key=lambda x: x[1]['mae'])
                best_model = ranked_models[0]
                
                comparison_table = []
                for rank, (model_key, model_data) in enumerate(ranked_models, 1):
                    comparison_table.append({
                        'rank': rank, 'model': model_data['model_name'], 'model_key': model_key,
                        'mae': model_data['mae'], 'rmse': model_data['rmse'],
                        'r2': model_data.get('r2'), 'mape': model_data.get('mape'),
                        'status': 'success'
                    })
                
                # Add failed models
                for model_key, model_data in comparison_results.items():
                    if model_data['status'] == 'failed':
                        comparison_table.append({
                            'rank': None, 'model': model_data['model_name'], 'model_key': model_key,
                            'mae': None, 'rmse': None, 'r2': None, 'mape': None,
                            'status': 'failed', 'error': model_data.get('error', 'Unknown error')
                        })
                
                return {
                    'comparison_results': comparison_results,
                    'comparison_table': comparison_table,
                    'best_model': {
                        'model_key': best_model[0], 'model_name': best_model[1]['model_name'],
                        'mae': best_model[1]['mae'], 'rmse': best_model[1]['rmse'],
                        'r2': best_model[1].get('r2'), 'mape': best_model[1].get('mape')
                    },
                    'total_models_tested': len(comparison_models),
                    'successful_models': len(successful_models),
                    'failed_models': len(comparison_models) - len(successful_models)
                }
            else:
                raise ValueError("All models failed to run successfully")
                
        except Exception as e:
            logging.error(f"Model comparison error: {str(e)}")
            raise