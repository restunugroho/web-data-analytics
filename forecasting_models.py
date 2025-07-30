import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import parse_datetime_flexible, clean_float, sanitize_for_json
import logging

class ForecastingModels:
    """Enhanced forecasting models with better feature engineering and model selection"""
    
    @staticmethod
    def create_features(df, date_col, value_col):
        """Create comprehensive time-based and seasonal features"""
        df = df.copy()
        
        # Basic time features
        df['days_numeric'] = (df[date_col] - df[date_col].min()).dt.days
        df['month'] = df[date_col].dt.month
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['quarter'] = df[date_col].dt.quarter
        df['day_of_year'] = df[date_col].dt.dayofyear
        df['week_of_year'] = df[date_col].dt.isocalendar().week
        df['is_weekend'] = (df[date_col].dt.dayofweek >= 5).astype(int)
        df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
        df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
        
        # Cyclical encoding for better seasonality capture
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
        
        # Lag features (if sufficient data)
        if len(df) > 7:
            df['lag_1'] = df[value_col].shift(1).fillna(df[value_col].mean())
            df['lag_7'] = df[value_col].shift(7).fillna(df[value_col].mean())
            
        if len(df) > 14:
            df['lag_14'] = df[value_col].shift(14).fillna(df[value_col].mean())
            
        if len(df) > 30:
            df['lag_30'] = df[value_col].shift(30).fillna(df[value_col].mean())
            
            # Rolling window features
            df['rolling_7_mean'] = df[value_col].rolling(7, min_periods=1).mean()
            df['rolling_7_std'] = df[value_col].rolling(7, min_periods=1).std().fillna(0)
            df['rolling_30_mean'] = df[value_col].rolling(30, min_periods=1).mean()
            df['rolling_30_std'] = df[value_col].rolling(30, min_periods=1).std().fillna(0)
            
            # Trend features
            df['rolling_7_trend'] = df['rolling_7_mean'].diff().fillna(0)
            df['rolling_30_trend'] = df['rolling_30_mean'].diff().fillna(0)
        
        # Statistical features
        if len(df) > 5:
            # Recent volatility
            df['recent_volatility'] = df[value_col].rolling(min(7, len(df)), min_periods=1).std().fillna(0)
            
            # Percentage change features
            df['pct_change_1'] = df[value_col].pct_change(1).fillna(0)
            if len(df) > 7:
                df['pct_change_7'] = df[value_col].pct_change(7).fillna(0)
        
        return df

    @staticmethod
    def validate_forecast_data(data, date_column, value_column):
        """Validate data quality for forecasting"""
        if len(data) < 10:
            return False, "Insufficient data for forecasting (minimum 10 points required)"
        
        # Check for missing values
        if data[value_column].isna().sum() > len(data) * 0.1:
            return False, "Too many missing values in target variable"
        
        # Check for constant values
        if data[value_column].nunique() <= 1:
            return False, "Target variable has no variation"
        
        # Check date continuity (warn but don't fail)
        date_gaps = data[date_column].diff().dt.days
        if date_gaps.max() > date_gaps.median() * 5:
            logging.warning("Large gaps detected in time series data")
        
        return True, "Data validation passed"

    @staticmethod
    def calculate_prediction_intervals(y_train, y_pred_train, y_pred_test, method='residual'):
        """Calculate prediction intervals using different methods"""
        if method == 'residual':
            # Use historical residuals
            residuals = y_train - y_pred_train
            std_error = np.sqrt(np.mean(residuals**2))
            
            # Adjust for prediction uncertainty
            n_train = len(y_train)
            prediction_variance = std_error * np.sqrt(1 + 1/n_train)
            
        elif method == 'quantile':
            # Use quantile-based approach
            residuals = y_train - y_pred_train
            prediction_variance = np.std(residuals)
        
        else:
            # Simple standard deviation
            prediction_variance = np.std(y_train - y_pred_train)
        
        # Confidence levels: 50%, 80%, 95%
        ci_levels = [0.674, 1.282, 1.96]
        confidence_intervals = {}
        
        for i, level in enumerate(['50', '80', '95']):
            lower = y_pred_test - ci_levels[i] * prediction_variance
            upper = y_pred_test + ci_levels[i] * prediction_variance
            confidence_intervals[f'ci_{level}_lower'] = lower.tolist()
            confidence_intervals[f'ci_{level}_upper'] = upper.tolist()
        
        return confidence_intervals, float(prediction_variance)

    @staticmethod
    def create_forecast(data, date_column, value_column, model_type='linear_regression', forecast_periods=None):
        """Enhanced forecast with multiple model options and better error handling"""
        
        # Validate input data
        is_valid, validation_message = ForecastingModels.validate_forecast_data(data, date_column, value_column)
        if not is_valid:
            return None, None, validation_message
        
        # Determine test size
        test_size = max(1, int(len(data) * 0.2))
        if forecast_periods:
            test_size = min(test_size, forecast_periods)
        
        train_data = data.iloc[:-test_size].copy()
        test_data = data.iloc[-test_size:].copy()
        
        if len(train_data) < 5:
            return None, None, "Insufficient training data for forecasting"
        
        try:
            logging.info(f"Creating forecast using {model_type} model")
            logging.info(f"Training data: {len(train_data)} points, Test data: {len(test_data)} points")
            
            # Handle different model types
            if model_type in ['naive', 'moving_average', 'ets']:
                return ForecastingModels._statistical_forecast(train_data, test_data, date_column, value_column, model_type)
            
            # Machine learning models
            train_features = ForecastingModels.create_features(train_data, date_column, value_column)
            test_features = ForecastingModels.create_features(test_data, date_column, value_column)
            
            # Select relevant features
            feature_cols = [col for col in train_features.columns
                          if col not in [date_column, value_column]
                          and pd.api.types.is_numeric_dtype(train_features[col])
                          and not train_features[col].isna().all()
                          and train_features[col].var() > 1e-10]  # Remove constant features
            
            X_train = train_features[feature_cols].fillna(0)
            y_train = train_features[value_column].values
            X_test = test_features[feature_cols].fillna(0)
            y_actual = test_features[value_column].values
            
            # Model selection and training
            models = {
                'linear_regression': (LinearRegression(), "Linear Regression"),
                'random_forest': (RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=min(10, len(feature_cols)), 
                    random_state=42,
                    n_jobs=-1
                ), "Random Forest"),
            }
            
            if model_type not in models:
                model_type = 'linear_regression'  # Fallback
            
            model, model_name = models[model_type]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            metrics = ForecastingModels._calculate_forecast_metrics(y_actual, y_pred_test, y_train)
            
            # Calculate prediction intervals
            confidence_intervals, prediction_variance = ForecastingModels.calculate_prediction_intervals(
                y_train, y_pred_train, y_pred_test
            )
            
            # Feature importance (for tree-based models)
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
                feature_importance = dict(zip(feature_cols, importance_scores))
                # Keep only top 10 most important features
                feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])
            
            forecast_data = {
                'model_name': model_name,
                'model_type': model_type,
                'train_dates': train_data[date_column].dt.strftime('%Y-%m-%d').tolist(),
                'train_values': train_data[value_column].tolist(),
                'train_predictions': y_pred_train.tolist(),
                'test_dates': test_data[date_column].dt.strftime('%Y-%m-%d').tolist(),
                'test_actual': y_actual.tolist(),
                'test_predicted': y_pred_test.tolist(),
                'split_date': train_data[date_column].iloc[-1].strftime('%Y-%m-%d'),
                'metrics': metrics,
                'confidence_intervals': confidence_intervals,
                'prediction_variance': prediction_variance,
                'feature_importance': feature_importance,
                'features_used': feature_cols,
                'data_quality': {
                    'train_size': len(train_data),
                    'test_size': len(test_data),
                    'features_count': len(feature_cols),
                    'missing_values_handled': True
                }
            }
            
            # Generate forecast insight
            forecast_insight = f"{model_name} - MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}"
            if metrics['mape'] is not None:
                forecast_insight += f", MAPE: {metrics['mape']:.1f}%"
            if metrics['r2'] is not None:
                forecast_insight += f", R²: {metrics['r2']:.3f}"
            
            return forecast_data, forecast_insight, None
            
        except Exception as e:
            logging.error(f"Forecasting error with {model_type}: {str(e)}")
            return None, None, f"Forecasting error: {str(e)}"

    @staticmethod
    def _calculate_forecast_metrics(y_actual, y_pred, y_train=None):
        """Calculate comprehensive forecast metrics"""
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        
        # R-squared
        try:
            r2 = r2_score(y_actual, y_pred)
        except:
            r2 = None
        
        # MAPE (Mean Absolute Percentage Error)
        try:
            mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
            if not np.isfinite(mape):
                mape = None
        except:
            mape = None
        
        # Additional metrics
        try:
            # Mean Error (bias)
            me = np.mean(y_pred - y_actual)
            
            # Symmetric Mean Absolute Percentage Error
            smape = np.mean(2 * np.abs(y_actual - y_pred) / (np.abs(y_actual) + np.abs(y_pred))) * 100
            if not np.isfinite(smape):
                smape = None
                
            # Theil's U statistic (if training data available)
            theil_u = None
            if y_train is not None and len(y_train) > 1:
                naive_forecast = np.full_like(y_actual, y_train[-1])
                mse_forecast = mean_squared_error(y_actual, y_pred)
                mse_naive = mean_squared_error(y_actual, naive_forecast)
                if mse_naive > 0:
                    theil_u = np.sqrt(mse_forecast) / np.sqrt(mse_naive)
        except:
            me = None
            smape = None
            theil_u = None
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2) if r2 is not None else None,
            'mape': float(mape) if mape is not None else None,
            'me': float(me) if me is not None else None,
            'smape': float(smape) if smape is not None else None,
            'theil_u': float(theil_u) if theil_u is not None else None
        }

    @staticmethod
    def _statistical_forecast(train_data, test_data, date_column, value_column, model_type):
        """Enhanced statistical forecasting methods"""
        values = train_data[value_column].values
        y_actual = test_data[value_column].values
        
        if model_type == 'naive':
            # Naive forecast - last value
            y_pred = np.array([values[-1]] * len(test_data))
            model_name = 'Naive (Last Value)'
            
        elif model_type == 'moving_average':
            # Adaptive moving average
            optimal_window = min(max(3, len(values) // 10), 30)
            
            # Try different window sizes and pick the best
            best_window = optimal_window
            best_mae = float('inf')
            
            for window in range(3, min(optimal_window + 5, len(values) // 2)):
                ma = pd.Series(values).rolling(window=window, min_periods=1).mean()
                test_mae = np.mean(np.abs(values[-window:] - ma.iloc[-window-1:-1]))
                if test_mae < best_mae:
                    best_mae = test_mae
                    best_window = window
            
            ma = pd.Series(values).rolling(window=best_window, min_periods=1).mean()
            y_pred = np.array([ma.iloc[-1]] * len(test_data))
            model_name = f'Moving Average (window={best_window})'
            
        elif model_type == 'ets':
            # Enhanced exponential smoothing with trend
            alpha = 0.3  # Level smoothing
            beta = 0.1   # Trend smoothing
            
            # Initialize
            level = values[0]
            trend = np.mean(np.diff(values[:min(12, len(values))]))
            
            # Update level and trend
            for i in range(1, len(values)):
                prev_level = level
                level = alpha * values[i] + (1 - alpha) * (level + trend)
                trend = beta * (level - prev_level) + (1 - beta) * trend
            
            # Forecast
            predictions = []
            for h in range(len(test_data)):
                forecast = level + (h + 1) * trend
                predictions.append(forecast)
            
            y_pred = np.array(predictions)
            model_name = 'Exponential Smoothing (ETS)'
            
        else:
            # Fallback to naive
            y_pred = np.array([values[-1]] * len(test_data))
            model_name = 'Naive (Fallback)'
        
        # Calculate metrics
        metrics = ForecastingModels._calculate_forecast_metrics(y_actual, y_pred, values)
        
        # Simple confidence intervals based on historical volatility
        historical_volatility = np.std(values)
        seasonal_adjustment = 1.0
        
        # Adjust for seasonality if detected
        if len(values) >= 12:
            monthly_std = np.std([np.std(values[i::12]) for i in range(min(12, len(values)))])
            if monthly_std > 0:
                seasonal_adjustment = 1 + (monthly_std / historical_volatility)
        
        adjusted_volatility = historical_volatility * seasonal_adjustment
        
        ci_levels = [0.674, 1.282, 1.96]
        confidence_intervals = {}
        for i, level in enumerate(['50', '80', '95']):
            confidence_intervals[f'ci_{level}_lower'] = (y_pred - ci_levels[i] * adjusted_volatility).tolist()
            confidence_intervals[f'ci_{level}_upper'] = (y_pred + ci_levels[i] * adjusted_volatility).tolist()
        
        forecast_data = {
            'model_name': model_name,
            'model_type': model_type,
            'train_dates': train_data[date_column].dt.strftime('%Y-%m-%d').tolist(),
            'train_values': train_data[value_column].tolist(),
            'train_predictions': values.tolist(),  # For statistical models, training "predictions" are the original values
            'test_dates': test_data[date_column].dt.strftime('%Y-%m-%d').tolist(),
            'test_actual': y_actual.tolist(),
            'test_predicted': y_pred.tolist(),
            'split_date': train_data[date_column].iloc[-1].strftime('%Y-%m-%d'),
            'metrics': metrics,
            'confidence_intervals': confidence_intervals,
            'prediction_variance': float(adjusted_volatility),
            'feature_importance': {},
            'features_used': [],
            'data_quality': {
                'train_size': len(train_data),
                'test_size': len(test_data),
                'features_count': 0,
                'missing_values_handled': True
            }
        }
        
        forecast_insight = f"{model_name} - MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}"
        if metrics['mape'] is not None:
            forecast_insight += f", MAPE: {metrics['mape']:.1f}%"
        
        return forecast_data, forecast_insight, None

    @staticmethod
    def compare_forecasting_models(df, params):
        """Enhanced model comparison with detailed analysis"""
        try:
            date_col = params.get('date_col', 'date')
            value_col = params.get('value_col', 'value')
            comparison_models = params.get('comparison_models', ['linear_regression', 'random_forest', 'ets', 'naive'])
            category_col = params.get('category_col')
            
            logging.info(f"Comparing models: {comparison_models}")
            
            # Parse and validate data
            df[date_col] = parse_datetime_flexible(df[date_col])
            df = df.dropna(subset=[date_col]).sort_values(date_col)
            
            if len(df) < 10:
                raise ValueError("Insufficient data for model comparison")
            
            # Handle category-based comparison
            if category_col and category_col in df.columns:
                return ForecastingModels._compare_models_by_category(df, params)
            
            # Single series comparison
            comparison_results = {}
            model_performances = []
            
            for model_type in comparison_models:
                try:
                    logging.info(f"Testing model: {model_type}")
                    forecast_data, forecast_insight, forecast_error = ForecastingModels.create_forecast(
                        df, date_col, value_col, model_type=model_type)
                    
                    if forecast_data and forecast_error is None:
                        metrics = forecast_data['metrics']
                        comparison_results[model_type] = {
                            'model_name': forecast_data['model_name'],
                            'metrics': metrics,
                            'insight': forecast_insight,
                            'status': 'success',
                            'forecast_data': forecast_data
                        }
                        
                        # Add to performance tracking
                        model_performances.append({
                            'model_type': model_type,
                            'model_name': forecast_data['model_name'],
                            'mae': metrics['mae'],
                            'rmse': metrics['rmse'],
                            'r2': metrics.get('r2'),
                            'mape': metrics.get('mape'),
                            'theil_u': metrics.get('theil_u')
                        })
                        
                    else:
                        comparison_results[model_type] = {
                            'model_name': model_type.replace('_', ' ').title(),
                            'status': 'failed',
                            'error': forecast_error or 'Unknown error'
                        }
                        
                except Exception as e:
                    logging.error(f"Error with model {model_type}: {str(e)}")
                    comparison_results[model_type] = {
                        'model_name': model_type.replace('_', ' ').title(),
                        'status': 'failed',
                        'error': str(e)
                    }
            
            # Analyze results
            successful_models = [p for p in model_performances if p['mae'] is not None]
            
            if not successful_models:
                raise ValueError("All models failed to run successfully")
            
            # Rank models by different metrics
            rankings = {
                'by_mae': sorted(successful_models, key=lambda x: x['mae']),
                'by_rmse': sorted(successful_models, key=lambda x: x['rmse']),
            }
            
            # Add R² ranking (higher is better)
            r2_models = [m for m in successful_models if m['r2'] is not None]
            if r2_models:
                rankings['by_r2'] = sorted(r2_models, key=lambda x: x['r2'], reverse=True)
            
            # Determine overall best model (using MAE as primary metric)
            best_model = rankings['by_mae'][0]
            
            # Create comparison table
            comparison_table = []
            for rank, model in enumerate(rankings['by_mae'], 1):
                comparison_table.append({
                    'rank': rank,
                    'model': model['model_name'],
                    'model_key': model['model_type'],
                    'mae': model['mae'],
                    'rmse': model['rmse'],
                    'r2': model['r2'],
                    'mape': model['mape'],
                    'theil_u': model['theil_u'],
                    'status': 'success'
                })
            
            # Add failed models
            for model_key, model_data in comparison_results.items():
                if model_data['status'] == 'failed':
                    comparison_table.append({
                        'rank': None,
                        'model': model_data['model_name'],
                        'model_key': model_key,
                        'mae': None, 'rmse': None, 'r2': None, 'mape': None, 'theil_u': None,
                        'status': 'failed',
                        'error': model_data.get('error', 'Unknown error')
                    })
            
            # Generate insights
            insights = [
                f"Compared {len(comparison_models)} forecasting models",
                f"Successfully trained {len(successful_models)} models",
                f"Best performing model: {best_model['model_name']} (MAE: {best_model['mae']:.2f})",
            ]
            
            if len(successful_models) > 1:
                worst_model = rankings['by_mae'][-1]
                improvement = ((worst_model['mae'] - best_model['mae']) / worst_model['mae']) * 100
                insights.append(f"Best model is {improvement:.1f}% better than worst performing model")
            
            # Add model-specific insights
            if any(m['model_type'] == 'naive' for m in successful_models):
                naive_performance = next(m for m in successful_models if m['model_type'] == 'naive')
                if best_model['model_type'] != 'naive':
                    improvement_over_naive = ((naive_performance['mae'] - best_model['mae']) / naive_performance['mae']) * 100
                    insights.append(f"Best model improves {improvement_over_naive:.1f}% over naive baseline")
            
            return {
                'comparison_type': 'single-series',
                'comparison_results': comparison_results,
                'comparison_table': comparison_table,
                'rankings': rankings,
                'best_model': {
                    'model_key': best_model['model_type'],
                    'model_name': best_model['model_name'],
                    'metrics': {k: v for k, v in best_model.items() if k not in ['model_type', 'model_name']}
                },
                'performance_summary': {
                    'total_models_tested': len(comparison_models),
                    'successful_models': len(successful_models),
                    'failed_models': len(comparison_models) - len(successful_models),
                    'best_mae': best_model['mae'],
                    'mae_range': [min(m['mae'] for m in successful_models), max(m['mae'] for m in successful_models)]
                },
                'insights': insights,
                'data_info': {
                    'total_points': len(df),
                    'date_range': f"{df[date_col].min().strftime('%Y-%m-%d')} to {df[date_col].max().strftime('%Y-%m-%d')}"
                }
            }
                
        except Exception as e:
            logging.error(f"Model comparison error: {str(e)}")
            raise

    @staticmethod
    def _compare_models_by_category(df, params):
        """Compare models for each category separately"""
        date_col = params.get('date_col', 'date')
        value_col = params.get('value_col', 'value')
        comparison_models = params.get('comparison_models', ['linear_regression', 'random_forest', 'ets', 'naive'])
        category_col = params.get('category_col')
        
        categories = df[category_col].unique()
        category_comparisons = {}
        overall_best_models = {}
        
        for cat in categories:
            cat_data = df[df[category_col] == cat].copy()
            
            if len(cat_data) < 10:
                category_comparisons[cat] = {
                    'status': 'failed',
                    'error': f'Insufficient data for category {cat} (need at least 10 points, got {len(cat_data)})'
                }
                continue
            
            try:
                # Create single-series comparison for this category
                single_params = {k: v for k, v in params.items() if k != 'category_col'}
                comparison_result = ForecastingModels.compare_forecasting_models(cat_data, single_params)
                
                category_comparisons[cat] = {
                    'comparison': comparison_result,
                    'status': 'success',
                    'data_points': len(cat_data)
                }
                
                # Track best model for this category
                if comparison_result.get('best_model'):
                    overall_best_models[cat] = comparison_result['best_model']
                    
            except Exception as e:
                logging.error(f"Category {cat} comparison failed: {str(e)}")
                category_comparisons[cat] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Summarize results
        successful_categories = len([c for c in category_comparisons.values() if c['status'] == 'success'])
        total_categories = len(categories)
        
        # Find most frequently best model across categories
        best_model_counts = {}
        for cat_result in category_comparisons.values():
            if cat_result['status'] == 'success' and 'comparison' in cat_result:
                best_model_key = cat_result['comparison']['best_model']['model_key']
                best_model_counts[best_model_key] = best_model_counts.get(best_model_key, 0) + 1
        
        most_frequent_best = max(best_model_counts.items(), key=lambda x: x[1]) if best_model_counts else None
        
        insights = [
            f"Analyzed {total_categories} categories with {len(comparison_models)} models each",
            f"Successfully compared models for {successful_categories} categories",
        ]
        
        if most_frequent_best:
            insights.append(f"Most frequently best model: {most_frequent_best[0]} (best in {most_frequent_best[1]} categories)")
        
        return {
            'comparison_type': 'multi-category',
            'categories': list(categories),
            'category_comparisons': category_comparisons,
            'models_tested': comparison_models,
            'summary': {
                'total_categories': total_categories,
                'successful_comparisons': successful_categories,
                'failed_comparisons': total_categories - successful_categories,
                'most_frequent_best_model': most_frequent_best[0] if most_frequent_best else None,
                'best_model_frequency': best_model_counts
            },
            'overall_best_models': overall_best_models,
            'insights': insights
        }