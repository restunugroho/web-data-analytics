import pandas as pd
import numpy as np
import plotly.express as px
from utils import parse_datetime_flexible, clean_float, sanitize_for_json
from forecasting_models import ForecastingModels
import logging

class AnalysisModules:
    @staticmethod
    def analyze_predictability(data, value_column):
        """Analyze how predictable the data is"""
        values = data[value_column].values
        cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else float('inf')
        
        if len(values) > 10:
            diffs = np.diff(values)
            try:
                autocorr_1 = np.corrcoef(diffs[:-1], diffs[1:])[0, 1] if len(diffs) > 1 else 0
                if np.isnan(autocorr_1):
                    autocorr_1 = 0
                
                # Predictability categories
                if cv < 0.1:
                    predictability = "Very High - Data is very stable and predictable"
                elif cv < 0.3:
                    predictability = "High - Data shows consistent patterns"
                elif cv < 0.6:
                    predictability = "Moderate - Some variability but patterns are detectable"
                elif cv < 1.0:
                    predictability = "Low - High variability makes forecasting challenging"
                else:
                    predictability = "Very Low - Data is highly volatile and unpredictable"
                
                # Random walk characteristics
                if abs(autocorr_1) < 0.1:
                    random_walk_insight = "Data shows random walk characteristics - changes are largely unpredictable"
                elif autocorr_1 > 0.3:
                    random_walk_insight = "Data shows momentum - recent changes tend to continue"
                elif autocorr_1 < -0.3:
                    random_walk_insight = "Data shows mean reversion - tends to return to average after changes"
                else:
                    random_walk_insight = "Data shows weak autocorrelation patterns"
                
                return predictability, random_walk_insight, float(cv)
            except:
                return "Cannot determine - insufficient variation in data", "Analysis inconclusive", float(cv)
        else:
            return "Cannot determine - insufficient data points", "Need more data for analysis", float(cv)

    @staticmethod
    def analyze_detailed_seasonality(data, date_column, value_column):
        """Analyze seasonality patterns with detailed insights"""
        seasonality_insights = []
        
        data['month'] = data[date_column].dt.month
        data['day_of_week'] = data[date_column].dt.day_of_week
        
        # Monthly analysis
        monthly_avg = data.groupby('month')[value_column].mean()
        if len(monthly_avg) >= 12:
            peak_month = monthly_avg.idxmax()
            low_month = monthly_avg.idxmin()
            month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            seasonality_insights.append(f"Monthly pattern: Peak in {month_names[peak_month]}, lowest in {month_names[low_month]}")
            
            monthly_strength = monthly_avg.std() / monthly_avg.mean() if monthly_avg.mean() != 0 else 0
            if monthly_strength > 0.2:
                seasonality_insights.append(f"Strong monthly seasonality detected (strength: {monthly_strength:.3f})")
            else:
                seasonality_insights.append(f"Weak monthly seasonality (strength: {monthly_strength:.3f})")
        else:
            monthly_strength = 0
        
        # Weekly analysis
        weekly_avg = data.groupby('day_of_week')[value_column].mean()
        if len(weekly_avg) >= 7:
            peak_day = weekly_avg.idxmax()
            low_day = weekly_avg.idxmin()
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            seasonality_insights.append(f"Weekly pattern: Peak on {day_names[peak_day]}, lowest on {day_names[low_day]}")
            
            weekly_strength = weekly_avg.std() / weekly_avg.mean() if weekly_avg.mean() != 0 else 0
            if weekly_strength > 0.1:
                seasonality_insights.append(f"Notable weekly seasonality (strength: {weekly_strength:.3f})")
        else:
            weekly_strength = 0
        
        return seasonality_insights, monthly_strength, weekly_strength

    @staticmethod
    def analyze_time_series(df, params):
        """Enhanced time series analysis"""
        try:
            date_col = params.get('date_col', 'date')
            value_col = params.get('value_col', 'value')
            category_col = params.get('category_col', None)
            model_type = params.get('model_type', 'linear_regression')

            logging.info(date_col)
            logging.info(value_col)
            logging.info(category_col)
            logging.info(model_type)
            
            logging.info('analyze time series in analysis module')
            logging.info(df.shape)
            logging.info(df)
            df[date_col] = parse_datetime_flexible(df[date_col])
            logging.info(df)
            df = df.dropna(subset=[date_col]).sort_values(date_col)
            
            logging.info(date_col)
            logging.info(df.shape)

            if df.empty:
                raise ValueError("No valid datetime data found")
            
            df['days_from_start'] = (df[date_col] - df[date_col].min()).dt.days
            
            # Multi-category analysis
            if category_col and category_col in df.columns:
                categories = df[category_col].unique()
                category_stats = {}
                category_decompositions = {}
                category_forecasts = {}
                category_seasonality_insights = {}
                
                for cat in categories:
                    cat_data = df[df[category_col] == cat].copy()
                    if len(cat_data) > 1:
                        # Basic stats
                        slope = np.polyfit(cat_data['days_from_start'], cat_data[value_col], 1)[0] if len(cat_data) > 1 and cat_data['days_from_start'].std() > 0 else 0
                        predictability, random_walk_insight, cv = AnalysisModules.analyze_predictability(cat_data, value_col)
                        
                        category_stats[cat] = {
                            "mean": clean_float(cat_data[value_col].mean()),
                            "std": clean_float(cat_data[value_col].std()),
                            "min": clean_float(cat_data[value_col].min()),
                            "max": clean_float(cat_data[value_col].max()),
                            "trend_slope": clean_float(slope),
                            "trend": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                            "predictability": predictability,
                            "random_walk_insight": random_walk_insight,
                            "coefficient_variation": clean_float(cv)
                        }
                        
                        # Seasonality analysis
                        if len(cat_data) >= 7:
                            seasonality_insights, monthly_strength, weekly_strength = AnalysisModules.analyze_detailed_seasonality(cat_data, date_col, value_col)
                            category_seasonality_insights[cat] = seasonality_insights
                        
                        # Forecasting
                        if len(cat_data) >= 10:
                            forecast_data, forecast_insight, forecast_error = ForecastingModels.create_forecast(cat_data, date_col, value_col, model_type=model_type)
                            if forecast_data:
                                category_forecasts[cat] = forecast_data
                
                # Overall analysis
                overall_slope = np.polyfit(df['days_from_start'], df[value_col], 1)[0] if len(df) > 1 and df['days_from_start'].std() > 0 else 0
                overall_predictability, overall_random_walk, overall_cv = AnalysisModules.analyze_predictability(df, value_col)
                overall_seasonality_insights, monthly_strength, weekly_strength = AnalysisModules.analyze_detailed_seasonality(df, date_col, value_col)
                
                stats = {
                    "mean": clean_float(df[value_col].mean()),
                    "std": clean_float(df[value_col].std()),
                    "min": clean_float(df[value_col].min()),
                    "max": clean_float(df[value_col].max()),
                    "trend_slope": clean_float(overall_slope),
                    "trend": "increasing" if overall_slope > 0 else "decreasing" if overall_slope < 0 else "stable",
                    "categories": category_stats,
                    "predictability": overall_predictability,
                    "random_walk_insight": overall_random_walk,
                    "coefficient_variation": clean_float(overall_cv)
                }
                
                fig = px.line(df, x=date_col, y=value_col, color=category_col, title="Multi Time Series Analysis")
                
            else:
                # Single series analysis
                slope = np.polyfit(df['days_from_start'], df[value_col], 1)[0] if len(df) > 1 and df['days_from_start'].std() > 0 else 0
                predictability, random_walk_insight, cv = AnalysisModules.analyze_predictability(df, value_col)
                seasonality_insights, monthly_strength, weekly_strength = AnalysisModules.analyze_detailed_seasonality(df, date_col, value_col)
                
                stats = {
                    "mean": float(df[value_col].mean()),
                    "std": float(df[value_col].std()),
                    "min": float(df[value_col].min()),
                    "max": float(df[value_col].max()),
                    "trend_slope": float(slope),
                    "trend": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                    "predictability": predictability,
                    "random_walk_insight": random_walk_insight,
                    "coefficient_variation": float(cv)
                }
                
                fig = px.line(df, x=date_col, y=value_col, title="Time Series Analysis")
                fig.update_traces(line=dict(color='#3B82F6', width=2))
                
                category_decompositions = {}
                category_forecasts = {}
                category_seasonality_insights = {}
            
            # Enhanced styling
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#2E4057'), title_font_size=16,
                xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
                yaxis=dict(gridcolor='rgba(128,128,128,0.2)')
            )
            
            # Decomposition and forecast
            overall_decomposition = None
            overall_forecast = None
            
            if len(df) >= 10:
                df_sorted = df.sort_values(date_col).reset_index(drop=True)
                x_vals = np.arange(len(df_sorted))
                trend_line = np.polyval(np.polyfit(x_vals, df_sorted[value_col], 1), x_vals)
                
                window_size = min(12, len(df_sorted) // 2)
                if window_size >= 3:
                    seasonal = df_sorted[value_col] - pd.Series(trend_line)
                    seasonal_smooth = seasonal.rolling(window=window_size, center=True).mean().fillna(0)
                else:
                    seasonal_smooth = np.zeros(len(df_sorted))
                
                overall_decomposition = {
                    'dates': df_sorted[date_col].dt.strftime('%Y-%m-%d').tolist(),
                    'original': df_sorted[value_col].tolist(),
                    'trend': trend_line.tolist(),
                    'seasonal': seasonal_smooth.tolist()
                }
                
                forecast_data, forecast_insight, forecast_error = ForecastingModels.create_forecast(df_sorted, date_col, value_col, model_type=model_type)
                if forecast_data:
                    overall_forecast = forecast_data
            
            # Additional metrics
            seasonality_strength = float(monthly_strength)
            weekly_seasonality = float(weekly_strength)
            volatility = float(df[value_col].std() / df[value_col].mean()) if df[value_col].mean() != 0 else 0
            
            # Growth rate
            if len(df) > 1:
                first_value = df[value_col].iloc[0]
                last_value = df[value_col].iloc[-1]
                growth_rate = ((last_value - first_value) / first_value) * 100 if first_value != 0 else 0
            else:
                growth_rate = 0
            
            # Enhanced insights
            insights = [
                f"Average value: {stats['mean']:.2f}",
                f"Overall trend: {stats['trend']} (slope: {stats['trend_slope']:.4f} per day)",
                f"Predictability: {stats['predictability']}",
                f"Random walk analysis: {stats['random_walk_insight']}",
                f"Data span: {df[date_col].min().strftime('%Y-%m-%d')} to {df[date_col].max().strftime('%Y-%m-%d')}",
                f"Total growth: {growth_rate:.1f}%",
                f"Volatility: {'High' if volatility > 0.5 else 'Moderate' if volatility > 0.2 else 'Low'} (CV: {volatility:.3f})"
            ]
            
            # Add seasonality insights
            if category_col and category_col in df.columns:
                insights.extend(overall_seasonality_insights[:2])
            else:
                insights.extend(seasonality_insights[:2])
            
            response_data = {
                "module": "time_series",
                "statistics": stats,
                "plot": fig.to_json(),
                "insights": insights,
                "seasonality_strength": seasonality_strength,
                "weekly_seasonality": weekly_seasonality,
                "volatility": volatility,
                "growth_rate": growth_rate,
                "has_categories": category_col is not None and category_col in df.columns,
                "category_decompositions": category_decompositions,
                "category_forecasts": category_forecasts,
                "category_seasonality_insights": category_seasonality_insights,
                "overall_decomposition": overall_decomposition,
                "overall_forecast": overall_forecast,
                "categories_list": list(categories) if category_col and category_col in df.columns else [],
                "seasonality_insights": overall_seasonality_insights if category_col and category_col in df.columns else seasonality_insights,
                "selected_model": model_type
            }
        
            return sanitize_for_json(response_data)
            
        except Exception as e:
            logging.error(f"Internal error in time series analysis: {e}")
            raise

    @staticmethod
    def analyze_customer(df, params):
        """Enhanced customer analysis (RFM-like)"""
        customer_col = params.get('customer_col', 'customer_id')
        amount_col = params.get('amount_col', 'amount')
        date_col = params.get('date_col', 'transaction_date')
        
        df[date_col] = parse_datetime_flexible(df[date_col])
        df = df.dropna(subset=[date_col])
        reference_date = df[date_col].max()
        
        # Calculate RFM metrics
        rfm = df.groupby(customer_col).agg({
            date_col: lambda x: (reference_date - x.max()).days,  # Recency
            amount_col: ['count', 'sum']  # Frequency, Monetary
        }).reset_index()
        
        rfm.columns = [customer_col, 'recency', 'frequency', 'monetary']
        
        # Simple segmentation
        rfm['recency_score'] = pd.qcut(rfm['recency'], 3, labels=['3', '2', '1'])
        rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 3, labels=['1', '2', '3'])
        rfm['monetary_score'] = pd.qcut(rfm['monetary'], 3, labels=['1', '2', '3'])
        
        # Create segments
        def segment_customers(row):
            if row['frequency_score'] == '3' and row['monetary_score'] == '3':
                return 'Champions'
            elif row['recency_score'] == '3' and row['frequency_score'] == '3':
                return 'Loyal Customers'
            elif row['recency_score'] == '1':
                return 'At Risk'
            else:
                return 'Others'
        
        rfm['segment'] = rfm.apply(segment_customers, axis=1)
        segment_counts = rfm['segment'].value_counts().to_dict()
        
        # Create visualization
        fig = px.scatter(rfm, x='frequency', y='monetary', color='segment', 
                         title="Customer Segmentation", hover_data=['recency'])
        plot_json = fig.to_json()
        
        insights = [
            f"Total customers analyzed: {len(rfm)}",
            f"Champions: {segment_counts.get('Champions', 0)} customers",
            f"At Risk customers: {segment_counts.get('At Risk', 0)} customers"
        ]
        
        return {
            "module": "customer",
            "segments": segment_counts,
            "plot": plot_json,
            "insights": insights,
            "rfm_summary": {
                "avg_recency": float(rfm['recency'].mean()),
                "avg_frequency": float(rfm['frequency'].mean()),
                "avg_monetary": float(rfm['monetary'].mean())
            }
        }