import pandas as pd
import numpy as np
import plotly.express as px
from utils import parse_datetime_flexible, clean_float, sanitize_for_json
import logging
from scipy import signal
from scipy.stats import pearsonr

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
    def analyze_pattern_types(data, value_column, date_column=None):
        """Deep analysis of pattern types in time series data"""
        values = data[value_column].values
        n = len(values)
        
        if n < 10:
            return {
                'pattern_type': 'Insufficient Data',
                'confidence': 0.0,
                'characteristics': ['Need at least 10 data points for pattern analysis'],
                'pattern_strength': 0.0,
                'volatility_type': 'Unknown',
                'trend_consistency': 0.0,
                'recommendations': ['Collect more data points for reliable pattern analysis']
            }
        
        # Calculate various metrics
        diffs = np.diff(values)
        returns = diffs / values[:-1] if np.all(values[:-1] != 0) else diffs
        
        # Autocorrelation analysis
        autocorr_1 = np.corrcoef(values[:-1], values[1:])[0, 1] if n > 1 else 0
        if np.isnan(autocorr_1):
            autocorr_1 = 0
        
        # Variance ratio test (for random walk)
        if n >= 4:
            returns_2 = np.array([np.sum(returns[i:i+2]) for i in range(0, len(returns)-1, 2)])
            var_1 = np.var(returns) if len(returns) > 1 else 0
            var_2 = np.var(returns_2) / 2 if len(returns_2) > 1 else 0
            variance_ratio = var_2 / var_1 if var_1 != 0 else 1
        else:
            variance_ratio = 1
        
        # Hurst exponent (for persistence/mean reversion)
        hurst_exp = AnalysisModules._calculate_hurst_exponent(values)
        
        # Trend consistency
        if date_column and date_column in data.columns:
            data_sorted = data.sort_values(date_column)
            trend_consistency = AnalysisModules._calculate_trend_consistency(data_sorted[value_column].values)
        else:
            trend_consistency = AnalysisModules._calculate_trend_consistency(values)
        
        # Volatility clustering
        volatility_clustering = AnalysisModules._calculate_volatility_clustering(returns)
        
        # Determine pattern type based on metrics
        pattern_analysis = AnalysisModules._classify_pattern_type(
            autocorr_1, variance_ratio, hurst_exp, trend_consistency, 
            volatility_clustering, returns
        )
        
        return pattern_analysis

    @staticmethod
    def _calculate_hurst_exponent(values):
        """Calculate Hurst exponent to detect persistence/mean reversion"""
        try:
            n = len(values)
            if n < 20:
                return 0.5  # Neutral value for insufficient data
            
            # R/S analysis
            lags = range(2, min(n//4, 20))
            rs_values = []
            
            for lag in lags:
                # Split series into non-overlapping windows
                reshaped = values[:n//lag * lag].reshape(-1, lag)
                
                # Calculate R/S for each window
                rs_window = []
                for window in reshaped:
                    mean_window = np.mean(window)
                    deviations = np.cumsum(window - mean_window)
                    R = np.max(deviations) - np.min(deviations)
                    S = np.std(window)
                    if S != 0:
                        rs_window.append(R/S)
                
                if rs_window:
                    rs_values.append(np.mean(rs_window))
            
            if len(rs_values) < 3:
                return 0.5
            
            # Linear regression of log(R/S) vs log(lag)
            log_lags = np.log(lags[:len(rs_values)])
            log_rs = np.log(rs_values)
            
            # Remove infinite or NaN values
            valid_mask = np.isfinite(log_lags) & np.isfinite(log_rs)
            if np.sum(valid_mask) < 3:
                return 0.5
            
            hurst = np.polyfit(log_lags[valid_mask], log_rs[valid_mask], 1)[0]
            return np.clip(hurst, 0, 1)  # Ensure valid range
            
        except:
            return 0.5  # Neutral value if calculation fails

    @staticmethod
    def _calculate_trend_consistency(values):
        """Calculate how consistent the trend direction is"""
        if len(values) < 3:
            return 0.0
        
        diffs = np.diff(values)
        if len(diffs) == 0:
            return 0.0
        
        # Count direction changes
        signs = np.sign(diffs)
        sign_changes = np.sum(np.diff(signs) != 0)
        max_possible_changes = len(signs) - 1
        
        if max_possible_changes == 0:
            return 1.0
        
        consistency = 1 - (sign_changes / max_possible_changes)
        return consistency

    @staticmethod
    def _calculate_volatility_clustering(returns):
        """Detect volatility clustering (GARCH effects)"""
        if len(returns) < 10:
            return 0.0
        
        try:
            # Calculate rolling volatility
            window = min(5, len(returns) // 3)
            if window < 2:
                return 0.0
            
            rolling_vol = pd.Series(returns).rolling(window=window).std().dropna()
            if len(rolling_vol) < 3:
                return 0.0
            
            # Autocorrelation of volatility
            vol_autocorr = np.corrcoef(rolling_vol[:-1], rolling_vol[1:])[0, 1]
            return vol_autocorr if not np.isnan(vol_autocorr) else 0.0
        except:
            return 0.0

    @staticmethod
    def _classify_pattern_type(autocorr_1, variance_ratio, hurst_exp, trend_consistency, 
                            volatility_clustering, returns):
        """Classify the pattern type based on calculated metrics"""
        
        characteristics = []
        recommendations = []
        
        # Determine primary pattern type
        if abs(variance_ratio - 1) < 0.1 and abs(autocorr_1) < 0.1 and abs(hurst_exp - 0.5) < 0.1:
            pattern_type = "Random Walk"
            confidence = 0.8
            characteristics = [
                "Changes are largely unpredictable",
                "Past values don't predict future values",
                "Variance increases linearly with time",
                "No systematic patterns detected"
            ]
            recommendations = [
                "Forecasting will be challenging",
                "Focus on risk management rather than prediction",
                "Consider ensemble methods for any forecasting attempts",
                "Monitor for regime changes"
            ]
            pattern_strength = abs(variance_ratio - 1) + abs(autocorr_1) + abs(hurst_exp - 0.5)
            
        elif autocorr_1 > 0.3 or hurst_exp > 0.65 or trend_consistency > 0.7:
            pattern_type = "Momentum/Persistence"
            confidence = min(0.9, max(abs(autocorr_1), hurst_exp, trend_consistency))
            characteristics = [
                "Trends tend to continue in the same direction",
                "Strong positive autocorrelation detected",
                "High trend consistency observed",
                "Persistent behavior in value changes"
            ]
            recommendations = [
                "Trend-following models may work well",
                "Consider ARIMA or exponential smoothing",
                "Monitor for trend reversals",
                "Momentum-based strategies could be effective"
            ]
            pattern_strength = max(autocorr_1, hurst_exp, trend_consistency)
            
        elif hurst_exp < 0.35 or autocorr_1 < -0.2:
            pattern_type = "Mean Reversion"
            confidence = min(0.9, max(abs(autocorr_1), 1 - hurst_exp))
            characteristics = [
                "Values tend to return to long-term average",
                "Negative autocorrelation indicates mean reversion",
                "Overshoots are typically corrected",
                "Anti-persistent behavior detected"
            ]
            recommendations = [
                "Mean reversion models are suitable",
                "Consider Ornstein-Uhlenbeck process",
                "Look for support/resistance levels",
                "Contrarian strategies may be effective"
            ]
            pattern_strength = max(abs(autocorr_1), 1 - hurst_exp)
            
        elif volatility_clustering > 0.3 and abs(autocorr_1) < 0.3:
            pattern_type = "Volatility Clustering"
            confidence = volatility_clustering
            characteristics = [
                "Periods of high volatility followed by high volatility",
                "Periods of low volatility followed by low volatility",
                "GARCH effects detected",
                "Volatility is predictable even if values aren't"
            ]
            recommendations = [
                "Consider GARCH models for volatility forecasting",
                "Risk management is crucial",
                "Focus on volatility rather than level predictions",
                "Adaptive models may work well"
            ]
            pattern_strength = volatility_clustering
            
        elif 0.1 <= abs(autocorr_1) <= 0.3 or 0.4 <= hurst_exp <= 0.6:
            pattern_type = "Weak Patterns"
            confidence = max(abs(autocorr_1), abs(hurst_exp - 0.5))
            characteristics = [
                "Some predictable elements detected",
                "Mixed signals from different metrics",
                "Moderate autocorrelation present",
                "Pattern strength is limited"
            ]
            recommendations = [
                "Try multiple modeling approaches",
                "Ensemble methods may capture weak signals",
                "Focus on short-term predictions",
                "Combine with external indicators"
            ]
            pattern_strength = max(abs(autocorr_1), abs(hurst_exp - 0.5))
            
        elif len(returns) > 0 and np.std(returns) > 2 * np.mean(np.abs(returns)):
            pattern_type = "High Volatility/Chaotic"
            confidence = min(0.8, np.std(returns) / (np.mean(np.abs(returns)) + 1e-10))
            characteristics = [
                "Extremely high volatility detected",
                "Potential chaotic behavior",
                "Large unpredictable swings",
                "Non-linear dynamics possible"
            ]
            recommendations = [
                "Extreme caution in forecasting",
                "Consider regime-switching models",
                "Focus on risk management",
                "Look for structural breaks"
            ]
            pattern_strength = np.std(returns) / (np.mean(np.abs(returns)) + 1e-10)
            
        else:
            pattern_type = "Mixed/Complex"
            confidence = 0.4
            characteristics = [
                "Multiple competing patterns detected",
                "Complex behavior that doesn't fit standard categories",
                "May have regime changes or structural breaks",
                "Requires deeper investigation"
            ]
            recommendations = [
                "Consider regime-switching models",
                "Analyze sub-periods separately",
                "Look for external factors",
                "Use robust forecasting methods"
            ]
            pattern_strength = 0.3
        
        # Determine volatility type
        if len(returns) > 0:
            cv = np.std(returns) / (np.mean(np.abs(returns)) + 1e-10)
            if cv < 0.5:
                volatility_type = "Low Volatility"
            elif cv < 1.5:
                volatility_type = "Moderate Volatility"
            elif cv < 3.0:
                volatility_type = "High Volatility"
            else:
                volatility_type = "Extreme Volatility"
        else:
            volatility_type = "Unknown"
        
        # Add specific metrics to characteristics
        characteristics.extend([
            f"Autocorrelation: {autocorr_1:.3f}",
            f"Hurst Exponent: {hurst_exp:.3f}",
            f"Trend Consistency: {trend_consistency:.3f}",
            f"Variance Ratio: {variance_ratio:.3f}"
        ])
        
        return {
            'pattern_type': pattern_type,
            'confidence': float(confidence),
            'characteristics': characteristics,
            'pattern_strength': float(pattern_strength),
            'volatility_type': volatility_type,
            'trend_consistency': float(trend_consistency),
            'recommendations': recommendations,
            'metrics': {
                'autocorrelation': float(autocorr_1),
                'hurst_exponent': float(hurst_exp),
                'variance_ratio': float(variance_ratio),
                'volatility_clustering': float(volatility_clustering)
            }
        }

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
    def create_time_series_decomposition(df, date_column, value_column):
        """Create time series decomposition (trend, seasonal, residual) with seasonality analysis"""
        if len(df) < 10:
            return None
            
        df_sorted = df.sort_values(date_column).reset_index(drop=True)
        x_vals = np.arange(len(df_sorted))
        
        # Trend component using linear regression
        trend_line = np.polyval(np.polyfit(x_vals, df_sorted[value_column], 1), x_vals)
        
        # Seasonal component using moving average
        window_size = min(12, len(df_sorted) // 2)
        if window_size >= 3:
            seasonal = df_sorted[value_column] - pd.Series(trend_line)
            seasonal_smooth = seasonal.rolling(window=window_size, center=True).mean().fillna(0)
        else:
            seasonal_smooth = np.zeros(len(df_sorted))
        
        # Residual component
        residual = df_sorted[value_column] - trend_line - seasonal_smooth
        
        # Calculate seasonality strength
        seasonality_strength = AnalysisModules.calculate_seasonality_strength(
            df_sorted[value_column].values, 
            seasonal_smooth.values if hasattr(seasonal_smooth, 'values') else seasonal_smooth,
            residual.values if hasattr(residual, 'values') else residual
        )
        
        # Detect seasonality period
        seasonality_period = AnalysisModules.detect_seasonality_period(df_sorted[value_column].values)
        
        # Classify seasonality strength
        strength_category = AnalysisModules.classify_seasonality_strength(seasonality_strength)
        
        return {
            'dates': df_sorted[date_column].dt.strftime('%Y-%m-%d').tolist(),
            'original': df_sorted[value_column].tolist(),
            'trend': trend_line.tolist(),
            'seasonal': seasonal_smooth.tolist() if hasattr(seasonal_smooth, 'tolist') else seasonal_smooth.tolist(),
            'residual': residual.tolist() if hasattr(residual, 'tolist') else residual.tolist(),
            'seasonality_analysis': {
                'strength': round(seasonality_strength, 4),
                'strength_category': strength_category,
                'period_points': seasonality_period,
                'interpretation': AnalysisModules.interpret_seasonality(seasonality_strength, seasonality_period, len(df_sorted))
            }
        }

    @staticmethod
    def calculate_seasonality_strength(original, seasonal, residual):
        """
        Calculate seasonality strength using variance-based method
        Seasonality Strength = Var(Seasonal) / (Var(Seasonal) + Var(Residual))
        """
        try:
            # Remove any NaN or infinite values
            seasonal_clean = seasonal[~np.isnan(seasonal) & ~np.isinf(seasonal)]
            residual_clean = residual[~np.isnan(residual) & ~np.isinf(residual)]
            
            if len(seasonal_clean) == 0 or len(residual_clean) == 0:
                return 0.0
                
            var_seasonal = np.var(seasonal_clean)
            var_residual = np.var(residual_clean)
            
            # Avoid division by zero
            if var_seasonal + var_residual == 0:
                return 0.0
                
            strength = var_seasonal / (var_seasonal + var_residual)
            return max(0.0, min(1.0, strength))  # Clamp between 0 and 1
            
        except Exception:
            return 0.0

    @staticmethod
    def detect_seasonality_period(data):
        """
        Detect seasonality period using autocorrelation and FFT
        """
        try:
            if len(data) < 6:
                return None
                
            # Method 1: Autocorrelation
            autocorr_period = AnalysisModules.find_autocorr_period(data)
            
            # Method 2: FFT (Frequency domain analysis)
            fft_period = AnalysisModules.find_fft_period(data)
            
            # Choose the most reliable period
            if autocorr_period is not None and fft_period is not None:
                # If both methods agree (within 20%), use autocorr result
                if abs(autocorr_period - fft_period) / max(autocorr_period, fft_period) < 0.2:
                    return autocorr_period
                # Otherwise, use the one that makes more sense given data length
                elif autocorr_period <= len(data) // 3:
                    return autocorr_period
                elif fft_period <= len(data) // 3:
                    return fft_period
            
            # Return whichever method found a reasonable period
            if autocorr_period is not None and autocorr_period <= len(data) // 3:
                return autocorr_period
            elif fft_period is not None and fft_period <= len(data) // 3:
                return fft_period
                
            return None
            
        except Exception:
            return None

    @staticmethod
    def find_autocorr_period(data):
        """Find period using autocorrelation"""
        try:
            # Remove trend first
            detrended = signal.detrend(data)
            
            # Calculate autocorrelation
            correlation = np.correlate(detrended, detrended, mode='full')
            correlation = correlation[correlation.size // 2:]
            
            # Find peaks in autocorrelation
            if len(correlation) < 3:
                return None
                
            # Look for the first significant peak after lag 1
            max_lag = min(len(correlation) - 1, len(data) // 2)
            
            for lag in range(2, max_lag):
                # Check if this is a local maximum
                if (correlation[lag] > correlation[lag-1] and 
                    correlation[lag] > correlation[lag+1] and
                    correlation[lag] > 0.1 * correlation[0]):  # At least 10% of max correlation
                    return lag
                    
            return None
            
        except Exception:
            return None

    @staticmethod
    def find_fft_period(data):
        """Find period using FFT"""
        try:
            # Remove trend
            detrended = signal.detrend(data)
            
            # Apply FFT
            fft = np.fft.fft(detrended)
            freqs = np.fft.fftfreq(len(data))
            
            # Get magnitude spectrum (positive frequencies only)
            magnitude = np.abs(fft[:len(fft)//2])
            freqs = freqs[:len(freqs)//2]
            
            # Find the dominant frequency (excluding DC component)
            if len(magnitude) < 2:
                return None
                
            # Skip DC component (index 0)
            dominant_freq_idx = np.argmax(magnitude[1:]) + 1
            dominant_freq = freqs[dominant_freq_idx]
            
            # Convert frequency to period
            if dominant_freq > 0:
                period = 1.0 / dominant_freq
                # Return period only if it's reasonable (between 2 and half the data length)
                if 2 <= period <= len(data) // 2:
                    return int(round(period))
                    
            return None
            
        except Exception:
            return None

    @staticmethod
    def classify_seasonality_strength(strength):
        """Classify seasonality strength into categories"""
        if strength >= 0.7:
            return "Very Strong"
        elif strength >= 0.5:
            return "Strong"
        elif strength >= 0.3:
            return "Moderate"
        elif strength >= 0.1:
            return "Weak"
        else:
            return "Very Weak/No Seasonality"

    @staticmethod
    def interpret_seasonality(strength, period, data_length):
        """Provide interpretation of seasonality analysis"""
        interpretation = []
        
        # Strength interpretation
        strength_desc = AnalysisModules.classify_seasonality_strength(strength)
        interpretation.append(f"Seasonality strength: {strength_desc} ({strength:.3f})")
        
        # Period interpretation
        if period is not None:
            interpretation.append(f"Seasonality repeats approximately every {period} data points")
            
            # Provide context based on period length
            if period <= 7:
                interpretation.append("This suggests a short-term cyclical pattern")
            elif period <= 30:
                interpretation.append("This suggests a medium-term cyclical pattern")
            else:
                interpretation.append("This suggests a long-term cyclical pattern")
                
            # Check if we have enough data to capture multiple cycles
            cycles_captured = data_length / period
            if cycles_captured < 2:
                interpretation.append(f"Warning: Only {cycles_captured:.1f} complete cycles in data - seasonality detection may be unreliable")
            else:
                interpretation.append(f"Data contains approximately {cycles_captured:.1f} complete seasonal cycles")
        else:
            interpretation.append("No clear seasonal period detected")
        
        # Recommendations
        if strength >= 0.3 and period is not None:
            interpretation.append("Recommendation: Seasonality is significant and should be considered in forecasting models")
        elif strength >= 0.1:
            interpretation.append("Recommendation: Weak seasonality detected - consider seasonal adjustment if needed")
        else:
            interpretation.append("Recommendation: No significant seasonality - seasonal models may not be necessary")
        
        return ". ".join(interpretation)

    @staticmethod
    def analyze_time_series(df, params):
        """Enhanced time series analysis - ANALYSIS ONLY (no forecasting)"""
        try:
            date_col = params.get('date_col', 'date')
            value_col = params.get('value_col', 'value')
            category_col = params.get('category_col', None)

            logging.info(f'Time series analysis - Date col: {date_col}, Value col: {value_col}')
            logging.info(f'Data shape: {df.shape}')
            
            # Parse datetime
            df[date_col] = parse_datetime_flexible(df[date_col])
            df = df.dropna(subset=[date_col]).sort_values(date_col)
            
            if df.empty:
                raise ValueError("No valid datetime data found")
            
            df['days_from_start'] = (df[date_col] - df[date_col].min()).dt.days
            
            # Multi-category analysis
            if category_col and category_col in df.columns:
                categories = df[category_col].unique()
                category_stats = {}
                category_seasonality_insights = {}
                category_decompositions = {}
                
                for cat in categories:
                    cat_data = df[df[category_col] == cat].copy()
                    if len(cat_data) > 1:
                        # Basic statistics
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
                            "coefficient_variation": clean_float(cv),
                            "data_points": len(cat_data)
                        }
                        
                        # Seasonality analysis
                        if len(cat_data) >= 7:
                            seasonality_insights, monthly_strength, weekly_strength = AnalysisModules.analyze_detailed_seasonality(cat_data, date_col, value_col)
                            category_seasonality_insights[cat] = {
                                "insights": seasonality_insights,
                                "monthly_strength": monthly_strength,
                                "weekly_strength": weekly_strength
                            }
                        
                        # Enhanced Decomposition with seasonality analysis
                        decomposition = AnalysisModules.create_time_series_decomposition(cat_data, date_col, value_col)
                        if decomposition:
                            category_decompositions[cat] = decomposition
                
                # Overall analysis
                overall_slope = np.polyfit(df['days_from_start'], df[value_col], 1)[0] if len(df) > 1 and df['days_from_start'].std() > 0 else 0
                overall_predictability, overall_random_walk, overall_cv = AnalysisModules.analyze_predictability(df, value_col)
                
                # Deep pattern analysis
                pattern_analysis = AnalysisModules.analyze_pattern_types(df, value_col, date_col)
                
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
                    "coefficient_variation": clean_float(overall_cv),
                    "data_points": len(df)
                }
                
                fig = px.line(df, x=date_col, y=value_col, color=category_col, title="Multi-Category Time Series Analysis")
                
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
                    "coefficient_variation": float(cv),
                    "data_points": len(df)
                }
                
                fig = px.line(df, x=date_col, y=value_col, title="Time Series Analysis")
                fig.update_traces(line=dict(color='#3B82F6', width=2))
                
                category_decompositions = {}
                category_seasonality_insights = {}
            
            # Enhanced styling
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#2E4057'), title_font_size=16,
                xaxis=dict(gridcolor='rgba(128,128,128,0.2)', title="Date"),
                yaxis=dict(gridcolor='rgba(128,128,128,0.2)', title=value_col.title())
            )
            
            # Enhanced Overall decomposition with seasonality analysis
            overall_decomposition = AnalysisModules.create_time_series_decomposition(df, date_col, value_col)
            
            # Additional metrics
            seasonality_strength = float(monthly_strength)
            weekly_seasonality = float(weekly_strength)
            volatility = float(df[value_col].std() / df[value_col].mean()) if df[value_col].mean() != 0 else 0
            
            # Growth rate calculation
            if len(df) > 1:
                first_value = df[value_col].iloc[0]
                last_value = df[value_col].iloc[-1]
                growth_rate = ((last_value - first_value) / first_value) * 100 if first_value != 0 else 0
                
                # Time span
                time_span = (df[date_col].max() - df[date_col].min()).days
                annualized_growth = (growth_rate / time_span) * 365 if time_span > 0 else 0
            else:
                growth_rate = 0
                annualized_growth = 0
                time_span = 0
            
            # Enhanced insights
            insights = [
                f"Dataset contains {len(df)} data points spanning {time_span} days",
                f"Average value: {stats['mean']:.2f} ± {stats['std']:.2f}",
                f"Overall trend: {stats['trend']} (slope: {stats['trend_slope']:.4f} per day)",
                f"Predictability assessment: {stats['predictability']}",
                f"Behavioral pattern: {stats['random_walk_insight']}",
                f"Data range: {df[date_col].min().strftime('%Y-%m-%d')} to {df[date_col].max().strftime('%Y-%m-%d')}",
                f"Total growth: {growth_rate:.1f}% (annualized: {annualized_growth:.1f}%)",
                f"Volatility level: {'High' if volatility > 0.5 else 'Moderate' if volatility > 0.2 else 'Low'} (CV: {volatility:.3f})"
            ]
            
            # Add seasonality insights
            if category_col and category_col in df.columns:
                insights.extend(overall_seasonality_insights[:2])
            else:
                insights.extend(seasonality_insights[:2])
            
            insights.append(f"Pattern type: {pattern_analysis['pattern_type']} (confidence: {pattern_analysis['confidence']:.1%})")
            insights.append(f"Volatility assessment: {pattern_analysis['volatility_type']} with {pattern_analysis['pattern_strength']:.3f} strength")

            # Add enhanced seasonality insights from decomposition
            if overall_decomposition and 'seasonality_analysis' in overall_decomposition:
                seasonality_info = overall_decomposition['seasonality_analysis']
                insights.append(f"Advanced seasonality: {seasonality_info['strength_category']} (strength: {seasonality_info['strength']:.3f})")
                if seasonality_info['period_points']:
                    insights.append(f"Seasonal cycle: Repeats every {seasonality_info['period_points']} data points")

            response_data = {
                "module": "time_series_analysis",
                "analysis_type": "multi-category" if category_col and category_col in df.columns else "single-series",
                "statistics": stats,
                "plot": fig.to_json(),
                "insights": insights,
                "seasonality_analysis": {
                    "monthly_strength": seasonality_strength,
                    "weekly_strength": weekly_seasonality,
                    "insights": overall_seasonality_insights if category_col and category_col in df.columns else seasonality_insights,
                    "advanced_analysis": overall_decomposition['seasonality_analysis'] if overall_decomposition and 'seasonality_analysis' in overall_decomposition else None
                },
                "volatility_metrics": {
                    "volatility": volatility,
                    "coefficient_variation": stats["coefficient_variation"]
                },
                "growth_metrics": {
                    "total_growth_percent": growth_rate,
                    "annualized_growth_percent": annualized_growth,
                    "time_span_days": time_span
                },
                "pattern_analysis": pattern_analysis,
                "decomposition": {
                    "overall": overall_decomposition,
                    "by_category": category_decompositions if category_col and category_col in df.columns else {}
                },
                'overall_decomposition': overall_decomposition,
                'category_decompositions': category_decompositions if category_col and category_col in df.columns else {},
                "category_analysis": {
                    "has_categories": category_col is not None and category_col in df.columns,
                    "categories_list": list(categories) if category_col and category_col in df.columns else [],
                    "category_seasonality": category_seasonality_insights
                }
            }
        
            return sanitize_for_json(response_data)
            
        except Exception as e:
            logging.error(f"Time series analysis error: {e}")
            raise

    @staticmethod
    def analyze_customer(df, params):
        """Enhanced customer analysis (RFM segmentation)"""
        try:
            customer_col = params.get('customer_col', 'customer_id')
            amount_col = params.get('amount_col', 'amount')
            date_col = params.get('date_col', 'transaction_date')
            
            logging.info(f'Customer analysis - Customer col: {customer_col}, Amount col: {amount_col}, Date col: {date_col}')
            logging.info(f'Data shape: {df.shape}')
            
            # Validate required columns
            required_cols = [customer_col, amount_col, date_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            df[date_col] = parse_datetime_flexible(df[date_col])
            df = df.dropna(subset=[date_col])
            
            if df.empty:
                raise ValueError("No valid data after date parsing")
            
            reference_date = df[date_col].max()
            
            # Calculate RFM metrics
            rfm = df.groupby(customer_col).agg({
                date_col: lambda x: (reference_date - x.max()).days,  # Recency
                amount_col: ['count', 'sum', 'mean']  # Frequency, Monetary, Avg Order Value
            }).reset_index()
            
            rfm.columns = [customer_col, 'recency', 'frequency', 'monetary', 'avg_order_value']
            
            # Calculate percentiles for scoring
            rfm['recency_score'] = pd.qcut(rfm['recency'], 5, labels=['5', '4', '3', '2', '1'], duplicates='drop')
            rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=['1', '2', '3', '4', '5'], duplicates='drop')
            rfm['monetary_score'] = pd.qcut(rfm['monetary'], 5, labels=['1', '2', '3', '4', '5'], duplicates='drop')
            
            # Create RFM combined score
            rfm['rfm_score'] = rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(str)
            
            # Enhanced customer segmentation
            def segment_customers(row):
                if row['frequency_score'] in ['4', '5'] and row['monetary_score'] in ['4', '5']:
                    return 'Champions'
                elif row['recency_score'] in ['4', '5'] and row['frequency_score'] in ['3', '4', '5']:
                    return 'Loyal Customers'
                elif row['recency_score'] in ['4', '5'] and row['frequency_score'] in ['1', '2']:
                    return 'Potential Loyalists'
                elif row['recency_score'] in ['3', '4'] and row['frequency_score'] in ['1', '2'] and row['monetary_score'] in ['3', '4', '5']:
                    return 'New Customers'
                elif row['recency_score'] in ['2', '3'] and row['frequency_score'] in ['2', '3']:
                    return 'Promising'
                elif row['recency_score'] in ['2', '3'] and row['frequency_score'] in ['1', '2']:
                    return 'Need Attention'
                elif row['recency_score'] == '1':
                    return 'At Risk'
                elif row['recency_score'] in ['1', '2'] and row['frequency_score'] in ['4', '5']:
                    return 'Cannot Lose Them'
                else:
                    return 'Others'
            
            rfm['segment'] = rfm.apply(segment_customers, axis=1)
            segment_counts = rfm['segment'].value_counts().to_dict()
            
            # Calculate segment statistics
            segment_stats = rfm.groupby('segment').agg({
                'recency': ['mean', 'median'],
                'frequency': ['mean', 'median'],
                'monetary': ['mean', 'median'],
                'avg_order_value': ['mean', 'median']
            }).round(2)
            
            # Create multiple visualizations
            fig_scatter = px.scatter(rfm, x='frequency', y='monetary', color='segment', 
                                   title="Customer Segmentation (Frequency vs Monetary)", 
                                   hover_data=['recency', 'avg_order_value'],
                                   labels={'frequency': 'Purchase Frequency', 'monetary': 'Total Spent'})
            
            fig_scatter.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#2E4057'), title_font_size=16
            )
            
            # Customer value distribution
            customer_lifetime_value = rfm['monetary'].sum()
            top_customers = rfm.nlargest(10, 'monetary')[customer_col].tolist()
            
            # Insights generation
            insights = [
                f"Total customers analyzed: {len(rfm)}",
                f"Customer segments identified: {len(segment_counts)}",
                f"Champions (high value): {segment_counts.get('Champions', 0)} customers ({(segment_counts.get('Champions', 0)/len(rfm)*100):.1f}%)",
                f"At Risk customers: {segment_counts.get('At Risk', 0)} customers ({(segment_counts.get('At Risk', 0)/len(rfm)*100):.1f}%)",
                f"Average customer lifetime value: ${rfm['monetary'].mean():.2f}",
                f"Top 10% customers generate ${rfm.nlargest(int(len(rfm)*0.1), 'monetary')['monetary'].sum():.2f} ({(rfm.nlargest(int(len(rfm)*0.1), 'monetary')['monetary'].sum()/customer_lifetime_value*100):.1f}% of total)",
                f"Average days since last purchase: {rfm['recency'].mean():.1f} days",
                f"Average purchase frequency: {rfm['frequency'].mean():.1f} transactions per customer"
            ]
            
            response_data = {
                "module": "customer_analysis",
                "total_customers": len(rfm),
                "segments": {
                    "counts": segment_counts,
                    "statistics": segment_stats.to_dict()
                },
                "plot": fig_scatter.to_json(),
                "insights": insights,
                "rfm_summary": {
                    "avg_recency": float(rfm['recency'].mean()),
                    "avg_frequency": float(rfm['frequency'].mean()),
                    "avg_monetary": float(rfm['monetary'].mean()),
                    "avg_order_value": float(rfm['avg_order_value'].mean()),
                    "median_recency": float(rfm['recency'].median()),
                    "median_frequency": float(rfm['frequency'].median()),
                    "median_monetary": float(rfm['monetary'].median())
                },
                "business_metrics": {
                    "total_revenue": float(customer_lifetime_value),
                    "top_10_percent_revenue": float(rfm.nlargest(int(len(rfm)*0.1), 'monetary')['monetary'].sum()),
                    "revenue_concentration": float(rfm.nlargest(int(len(rfm)*0.1), 'monetary')['monetary'].sum()/customer_lifetime_value*100),
                    "top_customers": top_customers[:5]  # Top 5 for brevity
                },
                "data_quality": {
                    "analysis_date": reference_date.strftime('%Y-%m-%d'),
                    "date_range": f"{df[date_col].min().strftime('%Y-%m-%d')} to {df[date_col].max().strftime('%Y-%m-%d')}",
                    "total_transactions": len(df)
                }
            }
            
            return sanitize_for_json(response_data)
            
        except Exception as e:
            logging.error(f"Customer analysis error: {e}")
            raise