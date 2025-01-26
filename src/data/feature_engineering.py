# feature_engineering.py
import pandas as pd
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features."""
    df = df.copy()
    df = df.sort_values('settlement_date')
    
    # Basic time features
    df['week'] = df['settlement_date'].dt.isocalendar().week
    df['quarter'] = df['settlement_date'].dt.quarter
    df['is_month_start'] = df['settlement_date'].dt.is_month_start
    df['is_month_end'] = df['settlement_date'].dt.is_month_end
    
    # Season (Australia)
    df['season'] = pd.cut(df['month'], 
                         bins=[0,2,5,8,11,12], 
                         labels=['Summer', 'Autumn', 'Winter', 'Spring', 'Summer'])
    
    return df

def add_lag_features(df: pd.DataFrame, periods: List[int] = [1, 2, 24, 48, 168]) -> pd.DataFrame:
    """Add lagged demand features."""
    df = df.copy()
    
    for period in periods:
        df[f'demand_lag_{period}'] = df['total_demand'].shift(period)
        df[f'rrp_lag_{period}'] = df['rrp'].shift(period)
    
    return df

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling statistics."""
    df = df.copy()
    
    windows = {
        '24h': 48,      # 48 30-min periods
        '7d': 336,      # 7 days
        '30d': 1440     # 30 days
    }
    
    for name, window in windows.items():
        df[f'demand_rolling_mean_{name}'] = df['total_demand'].rolling(window=window, min_periods=1).mean()
        df[f'demand_rolling_std_{name}'] = df['total_demand'].rolling(window=window, min_periods=1).std()
        
        # Rate of change
        df[f'demand_roc_{name}'] = df['total_demand'].pct_change(periods=window)
        
    return df

def add_periodicity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add periodicity features."""
    df = df.copy()

    # Daily periodicity
    df['day_of_year'] = df['settlement_date'].dt.dayofyear
    df['daily_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['daily_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    
    # Sub-daily periodicity (48 intervals per day)
    df['daily_sin'] = np.sin(2 * np.pi * df['time'] / 48.0)
    df['daily_cos'] = np.cos(2 * np.pi * df['time'] / 48.0)
    
    # Weekly periodicity
    df['week_progress'] = (df['weekday'].map({
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
        'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }) + df['time']/24.0) / 7.0
    df['weekly_sin'] = np.sin(2 * np.pi * df['week_progress'])
    df['weekly_cos'] = np.cos(2 * np.pi * df['week_progress'])
    
    return df

def add_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add demand-related features."""
    df = df.copy()
    
    # Peak/off-peak indicator (simplified)
    peak_hours = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    df['is_peak_hour'] = df['hour'].isin(peak_hours)
    
    # Demand patterns
    df['daily_avg_demand'] = df.groupby(['year', 'month', 'day'])['total_demand'].transform('mean')
    df['demand_vs_daily_avg'] = df['total_demand'] / df['daily_avg_demand']
    
    # Quadratic terms
    df['demand_squared'] = df['total_demand'] ** 2
    df['temperature_proxy'] = np.where(df['hour'].between(17, 20), 
                                     df['total_demand'], 
                                     df['total_demand'] * 0.8)
    
    return df

def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add holiday-related features."""
    df = df.copy()
    
    # Days since last holiday
    df['days_since_holiday'] = (~df['holiday']).cumsum()
    df['days_since_holiday'] = df['days_since_holiday'] - df['days_since_holiday'].where(df['holiday']).ffill()
    
    # Days until next holiday
    df['days_until_holiday'] = (~df['holiday'])[::-1].cumsum()[::-1]
    df['days_until_holiday'] = df['days_until_holiday'] - df['days_until_holiday'].where(df['holiday']).bfill()
    
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Main function to engineer all features."""
    logger.info("Starting feature engineering")
    
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_periodicity_features(df)
    df = add_demand_features(df)
    df = add_holiday_features(df)
    
    # Drop rows with NaN values from lagged features
    df = df.dropna()
    
    logger.info(f"Feature engineering completed. New shape: {df.shape}")
    return df