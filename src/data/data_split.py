# data_split.py
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def create_time_splits(
    df: pd.DataFrame,
    target_col: str = 'total_demand',
    test_size: float = 0.2,
    val_size: float = 0.1,
    timestamp_col: str = 'settlement_date'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/val/test splits respecting temporal order.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        test_size: Proportion of data for testing
        val_size: Proportion of data for validation
        timestamp_col: Name of timestamp column
    
    Returns:
        Tuple of (train, val, test) DataFrames
    """
    logger.info("Creating time-based data splits")
    
    # Ensure data is sorted by time
    df = df.sort_values(timestamp_col).copy()
    
    # Calculate split points
    n_samples = len(df)
    test_start = int(n_samples * (1 - test_size))
    val_start = int(test_start * (1 - val_size))
    
    # Split data
    train = df.iloc[:val_start]
    val = df.iloc[val_start:test_start]
    test = df.iloc[test_start:]
    
    # Log split sizes
    logger.info(f"Train set: {len(train)} samples ({train[timestamp_col].min()} to {train[timestamp_col].max()})")
    logger.info(f"Validation set: {len(val)} samples ({val[timestamp_col].min()} to {val[timestamp_col].max()})")
    logger.info(f"Test set: {len(test)} samples ({test[timestamp_col].min()} to {test[timestamp_col].max()})")
    
    return train, val, test

def create_sequences(
    df: pd.DataFrame,
    target_col: str = 'total_demand',
    sequence_length: int = 48,  # 24 hours of 30-min intervals
    horizon: int = 48,  # Predict next 24 hours
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series prediction.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        sequence_length: Number of time steps for input sequence
        horizon: Number of time steps to predict
        stride: Step size between sequences
    
    Returns:
        Tuple of (X, y) arrays where X is input sequences and y is target sequences
    """
    data = df[target_col].values
    X, y = [], []
    
    for i in range(0, len(data) - sequence_length - horizon + 1, stride):
        X.append(data[i:(i + sequence_length)])
        y.append(data[(i + sequence_length):(i + sequence_length + horizon)])
    
    return np.array(X), np.array(y)

def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    save_path: str = '../data/processed'
) -> None:
    """Save train/val/test splits to disk."""
    import os
    os.makedirs(save_path, exist_ok=True)
    
    train.to_csv(f"{save_path}/train.csv", index=False)
    val.to_csv(f"{save_path}/val.csv", index=False)
    test.to_csv(f"{save_path}/test.csv", index=False)
    
    logger.info(f"Splits saved to {save_path}")

if __name__ == "__main__":
    # Example usage
    from sqlalchemy import create_engine
    from feature_engineering import engineer_features
    
    # Load and prepare data
    engine = create_engine("postgresql://energy_user:energy_password@energy-postgres-1:5432/energy_db")
    df = pd.read_sql("SELECT * FROM processed_energy", engine)
    df = engineer_features(df)
    
    # Create splits
    train, val, test = create_time_splits(df)
    
    # Optionally save splits
    save_splits(train, val, test)