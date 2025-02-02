# data_split.py
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import logging
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TimeWindow:
    """Represents a time window for training or validation."""
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    
    def __str__(self):
        return f"{self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}"

@dataclass
class WalkForwardSplit:
    """Represents a single walk-forward split with training and validation periods."""
    train_window: TimeWindow
    val_window: TimeWindow
    train_data: pd.DataFrame
    val_data: pd.DataFrame
    
    def __str__(self):
        return f"Train: {self.train_window}\nValidation: {self.val_window}\n" \
               f"Train samples: {len(self.train_data)}, Validation samples: {len(self.val_data)}"

class WalkForwardValidator:
    """Handles the creation and management of walk-forward validation splits."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        date_column: str = 'datetime',
        initial_train_years: int = 3,
        validation_window: str = '3M',
        stride: str = '1M'
    ):
        """
        Initialize the walk-forward validator.
        
        Args:
            df: Input DataFrame
            date_column: Name of the datetime column
            initial_train_years: Number of years for initial training
            validation_window: Size of validation window (pandas offset string)
            stride: How far to move forward each time (pandas offset string)
        """
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found in DataFrame")
        
        self.df = df.copy()
        self.df[date_column] = pd.to_datetime(self.df[date_column])
        self.df = self.df.sort_values(date_column)
        
        self.date_column = date_column
        self.initial_train_years = initial_train_years
        self.validation_window = validation_window
        self.stride = stride
        
        # Calculate key dates
        self.start_date = self.df[date_column].min()
        self.end_date = self.df[date_column].max()
        self.initial_train_end = self.start_date + pd.DateOffset(years=initial_train_years)
        
        logger.info(f"Initialized WalkForwardValidator:")
        logger.info(f"Data range: {self.start_date} to {self.end_date}")
        logger.info(f"Initial training period ends: {self.initial_train_end}")
    
    def create_splits(self) -> List[WalkForwardSplit]:
        """
        Create all walk-forward splits based on the configured parameters.
        
        Returns:
            List of WalkForwardSplit objects
        """
        splits = []
        current_train_end = self.initial_train_end
        
        while current_train_end + pd.Timedelta(self.validation_window) <= self.end_date:
            # Define windows
            val_end = current_train_end + pd.Timedelta(self.validation_window)
            
            train_window = TimeWindow(self.start_date, current_train_end)
            val_window = TimeWindow(current_train_end, val_end)
            
            # Create split
            train_data = self.df[
                (self.df[self.date_column] >= train_window.start_date) & 
                (self.df[self.date_column] < train_window.end_date)
            ]
            val_data = self.df[
                (self.df[self.date_column] >= val_window.start_date) & 
                (self.df[self.date_column] < val_window.end_date)
            ]
            
            splits.append(WalkForwardSplit(train_window, val_window, train_data, val_data))
            
            # Move forward
            current_train_end += pd.Timedelta(self.stride)
        
        logger.info(f"Created {len(splits)} walk-forward splits")
        return splits

def create_sequences_for_split(
    split: WalkForwardSplit,
    target_col: str = 'total_demand',
    sequence_length: int = 48,
    horizon: int = 48,
    feature_columns: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sequences for a single walk-forward split.
    
    Args:
        split: WalkForwardSplit object containing train and validation data
        target_col: Name of target column
        sequence_length: Number of time steps for input sequence
        horizon: Number of time steps to predict
        feature_columns: Optional list of feature columns to include
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val) arrays
    """
    def create_sequences_from_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if feature_columns:
            data = df[feature_columns + [target_col]].values
            target_idx = len(feature_columns)
        else:
            data = df[[target_col]].values
            target_idx = 0
        
        X, y = [], []
        for i in range(0, len(data) - sequence_length - horizon + 1):
            X.append(data[i:(i + sequence_length)])
            y.append(data[(i + sequence_length):(i + sequence_length + horizon), target_idx])
        
        return np.array(X), np.array(y)
    
    X_train, y_train = create_sequences_from_df(split.train_data)
    X_val, y_val = create_sequences_from_df(split.val_data)
    
    return X_train, y_train, X_val, y_val

if __name__ == "__main__":
    # Example usage
    from sqlalchemy import create_engine
    
    # Load data
    engine = create_engine("postgresql://energy_user:energy_password@energy-postgres-1:5432/energy_db")
    df = pd.read_sql("SELECT * FROM processed_energy", engine)
    
    # Initialize validator
    validator = WalkForwardValidator(
        df,
        date_column='datetime',
        initial_train_years=3,
        validation_window='3M',
        stride='1M'
    )
    
    # Create splits
    splits = validator.create_splits()
    
    # Example: Create sequences for first split
    feature_cols = ['hour', 'day_of_week', 'month', 'is_holiday']
    X_train, y_train, X_val, y_val = create_sequences_for_split(
        splits[0],
        target_col='total_demand',
        sequence_length=48,
        horizon=48,
        feature_columns=feature_cols
    )
    
    # Log information about the first split
    logger.info(f"First split information:")
    logger.info(str(splits[0]))
    logger.info(f"Training sequences shape: {X_train.shape}")
    logger.info(f"Validation sequences shape: {X_val.shape}")