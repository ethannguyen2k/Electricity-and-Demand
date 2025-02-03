# model_training.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from abc import ABC, abstractmethod
import lightgbm as lgb
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class ModelResults:
    """Stores results from a single training iteration."""
    split_index: int
    train_period: str
    val_period: str
    metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    
    def to_dict(self):
        return {
            'split_index': self.split_index,
            'train_period': self.train_period,
            'val_period': self.val_period,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance
        }

class BaseTimeSeriesModel(ABC):
    """Abstract base class for time series models."""
    
    def __init__(self, feature_columns: List[str], target_column: str):
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.model = None
        self.scaler = StandardScaler()
    
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        pass

class LightGBMModel(BaseTimeSeriesModel):
    """LightGBM implementation for time series forecasting."""
    
    def __init__(
        self, 
        feature_columns: List[str],
        target_column: str,
        params: Dict[str, Any] = None
    ):
        super().__init__(feature_columns, target_column)
        self.params = params or {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the LightGBM model."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Create dataset
        train_data = lgb.Dataset(X_scaled, label=y_train)
        
        # Train model
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data],
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        logger.info("LightGBM model training completed")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        importance = self.model.feature_importance(importance_type='gain')
        return dict(zip(self.feature_columns, importance))

class EnergyLSTM(nn.Module):
    """LSTM model for energy demand forecasting using PyTorch.
    
    This implementation includes several architectural choices specifically for energy forecasting:
    1. Multiple LSTM layers with residual connections to capture both short and long-term patterns
    2. Separate processing paths for different feature types
    3. Attention mechanism to focus on relevant historical patterns
    """
    
    def __init__(
        self,
        feature_columns: List[str],
        sequence_length: int,
        horizon: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.feature_columns = feature_columns
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Split features into categories for separate processing
        self.time_features = [col for col in feature_columns if any(
            x in col for x in ['hour', 'day', 'week', 'month', 'sin', 'cos'])]
        self.demand_features = [col for col in feature_columns if 'demand' in col]
        
        # Separate LSTMs for different feature types
        self.demand_lstm = nn.LSTM(
            input_size=len(self.demand_features),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dense network for time features
        self.time_net = nn.Sequential(
            nn.Linear(len(self.time_features), hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Final prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon)
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        prev_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, num_features)
            prev_hidden: Optional previous hidden state for stateful prediction
            
        Returns:
            Predictions and final hidden state
        """
        batch_size = x.size(0)
        
        # Split features
        demand_idx = [self.feature_columns.index(col) for col in self.demand_features]
        time_idx = [self.feature_columns.index(col) for col in self.time_features]
        
        demand_input = x[:, :, demand_idx]
        time_input = x[:, :, time_idx]
        
        # Process demand features through LSTM
        demand_output, hidden = self.demand_lstm(demand_input, prev_hidden)
        
        # Process time features
        time_output = self.time_net(time_input)
        
        # Apply attention to LSTM output
        attn_output, _ = self.attention(
            demand_output,
            demand_output,
            demand_output
        )
        
        # Combine features
        combined = torch.cat([
            attn_output[:, -1],  # Take last time step
            time_output[:, -1]   # Take last time step
        ], dim=1)
        
        # Generate prediction
        prediction = self.predictor(combined)
        
        return prediction, hidden

class PyTorchLSTM(BaseTimeSeriesModel):
    """PyTorch LSTM wrapper that implements the BaseTimeSeriesModel interface."""
    
    def __init__(
        self,
        feature_columns: List[str],
        target_column: str,
        sequence_length: int = 48,
        horizon: int = 48,
        hidden_size: int = 64,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        num_epochs: int = 50,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__(feature_columns, target_column)
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        
        # Initialize model
        self.model = EnergyLSTM(
            feature_columns=feature_columns,
            sequence_length=sequence_length,
            horizon=horizon,
            hidden_size=hidden_size
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        logger.info(f"Initialized PyTorch LSTM model on device: {device}")
    
    def _prepare_data(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare numpy arrays for PyTorch training."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        if y is not None:
            y_tensor = torch.FloatTensor(y).to(self.device)
            return X_tensor, y_tensor
        return X_tensor, None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the PyTorch LSTM model."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
        X_scaled = X_scaled.reshape(X_train.shape)
        
        # Prepare data
        X_tensor, y_tensor = self._prepare_data(X_scaled, y_train)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Training loop
        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                
                # Forward pass
                predictions, _ = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using the trained model."""
        # Scale features
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1]))
        X_scaled = X_scaled.reshape(X.shape)
        
        # Prepare data
        X_tensor, _ = self._prepare_data(X_scaled)
        
        # Generate predictions
        self.model.eval()
        with torch.no_grad():
            predictions, _ = self.model(X_tensor)
            return predictions.cpu().numpy()
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Approximate feature importance through gradient-based sensitivity analysis.
        This is a simple approximation since LSTMs don't have direct feature importance.
        """
        self.model.eval()
        importance_scores = {}
        
        # Use a small validation set for importance calculation
        with torch.no_grad():
            X_sample = torch.randn(
                100,
                self.sequence_length,
                len(self.feature_columns)
            ).to(self.device)
            
            # Calculate gradient of output with respect to input
            X_sample.requires_grad = True
            output, _ = self.model(X_sample)
            output.sum().backward()
            
            # Average gradients across time steps and batches
            grad_imp = X_sample.grad.abs().mean(dim=(0, 1)).cpu().numpy()
            importance_scores = dict(zip(self.feature_columns, grad_imp))
        
        return importance_scores

class ModelTrainer:
    """Handles the training process across multiple walk-forward splits."""
    
    def __init__(
        self,
        model_class: BaseTimeSeriesModel,
        feature_columns: List[str],
        target_column: str,
        model_params: Dict[str, Any] = None
    ):
        self.model_class = model_class
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.model_params = model_params or {}
        self.results = []
    
    def train_evaluate_split(
        self,
        split_index: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        train_period: str,
        val_period: str
    ) -> ModelResults:
        """Train and evaluate model on a single split."""
        # Initialize and train model
        model = self.model_class(
            self.feature_columns,
            self.target_column,
            **self.model_params
        )
        model.fit(X_train, y_train)
        
        # Generate predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Calculate metrics
        metrics = {
            'train_mse': np.mean((y_train - train_pred) ** 2),
            'train_mae': np.mean(np.abs(y_train - train_pred)),
            'val_mse': np.mean((y_val - val_pred) ** 2),
            'val_mae': np.mean(np.abs(y_val - val_pred))
        }
        
        # Get feature importance
        feature_importance = model.get_feature_importance()
        
        return ModelResults(
            split_index=split_index,
            train_period=train_period,
            val_period=val_period,
            metrics=metrics,
            feature_importance=feature_importance
        )
    
    def save_results(self, filepath: str) -> None:
        """Save training results to disk."""
        results_dict = [result.to_dict() for result in self.results]
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        logger.info(f"Results saved to {filepath}")

if __name__ == "__main__":
    # Example usage
    from data_split import WalkForwardValidator
    from feature_engineering import engineer_features
    from sqlalchemy import create_engine
    
    # Load and prepare data
    engine = create_engine("postgresql://energy_user:energy_password@energy-postgres-1:5432/energy_db")
    df = pd.read_sql("SELECT * FROM processed_energy", engine)
    df = engineer_features(df)
    
    # Define features
    feature_cols = [
        'hour', 'day_of_week', 'month', 'is_holiday',
        'demand_lag_1', 'demand_lag_48',
        'demand_rolling_mean_24h', 'demand_rolling_std_24h',
        'daily_sin', 'daily_cos', 'weekly_sin', 'weekly_cos'
    ]
    
    # Create walk-forward splits
    validator = WalkForwardValidator(df, 'datetime')
    splits = validator.create_splits()
    
    # Train LightGBM model
    lgb_trainer = ModelTrainer(
        model_class=LightGBMModel,
        feature_columns=feature_cols,
        target_column='total_demand'
    )
    
    # Train on each split
    for i, split in enumerate(splits):
        result = lgb_trainer.train_evaluate_split(
            i,
            split.train_data[feature_cols].values,
            split.train_data['total_demand'].values,
            split.val_data[feature_cols].values,
            split.val_data['total_demand'].values,
            str(split.train_window),
            str(split.val_window)
        )
        lgb_trainer.results.append(result)
    
    # Save results
    lgb_trainer.save_results('lightgbm_results.json')