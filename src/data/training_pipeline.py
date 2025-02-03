# training_pipeline.py
import os
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
from sqlalchemy import create_engine
from feature_engineering import engineer_features
from data_split import WalkForwardValidator
from model_training import ModelTrainer, LightGBMModel, PyTorchLSTM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainingPipeline:
    """Pipeline for training energy demand forecasting models."""
    
    def __init__(
        self,
        db_url: str = "postgresql://energy_user:energy_password@energy-postgres-1:5432/energy_db",
        output_dir: str = "../models",
        config: Optional[Dict[str, Any]] = None
    ):
        self.db_url = db_url
        self.output_dir = output_dir
        self.config = config or self._default_config()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save config
        self._save_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for model training."""
        return {
            'feature_columns': [
                'hour', 'day_of_week', 'month', 'is_holiday',
                'demand_lag_1', 'demand_lag_48', 'demand_lag_168',
                'demand_rolling_mean_24h', 'demand_rolling_std_24h',
                'daily_sin', 'daily_cos', 'weekly_sin', 'weekly_cos',
                'is_peak_hour', 'demand_vs_daily_avg',
                'temperature_proxy'
            ],
            'target_column': 'total_demand',
            'sequence_length': 48,  # 24 hours
            'prediction_horizon': 48,
            'walk_forward_params': {
                'initial_train_years': 3,
                'validation_window': '3M',
                'stride': '1M'
            },
            'models': {
                'lightgbm': {
                    'enabled': True,
                    'params': {
                        'num_leaves': 31,
                        'learning_rate': 0.05,
                        'feature_fraction': 0.9
                    }
                },
                'lstm': {
                    'enabled': True,
                    'params': {
                        'hidden_size': 64,
                        'num_layers': 2,
                        'dropout': 0.1,
                        'learning_rate': 0.001,
                        'batch_size': 32,
                        'num_epochs': 50
                    }
                }
            }
        }
    
    def _save_config(self) -> None:
        """Save configuration to output directory."""
        config_path = os.path.join(self.output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Configuration saved to {config_path}")
    
    def load_data(self) -> pd.DataFrame:
        """Load data from database and engineer features."""
        logger.info("Loading data from database...")
        engine = create_engine(self.db_url)
        df = pd.read_sql("SELECT * FROM processed_energy", engine)
        
        logger.info("Engineering features...")
        df = engineer_features(df)
        
        return df
    
    def setup_models(self) -> Dict[str, ModelTrainer]:
        """Initialize model trainers based on configuration."""
        trainers = {}
        
        if self.config['models']['lightgbm']['enabled']:
            trainers['lightgbm'] = ModelTrainer(
                model_class=LightGBMModel,
                feature_columns=self.config['feature_columns'],
                target_column=self.config['target_column'],
                model_params=self.config['models']['lightgbm']['params']
            )
        
        if self.config['models']['lstm']['enabled']:
            trainers['lstm'] = ModelTrainer(
                model_class=PyTorchLSTM,
                feature_columns=self.config['feature_columns'],
                target_column=self.config['target_column'],
                model_params={
                    **self.config['models']['lstm']['params'],
                    'sequence_length': self.config['sequence_length'],
                    'horizon': self.config['prediction_horizon']
                }
            )
        
        return trainers
    
    def run(self) -> None:
        """Run the complete training pipeline."""
        start_time = datetime.now()
        run_id = start_time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.output_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        
        try:
            # Load and prepare data
            df = self.load_data()
            
            # Create walk-forward splits
            validator = WalkForwardValidator(
                df=df,
                date_column='settlement_date',
                **self.config['walk_forward_params']
            )
            splits = validator.create_splits()
            
            # Train models
            trainers = self.setup_models()
            for model_name, trainer in trainers.items():
                logger.info(f"Training {model_name} model...")
                model_dir = os.path.join(run_dir, model_name)
                os.makedirs(model_dir, exist_ok=True)
                
                # Train on each split
                for i, split in enumerate(splits):
                    result = trainer.train_evaluate_split(
                        split_index=i,
                        X_train=split.train_data[self.config['feature_columns']].values,
                        y_train=split.train_data[self.config['target_column']].values,
                        X_val=split.val_data[self.config['feature_columns']].values,
                        y_val=split.val_data[self.config['target_column']].values,
                        train_period=str(split.train_window),
                        val_period=str(split.val_window)
                    )
                    trainer.results.append(result)
                
                # Save results
                trainer.save_results(os.path.join(model_dir, 'results.json'))
            
            logger.info(f"Training pipeline completed successfully! Results saved in {run_dir}")
        
        except Exception as e:
            logger.error(f"Error in training pipeline: {str(e)}")
            raise

if __name__ == "__main__":
    # Run training pipeline
    pipeline = ModelTrainingPipeline()
    pipeline.run()