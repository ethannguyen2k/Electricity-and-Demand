# test_split.py
from data_split import create_time_splits
import pandas as pd
from sqlalchemy import create_engine
from feature_engineering import engineer_features
import logging

logging.basicConfig(level=logging.INFO)

# Load data
engine = create_engine("postgresql://energy_user:energy_password@energy-postgres-1:5432/energy_db")
df = pd.read_sql("SELECT * FROM processed_energy", engine)

# Engineer features
df_engineered = engineer_features(df)

# Create splits
train, val, test = create_time_splits(df_engineered)

# Basic checks
print("\nShapes:")
print(f"Total: {len(df_engineered)}")
print(f"Train: {len(train)} ({len(train)/len(df_engineered):.1%})")
print(f"Val: {len(val)} ({len(val)/len(df_engineered):.1%})")
print(f"Test: {len(test)} ({len(test)/len(df_engineered):.1%})")

print("\nDate ranges:")
print(f"Train: {train['settlement_date'].min()} to {train['settlement_date'].max()}")
print(f"Val: {val['settlement_date'].min()} to {val['settlement_date'].max()}")
print(f"Test: {test['settlement_date'].min()} to {test['settlement_date'].max()}")