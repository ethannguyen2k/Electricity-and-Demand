import pandas as pd
from feature_engineering import engineer_features
import logging

logging.basicConfig(level=logging.INFO)

# Load data from PostgreSQL
from sqlalchemy import create_engine
db_url = "postgresql://energy_user:energy_password@energy-postgres-1:5432/energy_db"
engine = create_engine(db_url)

# Test individual functions
df = pd.read_sql("SELECT * FROM processed_energy LIMIT 1000", engine)

# Look at a few rows before engineering
print("\nOriginal columns:", df.columns.tolist())
print("\nSample data shape:", df.shape)

# Apply feature engineering
df_engineered = engineer_features(df)

# Basic checks
print("\nNew columns:", [col for col in df_engineered.columns if col not in df.columns])
print("\nNew shape:", df_engineered.shape)

# Check for NaN values
print("\nNaN counts:")
print(df_engineered.isna().sum())

# Look at sample values for new features
print("\nSample of new features:")
print(df_engineered.sample(5))