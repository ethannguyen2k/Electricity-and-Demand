from sqlalchemy import create_engine, Table, Column, MetaData
from sqlalchemy.types import DateTime, Float, String, Boolean, Integer
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_to_database(df: pd.DataFrame, db_url: str = "postgresql://energy_user:energy_password@energy-postgres-1:5432/energy_db"):
    """Load data to PostgreSQL database."""
    engine = create_engine(db_url)
    metadata = MetaData()
    
    # Create raw data table
    raw_table = Table(
        'raw_energy', metadata,
        Column('id', Integer, primary_key=True),
        Column('settlement_date', DateTime),
        Column('total_demand', Float),
        Column('rrp', Float)
    )
    
    # Create processed data table
    processed_table = Table(
        'processed_energy', metadata,
        Column('id', Integer, primary_key=True),
        Column('settlement_date', DateTime),
        Column('total_demand', Float),
        Column('rrp', Float),
        Column('weekday', String),
        Column('holiday', Boolean),
        Column('hour', Integer),
        Column('minute', Integer),
        Column('time', Float),
        Column('day', Integer),
        Column('month', Integer),
        Column('year', Integer),
        Column('is_weekend', Boolean)
    )
    
    # Create tables
    metadata.create_all(engine)
    
    # Prepare and load raw data
    raw_df = df[['SETTLEMENTDATE', 'TOTALDEMAND', 'RRP']].copy()
    raw_df.columns = ['settlement_date', 'total_demand', 'rrp']
    
    # Prepare and load processed data
    processed_df = df.copy()
    processed_df.columns = processed_df.columns.str.lower()
    processed_df = processed_df.rename(columns={'settlementdate': 'settlement_date', 
                                              'totaldemand': 'total_demand'})
    
    # Insert data
    logger.info("Loading raw data to database")
    raw_df.to_sql('raw_energy', engine, if_exists='append', index=False)
    
    logger.info("Loading processed data to database")
    processed_df.to_sql('processed_energy', engine, if_exists='append', index=False)
    
    logger.info("Data loading completed")