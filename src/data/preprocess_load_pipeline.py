from preprocessing import preprocess_data
from ingestion import load_to_database
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_pipeline(db_url: str = "postgresql://energy_user:energy_password@energy-postgres-1:5432/energy_db"):
    """Run the complete data pipeline."""
    raw_data_path = os.path.join("src","data", "raw")
    processed_data_path = os.path.join("data", "processed", "energy_processed.csv")
    
    logger.info("Starting preprocessing...")
    processed_df = preprocess_data(raw_data_path)
    
    # Save processed CSV
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    processed_df.to_csv(processed_data_path, index=False)
    logger.info(f"Processed data saved to {processed_data_path}")
    
    # Load to database
    logger.info("Loading to database...")
    load_to_database(processed_df, db_url)
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()