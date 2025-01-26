import pandas as pd
import holidays
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_data(raw_data_path: str) -> pd.DataFrame:
    """Preprocess energy consumption data."""
    logger.info("Loading data from %s", raw_data_path)
    
    # Load and combine files
    energy_dfs = []
    for file_name in sorted(os.listdir(raw_data_path)):
        if file_name.endswith(".csv"):
            df = pd.read_csv(os.path.join(raw_data_path, file_name))
            energy_dfs.append(df)
    
    df = pd.concat(energy_dfs, axis=0, ignore_index=True)
    
    # Basic preprocessing
    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"])
    df.drop(columns=["REGION", "PERIODTYPE"], inplace=True)
    
    # Add time features
    vic_holidays = holidays.Australia(years=range(2018, 2024), state="VIC")
    df["Weekday"] = df["SETTLEMENTDATE"].dt.day_name()
    df["Holiday"] = df["SETTLEMENTDATE"].dt.date.map(lambda x: x in vic_holidays)
    df["Holiday"] = df["Holiday"].fillna(False)
    
    # Align to 30-minute intervals
    df["30min_interval"] = df["SETTLEMENTDATE"].apply(_get_30min_start)
    df = df.groupby("30min_interval").agg({
        "TOTALDEMAND": "mean",
        "RRP": "mean",
        "Weekday": "first",
        "Holiday": "first"
    }).reset_index()
    
    df = df.rename(columns={"30min_interval": "SETTLEMENTDATE"})
    df["TOTALDEMAND"] = df["TOTALDEMAND"].round(2)
    df["RRP"] = df["RRP"].round(2)
    
    # Additional features
    df["Hour"] = df["SETTLEMENTDATE"].dt.hour
    df["Minute"] = df["SETTLEMENTDATE"].dt.minute
    df["Time"] = df["Hour"] + df["Minute"] / 60
    df["Day"] = df["SETTLEMENTDATE"].dt.day
    df["Month"] = df["SETTLEMENTDATE"].dt.month
    df["Year"] = df["SETTLEMENTDATE"].dt.year
    df["is_weekend"] = df["Weekday"].isin(["Saturday", "Sunday"])
    
    logger.info("Preprocessing completed")
    return df

def _get_30min_start(timestamp):
    """Align timestamp to 30-minute intervals."""
    minute = timestamp.minute
    if minute in [5, 10, 15, 20, 25]:
        return timestamp.replace(minute=0, second=0, microsecond=0)
    elif minute in [35, 40, 45, 50, 55]:
        return timestamp.replace(minute=30, second=0, microsecond=0)
    return timestamp