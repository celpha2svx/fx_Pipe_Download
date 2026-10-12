import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    """
    The Serve Module: The single point of entry for your ML models.
    """
    def __init__(self, base_dir: str = "./data/processed"):
        self.processed_dir = Path(base_dir)
        if not self.processed_dir.exists():
            raise FileNotFoundError(f"Processed data directory not found: {self.processed_dir}")
    def get_data(self, pair: str, timeframe: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Loads clean, processed data for a specific pair and timeframe.
        Optionally filters by date range.
        """
        filename = f"{pair}_{timeframe}_clean.parquet"
        file_path = self.processed_dir / filename
        
        if not file_path.exists():
            logging.error(f"Data file not found: {file_path}")
            return pd.DataFrame()

        logging.info(f"Loading {pair} ({timeframe}) data from {file_path}")
        df = pd.read_parquet(file_path)
        
         # Filter by date range if provided
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        logging.info(f"Successfully loaded {len(df)} rows.")
        return df

if __name__ == "__main__":
    # TEST LOCALLY: Load the data we just cleaned
    loader = DataLoader()
    df = loader.get_data(pair="EURUSD", timeframe="H1", start_date="2023-01-02", end_date="2023-01-04")
    print(df.head())
    print(df.shape)