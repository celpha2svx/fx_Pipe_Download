import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TransformationPhase:
    def __init__(self, base_dir: str = "./data"):
        self.raw_dir = Path(base_dir) / "raw"
        self.processed_dir = Path(base_dir) / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def is_forex_market_open(self, dt):
        """
        Forex market hours: Sunday 22:00 UTC to Friday 22:00 UTC.
        Returns True if the timestamp is within market hours.
        """
        day = dt.dayofweek # 0=Monday, 1=Tuesday, ..., 4=Friday, 5=Saturday, 6=Sunday
        hour = dt.hour
        
        if day == 5: # Saturday
            return False
        elif day == 6: # Sunday
            return hour >= 22 # Market opens Sunday 22:00 UTC
        elif day == 4: # Friday
            return hour < 22 # Market closes Friday 22:00 UTC
        else: # Monday-Thursday
            return True
    
    def clean_data(self, pair: str, timeframe: str, filename: str):
        """
        Cleans the raw CSV and saves it as Parquet.
        """
        raw_path = self.raw_dir / filename
        if not raw_path.exists():
            logging.error(f"Raw file not found: {raw_path}")
            return None

        logging.info(f"Loading raw data from {raw_path}")
        df = pd.read_csv(raw_path, parse_dates=['timestamp'])

        # 1. Deduplication
        df = df.drop_duplicates(subset=['timestamp'], keep='first')

        # 2. Sorting
        df = df.sort_values('timestamp')

        # 3. Timezone Alignment
        # Convert to UTC and then drop timezone info for ML compatibility
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
        df.set_index('timestamp', inplace=True)

        # 4. Weekend & Holiday Logic
        # Filter out any bars that are outside Forex market hours
        market_open_mask = df.index.map(self.is_forex_market_open)
        df = df[market_open_mask]

        # 5. Data Validation
        # Check for negative prices
        if (df[['open', 'high', 'low', 'close']] < 0).any().any():
            logging.warning("Negative prices detected. Dropping invalid rows.")
            df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]

        # Check High >= Low
        invalid_hl = df['high'] < df['low']
        if invalid_hl.any():
            logging.warning(f"Found {invalid_hl.sum()} rows where High < Low. Dropping them.")
            df = df[~invalid_hl]

        # Check Close is between High and Low
        invalid_close = (df['close'] > df['high']) | (df['close'] < df['low'])
        if invalid_close.any():
            logging.warning(f"Found {invalid_close.sum()} rows where Close is outside High/Low range. Dropping them.")
            df = df[~invalid_close]

        # Check Volume is non-negative
        if (df['volume'] < 0).any():
            logging.warning("Negative volume detected. Dropping invalid rows.")
            df = df[df['volume'] >= 0]
    
        # Save to Parquet
        processed_filename = f"{pair}_{timeframe}_clean.parquet"
        processed_path = self.processed_dir / processed_filename
        df.to_parquet(processed_path)
        
        logging.info(f"Successfully cleaned and saved {len(df)} rows to {processed_path}")
        return processed_path

if __name__ == "__main__":
    # TEST LOCALLY: Clean the file we just downloaded
    transformation = TransformationPhase()
    transformation.clean_data(
        pair="EURUSD",
        timeframe="H1",
        filename="EURUSD_H1_20230101_20230107_raw.csv"
    )
    