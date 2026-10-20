import dukascopy_python
from dukascopy_python.instruments import (
    INSTRUMENT_FX_MAJORS_EUR_USD, 
    INSTRUMENT_FX_MAJORS_GBP_USD, 
    INSTRUMENT_FX_MAJORS_USD_JPY, 
    INSTRUMENT_FX_MAJORS_AUD_USD
)
from datetime import datetime
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IngestionPhase:
    def __init__(self, base_dir: str = "./data"):
        # This is where raw data will be saved
        self.raw_dir = Path(base_dir) / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def fetch_data(self, pair: str, timeframe: str, start_date: datetime, end_date: datetime):
        """
        Downloads historical data directly from Dukascopy.
        """
        # Map our simple parameters to the library's internal constants
        instrument_map = {
            "EURUSD": INSTRUMENT_FX_MAJORS_EUR_USD,
            "GBPUSD": INSTRUMENT_FX_MAJORS_GBP_USD,
            "USDJPY": INSTRUMENT_FX_MAJORS_USD_JPY,
            "AUDUSD": INSTRUMENT_FX_MAJORS_AUD_USD
        }
        interval_map = {
            "M1": dukascopy_python.INTERVAL_MIN_1,
            "H1": dukascopy_python.INTERVAL_HOUR_1,
            "H4": dukascopy_python.INTERVAL_HOUR_4,
            "D1": dukascopy_python.INTERVAL_DAY_1,
        }

        instrument = instrument_map.get(pair)
        interval = interval_map.get(timeframe)

        instrument = instrument_map.get(pair)
        interval = interval_map.get(timeframe)
        
        if not instrument or not interval:
            logging.error(f"Unsupported pair ({pair}) or timeframe ({timeframe})")
            return None

        logging.info(f"Fetching {pair} ({timeframe}) from {start_date.date()} to {end_date.date()}")
        
        try:
            # Fetch directly from Dukascopy (using Bid prices, standard for backtesting)
            df = dukascopy_python.fetch(
                instrument=instrument,
                interval=interval,
                offer_side=dukascopy_python.OFFER_SIDE_BID, 
                start=start_date,
                end=end_date
            )
            
            if df.empty:
                logging.error("No data returned. Check your internet or date range.")
                return None

            # Save to CSV in the raw directory
            filename = f"{pair}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_raw.csv"
            save_path = self.raw_dir / filename
            df.to_csv(save_path)
            
            logging.info(f"Successfully saved {len(df)} rows to {save_path}")
            return save_path
            
        except Exception as e:
            logging.error(f"Dukascopy fetch failed: {e}")
            return None

if __name__ == "__main__":
    # TEST LOCALLY:
    ingestion = IngestionPhase()
    
    #  1 week of EUR/USD 1-Hour data to test
    ingestion.fetch_data(
        pair="EURUSD", 
        timeframe="H1", 
        start_date=datetime(2023, 1, 1), 
        end_date=datetime(2023, 1, 7)
    )