import yaml
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path

# IMPORTS 
from fxpipe.phases.serve.loader import DataLoader
from fxpipe.backtest.strategy import VRMRStrategy
from fxpipe.backtest.costs import CostModel
from fxpipe.backtest.engine import BacktestEngine
from fxpipe.backtest.metrics import MetricsCalculator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_baseline(data_dir: str = "./data/processed", config_dir: str = "./config"):
    # 1. Load Configs
    with open(f"{config_dir}/strategy.yaml") as f: strategy_config = yaml.safe_load(f)
    with open(f"{config_dir}/costs.yaml") as f: cost_config = yaml.safe_load(f)
    with open(f"{config_dir}/backtest.yaml") as f: backtest_config = yaml.safe_load(f)

    # 2. Initialize Components
    strategy = VRMRStrategy(strategy_config)
    cost_model = CostModel(cost_config)
    metrics_calc = MetricsCalculator(backtest_config["capital"]["initial"])
    loader = DataLoader(base_dir=data_dir)

    pairs = backtest_config["instruments"]
    timeframes = backtest_config["timeframes"]
    start_date = pd.to_datetime(backtest_config["date_range"]["start"])
    end_date = pd.to_datetime(backtest_config["date_range"]["end"])

    all_results = []

    # 3. Loop through all combinations
    for pair in pairs:
        for timeframe in timeframes:
            logging.info(f"--- Running Baseline: {pair} | {timeframe} ---")
            
            # Load Data
            df = loader.get_data(pair=pair, timeframe=timeframe, start_date=str(start_date.date()), end_date=str(end_date.date()))
            if df.empty:
                logging.warning(f"No data found for {pair} {timeframe}. Skipping.")
                continue

            # Prepare Signals
            prepared_df = strategy.prepare(df)
            if prepared_df.empty:
                logging.warning(f"No valid signals for {pair} {timeframe}. Skipping.")
                continue

            # Run Engine
            engine = BacktestEngine(backtest_config["capital"]["initial"], strategy, cost_model)
            results = engine.run(prepared_df, pair=pair)

            # Calculate Metrics
            final_metrics = metrics_calc.calculate(
                trades=results["trades"],
                equity_curve=results["equity_curve"],
                final_equity=results["final_equity"],
                start_date=prepared_df.index[0],
                end_date=prepared_df.index[-1]
            )

            # Add identifiers to metrics
            final_metrics["pair"] = pair
            final_metrics["timeframe"] = timeframe
            
            all_results.append(final_metrics)

    # 4. Save Report
    if not all_results:
        logging.error("No trades generated across any pair/timeframe. Report not created.")
        return

    report_df = pd.DataFrame(all_results)
    
    # Reorder columns for readability
    cols = ["pair", "timeframe", "total_return_pct", "sharpe_ratio", "max_drawdown_pct", 
            "win_rate_pct", "profit_factor", "total_trades", "trades_per_year", 
            "avg_duration_bars", "avg_win", "avg_loss", "longest_win_streak", "longest_lose_streak"]
    report_df = report_df[cols]

    # Save to results directory
    results_dir = Path(backtest_config["output"]["results_dir"])
    results_dir.mkdir(exist_ok=True)
    report_path = results_dir / backtest_config["output"]["report_filename"]
    
    report_df.to_csv(report_path, index=False)
    logging.info(f"Successfully saved Baseline Report to {report_path}")

if __name__ == "__main__":
    # When running in Colab, we will pass the Google Drive path here
    run_baseline()