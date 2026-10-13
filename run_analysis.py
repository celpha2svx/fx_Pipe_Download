import yaml
import pandas as pd
import logging
from pathlib import Path

from fxpipe.phases.serve.loader import DataLoader
from fxpipe.backtest.strategy import VRMRStrategy
from fxpipe.backtest.costs import CostModel
from fxpipe.backtest.enhanced_engine import EnhancedBacktestEngine
from fxpipe.backtest.analyzer import TradeAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_deep_analysis(data_dir: str = "./data/processed", config_dir: str = "./config", results_dir: str = "./results"):
    # Load Configs
    with open(f"{config_dir}/strategy.yaml") as f:
        strategy_config = yaml.safe_load(f)
    with open(f"{config_dir}/costs.yaml") as f:
        cost_config = yaml.safe_load(f)
    with open(f"{config_dir}/backtest.yaml") as f:
        backtest_config = yaml.safe_load(f)

    # Initialize
    strategy = VRMRStrategy(strategy_config)
    cost_model = CostModel(cost_config)
    loader = DataLoader(base_dir=data_dir)

    # Create results directory
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)

    # Run on all H4 pairs + EURUSD H1 as comparison
    combinations = [
        ("EURUSD", "H4"),
        ("GBPUSD", "H4"),
        ("USDJPY", "H4"),
        ("AUDUSD", "H4"),
        ("EURUSD", "H1"),  # H1 comparison
    ]

    for pair, timeframe in combinations:
        logging.info(f"--- Deep Analysis: {pair} | {timeframe} ---")

        # Load Data
        df = loader.get_data(pair=pair, timeframe=timeframe)
        if df.empty:
            logging.warning(f"No data found for {pair} {timeframe}. Skipping.")
            continue

        # Prepare Signals
        prepared_df = strategy.prepare(df)
        if prepared_df.empty:
            logging.warning(f"No valid signals for {pair} {timeframe}. Skipping.")
            continue

        # Run Enhanced Engine
        engine = EnhancedBacktestEngine(backtest_config["capital"]["initial"], strategy, cost_model)
        results = engine.run(prepared_df, pair=pair, timeframe=timeframe)

        # Save Trade Log
        trades_df = pd.DataFrame(results["trades"])
        equity_df = pd.DataFrame(results["equity_curve"])

        trades_filename = results_path / f"trades_{pair}_{timeframe}.csv"
        trades_df.to_csv(trades_filename, index=False)
        logging.info(f"Saved trade log: {trades_filename} ({len(trades_df)} trades)")

        # Run Analyzer
        analyzer = TradeAnalyzer(trades_df, equity_df, pair, timeframe)
        analysis_text = analyzer.run_all_analyses()

        # Save Analysis Report
        analysis_filename = results_path / f"analysis_{pair}_{timeframe}.txt"
        with open(analysis_filename, "w") as f:
            f.write(analysis_text)
        logging.info(f"Saved analysis: {analysis_filename}")

    logging.info("Deep analysis complete. Check the results/ folder.")


if __name__ == "__main__":
    run_deep_analysis()