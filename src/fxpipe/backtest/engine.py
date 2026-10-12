import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BacktestEngine:
    """
    Event-driven backtest engine.
    Walks bar-by-bar, manages positions, applies costs, and tracks equity.
    """
    def __init__(self, initial_capital: float, strategy, cost_model):
        self.initial_capital = float(initial_capital)
        self.equity = self.initial_capital
        self.peak_equity = self.equity
        self.max_drawdown = 0.0
        
        self.strategy = strategy
        self.cost_model = cost_model
        
        self.position = None
        self.trades = []
        self.equity_curve = []
        
    def run(self, df: pd.DataFrame, pair: str) -> dict:
        if len(df) < 2:
            logging.warning("Not enough data to run backtest.")
            return self._get_results()

        # Ensure we are walking forward in time
        df = df.sort_index()

        for i in range(1, len(df)):
            prev_bar = df.iloc[i - 1]
            curr_bar = df.iloc[i]
            curr_time = df.index[i]

            # 1. Check Exits if we are in a position
            if self.position:
                self._check_exits(curr_bar, prev_bar, pair, curr_time)

            # 2. Check Entries if we are flat
            # We execute on the current bar's open based on the previous bar's close signal
            # This prevents lookahead bias!
            if not self.position and prev_bar["signal"] != 0:
                self._open_position(curr_bar, prev_bar, pair, curr_time)

            # Record equity
            self.equity_curve.append({"timestamp": curr_time, "equity": self.equity})
            if self.equity > self.peak_equity:
                self.peak_equity = self.equity
            
            drawdown = (self.peak_equity - self.equity) / self.peak_equity
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown

        # Force close any open position at the end of the dataset
        if self.position:
            last_bar = df.iloc[-1]
            self._close_position(last_bar, last_bar.name, pair, exit_type="end_of_data", price=last_bar["close"])

        return self._get_results()

    def _open_position(self, curr_bar, prev_bar, pair, entry_time):
        direction = int(prev_bar["signal"])  # 1 for long, -1 for short
        raw_entry = curr_bar["open"]
        
        # Calculate raw stop loss based on ATR from the previous bar
        raw_stop = self.strategy.get_stop_price(direction, raw_entry, prev_bar["atr_stop_distance"])
        
        dir_str = "long" if direction == 1 else "short"
        
        # Apply costs (spread on entry)
        adj_entry = self.cost_model.get_adjusted_entry(pair, raw_entry, dir_str)
        adj_stop = self.cost_model.get_adjusted_exit(pair, raw_stop, dir_str, "stop_loss")
        
        # Position Sizing: 1% risk of equity
        risk_per_unit = abs(adj_entry - adj_stop)
        if risk_per_unit <= 0:
            return # Impossible trade
            
        risk_amount = self.equity * 0.01
        units = risk_amount / risk_per_unit
        
        self.position = {
            "direction": direction,
            "entry_time": entry_time,
            "entry_price": adj_entry,
            "stop_price": raw_stop, # Store raw stop for intrabar checking
            "units": units,
            "bars_held": 0
        }

    def _check_exits(self, curr_bar, prev_bar, pair, curr_time):
        direction = self.position["direction"]
        dir_str = "long" if direction == 1 else "short"
        self.position["bars_held"] += 1
        
        # 1. Check Stop Loss (Intrabar)
        sl_hit = False
        if direction == 1 and curr_bar["low"] <= self.position["stop_price"]:
            sl_hit = True
        elif direction == -1 and curr_bar["high"] >= self.position["stop_price"]:
            sl_hit = True
            
        if sl_hit:
            # Exit at stop price, apply costs (slippage + half spread)
            adj_exit = self.cost_model.get_adjusted_exit(pair, self.position["stop_price"], dir_str, "stop_loss")
            self._close_position(curr_bar, curr_time, pair, exit_type="stop_loss", price=adj_exit)
            return

        # 2. Check Z-score Exit (Bar Close) - The Take Profit
        if self.strategy.zscore_exit_hit(direction, curr_bar["zscore"]):
            raw_exit = curr_bar["close"]
            adj_exit = self.cost_model.get_adjusted_exit(pair, raw_exit, dir_str, "take_profit")
            self._close_position(curr_bar, curr_time, pair, exit_type="take_profit", price=adj_exit)
            return

        # 3. Check Time Stop (Bar Close)
        if self.strategy.time_stop_hit(self.position["bars_held"]):
            raw_exit = curr_bar["close"]
            adj_exit = self.cost_model.get_adjusted_exit(pair, raw_exit, dir_str, "time_stop")
            self._close_position(curr_bar, curr_time, pair, exit_type="time_stop", price=adj_exit)
            return

    def _close_position(self, curr_bar, exit_time, pair, exit_type, price):
        direction = self.position["direction"]
        units = self.position["units"]
        entry_price = self.position["entry_price"]
        
        # PnL calculation
        if direction == 1:
            pnl = (price - entry_price) * units
        else:
            pnl = (entry_price - price) * units
            
        self.equity += pnl
        
        self.trades.append({
            "pair": pair,
            "direction": "long" if direction == 1 else "short",
            "entry_time": self.position["entry_time"],
            "exit_time": exit_time,
            "entry_price": entry_price,
            "exit_price": price,
            "units": units,
            "pnl": pnl,
            "exit_type": exit_type,
            "bars_held": self.position["bars_held"]
        })
        
        # Close position
        self.position = None

    def _get_results(self) -> dict:
        return {
            "trades": self.trades,
            "equity_curve": self.equity_curve,
            "final_equity": self.equity,
            "max_drawdown": self.max_drawdown
        }

if __name__ == "__main__":
    import yaml
    from pathlib import Path
    from fxpipe.backtest.strategy import VRMRStrategy
    from fxpipe.backtest.costs import CostModel

    # Load configs
    with open("config/strategy.yaml") as f:
        strategy_config = yaml.safe_load(f)
    with open("config/costs.yaml") as f:
        cost_config = yaml.safe_load(f)
    with open("config/backtest.yaml") as f:
        backtest_config = yaml.safe_load(f)

    # Initialize components
    strategy = VRMRStrategy(strategy_config)
    cost_model = CostModel(cost_config)
    engine = BacktestEngine(backtest_config["capital"]["initial"], strategy, cost_model)

    # Load data and prepare signals
    data_path = "data/processed/EURUSD_H1_clean.parquet"
    try:
        df = pd.read_parquet(data_path)
        prepared_df = strategy.prepare(df)
        
        # Run the engine
        results = engine.run(prepared_df, pair="EURUSD")
        
        print(f"\n--- Backtest Complete ---")
        print(f"Total Trades: {len(results['trades'])}")
        print(f"Starting Equity: ${backtest_config['capital']['initial']}")
        print(f"Final Equity: ${results['final_equity']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
        
    except FileNotFoundError:
        print(f"Data not found at {data_path}. Run Phase 0 first.")