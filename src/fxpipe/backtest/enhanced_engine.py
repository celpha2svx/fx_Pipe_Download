import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EnhancedBacktestEngine:
    """
    Enhanced event-driven backtest engine.
    Logs comprehensive trade data for deep analysis.
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

        # Streak tracking
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.trade_id = 0

    def _get_pip_value(self, pair: str) -> float:
        if pair == "USDJPY":
            return 0.01
        return 0.0001

    def _get_session(self, timestamp) -> str:
        hour = timestamp.hour
        if 0 <= hour < 8:
            return "Asia"
        elif 8 <= hour < 16:
            return "London"
        else:
            return "New York"

    def run(self, df: pd.DataFrame, pair: str, timeframe: str) -> dict:
        if len(df) < 2:
            logging.warning("Not enough data to run backtest.")
            return self._get_results()

        df = df.sort_index()
        pip_value = self._get_pip_value(pair)

        for i in range(1, len(df)):
            prev_bar = df.iloc[i - 1]
            curr_bar = df.iloc[i]
            curr_time = df.index[i]

            # 1. Check Exits
            if self.position:
                self._update_excursion(curr_bar, pip_value)
                self._check_exits(curr_bar, prev_bar, pair, timeframe, curr_time, pip_value)

            # 2. Check Entries
            if not self.position and prev_bar["signal"] != 0:
                self._open_position(curr_bar, prev_bar, pair, timeframe, curr_time, pip_value)

            # Record equity
            self.equity_curve.append({"timestamp": curr_time, "equity": self.equity})
            if self.equity > self.peak_equity:
                self.peak_equity = self.equity

            drawdown = (self.peak_equity - self.equity) / self.peak_equity
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown

        # Force close at end of data
        if self.position:
            last_bar = df.iloc[-1]
            self._close_position(
                last_bar, last_bar.name, pair, timeframe,
                exit_type="End of Data", price=last_bar["close"], pip_value=pip_value
            )

        return self._get_results()

    def _open_position(self, curr_bar, prev_bar, pair, timeframe, entry_time, pip_value):
        direction = int(prev_bar["signal"])
        raw_entry = curr_bar["open"]
        raw_stop = self.strategy.get_stop_price(direction, raw_entry, prev_bar["atr_stop_distance"])

        dir_str = "long" if direction == 1 else "short"
        adj_entry = self.cost_model.get_adjusted_entry(pair, raw_entry, dir_str)
        adj_stop = self.cost_model.get_adjusted_exit(pair, raw_stop, dir_str, "stop_loss")

        risk_per_unit = abs(adj_entry - adj_stop)
        if risk_per_unit <= 0:
            return

        risk_amount = self.equity * 0.01
        units = risk_amount / risk_per_unit

        # Calculate stop distance in pips
        stop_distance_pips = abs(raw_entry - raw_stop) / pip_value

        # Determine regime
        regime = "Ranging" if prev_bar["rv_current"] < prev_bar["rv_avg"] else "Trending"

        # Volume ratio
        volume_ratio = prev_bar["volume"] / prev_bar["volume_avg"] if prev_bar["volume_avg"] > 0 else 0

        self.position = {
            "trade_id": self.trade_id,
            "direction": direction,
            "entry_time": entry_time,
            "entry_price_raw": raw_entry,
            "entry_price_adj": adj_entry,
            "stop_price_raw": raw_stop,
            "stop_price_adj": adj_stop,
            "units": units,
            "bars_held": 0,
            "mfe_pips": 0.0,
            "mae_pips": 0.0,
            "zscore_at_entry": prev_bar["zscore"],
            "rv_current_at_entry": prev_bar["rv_current"],
            "rv_avg_at_entry": prev_bar["rv_avg"],
            "regime_at_entry": regime,
            "volume_ratio_at_entry": volume_ratio,
            "atr_at_entry": prev_bar["atr"],
            "stop_distance_pips": stop_distance_pips,
            "session_at_entry": self._get_session(entry_time),
            "equity_at_entry": self.equity,
            "consecutive_wins_before": self.consecutive_wins,
            "consecutive_losses_before": self.consecutive_losses,
        }

    def _update_excursion(self, curr_bar, pip_value):
        if self.position is None:
            return

        direction = self.position["direction"]
        entry_raw = self.position["entry_price_raw"]

        if direction == 1:  # Long
            favorable = (curr_bar["high"] - entry_raw) / pip_value
            adverse = (entry_raw - curr_bar["low"]) / pip_value
        else:  # Short
            favorable = (entry_raw - curr_bar["low"]) / pip_value
            adverse = (curr_bar["high"] - entry_raw) / pip_value

        self.position["mfe_pips"] = max(self.position["mfe_pips"], favorable)
        self.position["mae_pips"] = max(self.position["mae_pips"], adverse)

    def _check_exits(self, curr_bar, prev_bar, pair, timeframe, curr_time, pip_value):
        direction = self.position["direction"]
        dir_str = "long" if direction == 1 else "short"
        self.position["bars_held"] += 1

        # 1. Stop Loss (intrabar)
        sl_hit = False
        if direction == 1 and curr_bar["low"] <= self.position["stop_price_raw"]:
            sl_hit = True
        elif direction == -1 and curr_bar["high"] >= self.position["stop_price_raw"]:
            sl_hit = True

        if sl_hit:
            adj_exit = self.cost_model.get_adjusted_exit(
                pair, self.position["stop_price_raw"], dir_str, "stop_loss"
            )
            self._close_position(curr_bar, curr_time, pair, timeframe, "SL", adj_exit, pip_value)
            return

        # 2. Z-score exit (TP)
        if self.strategy.zscore_exit_hit(direction, curr_bar["zscore"]):
            raw_exit = curr_bar["close"]
            adj_exit = self.cost_model.get_adjusted_exit(pair, raw_exit, dir_str, "take_profit")
            self._close_position(curr_bar, curr_time, pair, timeframe, "TP", adj_exit, pip_value)
            return

        # 3. Time Stop
        if self.strategy.time_stop_hit(self.position["bars_held"]):
            raw_exit = curr_bar["close"]
            adj_exit = self.cost_model.get_adjusted_exit(pair, raw_exit, dir_str, "time_stop")
            self._close_position(curr_bar, curr_time, pair, timeframe, "Time Stop", adj_exit, pip_value)
            return

    def _close_position(self, curr_bar, exit_time, pair, timeframe, exit_type, adj_exit_price, pip_value):
        direction = self.position["direction"]
        units = self.position["units"]
        entry_raw = self.position["entry_price_raw"]
        entry_adj = self.position["entry_price_adj"]

        # Calculate P&L
        if direction == 1:
            gross_pnl_pips = (adj_exit_price - entry_adj) / pip_value
            net_pnl_account = (adj_exit_price - entry_adj) * units
        else:
            gross_pnl_pips = (entry_adj - adj_exit_price) / pip_value
            net_pnl_account = (entry_adj - adj_exit_price) * units

        # Calculate cost in pips
        spread_pips = self.cost_model.spread.get(pair, 1.5)
        slippage_map = {"TP": "take_profit", "SL": "stop_loss", "Time Stop": "time_stop", "End of Data": "time_stop"}
        slip_key = slippage_map.get(exit_type, "time_stop")
        slippage_pips = self.cost_model.slippage.get(slip_key, 1.0)
        cost_pips = spread_pips + slippage_pips

        net_pnl_pips = gross_pnl_pips - cost_pips

        self.equity += net_pnl_account

        # Update streaks
        if net_pnl_account > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

        self.trade_id += 1

        self.trades.append({
            # Identity
            "trade_id": self.position["trade_id"],
            "pair": pair,
            "timeframe": timeframe,
            "direction": "Long" if direction == 1 else "Short",

            # Entry
            "entry_time": self.position["entry_time"],
            "entry_price": entry_adj,
            "zscore_at_entry": self.position["zscore_at_entry"],
            "rv_current_at_entry": self.position["rv_current_at_entry"],
            "rv_avg_at_entry": self.position["rv_avg_at_entry"],
            "regime_at_entry": self.position["regime_at_entry"],
            "volume_ratio_at_entry": self.position["volume_ratio_at_entry"],
            "atr_at_entry": self.position["atr_at_entry"],
            "stop_distance_pips": self.position["stop_distance_pips"],
            "session_at_entry": self.position["session_at_entry"],

            # Exit
            "exit_time": exit_time,
            "exit_price": adj_exit_price,
            "exit_reason": exit_type,
            "bars_held": self.position["bars_held"],

            # Excursion
            "mfe_pips": self.position["mfe_pips"],
            "mae_pips": self.position["mae_pips"],

            # P&L
            "gross_pnl_pips": gross_pnl_pips,
            "cost_pips": cost_pips,
            "net_pnl_pips": net_pnl_pips,
            "net_pnl_account": net_pnl_account,
            "equity_at_entry": self.position["equity_at_entry"],
            "position_size_units": units,

            # Context
            "consecutive_losses_before": self.position["consecutive_losses_before"],
            "consecutive_wins_before": self.position["consecutive_wins_before"],
        })

        self.position = None

    def _get_results(self) -> dict:
        return {
            "trades": self.trades,
            "equity_curve": self.equity_curve,
            "final_equity": self.equity,
            "max_drawdown": self.max_drawdown,
        }