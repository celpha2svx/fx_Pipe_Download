import numpy as np
import pandas as pd


class VRMRStrategy:
    """
    Frozen VRMR baseline strategy.

    Rules:
    - Z-score entry at ±2.0, lookback 20.
    - Exit when Z-score returns to 0.
    - Realized volatility filter: current 20-bar RV < 100-bar average RV.
    - Volume filter: current volume > 1.5x 20-bar average volume.
    - Stop-loss: 1.5x ATR(14), fixed.
    - Time stop: 10 bars.
    - Position sizing handled by engine, not here.
    """

    def __init__(self, config: dict):
        # Z-score
        self.z_lookback = int(config["zscore"]["lookback"])
        self.entry_threshold = float(config["zscore"]["entry_threshold"])
        self.exit_threshold = float(config["zscore"]["exit_threshold"])

        # Realized volatility regime filter
        self.rv_current_lookback = int(config["realized_volatility"]["current_lookback"])
        self.rv_avg_lookback = int(config["realized_volatility"]["average_lookback"])

        # Volume filter
        self.volume_enabled = bool(config["volume_filter"]["enabled"])
        self.volume_multiplier = float(config["volume_filter"]["multiplier"])
        self.volume_lookback = int(config["volume_filter"]["lookback"])

        # ATR / stop
        self.atr_period = int(config["atr"]["period"])
        self.stop_loss_multiplier = float(config["atr"]["stop_loss_multiplier"])

        # Time stop
        self.time_stop_enabled = bool(config["time_stop"]["enabled"])
        self.max_bars = int(config["time_stop"]["max_bars"])

        # Warmup period required before indicators are valid
        self.min_history = max(
            self.z_lookback,
            self.rv_avg_lookback,
            self.volume_lookback,
            self.atr_period,
        ) + 1

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds all frozen VRMR indicators to the dataframe.

        Expected input columns:
        - open
        - high
        - low
        - close
        - volume

        Index should be datetime.
        """
        out = df.copy()
        out = out.sort_index()

        close = out["close"]

        # -------------------------
        # Z-score
        # -------------------------
        z_mean = close.rolling(self.z_lookback, min_periods=self.z_lookback).mean()
        z_std = close.rolling(self.z_lookback, min_periods=self.z_lookback).std(ddof=0)

        # Avoid division by zero
        z_std = z_std.replace(0, np.nan)

        out["zscore"] = (close - z_mean) / z_std

        # -------------------------
        # Realized volatility filter
        # -------------------------
        log_returns = np.log(close / close.shift(1))

        out["rv_current"] = log_returns.rolling(
            self.rv_current_lookback,
            min_periods=self.rv_current_lookback
        ).std(ddof=0)

        out["rv_avg"] = log_returns.rolling(
            self.rv_avg_lookback,
            min_periods=self.rv_avg_lookback
        ).std(ddof=0)

        # Mean reversion is safer when short-term volatility is below long-term average.
        out["rv_filter"] = out["rv_current"] < out["rv_avg"]

        # -------------------------
        # Volume filter
        # -------------------------
        if self.volume_enabled:
            volume_avg = out["volume"].rolling(
                self.volume_lookback,
                min_periods=self.volume_lookback
            ).mean()

            out["volume_filter"] = out["volume"] > (self.volume_multiplier * volume_avg)
        else:
            out["volume_filter"] = True

        # Fill NaN booleans as False
        out["rv_filter"] = out["rv_filter"].fillna(False)
        out["volume_filter"] = out["volume_filter"].fillna(False)

        # -------------------------
        # ATR
        # -------------------------
        high_low = out["high"] - out["low"]
        high_close = (out["high"] - close.shift()).abs()
        low_close = (out["low"] - close.shift()).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        out["atr"] = true_range.rolling(
            self.atr_period,
            min_periods=self.atr_period
        ).mean()

        # ATR stop distance in price units
        out["atr_stop_distance"] = out["atr"] * self.stop_loss_multiplier

        return out

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes indicators and frozen entry signals.

        signal:
            1  = long signal
           -1  = short signal
            0  = no signal

        Important:
        Signal is known at the close of this bar.
        The engine should execute on the next bar open.
        """
        out = self.compute_indicators(df)

        long_condition = (
            (out["zscore"] < -self.entry_threshold)
            & out["rv_filter"]
            & out["volume_filter"]
        )

        short_condition = (
            (out["zscore"] > self.entry_threshold)
            & out["rv_filter"]
            & out["volume_filter"]
        )

        out["signal"] = 0
        out.loc[long_condition, "signal"] = 1
        out.loc[short_condition, "signal"] = -1

        # Remove warmup rows where indicators are not valid yet
        out = out.dropna(
            subset=[
                "zscore",
                "rv_current",
                "rv_avg",
                "atr",
                "atr_stop_distance",
            ]
        )

        return out

    def get_stop_price(
        self,
        direction: int,
        entry_price: float,
        atr_stop_distance: float
    ) -> float:
        """
        Returns raw stop-loss price before costs.

        direction:
            1  = long
           -1  = short
        """
        if direction == 1:
            return entry_price - atr_stop_distance
        elif direction == -1:
            return entry_price + atr_stop_distance
        else:
            raise ValueError("direction must be 1 for long or -1 for short")

    def zscore_exit_hit(self, direction: int, zscore: float) -> bool:
        """
        Frozen mean-reversion exit.

        Long exits when Z-score returns upward to 0.
        Short exits when Z-score returns downward to 0.
        """
        if pd.isna(zscore):
            return False

        if direction == 1:
            return zscore >= self.exit_threshold

        if direction == -1:
            return zscore <= self.exit_threshold

        return False

    def time_stop_hit(self, bars_held: int) -> bool:
        """
        Hard time stop after max_bars.
        """
        if not self.time_stop_enabled:
            return False

        return bars_held >= self.max_bars


if __name__ == "__main__":
    import yaml

    config_path = "config/strategy.yaml"

    with open(config_path, "r") as file:
        strategy_config = yaml.safe_load(file)

    strategy = VRMRStrategy(strategy_config)

    # Quick local test using the Phase 0 processed data.
    # Change this path if you are testing another pair/timeframe.
    data_path = "data/processed/EURUSD_H1_clean.parquet"

    try:
        df = pd.read_parquet(data_path)
    except FileNotFoundError:
        print(f"No data found at {data_path}")
        print("Run Phase 0 ingestion and transformation first.")
        raise SystemExit

    prepared = strategy.prepare(df)

    print("Prepared rows:", len(prepared))
    print("\nSignal counts:")
    print(prepared["signal"].value_counts())

    print("\nFirst non-zero signals:")
    print(prepared[prepared["signal"] != 0].head())