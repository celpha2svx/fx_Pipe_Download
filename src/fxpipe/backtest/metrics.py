import pandas as pd
import numpy as np
from datetime import datetime

class MetricsCalculator:
    """
    Computes all baseline performance metrics from the engine's output.
    """
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital

    def calculate(self, trades: list, equity_curve: list, final_equity: float, start_date: datetime, end_date: datetime) -> dict:
        # Default metrics for when there are 0 trades (prevents division by zero errors)
        metrics = {
            "total_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0,
            "trades_per_year": 0.0,
            "avg_duration_bars": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "longest_win_streak": 0,
            "longest_lose_streak": 0
        }

        if not trades:
            return metrics

        df = pd.DataFrame(trades)
        metrics["total_trades"] = len(df)
        metrics["total_return_pct"] = ((final_equity - self.initial_capital) / self.initial_capital) * 100

        # 1. Win/Loss Analysis
        wins = df[df["pnl"] > 0]
        losses = df[df["pnl"] <= 0]

        metrics["win_rate_pct"] = (len(wins) / len(df)) * 100
        metrics["avg_win"] = wins["pnl"].mean() if not wins.empty else 0.0
        metrics["avg_loss"] = losses["pnl"].mean() if not losses.empty else 0.0
        metrics["avg_duration_bars"] = df["bars_held"].mean()

        gross_profit = wins["pnl"].sum() if not wins.empty else 0.0
        gross_loss = abs(losses["pnl"].sum()) if not losses.empty else 0.0
        metrics["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else np.inf

        # 2. Trades Per Year
        # Avoid division by zero if start and end dates are the same
        years = max((end_date - start_date).days / 365.25, 0.01) 
        metrics["trades_per_year"] = len(df) / years

        # 3. Streaks
        metrics["longest_win_streak"] = self._get_max_streak(df["pnl"], win=True)
        metrics["longest_lose_streak"] = self._get_max_streak(df["pnl"], win=False)

        # 4. Equity Curve Metrics (Sharpe Ratio & Max Drawdown)
        # We use daily returns for a mathematically accurate Sharpe Ratio
        if equity_curve:
            eq_df = pd.DataFrame(equity_curve)
            eq_df.set_index('timestamp', inplace=True)
            
            # Group by day to get daily returns
            daily_eq = eq_df['equity'].resample('D').last().dropna()
            daily_returns = daily_eq.pct_change().dropna()
            
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                # Annualize assuming 252 trading days
                metrics["sharpe_ratio"] = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
                
                # Recalculate Max DD from daily equity to be precise
                rolling_max = daily_eq.cummax()
                drawdowns = (daily_eq - rolling_max) / rolling_max
                metrics["max_drawdown_pct"] = abs(drawdowns.min()) * 100

        return metrics

    def _get_max_streak(self, pnls: pd.Series, win: bool) -> int:
        """Helper to calculate longest winning or losing streak."""
        max_streak = 0
        current_streak = 0
        for pnl in pnls:
            if (win and pnl > 0) or (not win and pnl <= 0):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak


if __name__ == "__main__":
    # Quick local test to ensure it handles 0 trades gracefully
    calc = MetricsCalculator(initial_capital=1000.0)
    
    dummy_trades = []
    dummy_equity = [{"timestamp": pd.Timestamp("2023-01-01"), "equity": 1000.0}]
    
    results = calc.calculate(
        trades=dummy_trades, 
        equity_curve=dummy_equity, 
        final_equity=1000.0, 
        start_date=pd.Timestamp("2023-01-01"), 
        end_date=pd.Timestamp("2023-01-07")
    )
    
    print("--- Metrics Test (0 Trades) ---")
    for k, v in results.items():
        print(f"{k}: {v}")