import pandas as pd
import numpy as np
from io import StringIO


class TradeAnalyzer:
    """
    Runs comprehensive analyses A-L on the trade log.
    Outputs a formatted text report.
    """

    def __init__(self, trades_df: pd.DataFrame, equity_curve_df: pd.DataFrame, pair: str, timeframe: str):
        self.df = trades_df
        self.equity_df = equity_curve_df
        self.pair = pair
        self.timeframe = timeframe

    def run_all_analyses(self) -> str:
        buffer = StringIO()
        buffer.write(f"{'='*70}\n")
        buffer.write(f"DEEP ANALYSIS REPORT: {self.pair} | {self.timeframe}\n")
        buffer.write(f"Total Trades: {len(self.df)}\n")
        buffer.write(f"{'='*70}\n\n")

        if self.df.empty:
            buffer.write("No trades to analyze.\n")
            return buffer.getvalue()

        buffer.write(self._analysis_a())
        buffer.write(self._analysis_b())
        buffer.write(self._analysis_c())
        buffer.write(self._analysis_d())
        buffer.write(self._analysis_e())
        buffer.write(self._analysis_f())
        buffer.write(self._analysis_g())
        buffer.write(self._analysis_h())
        buffer.write(self._analysis_i())
        buffer.write(self._analysis_j())
        buffer.write(self._analysis_k())
        buffer.write(self._analysis_l())

        return buffer.getvalue()

    def _analysis_a(self) -> str:
        buf = StringIO()
        buf.write("A. EXIT REASON BREAKDOWN\n")
        buf.write("-" * 40 + "\n")

        for reason in self.df["exit_reason"].unique():
            subset = self.df[self.df["exit_reason"] == reason]
            count = len(subset)
            pct = (count / len(self.df)) * 100
            avg_pnl = subset["net_pnl_pips"].mean()
            win_rate = (subset["net_pnl_account"] > 0).mean() * 100

            buf.write(f"  {reason:<12} | Count: {count:<5} | {pct:.1f}% | Avg Net P&L: {avg_pnl:.2f} pips | Win Rate: {win_rate:.1f}%\n")

        buf.write("\n")
        return buf.getvalue()

    def _analysis_b(self) -> str:
        buf = StringIO()
        buf.write("B. MFE / MAE ANALYSIS\n")
        buf.write("-" * 40 + "\n")

        winners = self.df[self.df["net_pnl_account"] > 0]
        losers = self.df[self.df["net_pnl_account"] <= 0]

        if not winners.empty:
            buf.write(f"  Winners ({len(winners)} trades):\n")
            buf.write(f"    Avg MFE: {winners['mfe_pips'].mean():.2f} pips\n")
            buf.write(f"    Avg MAE: {winners['mae_pips'].mean():.2f} pips\n")

        if not losers.empty:
            buf.write(f"  Losers ({len(losers)} trades):\n")
            buf.write(f"    Avg MFE: {losers['mfe_pips'].mean():.2f} pips\n")
            buf.write(f"    Avg MAE: {losers['mae_pips'].mean():.2f} pips\n")

        sl_trades = self.df[self.df["exit_reason"] == "SL"]
        if not sl_trades.empty:
            buf.write(f"  SL Trades ({len(sl_trades)} trades):\n")
            buf.write(f"    Avg MFE before stop: {sl_trades['mfe_pips'].mean():.2f} pips\n")
            buf.write(f"    (If MFE > 10 pips, stops may be too tight)\n")

        buf.write("\n")
        return buf.getvalue()

    def _analysis_c(self) -> str:
        buf = StringIO()
        buf.write("C. REGIME CLASSIFICATION CHECK\n")
        buf.write("-" * 40 + "\n")

        for regime in self.df["regime_at_entry"].unique():
            subset = self.df[self.df["regime_at_entry"] == regime]
            win_rate = (subset["net_pnl_account"] > 0).mean() * 100
            avg_pnl = subset["net_pnl_pips"].mean()
            buf.write(f"  {regime:<10} | Trades: {len(subset):<5} | Win Rate: {win_rate:.1f}% | Avg Net P&L: {avg_pnl:.2f} pips\n")

        buf.write("\n")
        return buf.getvalue()

    def _analysis_d(self) -> str:
        buf = StringIO()
        buf.write("D. Z-SCORE ENTRY DISTRIBUTION\n")
        buf.write("-" * 40 + "\n")

        # Bucket Z-scores by absolute value
        self.df["zscore_abs"] = self.df["zscore_at_entry"].abs()
        buckets = [(2.0, 2.2), (2.2, 2.5), (2.5, 3.0), (3.0, 999)]

        for low, high in buckets:
            subset = self.df[(self.df["zscore_abs"] >= low) & (self.df["zscore_abs"] < high)]
            if len(subset) > 0:
                win_rate = (subset["net_pnl_account"] > 0).mean() * 100
                avg_pnl = subset["net_pnl_pips"].mean()
                label = f"{low:.1f}-{high:.1f}" if high < 999 else f"{low:.1f}+"
                buf.write(f"  Z {label:<8} | Trades: {len(subset):<5} | Win Rate: {win_rate:.1f}% | Avg Net P&L: {avg_pnl:.2f} pips\n")

        buf.write("\n")
        return buf.getvalue()

    def _analysis_e(self) -> str:
        buf = StringIO()
        buf.write("E. VOLUME FILTER IMPACT\n")
        buf.write("-" * 40 + "\n")

        buckets = [(1.5, 1.8), (1.8, 2.5), (2.5, 999)]
        for low, high in buckets:
            subset = self.df[(self.df["volume_ratio_at_entry"] >= low) & (self.df["volume_ratio_at_entry"] < high)]
            if len(subset) > 0:
                win_rate = (subset["net_pnl_account"] > 0).mean() * 100
                avg_pnl = subset["net_pnl_pips"].mean()
                label = f"{low:.1f}-{high:.1f}" if high < 999 else f"{low:.1f}+"
                buf.write(f"  Vol Ratio {label:<8} | Trades: {len(subset):<5} | Win Rate: {win_rate:.1f}% | Avg Net P&L: {avg_pnl:.2f} pips\n")

        buf.write("\n")
        return buf.getvalue()

    def _analysis_f(self) -> str:
        buf = StringIO()
        buf.write("F. SESSION ANALYSIS\n")
        buf.write("-" * 40 + "\n")

        for session in ["Asia", "London", "New York"]:
            subset = self.df[self.df["session_at_entry"] == session]
            if len(subset) > 0:
                win_rate = (subset["net_pnl_account"] > 0).mean() * 100
                avg_pnl = subset["net_pnl_pips"].mean()
                buf.write(f"  {session:<8} | Trades: {len(subset):<5} | Win Rate: {win_rate:.1f}% | Avg Net P&L: {avg_pnl:.2f} pips\n")

        buf.write("\n")
        return buf.getvalue()

    def _analysis_g(self) -> str:
        buf = StringIO()
        buf.write("G. LONG vs SHORT ASYMMETRY\n")
        buf.write("-" * 40 + "\n")

        for direction in ["Long", "Short"]:
            subset = self.df[self.df["direction"] == direction]
            if len(subset) > 0:
                win_rate = (subset["net_pnl_account"] > 0).mean() * 100
                avg_pnl = subset["net_pnl_pips"].mean()
                avg_duration = subset["bars_held"].mean()
                buf.write(f"  {direction:<6} | Trades: {len(subset):<5} | Win Rate: {win_rate:.1f}% | Avg Net P&L: {avg_pnl:.2f} pips | Avg Duration: {avg_duration:.1f} bars\n")

        buf.write("\n")
        return buf.getvalue()

    def _analysis_h(self) -> str:
        buf = StringIO()
        buf.write("H. ROLLING PERFORMANCE (50-trade window)\n")
        buf.write("-" * 40 + "\n")

        if len(self.df) >= 50:
            rolling_win_rate = (self.df["net_pnl_account"] > 0).rolling(50).mean() * 100
            rolling_pnl = self.df["net_pnl_pips"].rolling(50).mean()

            buf.write(f"  First 50 trades Win Rate: {rolling_win_rate.iloc[49]:.1f}%\n")
            buf.write(f"  Last 50 trades Win Rate:  {rolling_win_rate.iloc[-1]:.1f}%\n")
            buf.write(f"  Best 50-trade Win Rate:   {rolling_win_rate.max():.1f}%\n")
            buf.write(f"  Worst 50-trade Win Rate:  {rolling_win_rate.min():.1f}%\n")
            buf.write(f"  (Check if edge degraded over time)\n")
        else:
            buf.write("  Not enough trades for rolling analysis.\n")

        buf.write("\n")
        return buf.getvalue()

    def _analysis_i(self) -> str:
        buf = StringIO()
        buf.write("I. DURATION ANALYSIS\n")
        buf.write("-" * 40 + "\n")

        winners = self.df[self.df["net_pnl_account"] > 0]
        losers = self.df[self.df["net_pnl_account"] <= 0]

        if not winners.empty:
            buf.write(f"  Winners avg duration: {winners['bars_held'].mean():.1f} bars\n")
        if not losers.empty:
            buf.write(f"  Losers avg duration:  {losers['bars_held'].mean():.1f} bars\n")

        buf.write(f"  Overall avg duration: {self.df['bars_held'].mean():.1f} bars\n")
        buf.write(f"  Median duration:      {self.df['bars_held'].median():.1f} bars\n")

        buf.write("\n")
        return buf.getvalue()

    def _analysis_j(self) -> str:
        buf = StringIO()
        buf.write("J. COST IMPACT\n")
        buf.write("-" * 40 + "\n")

        avg_gross = self.df["gross_pnl_pips"].mean()
        avg_cost = self.df["cost_pips"].mean()
        avg_net = self.df["net_pnl_pips"].mean()

        total_gross_profit = self.df[self.df["gross_pnl_pips"] > 0]["gross_pnl_pips"].sum()
        total_cost = self.df["cost_pips"].sum()

        cost_pct = (total_cost / total_gross_profit * 100) if total_gross_profit > 0 else 0

        buf.write(f"  Avg Gross P&L per trade: {avg_gross:.2f} pips\n")
        buf.write(f"  Avg Cost per trade:      {avg_cost:.2f} pips\n")
        buf.write(f"  Avg Net P&L per trade:   {avg_net:.2f} pips\n")
        buf.write(f"  Total Cost as % of Gross Profit: {cost_pct:.1f}%\n")

        buf.write("\n")
        return buf.getvalue()

    def _analysis_k(self) -> str:
        buf = StringIO()
        buf.write("K. CONSECUTIVE LOSS IMPACT\n")
        buf.write("-" * 40 + "\n")

        after_streak = self.df[self.df["consecutive_losses_before"] >= 3]
        normal = self.df[self.df["consecutive_losses_before"] < 3]

        if len(after_streak) > 0:
            wr_streak = (after_streak["net_pnl_account"] > 0).mean() * 100
            avg_streak = after_streak["net_pnl_pips"].mean()
            buf.write(f"  After 3+ losses ({len(after_streak)} trades): Win Rate {wr_streak:.1f}% | Avg Net P&L {avg_streak:.2f} pips\n")

        if len(normal) > 0:
            wr_normal = (normal["net_pnl_account"] > 0).mean() * 100
            avg_normal = normal["net_pnl_pips"].mean()
            buf.write(f"  After 0-2 losses ({len(normal)} trades): Win Rate {wr_normal:.1f}% | Avg Net P&L {avg_normal:.2f} pips\n")

        buf.write("\n")
        return buf.getvalue()

    def _analysis_l(self) -> str:
        buf = StringIO()
        buf.write("L. EQUITY CURVE SUMMARY\n")
        buf.write("-" * 40 + "\n")

        if not self.equity_df.empty:
            start_eq = self.equity_df["equity"].iloc[0]
            end_eq = self.equity_df["equity"].iloc[-1]
            min_eq = self.equity_df["equity"].min()
            max_eq = self.equity_df["equity"].max()

            buf.write(f"  Starting Equity: ${start_eq:.2f}\n")
            buf.write(f"  Ending Equity:   ${end_eq:.2f}\n")
            buf.write(f"  Peak Equity:     ${max_eq:.2f}\n")
            buf.write(f"  Trough Equity:   ${min_eq:.2f}\n")
            buf.write(f"  Total Return:    {((end_eq - start_eq) / start_eq) * 100:.2f}%\n")

        buf.write("\n")
        return buf.getvalue()