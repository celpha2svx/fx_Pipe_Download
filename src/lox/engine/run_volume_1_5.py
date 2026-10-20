import numpy as np
import pandas as pd
import logging
from pathlib import Path
from fxpipe.phases.serve.loader import DataLoader
from src.lox.engine.hurst_fast import build_hurst_dataframe

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ============================================================
# PHASE 1.5A: Multi-Timeframe Deployment
# ============================================================

def run_phase_1_5a(loader, results_path, pairs, timeframes):
    """Run Hurst on all pairs/timeframes and save distribution comparison."""
    logging.info("=" * 60)
    logging.info("PHASE 1.5A: MULTI-TIMEFRAME DEPLOYMENT")
    logging.info("=" * 60)
    
    all_distributions = []
    hurst_data = {}  # Store for later phases
    
    for pair in pairs:
        for tf in timeframes:
            logging.info(f"--- {pair} {tf} ---")
            
            try:
                df = loader.get_data(pair=pair, timeframe=tf)
                if df.empty:
                    logging.warning(f"No data for {pair} {tf}. Skipping.")
                    continue
            except Exception as e:
                logging.warning(f"Error loading {pair} {tf}: {e}")
                continue
            
            # Run Hurst
            hurst_df = build_hurst_dataframe(df)
            key = f"{pair}_{tf}"
            hurst_data[key] = hurst_df
            
            # Save CSV
            csv_path = results_path / f"hurst_{pair}_{tf}.csv"
            hurst_df.to_csv(csv_path)
            logging.info(f"Saved: {csv_path} ({len(hurst_df)} rows)")
            
            # Distribution
            verdict_counts = hurst_df['regime_verdict'].value_counts()
            verdict_pcts = (verdict_counts / len(hurst_df) * 100).round(2)
            
            for regime in ["TRENDING", "UNCLEAR", "RANDOM", "MEAN_REVERTING", "INSUFFICIENT_DATA"]:
                all_distributions.append({
                    "pair": pair,
                    "timeframe": tf,
                    "regime": regime,
                    "percentage": verdict_pcts.get(regime, 0.0),
                    "bars": int(verdict_counts.get(regime, 0))
                })
    
    # Save distribution report
    dist_df = pd.DataFrame(all_distributions)
    pivot_df = dist_df.pivot_table(index='regime', columns=['pair', 'timeframe'], values='percentage', fill_value=0.0)
    
    report_path = results_path / "regime_distribution_by_timeframe.txt"
    with open(report_path, "w") as f:
        f.write("PHASE 1.5A: REGIME DISTRIBUTION BY TIMEFRAME\n")
        f.write("=" * 80 + "\n\n")
        f.write("Question: Is 0.0% mean-reverting an H4 fact or a forex fact?\n\n")
        f.write("PERCENTAGE DISTRIBUTION:\n")
        f.write(pivot_df.to_string())
        f.write("\n")
    
    logging.info(f"Saved: {report_path}")
    return hurst_data, dist_df


# ============================================================
# PHASE 1.5B: Forward Return Analysis
# ============================================================

def run_phase_1_5b(hurst_data, results_path, pairs, timeframes):
    """Measure forward returns conditioned on regime."""
    logging.info("\n" + "=" * 60)
    logging.info("PHASE 1.5B: REGIME → FORWARD RETURN ANALYSIS")
    logging.info("=" * 60)
    
    horizons = [5, 10, 20, 50]
    all_results = []
    
    for pair in pairs:
        for tf in timeframes:
            key = f"{pair}_{tf}"
            if key not in hurst_data:
                continue
            
            df = hurst_data[key].copy()
            
            # Compute forward returns for each horizon
            for h in horizons:
                df[f'fwd_ret_{h}'] = df['log_return'].shift(-h).rolling(h).sum()
            
            # Analyze by regime
            for regime in df['regime_verdict'].unique():
                if regime == "INSUFFICIENT_DATA":
                    continue
                
                subset = df[df['regime_verdict'] == regime]
                
                for h in horizons:
                    col = f'fwd_ret_{h}'
                    valid = subset[col].dropna()
                    
                    if len(valid) > 0:
                        all_results.append({
                            "pair": pair,
                            "timeframe": tf,
                            "regime": regime,
                            "horizon": h,
                            "mean_return": valid.mean(),
                            "std_return": valid.std(),
                            "pct_positive": (valid > 0).mean() * 100,
                            "sample_count": len(valid)
                        })
    
    fwd_df = pd.DataFrame(all_results)
    fwd_path = results_path / "forward_returns_by_regime.csv"
    fwd_df.to_csv(fwd_path, index=False)
    logging.info(f"Saved: {fwd_path}")
    
    # Key question summary
    summary_path = results_path / "forward_return_summary.txt"
    with open(summary_path, "w") as f:
        f.write("PHASE 1.5B: FORWARD RETURN ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write("Question: Does TRENDING regime actually produce directional movement?\n\n")
        
        for pair in pairs:
            for tf in timeframes:
                subset = fwd_df[(fwd_df['pair'] == pair) & (fwd_df['timeframe'] == tf)]
                if subset.empty:
                    continue
                
                f.write(f"\n{pair} {tf}:\n")
                f.write("-" * 40 + "\n")
                
                for regime in ["TRENDING", "MEAN_REVERTING", "UNCLEAR", "RANDOM"]:
                    reg_data = subset[subset['regime'] == regime]
                    if reg_data.empty:
                        continue
                    f.write(f"  {regime}:\n")
                    for _, row in reg_data.iterrows():
                        f.write(f"    H={row['horizon']:>2} | mean={row['mean_return']:>8.5f} | "
                                f"std={row['std_return']:>8.5f} | "
                                f"%pos={row['pct_positive']:>5.1f}% | "
                                f"n={row['sample_count']}\n")
    
    logging.info(f"Saved: {summary_path}")
    return fwd_df


# ============================================================
# PHASE 1.5C: Volatility Regime Layer
# ============================================================

def run_phase_1_5c(hurst_data, results_path, pairs, timeframes):
    """Add volatility regime and create 2D combined regime map."""
    logging.info("\n" + "=" * 60)
    logging.info("PHASE 1.5C: VOLATILITY REGIME LAYER")
    logging.info("=" * 60)
    
    combined_results = []
    
    for pair in pairs:
        for tf in timeframes:
            key = f"{pair}_{tf}"
            if key not in hurst_data:
                continue
            
            df = hurst_data[key].copy()
            
            # Realized volatility (20-period)
            df['rv_20'] = df['log_return'].rolling(20).std()
            
            # 100-period average volatility
            df['rv_100_avg'] = df['log_return'].rolling(100).std().rolling(100).mean()
            
            # Volatility percentile rank (relative to last 500 bars)
            df['vol_pctile'] = df['rv_20'].rolling(500).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
            )
            
            # Volatility regime label
            def vol_regime(pctile):
                if pd.isna(pctile):
                    return "UNKNOWN"
                elif pctile < 25:
                    return "LOW_VOL"
                elif pctile < 75:
                    return "NORMAL_VOL"
                elif pctile < 95:
                    return "HIGH_VOL"
                else:
                    return "EXTREME_VOL"
            
            df['vol_regime'] = df['vol_pctile'].apply(vol_regime)
            
            # Combined 2D regime
            df['combined_regime'] = df['regime_verdict'] + "_" + df['vol_regime']
            
            # Save updated CSV
            csv_path = results_path / f"hurst_vol_{pair}_{tf}.csv"
            df.to_csv(csv_path)
            
            # Distribution of combined regimes
            combined_counts = df['combined_regime'].value_counts()
            combined_pcts = (combined_counts / len(df) * 100).round(2)
            
            for regime, pct in combined_pcts.items():
                combined_results.append({
                    "pair": pair,
                    "timeframe": tf,
                    "combined_regime": regime,
                    "percentage": pct,
                    "bars": int(combined_counts[regime])
                })
    
    # Save combined distribution
    combined_df = pd.DataFrame(combined_results)
    combined_path = results_path / "combined_regime_distribution.csv"
    combined_df.to_csv(combined_path, index=False)
    logging.info(f"Saved: {combined_path}")
    
    return combined_df


# ============================================================
# PHASE 1.5D: Regime Dynamics (Persistence & Transitions)
# ============================================================

def run_phase_1_5d(hurst_data, results_path, pairs, timeframes):
    """Analyze regime duration, transitions, and trend age."""
    logging.info("\n" + "=" * 60)
    logging.info("PHASE 1.5D: REGIME DYNAMICS")
    logging.info("=" * 60)
    
    dynamics_report = results_path / "regime_dynamics_report.txt"
    
    with open(dynamics_report, "w") as f:
        f.write("PHASE 1.5D: REGIME DYNAMICS\n")
        f.write("=" * 80 + "\n\n")
        
        for pair in pairs:
            for tf in timeframes:
                key = f"{pair}_{tf}"
                if key not in hurst_data:
                    continue
                
                df = hurst_data[key].copy()
                verdicts = df['regime_verdict'].values
                
                f.write(f"\n{'='*60}\n")
                f.write(f"{pair} {tf}\n")
                f.write(f"{'='*60}\n\n")
                
                # --- Duration Analysis ---
                f.write("REGIME DURATIONS:\n")
                f.write("-" * 40 + "\n")
                
                durations = {regime: [] for regime in ["TRENDING", "UNCLEAR", "RANDOM", "MEAN_REVERTING"]}
                current_regime = None
                current_duration = 0
                
                for v in verdicts:
                    if v == current_regime:
                        current_duration += 1
                    else:
                        if current_regime in durations and current_duration > 0:
                            durations[current_regime].append(current_duration)
                        current_regime = v
                        current_duration = 1
                
                # Don't forget the last run
                if current_regime in durations and current_duration > 0:
                    durations[current_regime].append(current_duration)
                
                for regime, durs in durations.items():
                    if durs:
                        f.write(f"  {regime:<16} | Avg: {np.mean(durs):>6.1f} bars | "
                                f"Median: {np.median(durs):>6.1f} | "
                                f"Max: {np.max(durs):>5} | "
                                f"Count: {len(durs)}\n")
                    else:
                        f.write(f"  {regime:<16} | No occurrences\n")
                
                # --- Transition Matrix ---
                f.write("\nTRANSITION PROBABILITIES:\n")
                f.write("-" * 40 + "\n")
                
                regimes = ["TRENDING", "UNCLEAR", "RANDOM", "MEAN_REVERTING"]
                transitions = {r: {r2: 0 for r2 in regimes} for r in regimes}
                
                for i in range(len(verdicts) - 1):
                    curr = verdicts[i]
                    nxt = verdicts[i + 1]
                    if curr in regimes and nxt in regimes:
                        transitions[curr][nxt] += 1
                
                for from_regime in regimes:
                    total = sum(transitions[from_regime].values())
                    if total > 0:
                        f.write(f"  From {from_regime}:\n")
                        for to_regime in regimes:
                            prob = transitions[from_regime][to_regime] / total * 100
                            f.write(f"    → {to_regime:<16}: {prob:>5.1f}%\n")
                
                # --- Trend Age Analysis ---
                f.write("\nTREND AGE ANALYSIS (Forward returns by trend duration):\n")
                f.write("-" * 40 + "\n")
                
                # Compute forward 10-bar return
                df['fwd_10'] = df['log_return'].shift(-10).rolling(10).sum()
                
                # Track trend age
                trend_ages = []
                current_age = 0
                for v in verdicts:
                    if v == "TRENDING":
                        current_age += 1
                        trend_ages.append(current_age)
                    else:
                        current_age = 0
                        trend_ages.append(0)
                
                df['trend_age'] = trend_ages
                
                # Bucket by age
                young = df[(df['trend_age'] >= 1) & (df['trend_age'] <= 10)]
                middle = df[(df['trend_age'] > 10) & (df['trend_age'] <= 30)]
                old = df[df['trend_age'] > 30]
                
                for label, subset in [("Young (1-10 bars)", young), ("Middle (11-30)", middle), ("Old (30+)", old)]:
                    valid_fwd = subset['fwd_10'].dropna()
                    if len(valid_fwd) > 0:
                        f.write(f"  {label:<20} | mean fwd return: {valid_fwd.mean():>8.5f} | "
                                f"n={len(valid_fwd)}\n")
    
    logging.info(f"Saved: {dynamics_report}")


# ============================================================
# MASTER ORCHESTRATOR
# ============================================================

def run_volume_1_5(data_dir: str, results_dir: str):
    """Run ALL of Phase 1.5 in one shot."""
    
    loader = DataLoader(base_dir=data_dir)
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)
    
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    timeframes = ["H1", "H4", "D1"]
    
    # Phase 1.5A
    hurst_data, dist_df = run_phase_1_5a(loader, results_path, pairs, timeframes)
    
    # Phase 1.5B
    fwd_df = run_phase_1_5b(hurst_data, results_path, pairs, timeframes)
    
    # Phase 1.5C
    combined_df = run_phase_1_5c(hurst_data, results_path, pairs, timeframes)
    
    # Phase 1.5D
    run_phase_1_5d(hurst_data, results_path, pairs, timeframes)
    
    logging.info("\n" + "=" * 60)
    logging.info("VOLUME 1.5 COMPLETE")
    logging.info("=" * 60)
    logging.info(f"All results saved to: {results_path}")


if __name__ == "__main__":
    run_volume_1_5(
        data_dir='/content/drive/MyDrive/fxpipe_data/processed',
        results_dir='./results_v1_5'
    )