import numpy as np
import pandas as pd
from pathlib import Path
import logging
from lox.engine.hurst import build_hurst_dataframe
from fxpipe.phases.serve.loader import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_hurst_on_forex(data_dir: str, results_dir: str):
    """
    Part C: Deploy validated Hurst engine on forex H4 data.
    Part D: Run forex-specific hypotheses H1-H5.
    """
    loader = DataLoader(base_dir=data_dir)
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)
    
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    timeframe = "H4"
    
    all_hurst_dfs = {}
    
    # =========================================================
    # PART C: Deploy on Forex Data
    # =========================================================
    logging.info("=" * 60)
    logging.info("PART C: DEPLOYING HURST ENGINE ON FOREX H4 DATA")
    logging.info("=" * 60)
    
    for pair in pairs:
        logging.info(f"\n--- Processing {pair} {timeframe} ---")
        
        df = loader.get_data(pair=pair, timeframe=timeframe)
        if df.empty:
            logging.warning(f"No data found for {pair} {timeframe}. Skipping.")
            continue
        
        # Build Hurst dataframe
        hurst_df = build_hurst_dataframe(df)
        all_hurst_dfs[pair] = hurst_df
        
        # Save raw Hurst readings
        output_file = results_path / f"hurst_{pair}_{timeframe}.csv"
        hurst_df.to_csv(output_file)
        logging.info(f"Saved: {output_file} ({len(hurst_df)} rows)")
        
        # Summary statistics
        verdict_counts = hurst_df['regime_verdict'].value_counts()
        verdict_pcts = (verdict_counts / len(hurst_df) * 100).round(2)
        
        diagnostic_file = results_path / f"hurst_diagnostic_{pair}.txt"
        with open(diagnostic_file, "w") as f:
            f.write(f"HURST DIAGNOSTIC REPORT: {pair} {timeframe}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total Bars: {len(hurst_df)}\n\n")
            
            f.write("Regime Distribution:\n")
            for verdict, count in verdict_counts.items():
                pct = verdict_pcts[verdict]
                f.write(f"  {verdict:<20} | {count:>6} bars | {pct:.1f}%\n")
            
            f.write(f"\nAverage Hurst Values:\n")
            f.write(f"  H_short:  {hurst_df['H_short'].mean():.3f}\n")
            f.write(f"  H_medium: {hurst_df['H_medium'].mean():.3f}\n")
            f.write(f"  H_long:   {hurst_df['H_long'].mean():.3f}\n")
            
            # Average duration of each regime
            f.write(f"\nAverage Regime Duration (bars):\n")
            for verdict in ["MEAN_REVERTING", "TRENDING", "RANDOM", "UNCLEAR"]:
                mask = hurst_df['regime_verdict'] == verdict
                if mask.sum() > 0:
                    # Count consecutive runs
                    runs = []
                    current_run = 0
                    for val in mask:
                        if val:
                            current_run += 1
                        else:
                            if current_run > 0:
                                runs.append(current_run)
                            current_run = 0
                    if current_run > 0:
                        runs.append(current_run)
                    
                    if runs:
                        f.write(f"  {verdict:<20} | Avg: {np.mean(runs):.1f} bars | Max: {np.max(runs)} bars\n")
        
        logging.info(f"Saved diagnostic: {diagnostic_file}")
    
    # =========================================================
    # PART D: Forex Hypotheses
    # =========================================================
    logging.info("\n" + "=" * 60)
    logging.info("PART D: FOREX-SPECIFIC HYPOTHESES")
    logging.info("=" * 60)
    
    hypotheses_report = results_path / "forex_hypotheses_report.txt"
    with open(hypotheses_report, "w") as f:
        f.write("FOREX HYPOTHESES REPORT (H1-H5)\n")
        f.write("=" * 60 + "\n\n")
        
        # H1: Session Effect
        f.write("H1: SESSION EFFECT\n")
        f.write("-" * 40 + "\n")
        for pair, hurst_df in all_hurst_dfs.items():
            # Derive session from index
            hours = hurst_df.index.hour
            sessions = pd.cut(hours, bins=[0, 8, 16, 24], labels=["Asia", "London", "NY"], right=False)
            hurst_df['session'] = sessions
            
            session_means = hurst_df.groupby('session')['H_medium'].mean()
            f.write(f"\n  {pair}:\n")
            for session, h_val in session_means.items():
                f.write(f"    {session:<8} | H_medium: {h_val:.3f}\n")
        f.write("\n")
        
        # H2: Weekend Discontinuity
        f.write("H2: WEEKEND DISCONTINUITY\n")
        f.write("-" * 40 + "\n")
        for pair, hurst_df in all_hurst_dfs.items():
            # Find Friday bars and Monday bars
            day_of_week = hurst_df.index.dayofweek  # 0=Mon, 4=Fri
            
            pre_weekend = hurst_df[day_of_week == 4]['H_medium'].mean()  # Friday
            post_weekend = hurst_df[day_of_week == 0]['H_medium'].mean()  # Monday
            
            f.write(f"  {pair}:\n")
            f.write(f"    Pre-weekend (Fri) H_medium:  {pre_weekend:.3f}\n")
            f.write(f"    Post-weekend (Mon) H_medium: {post_weekend:.3f}\n")
            f.write(f"    Difference: {abs(pre_weekend - post_weekend):.3f}\n\n")
        
        # H3: Pair Relativity
        f.write("H3: PAIR RELATIVITY (Correlation of H_medium)\n")
        f.write("-" * 40 + "\n")
        h_medium_series = {}
        for pair, hurst_df in all_hurst_dfs.items():
            h_medium_series[pair] = hurst_df['H_medium']
        
        if len(h_medium_series) >= 2:
            corr_df = pd.DataFrame(h_medium_series).corr()
            f.write(corr_df.to_string() + "\n\n")
        
        # H4: Structural Break Sensitivity
        f.write("H4: STRUCTURAL BREAK SENSITIVITY\n")
        f.write("-" * 40 + "\n")
        breaks = {
            "SNB 2015": pd.Timestamp("2015-01-15"),
            "Brexit 2016": pd.Timestamp("2016-06-24"),
            "COVID 2020": pd.Timestamp("2020-03-12"),
            "Rate Hikes 2022": pd.Timestamp("2022-03-16"),
        }
        
        for pair, hurst_df in all_hurst_dfs.items():
            f.write(f"\n  {pair}:\n")
            for event_name, event_date in breaks.items():
                # Find nearest bar to event
                idx = hurst_df.index.get_indexer([event_date], method='nearest')[0]
                
                if idx >= 100 and idx < len(hurst_df) - 100:
                    pre_H = hurst_df['H_medium'].iloc[idx-100:idx].mean()
                    post_H = hurst_df['H_medium'].iloc[idx:idx+100].mean()
                    f.write(f"    {event_name:<20} | Pre: {pre_H:.3f} | Post: {post_H:.3f} | Shift: {post_H - pre_H:+.3f}\n")
        
        f.write("\n")
        
        # H5: Bar Size vs Hurst Reliability
        f.write("H5: HIGH-ATR BAR IMPACT\n")
        f.write("-" * 40 + "\n")
        for pair, hurst_df in all_hurst_dfs.items():
            # We don't have ATR in hurst_df, so use log_return magnitude as proxy
            abs_returns = hurst_df['log_return'].abs()
            threshold = abs_returns.quantile(0.95)
            
            high_vol_bars = hurst_df[abs_returns > threshold]
            normal_bars = hurst_df[abs_returns <= threshold]
            
            f.write(f"  {pair}:\n")
            f.write(f"    Normal bars H_medium:    {normal_bars['H_medium'].mean():.3f}\n")
            f.write(f"    High-vol bars H_medium:  {high_vol_bars['H_medium'].mean():.3f}\n")
            f.write(f"    Difference: {abs(high_vol_bars['H_medium'].mean() - normal_bars['H_medium'].mean()):.3f}\n\n")
    
    logging.info(f"Hypotheses report saved to: {hypotheses_report}")
    
    # =========================================================
    # ONE-PAGE VERDICT
    # =========================================================
    verdict_file = results_path / "lox_volume1_verdict.txt"
    with open(verdict_file, "w") as f:
        f.write("OPERATION LOX — VOLUME 1 VERDICT\n")
        f.write("=" * 60 + "\n\n")
        f.write("QUESTION: Does Hurst classify forex regimes usefully?\n\n")
        
        # Aggregate stats
        total_bars = sum(len(df) for df in all_hurst_dfs.values())
        total_agreement = sum(df['agreement_flag'].sum() for df in all_hurst_dfs.values())
        agreement_pct = (total_agreement / total_bars * 100) if total_bars > 0 else 0
        
        f.write(f"Total Bars Analyzed: {total_bars}\n")
        f.write(f"Agreement Rate: {agreement_pct:.1f}%\n\n")
        
        f.write("REGIME DISTRIBUTION (All Pairs Combined):\n")
        all_verdicts = pd.concat([df['regime_verdict'] for df in all_hurst_dfs.values()])
        verdict_pcts = all_verdicts.value_counts(normalize=True) * 100
        for verdict, pct in verdict_pcts.items():
            f.write(f"  {verdict:<20} | {pct:.1f}%\n")
        
        f.write("\n\nCONCLUSION:\n")
        if agreement_pct > 60:
            f.write("The Hurst engine shows strong agreement across timeframes.\n")
            f.write("Regime classification appears reliable for forex H4 data.\n")
            f.write("RECOMMENDATION: Proceed to Volume 2 with Hurst as primary regime input.\n")
        elif agreement_pct > 30:
            f.write("The Hurst engine shows moderate agreement.\n")
            f.write("Regime classification has value but needs supplementary filters.\n")
            f.write("RECOMMENDATION: Proceed to Volume 2 but add trap detector as backup.\n")
        else:
            f.write("The Hurst engine shows weak agreement across timeframes.\n")
            f.write("Regime classification may not be reliable for forex H4 data alone.\n")
            f.write("RECOMMENDATION: Rebuild regime engine or add complementary indicators.\n")
        
        f.write("\n\nFOREX-SPECIFIC ERRORS FOUND:\n")
        f.write("(See forex_hypotheses_report.txt for detailed findings)\n")
        f.write("- Check H1 for session contamination\n")
        f.write("- Check H2 for weekend gap bias\n")
        f.write("- Check H3 for USD-driven correlation\n")
        f.write("- Check H4 for structural break corruption\n")
        f.write("- Check H5 for high-vol bar distortion\n")
    
    logging.info(f"Verdict saved to: {verdict_file}")
    logging.info("\n" + "=" * 60)
    logging.info("OPERATION LOX — VOLUME 1 COMPLETE")
    logging.info("=" * 60)


if __name__ == "__main__":
    # For Colab: point to Google Drive data
    run_hurst_on_forex(
        data_dir='/content/drive/MyDrive/fxpipe_data/processed',
        results_dir='./results'
    )