import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def compute_hurst(returns: np.ndarray, min_window: int, max_window: int, num_windows: int) -> float:
    """
    Compute Hurst exponent using Detrended Fluctuation Analysis (DFA).
    
    DFA is preferred over classic R/S for financial data because:
    - Unbiased for random walks (H=0.5)
    - Correctly measures anti-persistence (H<0.5)
    - Robust to non-stationarity
    
    Parameters:
        returns: Array of log returns (NOT raw prices)
        min_window: Minimum scale for DFA
        max_window: Maximum scale for DFA
        num_windows: Number of scale values to test
    
    Returns:
        Hurst exponent H
    """
    n = len(returns)
    
    if n < min_window * 4:
        return 0.5
    
    # Step 1: Compute cumulative sum of demeaned returns (the "profile")
    mean_ret = np.mean(returns)
    Y = np.cumsum(returns - mean_ret)
    
    # Step 2: Generate logarithmically spaced scales
    scales = np.logspace(np.log10(min_window), np.log10(max_window), num_windows).astype(int)
    scales = np.unique(scales)
    scales = scales[scales >= 4]  # Need at least 4 points to fit a line
    scales = scales[scales <= n // 4]  # Need at least 4 segments
    
    if len(scales) < 3:
        return 0.5
    
    # Step 3: For each scale, compute the fluctuation function
    fluctuations = []
    valid_scales = []
    
    for scale in scales:
        num_segments = n // scale
        if num_segments < 1:
            continue
        
        rms_values = []
        
        # Forward segments
        for i in range(num_segments):
            segment = Y[i * scale : (i + 1) * scale]
            
            # Fit linear trend to this segment
            x = np.arange(scale)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            
            # Residuals after removing trend
            residuals = segment - trend
            
            # Root mean square of residuals
            rms = np.sqrt(np.mean(residuals ** 2))
            rms_values.append(rms)
        
        # Backward segments (for better statistics)
        for i in range(num_segments):
            start_idx = n - (i + 1) * scale
            end_idx = n - i * scale
            if start_idx < 0:
                continue
            segment = Y[start_idx : end_idx]
            
            x = np.arange(scale)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            residuals = segment - trend
            rms = np.sqrt(np.mean(residuals ** 2))
            rms_values.append(rms)
        
        if rms_values:
            fluctuations.append(np.mean(rms_values))
            valid_scales.append(scale)
    
    if len(valid_scales) < 3:
        return 0.5
    
    # Step 4: Fit log(F(s)) vs log(s) — slope is the Hurst exponent
    log_scales = np.log(np.array(valid_scales, dtype=float))
    log_fluct = np.log(np.array(fluctuations, dtype=float))
    
    # Remove any NaN or inf
    mask = np.isfinite(log_scales) & np.isfinite(log_fluct)
    if mask.sum() < 3:
        return 0.5
    
    slope, intercept = np.polyfit(log_scales[mask], log_fluct[mask], 1)
    
    # Clamp H to valid range [0, 1]
    return float(np.clip(slope, 0.0, 1.0))


def rolling_hurst(df: pd.DataFrame, lookback: int, min_tau: int, max_tau: int, num_windows: int) -> pd.Series:
    """
    Compute rolling Hurst exponent using DFA.
    """
    returns = df['log_return'].values
    n = len(returns)
    hurst_values = np.full(n, np.nan)
    
    for i in range(lookback, n):
        window = returns[i - lookback : i]
        if np.any(np.isnan(window)):
            continue
        hurst_values[i] = compute_hurst(window, min_tau, max_tau, num_windows)
    
    return pd.Series(hurst_values, index=df.index, name=f'H_{lookback}')


def compute_regime_verdict(h_short: float, h_medium: float, h_long: float) -> str:
    """
    Determine regime verdict based on agreement of three Hurst values.
    """
    if np.isnan(h_short) or np.isnan(h_medium) or np.isnan(h_long):
        return "INSUFFICIENT_DATA"
    
    all_below = (h_short < 0.45) and (h_medium < 0.45) and (h_long < 0.45)
    all_above = (h_short > 0.55) and (h_medium > 0.55) and (h_long > 0.55)
    all_random = (0.45 <= h_short <= 0.55) and (0.45 <= h_medium <= 0.55) and (0.45 <= h_long <= 0.55)
    
    if all_below:
        return "MEAN_REVERTING"
    elif all_above:
        return "TRENDING"
    elif all_random:
        return "RANDOM"
    else:
        return "UNCLEAR"


def build_hurst_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build complete Hurst regime dataframe from OHLCV data.
    """
    out = df.copy()
    out['log_return'] = np.log(out['close'] / out['close'].shift(1))
    
    logging.info("Computing H_short (lookback=50, tau=4-25)...")
    out['H_short'] = rolling_hurst(out, lookback=50, min_tau=4, max_tau=25, num_windows=10)
    
    logging.info("Computing H_medium (lookback=100, tau=4-50)...")
    out['H_medium'] = rolling_hurst(out, lookback=100, min_tau=4, max_tau=50, num_windows=15)
    
    logging.info("Computing H_long (lookback=200, tau=4-100)...")
    out['H_long'] = rolling_hurst(out, lookback=200, min_tau=4, max_tau=100, num_windows=15)
    
    # Agreement flag and verdict
    verdicts = []
    agreement_flags = []
    
    for i in range(len(out)):
        h_s = out['H_short'].iloc[i]
        h_m = out['H_medium'].iloc[i]
        h_l = out['H_long'].iloc[i]
        
        verdict = compute_regime_verdict(h_s, h_m, h_l)
        verdicts.append(verdict)
        
        if verdict in ["MEAN_REVERTING", "TRENDING", "RANDOM"]:
            agreement_flags.append(1)
        else:
            agreement_flags.append(0)
    
    out['agreement_flag'] = agreement_flags
    out['regime_verdict'] = verdicts
    
    output_cols = ['close', 'log_return', 'H_short', 'H_medium', 'H_long', 'agreement_flag', 'regime_verdict']
    return out[output_cols]