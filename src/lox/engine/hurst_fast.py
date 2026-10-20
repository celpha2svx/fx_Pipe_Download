import numpy as np
import pandas as pd
from numba import njit
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ============================================================
# NUMBA-COMPILED CORE (C-speed)
# ============================================================

@njit(cache=True)
def _linear_detrend_rms(segment, x_arr, x_mean, x_var):
    """
    Compute RMS after removing linear trend.
    Fully Numba-compiled. No np.polyfit (too slow).
    Uses manual least-squares: slope = cov(x,y) / var(x)
    """
    n = len(segment)
    
    # Mean of segment
    y_mean = 0.0
    for i in range(n):
        y_mean += segment[i]
    y_mean /= n
    
    # Slope via least squares
    xy_sum = 0.0
    for i in range(n):
        xy_sum += (x_arr[i] - x_mean) * (segment[i] - y_mean)
    
    slope = xy_sum / x_var if x_var > 1e-12 else 0.0
    intercept = y_mean - slope * x_mean
    
    # RMS of residuals
    rms_sq = 0.0
    for i in range(n):
        residual = segment[i] - (slope * x_arr[i] + intercept)
        rms_sq += residual * residual
    
    return np.sqrt(rms_sq / n)


@njit(cache=True)
def _dfa_core(profile, scales, n):
    """
    Core DFA loop. Fully compiled.
    For each scale: split profile into segments, detrend, compute fluctuation.
    """
    num_scales = len(scales)
    fluctuations = np.zeros(num_scales)
    
    for s_idx in range(num_scales):
        scale = scales[s_idx]
        num_segments = n // scale
        
        if num_segments < 1:
            fluctuations[s_idx] = 0.0
            continue
        
        # Pre-compute x values for this scale (linear trend basis)
        x_arr = np.empty(scale, dtype=np.float64)
        for i in range(scale):
            x_arr[i] = float(i)
        
        x_mean = 0.0
        for i in range(scale):
            x_mean += x_arr[i]
        x_mean /= scale
        
        x_var = 0.0
        for i in range(scale):
            x_var += (x_arr[i] - x_mean) ** 2
        
        total_rms_sq = 0.0
        count = 0
        
        # Forward segments
        for i in range(num_segments):
            start = i * scale
            end = start + scale
            segment = profile[start:end]
            rms = _linear_detrend_rms(segment, x_arr, x_mean, x_var)
            total_rms_sq += rms * rms
            count += 1
        
        # Backward segments (better statistics)
        for i in range(num_segments):
            end = n - i * scale
            start = end - scale
            if start < 0:
                continue
            segment = profile[start:end]
            rms = _linear_detrend_rms(segment, x_arr, x_mean, x_var)
            total_rms_sq += rms * rms
            count += 1
        
        if count > 0:
            fluctuations[s_idx] = np.sqrt(total_rms_sq / count)
        else:
            fluctuations[s_idx] = 0.0
    
    return fluctuations


@njit(cache=True)
def _compute_hurst_numba(returns, min_window, max_window, num_windows):
    """
    Full DFA Hurst computation in compiled code.
    """
    n = len(returns)
    
    if n < min_window * 4:
        return 0.5
    
    # Step 1: Build profile (cumulative sum of demeaned returns)
    mean_ret = 0.0
    for i in range(n):
        mean_ret += returns[i]
    mean_ret /= n
    
    profile = np.empty(n, dtype=np.float64)
    cumsum = 0.0
    for i in range(n):
        cumsum += returns[i] - mean_ret
        profile[i] = cumsum
    
    # Step 2: Generate log-spaced scales
    log_min = np.log10(float(min_window))
    log_max = np.log10(float(max_window))
    
    if num_windows < 3:
        num_windows = 3
    
    scales_raw = np.empty(num_windows, dtype=np.float64)
    for i in range(num_windows):
        scales_raw[i] = 10.0 ** (log_min + i * (log_max - log_min) / (num_windows - 1))
    
    # Convert to int and filter
    scales_int = np.empty(num_windows, dtype=np.int64)
    for i in range(num_windows):
        scales_int[i] = int(scales_raw[i])
    
    # Filter valid scales: >= 4 and <= n//4
    valid_scales = []
    for i in range(num_windows):
        s = scales_int[i]
        if s >= 4 and s <= n // 4:
            valid_scales.append(s)
    
    # Remove duplicates
    unique_scales = []
    for s in valid_scales:
        if s not in unique_scales:
            unique_scales.append(s)
    
    if len(unique_scales) < 3:
        return 0.5
    
    scales = np.array(unique_scales, dtype=np.int64)
    
    # Step 3: Run DFA core
    fluctuations = _dfa_core(profile, scales, n)
    
    # Step 4: Filter valid fluctuations and fit log-log
    valid_count = 0
    for i in range(len(fluctuations)):
        if fluctuations[i] > 0:
            valid_count += 1
    
    if valid_count < 3:
        return 0.5
    
    log_scales = np.empty(valid_count, dtype=np.float64)
    log_fluct = np.empty(valid_count, dtype=np.float64)
    idx = 0
    for i in range(len(fluctuations)):
        if fluctuations[i] > 0:
            log_scales[idx] = np.log(float(scales[i]))
            log_fluct[idx] = np.log(fluctuations[i])
            idx += 1
    
    # Manual linear regression (Numba-compatible)
    n_pts = valid_count
    x_mean = 0.0
    y_mean = 0.0
    for i in range(n_pts):
        x_mean += log_scales[i]
        y_mean += log_fluct[i]
    x_mean /= n_pts
    y_mean /= n_pts
    
    xy_sum = 0.0
    xx_sum = 0.0
    for i in range(n_pts):
        xy_sum += (log_scales[i] - x_mean) * (log_fluct[i] - y_mean)
        xx_sum += (log_scales[i] - x_mean) ** 2
    
    slope = xy_sum / xx_sum if xx_sum > 1e-12 else 0.5
    
    # Clamp to [0, 1]
    if slope < 0.0:
        slope = 0.0
    elif slope > 1.0:
        slope = 1.0
    
    return slope


# ============================================================
# PUBLIC INTERFACE (same signature as original hurst.py)
# ============================================================

def compute_hurst(returns, min_window, max_window, num_windows):
    """
    Compute Hurst exponent using Numba-accelerated DFA.
    Drop-in replacement for the original compute_hurst.
    """
    returns = np.asarray(returns, dtype=np.float64)
    returns = np.nan_to_num(returns, nan=0.0)
    return _compute_hurst_numba(returns, min_window, max_window, num_windows)


def rolling_hurst(df, lookback, min_tau, max_tau, num_windows):
    """
    Rolling Hurst exponent using Numba-optimized core.
    """
    returns = df['log_return'].values.astype(np.float64)
    n = len(returns)
    hurst_values = np.full(n, np.nan)
    
    for i in range(lookback, n):
        window = returns[i - lookback:i]
        if np.any(np.isnan(window)):
            continue
        hurst_values[i] = compute_hurst(window, min_tau, max_tau, num_windows)
    
    return pd.Series(hurst_values, index=df.index, name=f'H_{lookback}')


def compute_regime_verdict(h_short, h_medium, h_long):
    """Same agreement logic as original."""
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


def build_hurst_dataframe(df):
    """
    Build complete Hurst regime dataframe.
    Uses Numba-optimized rolling_hurst for speed.
    """
    out = df.copy()
    out['log_return'] = np.log(out['close'] / out['close'].shift(1))
    
    logging.info("Computing H_short (lookback=50, tau=4-25)...")
    out['H_short'] = rolling_hurst(out, lookback=50, min_tau=4, max_tau=25, num_windows=10)
    
    logging.info("Computing H_medium (lookback=100, tau=4-50)...")
    out['H_medium'] = rolling_hurst(out, lookback=100, min_tau=4, max_tau=50, num_windows=15)
    
    logging.info("Computing H_long (lookback=200, tau=4-100)...")
    out['H_long'] = rolling_hurst(out, lookback=200, min_tau=4, max_tau=100, num_windows=15)
    
    # Agreement and verdict
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