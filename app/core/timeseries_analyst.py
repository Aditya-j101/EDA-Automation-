import os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from typing import Dict, Any, Optional

from app.tools.ingester import load_dataset

def run_timeseries_analysis(dataset_path: str, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Performs deterministic time-series and chronological data drift analysis:
      - Datetime column identification & chronological sorting
      - Rolling 30-period mean on continuous feature
      - 2-sample Kolmogorov-Smirnov (KS) test between early 50% and late 50% data
    """
    df = load_dataset(dataset_path)
        
    non_numeric_cols = [c for c in df.select_dtypes(exclude='number').columns if c != target_col]
    numeric_cols = [c for c in df.select_dtypes(include='number').columns if c != target_col and not c.endswith('_was_missing')]
    
    date_col = None
    for col in non_numeric_cols:
        try:
            parsed = pd.to_datetime(df[col], errors='coerce')
            if parsed.notna().mean() > 0.5:
                date_col = col
                df[date_col] = parsed
                break
        except Exception:
            pass
            
    if date_col is None:
        summary_text = "=== TIME-SERIES ANALYSIS REPORT ===\nNo temporal data found. Skipping time-series analysis.\n=== TIME-SERIES COMPLETE ==="
        return {
            "has_datetime": False,
            "date_col": None,
            "rolling_mean_last": None,
            "drift_detected": False,
            "ks_stat": None,
            "ks_pvalue": None,
            "summary_text": summary_text
        }
        
    df = df.sort_values(date_col).reset_index(drop=True)
    rolling_mean_last = None
    ks_stat = None
    ks_pvalue = None
    drift_detected = False
    target_num_col = numeric_cols[0] if numeric_cols else None
    
    if target_num_col:
        rolling_series = df[target_num_col].rolling(window=30, min_periods=1).mean()
        rolling_mean_last = round(float(rolling_series.iloc[-1]), 4)
        
        mid = len(df) // 2
        early = df[target_num_col].iloc[:mid].dropna()
        late = df[target_num_col].iloc[mid:].dropna()
        
        if len(early) >= 5 and len(late) >= 5:
            stat_val, p_val = ks_2samp(early, late)
            ks_stat = round(float(stat_val), 4)
            ks_pvalue = round(float(p_val), 4)
            drift_detected = bool(p_val < 0.05)

    summary_lines = [
        "=== TIME-SERIES ANALYSIS REPORT ===",
        f"Datetime Column Identified: '{date_col}'",
        f"Chronological Range: {df[date_col].min()} to {df[date_col].max()}",
    ]
    if target_num_col:
        summary_lines.append(f"Analyzed Continuous Metric: '{target_num_col}'")
        summary_lines.append(f"  30-Period Rolling Mean (Final): {rolling_mean_last}")
        summary_lines.append(f"  Data Drift Analysis (Kolmogorov-Smirnov Test between early & late halves):")
        summary_lines.append(f"    KS Statistic: {ks_stat}, p-value: {ks_pvalue}")
        summary_lines.append(f"    Conclusion: {'DRIFT DETECTED (Statistically Significant)' if drift_detected else 'No Significant Drift Detected'}")
        
    summary_lines.append("=== TIME-SERIES COMPLETE ===")
    summary_text = "\n".join(summary_lines)
    
    return {
        "has_datetime": True,
        "date_col": date_col,
        "rolling_mean_last": rolling_mean_last,
        "drift_detected": drift_detected,
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pvalue,
        "summary_text": summary_text
    }
