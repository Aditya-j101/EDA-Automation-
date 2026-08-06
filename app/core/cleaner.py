import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

def _compute_mad_outliers(series: pd.Series) -> pd.Series:
    """Computes boolean Series indicating outliers using Modified Z-Score via MAD."""
    s = series.dropna()
    if len(s) < 3:
        return pd.Series(False, index=series.index)
    med = s.median()
    mad = np.median(np.abs(s - med))
    if mad == 0:
        return pd.Series(False, index=series.index)
    mod_z = 0.6745 * np.abs(series - med) / mad
    return mod_z > 3.5

from app.tools.ingester import load_dataset

def run_cleaning(
    dataset_path: str,
    target_col: Optional[str] = None,
    workspace_dir: Optional[str] = None,
    output_path: Optional[str] = None,
    cap_outliers: bool = False,
    run_id: str = "default"
) -> Dict[str, Any]:
    """
    Performs deterministic data cleaning, missingness contract handling, MAD outlier detection,
    and missingness mechanism diagnosis.
    """
    df = load_dataset(dataset_path)
        
    pre_shape = df.shape
    total_rows = len(df) if len(df) > 0 else 1
    
    # Target protection log
    if target_col:
        logging.info(f"[run_id={run_id}] Target variable '{target_col}' protected from cleaning/capping.")
        
    # Determine output paths
    if workspace_dir:
        data_dir = os.path.join(workspace_dir, "data")
        final_output_path = os.path.join(data_dir, "cleaned_data.csv")
        exploratory_path = os.path.join(data_dir, "eda_exploratory_data.csv")
    else:
        final_output_path = output_path if output_path else "data/cleaned_data.csv"
        exploratory_path = "data/eda_exploratory_data.csv"
        
    high_missing_cols = []
    imputed_cols = {}
    missing_indicators_added = []
    
    numeric_cols = list(df.select_dtypes(include='number').columns)
    non_numeric_cols = list(df.select_dtypes(exclude='number').columns)
    
    for col in df.columns:
        if col == target_col:
            continue
        null_count = df[col].isnull().sum()
        pct = (null_count / total_rows) * 100
        
        if pct > 40.0:
            high_missing_cols.append(col)
        elif pct > 0:
            indicator_col = f"{col}_was_missing"
            df[indicator_col] = df[col].isnull().astype(int)
            missing_indicators_added.append(indicator_col)
            
            if col in numeric_cols:
                med_val = df[col].median()
                df[col] = df[col].fillna(med_val)
                imputed_cols[col] = f"median ({med_val:.2f})"
            else:
                mode_s = df[col].mode()
                mode_val = mode_s[0] if not mode_s.empty else "Unknown"
                df[col] = df[col].fillna(mode_val)
                imputed_cols[col] = f"mode ({mode_val})"

    missingness_mechanism_notes = []
    if len(missing_indicators_added) >= 2:
        ind_df = df[missing_indicators_added]
        corr_matrix = ind_df.corr().abs()
        max_corr = 0.0
        for i in range(len(missing_indicators_added)):
            for j in range(i + 1, len(missing_indicators_added)):
                val = corr_matrix.iloc[i, j]
                if not np.isnan(val) and val > max_corr:
                    max_corr = val
        if max_corr > 0.3:
            missingness_mechanism_notes.append(
                f"Missing indicators show correlation (max r={max_corr:.2f}) -> Suggests MAR/MNAR mechanism."
            )
        else:
            missingness_mechanism_notes.append(
                "Missing indicators are largely uncorrelated -> Suggests MCAR mechanism heuristic."
            )
    elif len(missing_indicators_added) == 1:
        missingness_mechanism_notes.append("Single missing indicator added -> MCAR/MAR baseline heuristic.")
    else:
        missingness_mechanism_notes.append("No missing indicators needed (0 missing values or >40% threshold).")

    outlier_counts = {}
    winsorized_cols = []
    
    for col in numeric_cols:
        if col.endswith('_was_missing'):
            continue
            
        mad_outliers = _compute_mad_outliers(df[col])
        iqr_q1 = df[col].quantile(0.25)
        iqr_q3 = df[col].quantile(0.75)
        iqr = iqr_q3 - iqr_q1
        iqr_outliers = (df[col] < (iqr_q1 - 1.5 * iqr)) | (df[col] > (iqr_q3 + 1.5 * iqr))
        
        combined_outliers = mad_outliers | iqr_outliers
        count = int(combined_outliers.sum())
        outlier_counts[col] = count
        
        if cap_outliers and col != target_col and count > 0:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower, upper)
            winsorized_cols.append(col)

    duplicates_before = df.duplicated().sum()
    df = df.drop_duplicates()
    duplicates_removed = int(duplicates_before)
    
    post_shape = df.shape
    
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    df.to_csv(final_output_path, index=False)
    
    os.makedirs(os.path.dirname(exploratory_path), exist_ok=True)
    df.to_csv(exploratory_path, index=False)

    summary_lines = [
        "=== DATA CLEANING & MISSINGNESS REPORT ===",
        f"[run_id={run_id}] Workspace: {workspace_dir if workspace_dir else 'Default'}",
        f"Pre-Cleaning Shape: {pre_shape[0]} rows x {pre_shape[1]} columns",
        f"Post-Cleaning Shape: {post_shape[0]} rows x {post_shape[1]} columns",
        f"Protected Target Variable: '{target_col}'" if target_col else "Protected Target Variable: None",
        f"\nHigh Missingness Columns (>40% - Unimputed): {high_missing_cols if high_missing_cols else 'None'}",
        f"Missing Indicator Flags Added ({len(missing_indicators_added)}): {missing_indicators_added}",
        f"Missingness Mechanism Diagnosis: {' '.join(missingness_mechanism_notes)}",
        f"\nOutlier Detection (MAD & IQR bounds):",
    ]
    for col, count in outlier_counts.items():
        summary_lines.append(f"  - {col}: {count} outliers detected")
    if cap_outliers:
        summary_lines.append(f"Opt-in Winsorization Capped Columns ({len(winsorized_cols)}): {winsorized_cols}")
    else:
        summary_lines.append("Opt-in Outlier Capping: OFF (Outliers reported non-destructively)")
        
    summary_lines.append(f"\nDuplicate Rows Removed: {duplicates_removed}")
    summary_lines.append(f"Saved Cleaned Artifact: {final_output_path}")
    summary_lines.append(f"Saved Exploratory Artifact: {exploratory_path}")
    summary_lines.append("=== CLEANING COMPLETE ===")
    
    summary_text = "\n".join(summary_lines)
    
    return {
        "pre_shape": pre_shape,
        "post_shape": post_shape,
        "target_col": target_col,
        "high_missing_cols": high_missing_cols,
        "missing_indicators_added": missing_indicators_added,
        "imputed_cols": imputed_cols,
        "missingness_mechanism_notes": missingness_mechanism_notes,
        "outlier_counts": outlier_counts,
        "winsorized_cols": winsorized_cols,
        "duplicates_removed": duplicates_removed,
        "output_path": final_output_path,
        "exploratory_path": exploratory_path,
        "summary_text": summary_text
    }
