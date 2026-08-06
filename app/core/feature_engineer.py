import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from app.tools.ingester import load_dataset

def run_feature_engineering(
    dataset_path: str,
    target_col: Optional[str] = None,
    workspace_dir: Optional[str] = None,
    output_path: Optional[str] = None,
    run_id: str = "default"
) -> Dict[str, Any]:
    """
    Performs deterministic feature engineering with workspace directory support.
    """
    df = load_dataset(dataset_path)
        
    orig_cols_count = len(df.columns)
    
    if workspace_dir:
        final_output_path = os.path.join(workspace_dir, "data", "engineered_data.csv")
    else:
        final_output_path = output_path if output_path else "data/engineered_data.csv"
        
    new_features = []
    strategies = {
        "centered_interaction_terms": [],
        "skew_transformations": [],
        "date_part_extractions": []
    }
    
    numeric_cols = [c for c in df.select_dtypes(include='number').columns if c != target_col and not c.endswith('_was_missing')]
    non_numeric_cols = [c for c in df.select_dtypes(exclude='number').columns if c != target_col]
    
    # 1. MEAN-CENTERED INTERACTION TERMS
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().abs()
        pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col_a = numeric_cols[i]
                col_b = numeric_cols[j]
                val = corr_matrix.loc[col_a, col_b]
                if not np.isnan(val) and 0.2 < val < 0.85:
                    pairs.append((col_a, col_b, val))
                    
        pairs.sort(key=lambda x: x[2], reverse=True)
        pairs = pairs[:5]
        
        for col_a, col_b, c_val in pairs:
            mean_a = df[col_a].mean()
            mean_b = df[col_b].mean()
            centered_a = df[col_a] - mean_a
            centered_b = df[col_b] - mean_b
            
            new_col = f"{col_a}_x_{col_b}"
            df[new_col] = centered_a * centered_b
            new_features.append(new_col)
            strategies["centered_interaction_terms"].append(
                f"{new_col} (Mean-Centered, base r={c_val:.3f})"
            )
            
    # 2. SKEW NORMALIZATION TRANSFORMATION
    for col in numeric_cols:
        skew_before = df[col].skew()
        if not np.isnan(skew_before) and abs(skew_before) > 1.0:
            min_val = df[col].min()
            shift = abs(min_val) + 1.0 if min_val <= 0 else 0.0
            
            new_col = f"{col}_log1p"
            df[new_col] = np.log1p(df[col] + shift)
            skew_after = df[new_col].skew()
            
            new_features.append(new_col)
            strategies["skew_transformations"].append(
                f"{new_col} (skew: {skew_before:.2f} -> {skew_after:.2f})"
            )

    # 3. DATE-PART EXTRACTION
    for col in non_numeric_cols:
        try:
            parsed = pd.to_datetime(df[col], errors='coerce')
            if parsed.notna().mean() > 0.5:
                df[f"{col}_year"] = parsed.dt.year
                df[f"{col}_month"] = parsed.dt.month
                df[f"{col}_dayofweek"] = parsed.dt.dayofweek
                df[f"{col}_is_weekend"] = (parsed.dt.dayofweek >= 5).astype(int)
                
                added_date_cols = [f"{col}_year", f"{col}_month", f"{col}_dayofweek", f"{col}_is_weekend"]
                new_features.extend(added_date_cols)
                strategies["date_part_extractions"].append(f"Extracted from {col}: {added_date_cols}")
                
                df = df.drop(columns=[col])
        except Exception as e:
            logging.warning(f"[run_id={run_id}] Failed date extraction for '{col}': {e}")

    final_cols_count = len(df.columns)
    
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    df.to_csv(final_output_path, index=False)
    
    summary_lines = [
        "=== AUTOMATED FEATURE ENGINEERING REPORT ===",
        f"[run_id={run_id}] Original Features Count: {orig_cols_count}",
        f"Protected Target Variable: '{target_col}'" if target_col else "Protected Target Variable: None",
        f"New Features Created ({len(new_features)}): {new_features}",
        "\n--- Strategy Breakdowns ---",
        f"Mean-Centered Interaction Terms ({len(strategies['centered_interaction_terms'])}): {strategies['centered_interaction_terms'] if strategies['centered_interaction_terms'] else 'None (0.2 < |r| < 0.85)'}",
        f"Skew Normalization Transformations ({len(strategies['skew_transformations'])}): {strategies['skew_transformations'] if strategies['skew_transformations'] else 'None (|skew| > 1.0)'}",
        f"Date-Part Extractions ({len(strategies['date_part_extractions'])}): {strategies['date_part_extractions'] if strategies['date_part_extractions'] else 'None (no datetime cols detected)'}",
        f"Final Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns",
        f"Saved Engineered Dataset to: {final_output_path}",
        "=== FEATURE ENGINEERING COMPLETE ==="
    ]
    summary_text = "\n".join(summary_lines)
    
    return {
        "new_features": new_features,
        "strategies": strategies,
        "original_cols_count": orig_cols_count,
        "final_cols_count": final_cols_count,
        "output_path": final_output_path,
        "summary_text": summary_text
    }
