import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

def run_structural_ml_prep(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    workspace_dir: Optional[str] = None,
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    run_id: str = "default"
) -> Dict[str, Any]:
    """
    Performs structural ML preparation with an 80/20 train/test split.
    Fits imputers strictly on train set and transforms train & test sets independently
    to guarantee zero data leakage.
    """
    if len(df) < 10:
        return {"status": "SKIPPED", "reason": "Insufficient rows for train/test split (<10 rows)"}
        
    if workspace_dir:
        final_train_path = os.path.join(workspace_dir, "data", "ml_ready_train.csv")
        final_test_path = os.path.join(workspace_dir, "data", "ml_ready_test.csv")
    else:
        final_train_path = train_path if train_path else "data/ml_ready_train.csv"
        final_test_path = test_path if test_path else "data/ml_ready_test.csv"

    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    split_idx = int(len(shuffled) * 0.8)
    
    train_df = shuffled.iloc[:split_idx].copy()
    test_df = shuffled.iloc[split_idx:].copy()
    
    numeric_cols = [c for c in train_df.select_dtypes(include='number').columns if c != target_col and not c.endswith('_was_missing')]
    non_numeric_cols = [c for c in train_df.select_dtypes(exclude='number').columns if c != target_col]
    
    numeric_medians = {col: train_df[col].median() for col in numeric_cols}
    categorical_modes = {}
    for col in non_numeric_cols:
        m_s = train_df[col].mode()
        categorical_modes[col] = m_s[0] if not m_s.empty else "Unknown"
        
    for col, med in numeric_medians.items():
        train_df[col] = train_df[col].fillna(med)
    for col, mode_val in categorical_modes.items():
        train_df[col] = train_df[col].fillna(mode_val)
        
    for col, med in numeric_medians.items():
        test_df[col] = test_df[col].fillna(med)
    for col, mode_val in categorical_modes.items():
        test_df[col] = test_df[col].fillna(mode_val)
        
    os.makedirs(os.path.dirname(final_train_path), exist_ok=True)
    train_df.to_csv(final_train_path, index=False)
    test_df.to_csv(final_test_path, index=False)
    
    return {
        "status": "COMPLETED",
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "train_path": final_train_path,
        "test_path": final_test_path
    }


from app.tools.ingester import load_dataset

def run_ml_readiness(
    dataset_path: str,
    target_col: Optional[str] = None,
    workspace_dir: Optional[str] = None,
    run_id: str = "default"
) -> Dict[str, Any]:
    """
    Performs deterministic ML preparation & readiness analysis, and outputs structural ML artifacts.
    """
    df = load_dataset(dataset_path)
        
    numeric_df = df.select_dtypes(include='number')
    non_numeric_df = df.select_dtypes(exclude='number')
    
    leakage_warnings = []
    redundant_features = []
    vif_scores = {}
    class_dist = None
    imbalance_ratio = None
    
    dim_ratio = float(len(df.columns) / len(df)) if len(df) > 0 else 0.0
    high_dim_flag = bool(dim_ratio > 0.1)
    
    if target_col:
        mode = "SUPERVISED"
        counts = df[target_col].value_counts().to_dict()
        class_dist = {str(k): int(v) for k, v in counts.items()}
        if len(counts) > 0:
            min_count = min(counts.values())
            max_count = max(counts.values())
            imbalance_ratio = round(float(min_count / max_count), 4)
            
        target_series = df[target_col]
        if target_col in numeric_df.columns:
            target_num = target_series
        else:
            target_num = pd.Series(pd.factorize(target_series)[0], index=df.index)
            
        for col in numeric_df.columns:
            if col != target_col:
                corr_val = abs(numeric_df[col].corr(target_num))
                if not np.isnan(corr_val) and corr_val > 0.95:
                    leakage_warnings.append(f"'{col}' has correlation {corr_val:.4f} with target '{target_col}'")
    else:
        mode = "UNSUPERVISED"
        if len(numeric_df.columns) >= 2:
            corr_matrix = numeric_df.corr().abs()
            cols = list(numeric_df.columns)
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    c1, c2 = cols[i], cols[j]
                    val = corr_matrix.loc[c1, c2]
                    if not np.isnan(val) and val > 0.90:
                        redundant_features.append({
                            "col1": c1, "col2": c2, "corr": round(float(val), 4)
                        })
                        
        for col in numeric_df.columns:
            others = numeric_df.drop(columns=[col])
            if len(others.columns) > 0:
                corr_with_others = others.corrwith(numeric_df[col]).abs()
                max_corr = corr_with_others.max()
                r_sq = 0.0 if np.isnan(max_corr) else float(max_corr ** 2)
                vif = 1.0 / (1.0 - r_sq) if r_sq < 0.9999 else 999.0
                vif_scores[col] = round(float(vif), 2)

    ml_prep_res = run_structural_ml_prep(df, target_col=target_col, workspace_dir=workspace_dir, run_id=run_id)

    summary_lines = [
        "=== ML PREPARATION & READINESS REPORT ===",
        f"[run_id={run_id}] MODE: {mode}",
        f"Protected Target Variable: '{target_col}'" if target_col else "Protected Target Variable: None"
    ]
    if mode == "SUPERVISED":
        summary_lines.append(f"Class Distribution: {class_dist}")
        summary_lines.append(f"Class Imbalance Ratio (minority/majority): {imbalance_ratio}")
        summary_lines.append(f"Data Leakage Warnings ({len(leakage_warnings)}): {leakage_warnings if leakage_warnings else 'None detected'}")
    else:
        summary_lines.append(f"Feature Redundancy (>0.90 corr) ({len(redundant_features)} pairs): {redundant_features if redundant_features else 'None'}")
        summary_lines.append(f"Multicollinearity (VIF Scores): {vif_scores}")
        
    summary_lines.append(f"Dimensionality Ratio: {dim_ratio:.4f} {'(HIGH DIMENSIONALITY >0.1)' if high_dim_flag else '(Healthy)'}")
    summary_lines.append(f"\nStructural ML Preparation Artifacts:")
    summary_lines.append(f"  - Train Set (80%): {ml_prep_res.get('train_path')} ({ml_prep_res.get('train_shape')})")
    summary_lines.append(f"  - Test Set (20%): {ml_prep_res.get('test_path')} ({ml_prep_res.get('test_shape')})")
    summary_lines.append("  - Guarantee: Fits executed strictly on Train set; zero Data Leakage to Test set.")
    summary_lines.append("=== ML READINESS COMPLETE ===")
    
    summary_text = "\n".join(summary_lines)
    
    return {
        "mode": mode,
        "target_col": target_col,
        "class_distribution": class_dist,
        "imbalance_ratio": imbalance_ratio,
        "leakage_warnings": leakage_warnings,
        "redundant_features": redundant_features,
        "vif_scores": vif_scores,
        "dimensionality_ratio": round(dim_ratio, 4),
        "high_dimensionality_flag": high_dim_flag,
        "ml_prep_res": ml_prep_res,
        "summary_text": summary_text
    }
