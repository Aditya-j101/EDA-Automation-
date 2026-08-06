import os
import re
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

def validate_luhn(card_number: str) -> bool:
    """Validates credit card numbers using the Luhn algorithm."""
    digits = [int(c) for c in str(card_number) if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for i, digit in enumerate(reverse_digits):
        if i % 2 == 1:
            doubled = digit * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        else:
            checksum += digit
    return checksum % 10 == 0


def detect_pii_shield(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Scans column names and up to 500 sampled rows for PII patterns:
    Email, Phone, SSN/Tax ID, Credit Card (with Luhn check), IP address, Full Name.
    Assigns confidence score HIGH, MEDIUM, or LOW.
    """
    pii_findings = []
    sample_df = df.sample(n=min(500, len(df)), random_state=42) if len(df) > 0 else df
    
    email_regex = re.compile(r'^[\w\.-]+@[\w\.-]+\.\w+$')
    phone_regex = re.compile(r'^\+?1?\d{9,15}$|^\d{3}[-\s]?\d{3}[-\s]?\d{4}$')
    ssn_regex = re.compile(r'^\d{3}-\d{2}-\d{4}$|^\d{9}$')
    ip_regex = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
    
    for col in df.columns:
        col_lower = col.lower().strip()
        raw_samples = sample_df[col].dropna().tolist()
        non_null_samples = [str(s).strip() for s in raw_samples if str(s).strip() and str(s).strip().lower() not in ['nan', 'none', 'null', '']]
        
        if not non_null_samples:
            continue
            
        n_sample = len(non_null_samples)
        
        # 1. Credit Card
        if 'card' in col_lower or 'cc' in col_lower or 'credit' in col_lower:
            luhn_matches = sum(1 for s in non_null_samples if validate_luhn(s))
            if luhn_matches > 0:
                conf = "HIGH" if (luhn_matches / n_sample) > 0.2 else "MEDIUM"
                pii_findings.append({
                    "column": col, "type": "Credit Card Number", "confidence": conf,
                    "matched_samples": luhn_matches, "recommendation": "Mask or Hash prior to model training"
                })
                continue
                
        # 2. Email
        email_matches = sum(1 for s in non_null_samples if email_regex.match(s))
        if 'email' in col_lower or email_matches > 0:
            pct = email_matches / n_sample if n_sample > 0 else 0
            conf = "HIGH" if 'email' in col_lower or pct > 0.3 else ("MEDIUM" if pct > 0.1 else "LOW")
            if pct > 0.05 or 'email' in col_lower:
                pii_findings.append({
                    "column": col, "type": "Email Address", "confidence": conf,
                    "matched_samples": email_matches, "recommendation": "Anonymize or Redact"
                })
                continue
                
        # 3. Phone Number
        phone_matches = sum(1 for s in non_null_samples if phone_regex.match(s))
        if 'phone' in col_lower or 'mobile' in col_lower or 'tel' in col_lower or phone_matches > 0:
            pct = phone_matches / n_sample if n_sample > 0 else 0
            conf = "HIGH" if any(k in col_lower for k in ['phone', 'mobile', 'tel']) or pct > 0.3 else ("MEDIUM" if pct > 0.1 else "LOW")
            if pct > 0.05 or any(k in col_lower for k in ['phone', 'mobile', 'tel']):
                pii_findings.append({
                    "column": col, "type": "Phone Number", "confidence": conf,
                    "matched_samples": phone_matches, "recommendation": "Mask or Redact"
                })
                continue
                
        # 4. SSN / Tax ID
        ssn_matches = sum(1 for s in non_null_samples if ssn_regex.match(s))
        if 'ssn' in col_lower or 'tax_id' in col_lower or 'social' in col_lower or ssn_matches > 0:
            pct = ssn_matches / n_sample if n_sample > 0 else 0
            conf = "HIGH" if 'ssn' in col_lower or pct > 0.2 else ("MEDIUM" if pct > 0.05 else "LOW")
            if pct > 0.05 or 'ssn' in col_lower:
                pii_findings.append({
                    "column": col, "type": "SSN / National Identification Number", "confidence": conf,
                    "matched_samples": ssn_matches, "recommendation": "Hash or Remove"
                })
                continue

        # 5. IP Address
        ip_matches = sum(1 for s in non_null_samples if ip_regex.match(s))
        if 'ip' in col_lower or ip_matches > 0:
            pct = ip_matches / n_sample if n_sample > 0 else 0
            if pct > 0.2 or 'ip_address' in col_lower:
                pii_findings.append({
                    "column": col, "type": "IP Address", "confidence": "HIGH" if pct > 0.3 else "MEDIUM",
                    "matched_samples": ip_matches, "recommendation": "Anonymize IP prefix"
                })
                continue
                
        # 6. Full Name
        if any(k in col_lower for k in ['first_name', 'last_name', 'full_name', 'customer_name', 'user_name']):
            pii_findings.append({
                "column": col, "type": "Full Name / Person Name", "confidence": "HIGH",
                "matched_samples": n_sample, "recommendation": "Pseudonymize or Hash"
            })
            
    return pii_findings


def detect_sentinel_values(df: pd.DataFrame) -> Dict[str, List[Any]]:
    """Detects dummy or sentinel missing values like -999, 9999, ?, N/A."""
    sentinels = [-999, -9999, 999, 9999, "-999", "-9999", "999", "9999", "N/A", "null", "none", "NA", "?", "0000-00-00"]
    found = {}
    for col in df.columns:
        s = df[col].astype(str).str.strip().tolist()
        detected = [val for val in set(s) if val in [str(x) for x in sentinels]]
        if detected:
            found[col] = detected
    return found


def detect_category_variants(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Detects string case or whitespace typos (e.g. 'Male' vs 'male' vs ' Male')."""
    variants = {}
    for col in df.select_dtypes(exclude='number').columns:
        vals = df[col].dropna().astype(str).tolist()
        if not vals:
            continue
        orig_unique = set(vals)
        norm_map = {}
        for v in orig_unique:
            norm = v.strip().lower()
            if norm not in norm_map:
                norm_map[norm] = []
            norm_map[norm].append(v)
            
        conflicts = [v_list for norm, v_list in norm_map.items() if len(v_list) > 1]
        if conflicts:
            variants[col] = [item for sublist in conflicts for item in sublist]
    return variants


def detect_near_constants(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Flags features where a single value occupies > 95% of all non-null rows."""
    near_constants = []
    for col in df.columns:
        s = df[col].dropna()
        if len(s) > 0:
            top_count = s.value_counts().max()
            pct = (top_count / len(s)) * 100
            if pct >= 95.0 and len(s.unique()) > 1:
                top_val = s.value_counts().index[0]
                near_constants.append({
                    "column": col, "top_value": str(top_val), "percentage": round(pct, 2)
                })
    return near_constants


def compute_quality_score(df: pd.DataFrame, missing_info: dict, duplicate_rows: int, negative_val_cols: list, category_variants: dict) -> Dict[str, Any]:
    """
    Computes a headline Data Quality Score (0–100%) and Letter Grade.
      - Completeness Sub-Score (40%): Missing value penalty
      - Uniqueness Sub-Score (20%): Duplicate row penalty
      - Validity Sub-Score (20%): Negative/range/type constraint violations (outliers not penalized)
      - Consistency Sub-Score (20%): String formatting & category variant penalty
    """
    rows, cols = df.shape
    total_cells = rows * cols if rows * cols > 0 else 1
    
    # 1. Completeness
    total_nulls = sum(info["null_count"] for info in missing_info.values())
    completeness = max(0.0, 100.0 * (1.0 - total_nulls / total_cells))
    
    # 2. Uniqueness
    duplicate_pct = (duplicate_rows / rows) if rows > 0 else 0.0
    uniqueness = max(0.0, 100.0 * (1.0 - duplicate_pct))
    
    # 3. Validity (Type & Range Constraint Violations)
    validity_violations = len(negative_val_cols)
    validity = max(0.0, 100.0 * (1.0 - validity_violations / (len(df.columns) if len(df.columns) > 0 else 1)))
    
    # 4. Consistency
    consistency_violations = len(category_variants)
    consistency = max(0.0, 100.0 * (1.0 - consistency_violations / (len(df.columns) if len(df.columns) > 0 else 1)))
    
    overall_score = round(0.40 * completeness + 0.20 * uniqueness + 0.20 * validity + 0.20 * consistency, 1)
    
    if overall_score >= 95.0:
        grade = "A+"
    elif overall_score >= 90.0:
        grade = "A"
    elif overall_score >= 80.0:
        grade = "B"
    elif overall_score >= 70.0:
        grade = "C"
    elif overall_score >= 60.0:
        grade = "D"
    else:
        grade = "F"
        
    return {
        "overall_score": overall_score,
        "grade": grade,
        "completeness_subscore": round(completeness, 1),
        "uniqueness_subscore": round(uniqueness, 1),
        "validity_subscore": round(validity, 1),
        "consistency_subscore": round(consistency, 1)
    }


from app.tools.ingester import load_dataset

def run_profiling(dataset_path: str) -> Dict[str, Any]:
    """
    Performs deterministic data profiling, pre-LLM PII detection, and quality scoring.
    """
    df = load_dataset(dataset_path)
        
    rows, cols = df.shape
    numeric_df = df.select_dtypes(include='number')
    non_numeric_df = df.select_dtypes(exclude='number')
    
    numeric_cols = list(numeric_df.columns)
    non_numeric_cols = list(non_numeric_df.columns)
    
    # 1. PII Detection Shield (Runs BEFORE any LLM call)
    pii_findings = detect_pii_shield(df)
    
    # 2. Advanced Profiling Discoveries
    sentinel_findings = detect_sentinel_values(df)
    category_variants = detect_category_variants(df)
    near_constants = detect_near_constants(df)
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    
    # Target Detection
    target_col = None
    target_names = ['target', 'label', 'class', 'y']
    for col in df.columns:
        if col.lower() in target_names:
            target_col = col
            break
    if not target_col:
        for col in df.columns:
            if df[col].nunique() == 2:
                target_col = col
                break
                
    dtypes = {col: str(df[col].dtype) for col in df.columns}
    
    missing_info = {}
    total_rows = len(df) if len(df) > 0 else 1
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        pct = (null_count / total_rows) * 100
        level = "HIGH (>40%)" if pct > 40.0 else ("MODERATE (1-40%)" if pct >= 1.0 else "LOW (<1%)")
        missing_info[col] = {"null_count": null_count, "percentage": round(pct, 2), "level": level}
        
    duplicate_rows = int(df.duplicated().sum())
    
    negative_val_cols = []
    for col in numeric_cols:
        if (df[col].dropna() < 0).any():
            negative_val_cols.append(col)
            
    # 3. Data Quality Score calculation
    quality_score = compute_quality_score(df, missing_info, duplicate_rows, negative_val_cols, category_variants)

    summary_lines = [
        "=== DATA PROFILING & QUALITY REPORT ===",
        f"HEADLINE DATA QUALITY SCORE: {quality_score['overall_score']}/100 [Grade: {quality_score['grade']}]",
        f"  - Completeness: {quality_score['completeness_subscore']}%",
        f"  - Uniqueness: {quality_score['uniqueness_subscore']}%",
        f"  - Validity (Constraint Violations): {quality_score['validity_subscore']}%",
        f"  - Consistency (Format Uniformity): {quality_score['consistency_subscore']}%",
        f"\nTarget Column Detected Early: '{target_col}'" if target_col else "\nTarget Column Detected Early: None",
        f"\n[PII PRIVACY SHIELD DETECTED] ({len(pii_findings)} PII Columns):"
    ]
    for pii in pii_findings:
        summary_lines.append(f"  ⚠️ [{pii['confidence']} CONFIDENCE] {pii['column']}: {pii['type']} ({pii['recommendation']})")
    if not pii_findings:
        summary_lines.append("  ✓ No high-confidence PII columns detected.")
        
    summary_lines.append(f"\nConstant/Zero-Variance Columns ({len(constant_cols)}): {constant_cols}")
    summary_lines.append(f"Near-Constant Columns (>95% single value) ({len(near_constants)}): {[item['column'] for item in near_constants]}")
    summary_lines.append(f"Sentinel/Dummy Values Detected ({len(sentinel_findings)}): {sentinel_findings if sentinel_findings else 'None'}")
    summary_lines.append(f"Category Variants / Case Typos ({len(category_variants)}): {category_variants if category_variants else 'None'}")
    summary_lines.append(f"Duplicate Rows: {duplicate_rows} ({(duplicate_rows/total_rows)*100:.2f}%)")
    summary_lines.append("=== PROFILING COMPLETE ===")
    
    summary_text = "\n".join(summary_lines)
    
    return {
        "shape": (rows, cols),
        "target_col": target_col,
        "dtypes": dtypes,
        "numeric_cols": numeric_cols,
        "non_numeric_cols": non_numeric_cols,
        "missing_info": missing_info,
        "duplicate_rows": duplicate_rows,
        "negative_val_cols": negative_val_cols,
        "constant_cols": constant_cols,
        "near_constants": near_constants,
        "sentinel_findings": sentinel_findings,
        "category_variants": category_variants,
        "pii_findings": pii_findings,
        "quality_score": quality_score,
        "summary_text": summary_text
    }
