import os
import re
import pandas as pd
import numpy as np
from scipy import stats
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
    """Scans column names and samples for PII patterns."""
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
        
        # Credit Card
        if any(k in col_lower for k in ['card', 'cc', 'credit']):
            luhn_matches = sum(1 for s in non_null_samples if validate_luhn(s))
            if luhn_matches > 0:
                conf = "HIGH" if (luhn_matches / n_sample) > 0.2 else "MEDIUM"
                pii_findings.append({
                    "column": col, "type": "Credit Card Number", "confidence": conf,
                    "matched_samples": luhn_matches, "recommendation": "Mask or Hash prior to model training"
                })
                continue
                
        # Email
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
                
        # Phone
        phone_matches = sum(1 for s in non_null_samples if phone_regex.match(s))
        if any(k in col_lower for k in ['phone', 'mobile', 'tel']) or phone_matches > 0:
            pct = phone_matches / n_sample if n_sample > 0 else 0
            conf = "HIGH" if any(k in col_lower for k in ['phone', 'mobile', 'tel']) or pct > 0.3 else ("MEDIUM" if pct > 0.1 else "LOW")
            if pct > 0.05 or any(k in col_lower for k in ['phone', 'mobile', 'tel']):
                pii_findings.append({
                    "column": col, "type": "Phone Number", "confidence": conf,
                    "matched_samples": phone_matches, "recommendation": "Mask or Redact"
                })
                continue
                
        # SSN
        ssn_matches = sum(1 for s in non_null_samples if ssn_regex.match(s))
        if any(k in col_lower for k in ['ssn', 'tax_id', 'social']) or ssn_matches > 0:
            pct = ssn_matches / n_sample if n_sample > 0 else 0
            conf = "HIGH" if 'ssn' in col_lower or pct > 0.2 else ("MEDIUM" if pct > 0.05 else "LOW")
            if pct > 0.05 or 'ssn' in col_lower:
                pii_findings.append({
                    "column": col, "type": "SSN / National Identification Number", "confidence": conf,
                    "matched_samples": ssn_matches, "recommendation": "Hash or Remove"
                })
                continue

        # IP Address
        ip_matches = sum(1 for s in non_null_samples if ip_regex.match(s))
        if 'ip' in col_lower or ip_matches > 0:
            pct = ip_matches / n_sample if n_sample > 0 else 0
            if pct > 0.2 or 'ip_address' in col_lower:
                pii_findings.append({
                    "column": col, "type": "IP Address", "confidence": "HIGH" if pct > 0.3 else "MEDIUM",
                    "matched_samples": ip_matches, "recommendation": "Anonymize IP prefix"
                })
                continue
                
        # Full Name
        if any(k in col_lower for k in ['first_name', 'last_name', 'full_name', 'customer_name', 'user_name']):
            pii_findings.append({
                "column": col, "type": "Full Name / Person Name", "confidence": "HIGH",
                "matched_samples": n_sample, "recommendation": "Pseudonymize or Hash"
            })
            
    return pii_findings


def compute_missing_severity(pct: float) -> str:
    """Deterministic threshold rules for missingness severity."""
    if pct >= 40.0:
        return "critical"
    elif pct >= 20.0:
        return "high"
    elif pct >= 5.0:
        return "moderate"
    else:
        return "low"


def compute_outlier_severity(pct: float) -> str:
    """Deterministic threshold rules for outlier percentage severity."""
    if pct >= 10.0:
        return "critical"
    elif pct >= 5.0:
        return "high"
    elif pct >= 1.0:
        return "moderate"
    else:
        return "low"


def _check_normality(series: pd.Series) -> Dict[str, Any]:
    """Tests a series for normality using Shapiro-Wilk or D'Agostino's K^2 test."""
    s = series.dropna()
    if len(s) < 8:
        return {
            "is_normal": False,
            "p_value": 0.0,
            "statistic": 0.0,
            "test_used": "Sample Check",
            "reason": "Insufficient sample size (N < 8)"
        }
        
    skew_val = float(s.skew())
    if len(s) <= 5000:
        stat, p = stats.shapiro(s)
        test_used = "Shapiro-Wilk"
    else:
        stat, p = stats.normaltest(s)
        test_used = "D'Agostino K^2"
        
    p_val = float(p) if not np.isnan(p) else 0.0
    stat_val = float(stat) if not np.isnan(stat) else 0.0
    is_normal = bool(p_val > 0.05 and abs(skew_val) < 0.5)
    
    return {
        "is_normal": is_normal,
        "p_value": round(p_val, 4),
        "statistic": round(stat_val, 4),
        "skewness": round(skew_val, 4),
        "test_used": test_used
    }


def _cramers_v(contingency_table: pd.DataFrame) -> float:
    chi2, _, _, _ = stats.chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    if n == 0:
        return 0.0
    r, k = contingency_table.shape
    min_dim = min(r - 1, k - 1)
    if min_dim <= 0:
        return 0.0
    return float(np.sqrt((chi2 / n) / min_dim))


def _cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    g1, g2 = group1.dropna(), group2.dropna()
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0
    s1, s2 = g1.std(), g2.std()
    pooled_std = np.sqrt(((n1 - 1) * (s1 ** 2) + (n2 - 1) * (s2 ** 2)) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((g1.mean() - g2.mean()) / pooled_std)


def detect_simpsons_paradox(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> List[Dict[str, Any]]:
    """Checks overall correlation vs subgroup correlations for sign reversals."""
    reversals = []
    if len(num_cols) < 2 or not cat_cols:
        return reversals
        
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            c1, c2 = num_cols[i], num_cols[j]
            s1, s2 = df[c1].dropna(), df[c2].dropna()
            common_idx = s1.index.intersection(s2.index)
            if len(common_idx) < 15:
                continue
                
            overall_r = df.loc[common_idx, c1].corr(df.loc[common_idx, c2])
            if np.isnan(overall_r) or abs(overall_r) < 0.20:
                continue
                
            overall_sign = 1 if overall_r > 0 else -1
            for z_col in cat_cols:
                u_vals = df[z_col].dropna().unique()
                if 2 <= len(u_vals) <= 5:
                    subgroup_reversals = []
                    for val in u_vals:
                        sub_df = df[df[z_col] == val]
                        sub_s1, sub_s2 = sub_df[c1].dropna(), sub_df[c2].dropna()
                        sub_idx = sub_s1.index.intersection(sub_s2.index)
                        if len(sub_idx) >= 6:
                            sub_r = sub_df.loc[sub_idx, c1].corr(sub_df.loc[sub_idx, c2])
                            if not np.isnan(sub_r) and abs(sub_r) > 0.15:
                                sub_sign = 1 if sub_r > 0 else -1
                                if sub_sign != overall_sign:
                                    subgroup_reversals.append((val, sub_r))
                                    
                    if subgroup_reversals:
                        reversals.append({
                            "feature_x": c1,
                            "feature_y": c2,
                            "group_var": z_col,
                            "overall_corr": round(float(overall_r), 4),
                            "subgroup_reversals": [
                                {"category": str(v), "subgroup_corr": round(float(r), 4)} for v, r in subgroup_reversals
                            ],
                            "headline": f"SIMPSON'S PARADOX DETECTED: Correlation of '{c1}' & '{c2}' (r={overall_r:.2f}) reverses in subgroups of '{z_col}'!"
                        })
                        if len(reversals) >= 3:
                            return reversals
    return reversals


from app.tools.ingester import load_dataset

def build_structured_evidence(dataset_path: str, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Performs ALL deterministic data profiling, statistics, distribution shape metrics,
    hypothesis tests, PII audits, and anomaly detection. Returns the complete Structured EDA State.
    """
    df = load_dataset(dataset_path)
        
    rows, cols = df.shape
    total_cells = rows * cols if rows * cols > 0 else 1
    
    # Identify target column early if unspecified
    if not target_col:
        target_candidates = ['target', 'label', 'class', 'y']
        for col in df.columns:
            if col.lower() in target_candidates:
                target_col = col
                break
        if not target_col:
            for col in df.columns:
                if df[col].nunique() == 2:
                    target_col = col
                    break

    numeric_df = df.select_dtypes(include='number')
    non_numeric_df = df.select_dtypes(exclude='number')
    
    numeric_cols = list(numeric_df.columns)
    non_numeric_cols = list(non_numeric_df.columns)
    
    # 1. Schema Evidence
    dtypes = {col: str(df[col].dtype) for col in df.columns}
    schema_evidence = {
        "shape": (rows, cols),
        "total_rows": rows,
        "total_cols": cols,
        "target_col": target_col,
        "dtypes": dtypes,
        "numeric_cols": numeric_cols,
        "non_numeric_cols": non_numeric_cols,
    }

    # 2. Quality Evidence
    missing_info = {}
    total_nulls = 0
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        total_nulls += null_count
        pct = round((null_count / rows) * 100, 4) if rows > 0 else 0.0
        severity = compute_missing_severity(pct)
        missing_info[col] = {
            "null_count": null_count,
            "percentage": pct,
            "severity": severity
        }
        
    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_rows / rows) * 100, 4) if rows > 0 else 0.0
    duplicate_severity = "high" if duplicate_pct >= 10.0 else ("moderate" if duplicate_pct >= 2.0 else "low")
    
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    quasi_constant_cols = []
    for col in df.columns:
        s = df[col].dropna()
        if len(s) > 0:
            top_pct = (s.value_counts().max() / len(s)) * 100
            if top_pct >= 95.0 and len(s.unique()) > 1:
                quasi_constant_cols.append({
                    "column": col,
                    "top_value": str(s.value_counts().index[0]),
                    "percentage": round(top_pct, 2)
                })

    # Sentinels & Category Variants
    sentinels = [-999, -9999, 999, 9999, "-999", "-9999", "999", "9999", "N/A", "null", "none", "NA", "?", "0000-00-00"]
    sentinel_findings = {}
    for col in df.columns:
        s = df[col].astype(str).str.strip().tolist()
        detected = [val for val in set(s) if val in [str(x) for x in sentinels]]
        if detected:
            sentinel_findings[col] = detected

    category_variants = {}
    for col in non_numeric_cols:
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
            category_variants[col] = [item for sublist in conflicts for item in sublist]

    pii_findings = detect_pii_shield(df)

    # Headline Data Quality Score calculation
    completeness = max(0.0, 100.0 * (1.0 - total_nulls / total_cells))
    uniqueness = max(0.0, 100.0 * (1.0 - (duplicate_rows / rows if rows > 0 else 0)))
    
    negative_val_cols = [c for c in numeric_cols if (df[c].dropna() < 0).any()]
    validity = max(0.0, 100.0 * (1.0 - len(negative_val_cols) / (cols if cols > 0 else 1)))
    consistency = max(0.0, 100.0 * (1.0 - len(category_variants) / (cols if cols > 0 else 1)))
    
    overall_score = round(0.40 * completeness + 0.20 * uniqueness + 0.20 * validity + 0.20 * consistency, 1)
    if overall_score >= 95.0: grade = "A+"
    elif overall_score >= 90.0: grade = "A"
    elif overall_score >= 80.0: grade = "B"
    elif overall_score >= 70.0: grade = "C"
    elif overall_score >= 60.0: grade = "D"
    else: grade = "F"
    
    quality_evidence = {
        "missing_info": missing_info,
        "total_nulls": total_nulls,
        "overall_missing_pct": round((total_nulls / total_cells) * 100, 2),
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": duplicate_pct,
        "duplicate_severity": duplicate_severity,
        "constant_cols": constant_cols,
        "quasi_constant_cols": quasi_constant_cols,
        "sentinel_findings": sentinel_findings,
        "category_variants": category_variants,
        "pii_findings": pii_findings,
        "quality_score": {
            "overall_score": overall_score,
            "grade": grade,
            "completeness": round(completeness, 1),
            "uniqueness": round(uniqueness, 1),
            "validity": round(validity, 1),
            "consistency": round(consistency, 1)
        }
    }

    # 3. Distributions Evidence
    distribution_evidence = {}
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        mean_val = float(s.mean())
        std_val = float(s.std()) if len(s) > 1 else 0.0
        min_val = float(s.min())
        max_val = float(s.max())
        median_val = float(s.median())
        q25_val = float(s.quantile(0.25))
        q75_val = float(s.quantile(0.75))
        iqr_val = float(q75_val - q25_val)
        var_val = float(s.var()) if len(s) > 1 else 0.0
        skew_val = float(s.skew()) if len(s) > 2 else 0.0
        kurt_val = float(s.kurtosis()) if len(s) > 3 else 0.0
        
        if abs(skew_val) > 1.0:
            skew_severity = "strongly_skewed"
        elif abs(skew_val) > 0.5:
            skew_severity = "moderately_skewed"
        else:
            skew_severity = "symmetric"

        norm_test = _check_normality(s)
        
        # Outliers calculation via MAD & IQR
        med = s.median()
        mad = np.median(np.abs(s - med))
        mod_z = (0.6745 * np.abs(s - med) / mad) if mad > 0 else np.zeros(len(s))
        mad_outliers = int((mod_z > 3.5).sum())
        
        iqr_outliers = int(((s < (q25_val - 1.5 * iqr_val)) | (s > (q75_val + 1.5 * iqr_val))).sum())
        combined_outliers = int(((mod_z > 3.5) | (s < (q25_val - 1.5 * iqr_val)) | (s > (q75_val + 1.5 * iqr_val))).sum())
        outlier_pct = round((combined_outliers / len(s)) * 100, 2) if len(s) > 0 else 0.0
        
        distribution_evidence[col] = {
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "min": round(min_val, 4),
            "max": round(max_val, 4),
            "median": round(median_val, 4),
            "q25": round(q25_val, 4),
            "q75": round(q75_val, 4),
            "iqr": round(iqr_val, 4),
            "variance": round(var_val, 4),
            "skewness": round(skew_val, 4),
            "kurtosis": round(kurt_val, 4),
            "skewness_severity": skew_severity,
            "normality_test": norm_test,
            "outliers": {
                "mad_count": mad_outliers,
                "iqr_count": iqr_outliers,
                "total_outliers": combined_outliers,
                "percentage": outlier_pct,
                "severity": compute_outlier_severity(outlier_pct)
            }
        }
        
    for col in non_numeric_cols:
        s = df[col].dropna()
        n_uniq = int(s.nunique())
        uniq_ratio = round(n_uniq / len(s), 4) if len(s) > 0 else 0.0
        top_vc = s.value_counts().head(5).to_dict()
        top_cats = {str(k): {"count": int(v), "percentage": round((v / len(s)) * 100, 2)} for k, v in top_vc.items()}
        distribution_evidence[col] = {
            "nunique": n_uniq,
            "unique_ratio": uniq_ratio,
            "top_categories": top_cats
        }

    # 4. Relationships Evidence
    cat_cols_for_rel = [c for c in non_numeric_cols if df[c].nunique() < 20]
    simpsons_reversals = detect_simpsons_paradox(df, numeric_cols, cat_cols_for_rel)
    
    correlation_family = []
    if len(numeric_cols) >= 2:
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                c1, c2 = numeric_cols[i], numeric_cols[j]
                s1, s2 = numeric_df[c1].dropna(), numeric_df[c2].dropna()
                common_idx = s1.index.intersection(s2.index)
                n_eff = len(common_idx)
                if n_eff >= 8:
                    norm1 = distribution_evidence.get(c1, {}).get("normality_test", {}).get("is_normal", False)
                    norm2 = distribution_evidence.get(c2, {}).get("normality_test", {}).get("is_normal", False)
                    if norm1 and norm2:
                        r_val, p_val = stats.pearsonr(df.loc[common_idx, c1], df.loc[common_idx, c2])
                        method = "Pearson Correlation"
                    else:
                        r_val, p_val = stats.spearmanr(df.loc[common_idx, c1], df.loc[common_idx, c2])
                        method = "Spearman Correlation"
                    
                    r_clean = float(r_val) if not np.isnan(r_val) else 0.0
                    abs_r = abs(r_clean)
                    strength = "strong" if abs_r >= 0.7 else ("moderate" if abs_r >= 0.3 else "weak")
                    
                    correlation_family.append({
                        "pair": f"{c1} & {c2}",
                        "col1": c1,
                        "col2": c2,
                        "method": method,
                        "corr": round(r_clean, 4),
                        "raw_p_value": float(p_val) if not np.isnan(p_val) else 1.0,
                        "n_eff": n_eff,
                        "strength": strength
                    })

    group_tests_family = []
    for cat_col in cat_cols_for_rel:
        u_vals = df[cat_col].dropna().unique()
        if len(u_vals) == 2:
            v1, v2 = u_vals[0], u_vals[1]
            for num_col in numeric_cols:
                g1 = df[df[cat_col] == v1][num_col].dropna()
                g2 = df[df[cat_col] == v2][num_col].dropna()
                n_eff = len(g1) + len(g2)
                if len(g1) >= 8 and len(g2) >= 8:
                    norm1 = _check_normality(g1)["is_normal"]
                    norm2 = _check_normality(g2)["is_normal"]
                    if norm1 and norm2:
                        lev_stat, lev_p = stats.levene(g1, g2)
                        equal_var = bool(lev_p > 0.05)
                        t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=equal_var)
                        test_name = "Independent 2-Sample T-Test" if equal_var else "Welch's T-Test"
                        eff_size = _cohens_d(g1, g2)
                    else:
                        u_stat, p_val = stats.mannwhitneyu(g1, g2)
                        test_name = "Mann-Whitney U Test"
                        t_stat = u_stat
                        eff_size = _cohens_d(g1, g2)
                        
                    group_tests_family.append({
                        "test_name": test_name,
                        "variables": f"'{num_col}' by '{cat_col}' ({v1} vs {v2})",
                        "num_col": num_col,
                        "cat_col": cat_col,
                        "statistic": round(float(t_stat), 4) if not np.isnan(t_stat) else 0.0,
                        "raw_p_value": float(p_val) if not np.isnan(p_val) else 1.0,
                        "effect_size": round(float(eff_size), 4),
                        "effect_size_type": "Cohen's d",
                        "n_eff": n_eff
                    })
                    if len(group_tests_family) >= 5: break
            if len(group_tests_family) >= 5: break

    # Benjamini-Hochberg FDR correction on raw p-values
    def _apply_fdr(family: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not family:
            return []
        p_vals = np.array([item["raw_p_value"] for item in family])
        m = len(p_vals)
        if m > 1:
            try:
                adj_p = stats.false_discovery_control(p_vals, method='bh')
            except Exception:
                sorted_idx = np.argsort(p_vals)
                ranks = np.empty_like(sorted_idx)
                ranks[sorted_idx] = np.arange(1, m + 1)
                adj_p = p_vals * m / ranks
                adj_p = np.minimum.accumulate(adj_p[::-1])[::-1]
                adj_p = np.clip(adj_p, 0, 1)
        else:
            adj_p = p_vals
            
        for i, item in enumerate(family):
            p_adj_val = round(float(adj_p[i]), 4)
            item["p_adj"] = p_adj_val
            item["is_significant"] = bool(p_adj_val < 0.05)
            item["raw_p_value"] = round(item["raw_p_value"], 4)
        return family

    correlation_family = _apply_fdr(correlation_family)
    group_tests_family = _apply_fdr(group_tests_family)

    # VIF Multicollinearity and Redundancy
    redundant_features = []
    vif_scores = {}
    if len(numeric_cols) >= 2:
        corr_matrix = numeric_df.corr().abs()
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                c1, c2 = numeric_cols[i], numeric_cols[j]
                val = corr_matrix.loc[c1, c2]
                if not np.isnan(val) and val > 0.90:
                    redundant_features.append({"col1": c1, "col2": c2, "corr": round(float(val), 4)})
                    
        for col in numeric_cols:
            others = numeric_df.drop(columns=[col])
            if len(others.columns) > 0:
                corr_with_others = others.corrwith(numeric_df[col]).abs()
                max_corr = corr_with_others.max()
                r_sq = 0.0 if np.isnan(max_corr) else float(max_corr ** 2)
                vif = 1.0 / (1.0 - r_sq) if r_sq < 0.9999 else 999.0
                vif_scores[col] = round(float(vif), 2)

    relationship_evidence = {
        "correlation_family": correlation_family,
        "group_tests_family": group_tests_family,
        "simpsons_reversals": simpsons_reversals,
        "vif_scores": vif_scores,
        "redundant_features": redundant_features
    }

    # 5. Anomalies & ML Readiness Evidence
    class_dist = None
    imbalance_ratio = None
    imbalance_severity = "balanced"
    leakage_warnings = []
    
    if target_col and target_col in df.columns:
        counts = df[target_col].value_counts().to_dict()
        class_dist = {str(k): int(v) for k, v in counts.items()}
        if len(counts) > 1:
            min_c = min(counts.values())
            max_c = max(counts.values())
            imbalance_ratio = round(float(min_c / max_c), 4)
            if imbalance_ratio < 0.2:
                imbalance_severity = "severe"
            elif imbalance_ratio < 0.5:
                imbalance_severity = "moderate"
                
        target_series = df[target_col]
        target_num = target_series if target_col in numeric_cols else pd.Series(pd.factorize(target_series)[0], index=df.index)
        for col in numeric_cols:
            if col != target_col:
                c_val = abs(numeric_df[col].corr(target_num))
                if not np.isnan(c_val) and c_val > 0.95:
                    leakage_warnings.append({"column": col, "correlation": round(float(c_val), 4)})

    dim_ratio = round(float(cols / rows), 4) if rows > 0 else 0.0
    high_dim_flag = bool(dim_ratio > 0.1)

    anomalies_evidence = {
        "class_distribution": class_dist,
        "imbalance_ratio": imbalance_ratio,
        "imbalance_severity": imbalance_severity,
        "high_dimensionality": {
            "dimensionality_ratio": dim_ratio,
            "is_high_dimensional": high_dim_flag
        },
        "leakage_warnings": leakage_warnings
    }

    return {
        "schema": schema_evidence,
        "quality": quality_evidence,
        "distributions": distribution_evidence,
        "relationships": relationship_evidence,
        "anomalies": anomalies_evidence
    }
