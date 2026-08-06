import os
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional

def _cramers_v(contingency_table: pd.DataFrame) -> float:
    """Computes Cramer's V for a contingency table."""
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
    """Computes Cohen's d effect size between two groups."""
    g1, g2 = group1.dropna(), group2.dropna()
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0
    s1, s2 = g1.std(), g2.std()
    pooled_std = np.sqrt(((n1 - 1) * (s1 ** 2) + (n2 - 1) * (s2 ** 2)) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((g1.mean() - g2.mean()) / pooled_std)


def _check_normality(series: pd.Series) -> Dict[str, Any]:
    """Tests a series for normality using Shapiro-Wilk or D'Agostino's test."""
    s = series.dropna()
    if len(s) < 8:
        return {"is_normal": False, "p_value": 0.0, "test_used": "Sample Check", "reason": "Insufficient sample size (N < 8)"}
        
    skew_val = abs(s.skew())
    if len(s) <= 5000:
        stat, p = stats.shapiro(s)
    else:
        stat, p = stats.normaltest(s)
        
    is_normal = bool(p > 0.05 and skew_val < 0.5)
    return {
        "is_normal": is_normal,
        "p_value": round(float(p), 4),
        "skewness": round(float(skew_val), 4),
        "test_used": "Shapiro-Wilk" if len(s) <= 5000 else "D'Agostino K^2"
    }


def detect_simpsons_paradox(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> List[Dict[str, Any]]:
    """
    Automatically checks overall correlation corr(X, Y) vs subgroup correlations corr(X, Y | Z = z).
    If a sign reversal occurs (overall positive -> subgroup negative), elevates to a Simpson's Paradox finding!
    """
    reversals = []
    if len(num_cols) < 2 or not cat_cols:
        return reversals
        
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            c1, c2 = num_cols[i], num_cols[j]
            overall_series1, overall_series2 = df[c1].dropna(), df[c2].dropna()
            common_idx = overall_series1.index.intersection(overall_series2.index)
            
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
                        s1, s2 = sub_df[c1].dropna(), sub_df[c2].dropna()
                        sub_idx = s1.index.intersection(s2.index)
                        
                        if len(sub_idx) >= 6:
                            sub_r = sub_df.loc[sub_idx, c1].corr(sub_df.loc[sub_idx, c2])
                            if not np.isnan(sub_r) and abs(sub_r) > 0.15:
                                sub_sign = 1 if sub_r > 0 else -1
                                if sub_sign != overall_sign:
                                    subgroup_reversals.append((val, sub_r))
                                    
                    if len(subgroup_reversals) >= 1:
                        reversals.append({
                            "feature_x": c1,
                            "feature_y": c2,
                            "group_var": z_col,
                            "overall_corr": round(float(overall_r), 4),
                            "subgroup_reversals": [
                                {"category": str(v), "subgroup_corr": round(float(r), 4)} for v, r in subgroup_reversals
                            ],
                            "headline": f"SIMPSON'S PARADOX DETECTED: Overall correlation of '{c1}' & '{c2}' (r={overall_r:.2f}) REVERSES direction within subgroups of '{z_col}'!"
                        })
                        if len(reversals) >= 3:
                            return reversals
    return reversals


def rank_prioritized_insights(group_tests_family: List[dict], correlation_family: List[dict]) -> List[dict]:
    """
    Ranks analytical findings by 95% Confidence Interval Lower Bound (L_CI) of normalized effect size magnitude.
    Filters out redundant findings.
    """
    all_insights = []
    
    # 1. Process Group Tests
    for t in group_tests_family:
        if not t.get("is_significant"):
            continue
            
        n_eff = t.get("n_eff", 30)
        eff_size = t.get("effect_size", 0.0)
        
        # Convert Cohen's d to r-equivalent magnitude [0, 1]
        # r = d / sqrt(d^2 + 4)
        if t.get("effect_size_type") == "Cramer's V":
            norm_mag = eff_size ** 2
        else:
            d_val = abs(eff_size)
            norm_mag = (d_val ** 2) / (d_val ** 2 + 4.0)
            
        se = 1.0 / np.sqrt(max(n_eff, 4))
        l_ci = max(0.0, norm_mag - 1.96 * se)
        
        all_insights.append({
            "type": "Group Difference Test",
            "headline": f"{t['test_name']} on {t['variables']}",
            "variables": t['variables'],
            "normalized_magnitude": round(norm_mag, 4),
            "l_ci": round(l_ci, 4),
            "p_adj": t.get("adj_p_value"),
            "n_eff": n_eff,
            "raw": t
        })
        
    # 2. Process Correlations
    for c in correlation_family:
        if not c.get("is_significant"):
            continue
            
        n_eff = c.get("n_eff", 30)
        corr_val = abs(c.get("corr", 0.0))
        norm_mag = corr_val ** 2 # R^2 variance explained
        
        se = 1.0 / np.sqrt(max(n_eff, 4))
        l_ci = max(0.0, norm_mag - 1.96 * se)
        
        all_insights.append({
            "type": "Pairwise Correlation",
            "headline": f"{c['method']}: {c['pair']} (r={c['corr']})",
            "variables": c['pair'],
            "normalized_magnitude": round(norm_mag, 4),
            "l_ci": round(l_ci, 4),
            "p_adj": c.get("adj_p_value"),
            "n_eff": n_eff,
            "raw": c
        })

    # Sort by Confidence Interval Lower Bound (L_CI) descending
    all_insights.sort(key=lambda x: x["l_ci"], reverse=True)
    
    # De-duplicate redundant findings
    seen_vars = set()
    deduped = []
    for item in all_insights:
        v_key = item["variables"]
        if v_key not in seen_vars:
            seen_vars.add(v_key)
            deduped.append(item)
            
    return deduped[:5]


from app.tools.ingester import load_dataset

def run_statistical_analysis(dataset_path: str, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Performs assumption-driven statistical analysis, Simpson's Paradox detection,
    and family-wise Benjamini-Hochberg FDR correction.
    """
    df = load_dataset(dataset_path)
        
    numeric_df = df.select_dtypes(include='number')
    non_numeric_df = df.select_dtypes(exclude='number')
    
    # 1. Normality Assessment per Numeric Column
    normality_results = {}
    for col in numeric_df.columns:
        if not col.endswith('_was_missing'):
            normality_results[col] = _check_normality(numeric_df[col])
            
    num_cols = [c for c in numeric_df.columns if not c.endswith('_was_missing')]
    cat_cols = [c for c in non_numeric_df.columns if df[c].nunique() < 20]
    
    # 2. Simpson's Paradox Reversal Finding Detection
    simpsons_reversals = detect_simpsons_paradox(df, num_cols, cat_cols)

    # 3. Family 1: Pairwise Correlation Family
    correlation_family = []
    if len(num_cols) >= 2:
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                c1, c2 = num_cols[i], num_cols[j]
                s1, s2 = numeric_df[c1].dropna(), numeric_df[c2].dropna()
                common_idx = s1.index.intersection(s2.index)
                n_eff = len(common_idx)
                if n_eff >= 8:
                    norm1 = normality_results[c1]["is_normal"]
                    norm2 = normality_results[c2]["is_normal"]
                    
                    if norm1 and norm2:
                        r_val, p_val = stats.pearsonr(df.loc[common_idx, c1], df.loc[common_idx, c2])
                        method = "Pearson Correlation (Parametric)"
                    else:
                        r_val, p_val = stats.spearmanr(df.loc[common_idx, c1], df.loc[common_idx, c2])
                        method = "Spearman Correlation (Non-Parametric)"
                        
                    correlation_family.append({
                        "pair": f"{c1} & {c2}",
                        "method": method,
                        "corr": round(float(r_val), 4) if not np.isnan(r_val) else 0.0,
                        "raw_p_value": float(p_val) if not np.isnan(p_val) else 1.0,
                        "n_eff": n_eff
                    })
                    
    # 4. Family 2: Group Differences & Hypothesis Tests Family
    group_tests_family = []
    
    for cat_col in cat_cols:
        u_vals = df[cat_col].dropna().unique()
        if len(u_vals) == 2:
            v1, v2 = u_vals[0], u_vals[1]
            for num_col in num_cols:
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
                        test_name = "Independent 2-Sample T-Test" if equal_var else "Welch's T-Test (Unequal Variance)"
                        eff_size = _cohens_d(g1, g2)
                        eff_type = "Cohen's d"
                    else:
                        u_stat, p_val = stats.mannwhitneyu(g1, g2)
                        test_name = "Mann-Whitney U Test (Non-Parametric)"
                        t_stat = u_stat
                        eff_size = _cohens_d(g1, g2)
                        eff_type = "Rank Effect Size (Cohen's d equivalent)"
                        
                    group_tests_family.append({
                        "test_name": test_name,
                        "variables": f"Numeric: '{num_col}' by Cat: '{cat_col}' ({v1} vs {v2})",
                        "H0": f"Distribution/Mean of '{num_col}' is equal across '{cat_col}' groups",
                        "statistic": round(float(t_stat), 4) if not np.isnan(t_stat) else 0.0,
                        "raw_p_value": float(p_val) if not np.isnan(p_val) else 1.0,
                        "effect_size": round(float(eff_size), 4),
                        "effect_size_type": eff_type,
                        "n_eff": n_eff
                    })
                    if len(group_tests_family) >= 5:
                        break
            if len(group_tests_family) >= 5:
                break
                
    if len(cat_cols) >= 2 and len(group_tests_family) < 8:
        for i in range(len(cat_cols)):
            for j in range(i + 1, len(cat_cols)):
                c1, c2 = cat_cols[i], cat_cols[j]
                tbl = pd.crosstab(df[c1], df[c2])
                if tbl.shape[0] > 1 and tbl.shape[1] > 1:
                    exp_freq = stats.contingency.expected_freq(tbl)
                    total_cells = exp_freq.size
                    small_cells = (exp_freq < 5).sum()
                    min_exp = exp_freq.min()
                    
                    cochran_pass = bool(min_exp >= 1.0 and (small_cells / total_cells) <= 0.20)
                    n_eff = int(tbl.sum().sum())
                    
                    if tbl.shape == (2, 2) and not cochran_pass:
                        odds_ratio, p_val = stats.fisher_exact(tbl)
                        test_name = "Fisher's Exact Test (Cochran Violation Fallback)"
                        stat_val = odds_ratio
                        eff_val = _cramers_v(tbl)
                        eff_type = "Cramer's V"
                    else:
                        chi2, p_val, dof, _ = stats.chi2_contingency(tbl)
                        test_name = "Chi-Square Test of Independence" if cochran_pass else "Chi-Square (Warning: Cochran Rule Violated)"
                        stat_val = chi2
                        eff_val = _cramers_v(tbl)
                        eff_type = "Cramer's V"
                        
                    group_tests_family.append({
                        "test_name": test_name,
                        "variables": f"Categorical: '{c1}' & '{c2}'",
                        "H0": f"No association exists between '{c1}' and '{c2}'",
                        "statistic": round(float(stat_val), 4) if not np.isnan(stat_val) else 0.0,
                        "raw_p_value": float(p_val) if not np.isnan(p_val) else 1.0,
                        "effect_size": round(float(eff_val), 4),
                        "effect_size_type": eff_type,
                        "n_eff": n_eff
                    })
                    if len(group_tests_family) >= 8:
                        break
            if len(group_tests_family) >= 8:
                break

    # 5. Family-Wise Benjamini-Hochberg (BH) FDR Correction
    def _apply_fdr_correction(family_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not family_list:
            return []
        p_vals = np.array([item["raw_p_value"] for item in family_list])
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
            
        for i, item in enumerate(family_list):
            p_adj = round(float(adj_p[i]), 4)
            item["adj_p_value"] = p_adj
            item["family_size_m"] = m
            item["is_significant"] = bool(p_adj < 0.05)
            item["raw_p_value"] = round(item["raw_p_value"], 4)
            
        return family_list

    correlation_family = _apply_fdr_correction(correlation_family)
    group_tests_family = _apply_fdr_correction(group_tests_family)

    # 6. Rank Prioritized Insights based on 95% CI Lower Bound
    top_insights = rank_prioritized_insights(group_tests_family, correlation_family)

    summary_lines = [
        "=== ASSUMPTION-DRIVEN STATISTICAL ANALYSIS REPORT ===",
        f"Protected Target Variable: '{target_col}'" if target_col else "Protected Target Variable: None",
        f"\n--- MAJOR STATISTICAL FINDINGS & SIMPSON'S PARADOX ({len(simpsons_reversals)}) ---"
    ]
    for rev in simpsons_reversals:
        summary_lines.append(f"  🚨 {rev['headline']}")
    if not simpsons_reversals:
        summary_lines.append("  ✓ No Simpson's Paradox subgroup reversals detected.")
        
    summary_lines.append(f"\n--- TOP PRIORITIZED INSIGHTS (Ranked by 95% CI Lower Bound L_CI) ---")
    for idx, insight in enumerate(top_insights, 1):
        summary_lines.append(f"  Rank {idx} [{insight['type']}]: {insight['headline']} (L_CI={insight['l_ci']}, p_adj={insight['p_adj']}, N_eff={insight['n_eff']})")
    if not top_insights:
        summary_lines.append("  - No statistically significant prioritized insights found.")
        
    summary_lines.append("=== STATISTICAL ANALYSIS COMPLETE ===")
    summary_text = "\n".join(summary_lines)
    
    return {
        "normality_results": normality_results,
        "correlation_family": correlation_family,
        "group_tests_family": group_tests_family,
        "simpsons_reversals": simpsons_reversals,
        "top_insights": top_insights,
        "summary_text": summary_text
    }
