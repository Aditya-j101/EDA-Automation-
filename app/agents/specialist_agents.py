from langchain_core.messages import AIMessage
from app.agents.state import AgentState
from typing import Dict, Any, List

def data_quality_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Data Quality Specialist Agent.
    Interprets pre-computed quality evidence: missingness, duplicates, constants,
    PII shield findings, and headline quality scores without recalculating any metrics.
    """
    evidence = state.get("eda_evidence", {}).get("quality", {})
    missing_info = evidence.get("missing_info", {})
    duplicate_rows = evidence.get("duplicate_rows", 0)
    duplicate_pct = evidence.get("duplicate_pct", 0.0)
    duplicate_severity = evidence.get("duplicate_severity", "low")
    constant_cols = evidence.get("constant_cols", [])
    quasi_constant_cols = evidence.get("quasi_constant_cols", [])
    pii_findings = evidence.get("pii_findings", [])
    quality_score = evidence.get("quality_score", {})

    interpretations = []
    
    # 1. Missingness interpretation
    high_missing = [f"{col} ({info['percentage']}% nulls - severity: {info['severity']})" for col, info in missing_info.items() if info["null_count"] > 0]
    if high_missing:
        interpretations.append(f"Missingness Breakdown: {', '.join(high_missing)}")
    else:
        interpretations.append("Missingness Breakdown: Complete dataset with 0 missing values.")

    # 2. Duplicates interpretation
    interpretations.append(f"Duplicates: {duplicate_rows} duplicate rows detected ({duplicate_pct}% - severity: {duplicate_severity}).")

    # 3. Constant/Quasi-constant columns
    if constant_cols:
        interpretations.append(f"Constant Columns (0 variance): {constant_cols}")
    if quasi_constant_cols:
        q_str = ", ".join([f"{item['column']} ({item['percentage']}% top value '{item['top_value']}')" for item in quasi_constant_cols])
        interpretations.append(f"Quasi-Constant Columns (>95% single value): {q_str}")

    # 4. PII Shield findings
    if pii_findings:
        pii_str = ", ".join([f"{p['column']} ({p['type']} - {p['confidence']} confidence)" for p in pii_findings])
        interpretations.append(f"PII Privacy Shield Alerts: {pii_str}")
    else:
        interpretations.append("PII Privacy Shield Alerts: No high-confidence PII columns detected.")

    # 5. Quality Score
    if quality_score:
        interpretations.append(
            f"Headline Data Quality Score: {quality_score.get('overall_score', 0)}/100 [Grade: {quality_score.get('grade', 'N/A')}] "
            f"(Completeness: {quality_score.get('completeness')}%, Uniqueness: {quality_score.get('uniqueness')}%, "
            f"Validity: {quality_score.get('validity')}%, Consistency: {quality_score.get('consistency')}%)"
        )

    summary_text = "\n".join(interpretations)
    
    existing_findings = state.get("specialist_findings", {})
    updated_findings = dict(existing_findings)
    updated_findings["data_quality"] = interpretations

    return {
        "specialist_findings": updated_findings,
        "messages": [AIMessage(content=f"Data Quality Specialist Interpretation Complete:\n{summary_text}")]
    }


def distribution_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Distribution Specialist Agent.
    Interprets pre-computed distribution metrics: descriptive stats, shape (skewness/kurtosis),
    normality test results, and outlier presence without calculating any statistics.
    """
    distributions = state.get("eda_evidence", {}).get("distributions", {})
    interpretations = []

    for col, dist in distributions.items():
        if "mean" in dist: # Numeric column
            skew_sev = dist.get("skewness_severity", "symmetric")
            norm_test = dist.get("normality_test", {})
            outliers = dist.get("outliers", {})
            
            is_norm = norm_test.get("is_normal", False)
            norm_desc = f"Normal via {norm_test.get('test_used', 'test')} (p={norm_test.get('p_value', 0.0)})" if is_norm else f"Non-normal via {norm_test.get('test_used', 'test')} (p={norm_test.get('p_value', 0.0)})"
            
            interpretations.append(
                f"Feature '{col}': Mean={dist['mean']}, Std={dist['std']}, Median={dist['median']}, "
                f"IQR={dist['iqr']} (Q25={dist['q25']}, Q75={dist['q75']}), Skewness={dist['skewness']} ({skew_sev}), "
                f"Kurtosis={dist['kurtosis']}, Normality: {norm_desc}, Outliers={outliers.get('total_outliers', 0)} ({outliers.get('percentage', 0.0)}% - severity: {outliers.get('severity', 'low')})."
            )
        elif "nunique" in dist: # Categorical column
            top_cats = dist.get("top_categories", {})
            top_str = ", ".join([f"'{k}': {v['count']} ({v['percentage']}%)" for k, v in top_cats.items()])
            interpretations.append(
                f"Feature '{col}': Categorical with {dist['nunique']} unique values (ratio={dist['unique_ratio']}). Top categories: {top_str}."
            )

    summary_text = "\n".join(interpretations)
    
    existing_findings = state.get("specialist_findings", {})
    updated_findings = dict(existing_findings)
    updated_findings["distributions"] = interpretations

    return {
        "specialist_findings": updated_findings,
        "messages": [AIMessage(content=f"Distribution Specialist Interpretation Complete:\n{summary_text}")]
    }


def relationship_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Relationship Specialist Agent.
    Interprets pre-computed correlation matrices, hypothesis test findings,
    Simpson's paradox reversals, feature redundancy, and class imbalance.
    """
    relationships = state.get("eda_evidence", {}).get("relationships", {})
    anomalies = state.get("eda_evidence", {}).get("anomalies", {})
    
    correlations = relationships.get("correlation_family", [])
    group_tests = relationships.get("group_tests_family", [])
    simpsons = relationships.get("simpsons_reversals", [])
    vif_scores = relationships.get("vif_scores", {})
    redundant = relationships.get("redundant_features", [])
    
    interpretations = []

    # 1. Simpson's Paradox
    if simpsons:
        for s in simpsons:
            interpretations.append(f"Simpson's Paradox Alert: {s['headline']}")
    else:
        interpretations.append("Simpson's Paradox: No subgroup correlation reversals detected.")

    # 2. Significant Correlations
    sig_corrs = [c for c in correlations if c.get("is_significant")]
    if sig_corrs:
        for c in sig_corrs:
            interpretations.append(
                f"Correlation Finding: {c['pair']} has {c['strength']} {c['method']} r={c['corr']} (raw p={c['raw_p_value']}, p_adj={c['p_adj']})."
            )
    else:
        interpretations.append("Correlations: No statistically significant pairwise correlations after FDR correction.")

    # 3. Group Tests
    sig_tests = [t for t in group_tests if t.get("is_significant")]
    if sig_tests:
        for t in sig_tests:
            interpretations.append(
                f"Group Difference Finding: {t['test_name']} on {t['variables']} showed statistically significant difference (p_adj={t['p_adj']}, effect size {t['effect_size_type']}={t['effect_size']})."
            )

    # 4. Multicollinearity & Redundancy
    if redundant:
        r_str = ", ".join([f"{r['col1']} & {r['col2']} (r={r['corr']})" for r in redundant])
        interpretations.append(f"Feature Redundancy (>0.90 corr): {r_str}")
    high_vif = {col: score for col, score in vif_scores.items() if score > 5.0}
    if high_vif:
        interpretations.append(f"High Multicollinearity (VIF > 5.0): {high_vif}")

    # 5. Class Imbalance & Data Leakage
    class_dist = anomalies.get("class_distribution")
    imbalance_ratio = anomalies.get("imbalance_ratio")
    imbalance_severity = anomalies.get("imbalance_severity", "balanced")
    if class_dist:
        interpretations.append(
            f"Class Imbalance: Target distribution {class_dist} (imbalance ratio={imbalance_ratio} - severity: {imbalance_severity})."
        )
        
    leakage = anomalies.get("leakage_warnings", [])
    if leakage:
        l_str = ", ".join([f"{item['column']} (r={item['correlation']})" for item in leakage])
        interpretations.append(f"Data Leakage Risk (>0.95 target correlation): {l_str}")

    summary_text = "\n".join(interpretations)
    
    existing_findings = state.get("specialist_findings", {})
    updated_findings = dict(existing_findings)
    updated_findings["relationships"] = interpretations

    return {
        "specialist_findings": updated_findings,
        "messages": [AIMessage(content=f"Relationship Specialist Interpretation Complete:\n{summary_text}")]
    }
