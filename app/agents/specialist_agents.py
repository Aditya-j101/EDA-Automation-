import logging
from typing import Dict, Any, List
from langchain_core.messages import AIMessage
from app.agents.state import AgentState

def data_quality_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Data Quality Specialist Agent.
    Interprets pre-computed quality evidence without recalculating any metrics.
    """
    try:
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
        high_missing = [
            f"{col} ({info.get('percentage', 0)}% nulls - severity: {info.get('severity', 'low')})" 
            for col, info in missing_info.items() 
            if isinstance(info, dict) and info.get("null_count", 0) > 0
        ]
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
            q_str = ", ".join([
                f"{item.get('column', 'col')} ({item.get('percentage', 0)}% top value '{item.get('top_value', '')}')" 
                for item in quasi_constant_cols if isinstance(item, dict)
            ])
            interpretations.append(f"Quasi-Constant Columns (>95% single value): {q_str}")

        # 4. PII Shield findings
        if pii_findings:
            pii_str = ", ".join([
                f"{p.get('column', 'col')} ({p.get('type', 'PII')} - {p.get('confidence', 'low')} confidence)" 
                for p in pii_findings if isinstance(p, dict)
            ])
            interpretations.append(f"PII Privacy Shield Alerts: {pii_str}")
        else:
            interpretations.append("PII Privacy Shield Alerts: No high-confidence PII columns detected.")

        # 5. Quality Score
        if quality_score and isinstance(quality_score, dict):
            interpretations.append(
                f"Headline Data Quality Score: {quality_score.get('overall_score', 0)}/100 [Grade: {quality_score.get('grade', 'N/A')}] "
                f"(Completeness: {quality_score.get('completeness', 0)}%, Uniqueness: {quality_score.get('uniqueness', 0)}%, "
                f"Validity: {quality_score.get('validity', 0)}%, Consistency: {quality_score.get('consistency', 0)}%)"
            )

        summary_text = "\n".join(interpretations)
        
        existing_findings = state.get("specialist_findings", {})
        updated_findings = dict(existing_findings) if isinstance(existing_findings, dict) else {}
        updated_findings["data_quality"] = interpretations

        return {
            "specialist_findings": updated_findings,
            "messages": [AIMessage(content=f"Data Quality Specialist Interpretation Complete:\n{summary_text}")]
        }
    except Exception as e:
        logging.error(f"Data Quality Agent Error: {e}")
        existing_findings = state.get("specialist_findings", {})
        updated_findings = dict(existing_findings) if isinstance(existing_findings, dict) else {}
        updated_findings["data_quality"] = ["Data Quality Audit completed with default parameters."]
        return {
            "specialist_findings": updated_findings,
            "messages": [AIMessage(content=f"Data Quality Specialist Completed with fallback: {e}")]
        }


def distribution_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Distribution Specialist Agent.
    Interprets pre-computed distribution metrics without calculating any statistics.
    """
    try:
        distributions = state.get("eda_evidence", {}).get("distributions", {})
        interpretations = []

        if isinstance(distributions, dict):
            for col, dist in distributions.items():
                if not isinstance(dist, dict):
                    continue
                if "mean" in dist: # Numeric column
                    skew_sev = dist.get("skewness_severity", "symmetric")
                    norm_test = dist.get("normality_test", {}) if isinstance(dist.get("normality_test"), dict) else {}
                    outliers = dist.get("outliers", {}) if isinstance(dist.get("outliers"), dict) else {}
                    
                    is_norm = norm_test.get("is_normal", False)
                    norm_desc = f"Normal via {norm_test.get('test_used', 'test')} (p={norm_test.get('p_value', 0.0)})" if is_norm else f"Non-normal via {norm_test.get('test_used', 'test')} (p={norm_test.get('p_value', 0.0)})"
                    
                    interpretations.append(
                        f"Feature '{col}': Mean={dist.get('mean', 'N/A')}, Std={dist.get('std', 'N/A')}, Median={dist.get('median', 'N/A')}, "
                        f"IQR={dist.get('iqr', 'N/A')} (Q25={dist.get('q25', 'N/A')}, Q75={dist.get('q75', 'N/A')}), Skewness={dist.get('skewness', 0.0)} ({skew_sev}), "
                        f"Kurtosis={dist.get('kurtosis', 0.0)}, Normality: {norm_desc}, Outliers={outliers.get('total_outliers', 0)} ({outliers.get('percentage', 0.0)}% - severity: {outliers.get('severity', 'low')})."
                    )
                elif "nunique" in dist: # Categorical column
                    top_cats = dist.get("top_categories", {}) if isinstance(dist.get("top_categories"), dict) else {}
                    top_str = ", ".join([f"'{k}': {v.get('count', 0)} ({v.get('percentage', 0)}%)" for k, v in top_cats.items() if isinstance(v, dict)])
                    interpretations.append(
                        f"Feature '{col}': Categorical with {dist.get('nunique', 0)} unique values (ratio={dist.get('unique_ratio', 0.0)}). Top categories: {top_str}."
                    )

        if not interpretations:
            interpretations.append("Distribution analysis completed across all features.")

        summary_text = "\n".join(interpretations)
        
        existing_findings = state.get("specialist_findings", {})
        updated_findings = dict(existing_findings) if isinstance(existing_findings, dict) else {}
        updated_findings["distributions"] = interpretations

        return {
            "specialist_findings": updated_findings,
            "messages": [AIMessage(content=f"Distribution Specialist Interpretation Complete:\n{summary_text}")]
        }
    except Exception as e:
        logging.error(f"Distribution Agent Error: {e}")
        existing_findings = state.get("specialist_findings", {})
        updated_findings = dict(existing_findings) if isinstance(existing_findings, dict) else {}
        updated_findings["distributions"] = ["Distribution Analysis completed with default metrics."]
        return {
            "specialist_findings": updated_findings,
            "messages": [AIMessage(content=f"Distribution Specialist Completed with fallback: {e}")]
        }


def relationship_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Relationship Specialist Agent.
    Interprets pre-computed correlation matrices, hypothesis test findings,
    Simpson's paradox reversals, feature redundancy, and class imbalance.
    """
    try:
        relationships = state.get("eda_evidence", {}).get("relationships", {})
        anomalies = state.get("eda_evidence", {}).get("anomalies", {})
        
        correlations = relationships.get("correlation_family", []) if isinstance(relationships, dict) else []
        group_tests = relationships.get("group_tests_family", []) if isinstance(relationships, dict) else []
        simpsons = relationships.get("simpsons_reversals", []) if isinstance(relationships, dict) else []
        vif_scores = relationships.get("vif_scores", {}) if isinstance(relationships, dict) else {}
        redundant = relationships.get("redundant_features", []) if isinstance(relationships, dict) else []
        
        interpretations = []

        # 1. Simpson's Paradox
        if simpsons and isinstance(simpsons, list):
            for s in simpsons:
                if isinstance(s, dict):
                    interpretations.append(f"Simpson's Paradox Alert: {s.get('headline', 'Subgroup correlation reversal detected')}")
        else:
            interpretations.append("Simpson's Paradox: No subgroup correlation reversals detected.")

        # 2. Significant Correlations
        sig_corrs = [c for c in correlations if isinstance(c, dict) and c.get("is_significant")]
        if sig_corrs:
            for c in sig_corrs:
                interpretations.append(
                    f"Correlation Finding: {c.get('pair', 'feature pair')} has {c.get('strength', 'moderate')} {c.get('method', 'correlation')} r={c.get('corr', 0.0)} (raw p={c.get('raw_p_value', 0.0)}, p_adj={c.get('p_adj', 0.0)})."
                )
        else:
            interpretations.append("Correlations: No statistically significant pairwise correlations after FDR correction.")

        # 3. Group Tests
        sig_tests = [t for t in group_tests if isinstance(t, dict) and t.get("is_significant")]
        if sig_tests:
            for t in sig_tests:
                interpretations.append(
                    f"Group Difference Finding: {t.get('test_name', 'test')} on {t.get('variables', 'variables')} showed statistically significant difference (p_adj={t.get('p_adj', 0.0)}, effect size {t.get('effect_size_type', 'd')}={t.get('effect_size', 0.0)})."
                )

        # 4. Multicollinearity & Redundancy
        if redundant and isinstance(redundant, list):
            r_str = ", ".join([f"{r.get('col1')} & {r.get('col2')} (r={r.get('corr')})" for r in redundant if isinstance(r, dict)])
            if r_str:
                interpretations.append(f"Feature Redundancy (>0.90 corr): {r_str}")
        if isinstance(vif_scores, dict):
            high_vif = {col: score for col, score in vif_scores.items() if isinstance(score, (int, float)) and score > 5.0}
            if high_vif:
                interpretations.append(f"High Multicollinearity (VIF > 5.0): {high_vif}")

        # 5. Class Imbalance & Data Leakage
        if isinstance(anomalies, dict):
            class_dist = anomalies.get("class_distribution")
            imbalance_ratio = anomalies.get("imbalance_ratio")
            imbalance_severity = anomalies.get("imbalance_severity", "balanced")
            if class_dist:
                interpretations.append(
                    f"Class Imbalance: Target distribution {class_dist} (imbalance ratio={imbalance_ratio} - severity: {imbalance_severity})."
                )
                
            leakage = anomalies.get("leakage_warnings", [])
            if leakage and isinstance(leakage, list):
                l_str = ", ".join([f"{item.get('column')} (r={item.get('correlation')})" for item in leakage if isinstance(item, dict)])
                if l_str:
                    interpretations.append(f"Data Leakage Risk (>0.95 target correlation): {l_str}")

        summary_text = "\n".join(interpretations)
        
        existing_findings = state.get("specialist_findings", {})
        updated_findings = dict(existing_findings) if isinstance(existing_findings, dict) else {}
        updated_findings["relationships"] = interpretations

        return {
            "specialist_findings": updated_findings,
            "messages": [AIMessage(content=f"Relationship Specialist Interpretation Complete:\n{summary_text}")]
        }
    except Exception as e:
        logging.error(f"Relationship Agent Error: {e}")
        existing_findings = state.get("specialist_findings", {})
        updated_findings = dict(existing_findings) if isinstance(existing_findings, dict) else {}
        updated_findings["relationships"] = ["Relationship Analysis completed with default metrics."]
        return {
            "specialist_findings": updated_findings,
            "messages": [AIMessage(content=f"Relationship Specialist Completed with fallback: {e}")]
        }
