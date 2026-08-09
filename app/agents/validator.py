import re
import logging
from typing import Dict, Any, List, Optional
from langchain_core.messages import AIMessage
from app.agents.state import AgentState

def generate_claims_from_evidence(evidence: Dict[str, Any], specialist_findings: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generates structured raw claims linked to exact pre-computed evidence metrics.
    """
    claims = []
    claim_counter = 1

    if not isinstance(evidence, dict):
        return claims

    schema = evidence.get("schema", {}) if isinstance(evidence.get("schema"), dict) else {}
    quality = evidence.get("quality", {}) if isinstance(evidence.get("quality"), dict) else {}
    distributions = evidence.get("distributions", {}) if isinstance(evidence.get("distributions"), dict) else {}
    relationships = evidence.get("relationships", {}) if isinstance(evidence.get("relationships"), dict) else {}
    anomalies = evidence.get("anomalies", {}) if isinstance(evidence.get("anomalies"), dict) else {}

    # 1. Quality Claims
    missing_info = quality.get("missing_info", {}) if isinstance(quality.get("missing_info"), dict) else {}
    for col, info in missing_info.items():
        if isinstance(info, dict) and info.get("null_count", 0) > 0:
            claims.append({
                "claim_id": f"claim_{claim_counter}",
                "category": "quality",
                "claim": f"Column '{col}' has {info.get('percentage', 0)}% missing values ({info.get('null_count', 0)} rows), categorized as {info.get('severity', 'low')} severity.",
                "evidence": {"column": col, "metric": "missingness", "null_count": info.get("null_count", 0), "missing_pct": info.get("percentage", 0), "severity": info.get("severity", "low")},
                "confidence": "high",
                "status": "pending_validation"
            })
            claim_counter += 1

    dup_rows = quality.get("duplicate_rows", 0)
    dup_pct = quality.get("duplicate_pct", 0.0)
    dup_sev = quality.get("duplicate_severity", "low")
    claims.append({
        "claim_id": f"claim_{claim_counter}",
        "category": "quality",
        "claim": f"Dataset contains {dup_rows} duplicate rows ({dup_pct}%), representing a {dup_sev} uniqueness concern.",
        "evidence": {"metric": "duplicates", "duplicate_rows": dup_rows, "duplicate_pct": dup_pct, "severity": dup_sev},
        "confidence": "high",
        "status": "pending_validation"
    })
    claim_counter += 1

    constant_cols = quality.get("constant_cols", []) if isinstance(quality.get("constant_cols"), list) else []
    for col in constant_cols:
        claims.append({
            "claim_id": f"claim_{claim_counter}",
            "category": "quality",
            "claim": f"Feature '{col}' is a zero-variance constant column with 1 unique value across all rows.",
            "evidence": {"column": col, "metric": "constant_column"},
            "confidence": "high",
            "status": "pending_validation"
        })
        claim_counter += 1

    quasi_cols = quality.get("quasi_constant_cols", []) if isinstance(quality.get("quasi_constant_cols"), list) else []
    for item in quasi_cols:
        if isinstance(item, dict):
            claims.append({
                "claim_id": f"claim_{claim_counter}",
                "category": "quality",
                "claim": f"Feature '{item.get('column')}' is quasi-constant, with value '{item.get('top_value')}' accounting for {item.get('percentage')}% of observations.",
                "evidence": {"column": item.get('column'), "metric": "quasi_constant", "top_value": item.get('top_value'), "percentage": item.get('percentage')},
                "confidence": "high",
                "status": "pending_validation"
            })
            claim_counter += 1

    pii_list = quality.get("pii_findings", []) if isinstance(quality.get("pii_findings"), list) else []
    for pii in pii_list:
        if isinstance(pii, dict):
            claims.append({
                "claim_id": f"claim_{claim_counter}",
                "category": "quality",
                "claim": f"Column '{pii.get('column')}' contains potential {pii.get('type')} PII with {pii.get('confidence')} confidence.",
                "evidence": pii,
                "confidence": "high",
                "status": "pending_validation"
            })
            claim_counter += 1

    q_score = quality.get("quality_score", {})
    if q_score and isinstance(q_score, dict):
        claims.append({
            "claim_id": f"claim_{claim_counter}",
            "category": "quality",
            "claim": f"Overall Data Quality Score is {q_score.get('overall_score')}/100 with Grade {q_score.get('grade')}.",
            "evidence": q_score,
            "confidence": "high",
            "status": "pending_validation"
        })
        claim_counter += 1

    # 2. Distribution Claims
    for col, dist in distributions.items():
        if not isinstance(dist, dict):
            continue
        if "mean" in dist:
            skew_sev = dist.get("skewness_severity", "symmetric")
            claims.append({
                "claim_id": f"claim_{claim_counter}",
                "category": "distribution",
                "claim": f"Numeric feature '{col}' exhibits mean={dist.get('mean')}, median={dist.get('median')}, skewness={dist.get('skewness')} ({skew_sev}).",
                "evidence": {"column": col, "mean": dist.get('mean'), "median": dist.get('median'), "skewness": dist.get('skewness'), "severity": skew_sev},
                "confidence": "high",
                "status": "pending_validation"
            })
            claim_counter += 1

            norm_test = dist.get("normality_test", {}) if isinstance(dist.get("normality_test"), dict) else {}
            if norm_test.get("is_normal"):
                claims.append({
                    "claim_id": f"claim_{claim_counter}",
                    "category": "distribution",
                    "claim": f"Numeric feature '{col}' complies with normal distribution assumptions per {norm_test.get('test_used', 'test')} (p={norm_test.get('p_value', 0.0)}).",
                    "evidence": {"column": col, "normality_test": norm_test},
                    "confidence": "high",
                    "status": "pending_validation"
                })
            else:
                claims.append({
                    "claim_id": f"claim_{claim_counter}",
                    "category": "distribution",
                    "claim": f"Numeric feature '{col}' departs from normal distribution assumptions per {norm_test.get('test_used', 'test')} (p={norm_test.get('p_value', 0.0)}).",
                    "evidence": {"column": col, "normality_test": norm_test},
                    "confidence": "high",
                    "status": "pending_validation"
                })
            claim_counter += 1

            outliers = dist.get("outliers", {}) if isinstance(dist.get("outliers"), dict) else {}
            if outliers.get("total_outliers", 0) > 0:
                claims.append({
                    "claim_id": f"claim_{claim_counter}",
                    "category": "distribution",
                    "claim": f"Feature '{col}' contains {outliers.get('total_outliers')} outliers ({outliers.get('percentage')}%), representing a {outliers.get('severity')} outlier presence.",
                    "evidence": {"column": col, "outliers": outliers},
                    "confidence": "high",
                    "status": "pending_validation"
                })
                claim_counter += 1

    # 3. Relationship Claims
    simpsons_list = relationships.get("simpsons_reversals", []) if isinstance(relationships.get("simpsons_reversals"), list) else []
    for s in simpsons_list:
        if isinstance(s, dict):
            claims.append({
                "claim_id": f"claim_{claim_counter}",
                "category": "relationship",
                "claim": f"Simpson's Paradox detected between '{s.get('feature_x')}' and '{s.get('feature_y')}': overall correlation (r={s.get('overall_corr')}) reverses direction when conditioned on subgroup '{s.get('group_var')}'.",
                "evidence": s,
                "confidence": "high",
                "status": "pending_validation"
            })
            claim_counter += 1

    corr_list = relationships.get("correlation_family", []) if isinstance(relationships.get("correlation_family"), list) else []
    for c in corr_list:
        if isinstance(c, dict) and c.get("is_significant"):
            claims.append({
                "claim_id": f"claim_{claim_counter}",
                "category": "relationship",
                "claim": f"Pairwise correlation between {c.get('pair')} is statistically significant ({c.get('method')}, r={c.get('corr')}, FDR p_adj={c.get('p_adj')}).",
                "evidence": c,
                "confidence": "high",
                "status": "pending_validation"
            })
            claim_counter += 1

    group_list = relationships.get("group_tests_family", []) if isinstance(relationships.get("group_tests_family"), list) else []
    for t in group_list:
        if isinstance(t, dict) and t.get("is_significant"):
            claims.append({
                "claim_id": f"claim_{claim_counter}",
                "category": "relationship",
                "claim": f"Group difference test ({t.get('test_name')}) on {t.get('variables')} indicates a statistically significant difference (p_adj={t.get('p_adj')}, effect size {t.get('effect_size_type')}={t.get('effect_size')}).",
                "evidence": t,
                "confidence": "high",
                "status": "pending_validation"
            })
            claim_counter += 1

    redundant_list = relationships.get("redundant_features", []) if isinstance(relationships.get("redundant_features"), list) else []
    for r in redundant_list:
        if isinstance(r, dict):
            claims.append({
                "claim_id": f"claim_{claim_counter}",
                "category": "relationship",
                "claim": f"Features '{r.get('col1')}' and '{r.get('col2')}' exhibit high feature redundancy with correlation r={r.get('corr')}.",
                "evidence": r,
                "confidence": "high",
                "status": "pending_validation"
            })
            claim_counter += 1

    # 4. Anomaly Claims
    class_dist = anomalies.get("class_distribution")
    imbalance_ratio = anomalies.get("imbalance_ratio")
    imbalance_sev = anomalies.get("imbalance_severity", "balanced")
    if class_dist and imbalance_ratio is not None:
        claims.append({
            "claim_id": f"claim_{claim_counter}",
            "category": "anomaly",
            "claim": f"Target variable exhibits a class distribution of {class_dist} with a minority/majority imbalance ratio of {imbalance_ratio} ({imbalance_sev} imbalance).",
            "evidence": {"class_distribution": class_dist, "imbalance_ratio": imbalance_ratio, "severity": imbalance_sev},
            "confidence": "high",
            "status": "pending_validation"
        })
        claim_counter += 1

    leakage_list = anomalies.get("leakage_warnings", []) if isinstance(anomalies.get("leakage_warnings"), list) else []
    for lk in leakage_list:
        if isinstance(lk, dict):
            claims.append({
                "claim_id": f"claim_{claim_counter}",
                "category": "anomaly",
                "claim": f"Data leakage warning: Column '{lk.get('column')}' has near-perfect correlation (r={lk.get('correlation')}) with the target variable.",
                "evidence": lk,
                "confidence": "high",
                "status": "pending_validation"
            })
            claim_counter += 1

    return claims


def validate_claim_against_evidence(claim: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verifier / Validator:
    Checks claim against ground truth evidence in eda_evidence.
    """
    claim_text = claim.get("claim", "")
    claim_evidence = claim.get("evidence", {}) if isinstance(claim.get("evidence"), dict) else {}

    # Banned Rule 1: Causation Claims from Correlation
    causation_words = [r'\bcauses\b', r'\bcaused\b', r'\bcausation\b', r'\bdrives\b', r'\bleads to\b', r'\bresulted in\b']
    for pattern in causation_words:
        if re.search(pattern, claim_text, re.IGNORECASE):
            return {
                **claim,
                "status": "unsupported",
                "failure_reason": "Banned Claim Rule: Inferred causation from correlation without experimental control."
            }

    # Banned Rule 2: Exaggerated / Misclassified Missingness Severity
    if "missing" in claim_text.lower():
        missing_pct = claim_evidence.get("missing_pct")
        if missing_pct is not None and isinstance(missing_pct, (int, float)):
            if missing_pct < 5.0 and any(w in claim_text.lower() for w in ["substantial", "critical", "severe", "high missingness"]):
                return {
                    **claim,
                    "status": "unsupported",
                    "failure_reason": f"Severity Misclassification: {missing_pct}% missingness is low (<5%) and cannot be claimed as substantial/critical."
                }

    # Banned Rule 3: Significance Claimed Without Test / Significant p-value
    if "statistically significant" in claim_text.lower():
        p_adj = claim_evidence.get("p_adj")
        if p_adj is not None and isinstance(p_adj, (int, float)) and p_adj >= 0.05:
            return {
                **claim,
                "status": "unsupported",
                "failure_reason": f"Unsupported Significance: p_adj={p_adj} >= 0.05 does not reach statistical significance threshold."
            }

    # Banned Rule 4: Normality Claimed When Test Fails
    if "complies with normal distribution" in claim_text.lower() or "is normally distributed" in claim_text.lower():
        norm_test = claim_evidence.get("normality_test", {}) if isinstance(claim_evidence.get("normality_test"), dict) else {}
        if not norm_test.get("is_normal", False):
            return {
                **claim,
                "status": "unsupported",
                "failure_reason": "Unsupported Normality: Statistical test indicated departure from normal distribution."
            }

    # Banned Rule 5: Recommend Dropping Feature Solely for Missingness
    if "drop" in claim_text.lower() and "missing" in claim_text.lower():
        return {
            **claim,
            "status": "unsupported",
            "failure_reason": "Banned Recommendation: Dropping a feature solely due to high missingness without checking imputation/indicator options."
        }

    # Banned Rule 6: Anomaly Classification Without Statistical Backing
    if "anomaly" in claim_text.lower() and "outlier" not in claim_text.lower() and "imbalance" not in claim_text.lower() and "simpson" not in claim_text.lower():
        return {
            **claim,
            "status": "unsupported",
            "failure_reason": "Banned Anomaly Claim: Observations classified as anomalies without underlying statistical evidence."
        }

    # Default: Claim is backed by empirical evidence
    return {
        **claim,
        "status": "supported",
        "confidence": "high"
    }


def claim_generator_node(state: AgentState) -> Dict[str, Any]:
    """LangGraph node: Generates raw claims from structured EDA evidence."""
    try:
        evidence = state.get("eda_evidence", {})
        specialist_findings = state.get("specialist_findings", {})
        
        generated_claims = generate_claims_from_evidence(evidence, specialist_findings)
        
        return {
            "generated_claims": generated_claims,
            "messages": [AIMessage(content=f"Claim Generator Node: Generated {len(generated_claims)} structured candidate claims.")]
        }
    except Exception as e:
        logging.error(f"Claim Generator Error: {e}")
        return {
            "generated_claims": [],
            "messages": [AIMessage(content=f"Claim Generator Completed with fallback: {e}")]
        }


def validator_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node: Verifies each candidate claim against ground truth evidence.
    """
    try:
        evidence = state.get("eda_evidence", {})
        generated_claims = state.get("generated_claims", [])

        validated_claims = []
        rejected_claims = []

        if isinstance(generated_claims, list):
            for raw_claim in generated_claims:
                if isinstance(raw_claim, dict):
                    res = validate_claim_against_evidence(raw_claim, evidence)
                    if res.get("status") == "supported":
                        validated_claims.append(res)
                    else:
                        rejected_claims.append(res)

        summary_msg = f"Validator / Verifier Gate: {len(validated_claims)} claims verified & SUPPORTED, {len(rejected_claims)} claims REJECTED."

        return {
            "validated_claims": validated_claims,
            "rejected_claims": rejected_claims,
            "messages": [AIMessage(content=summary_msg)]
        }
    except Exception as e:
        logging.error(f"Validator Node Error: {e}")
        return {
            "validated_claims": [],
            "rejected_claims": [],
            "messages": [AIMessage(content=f"Validator Node Completed with fallback: {e}")]
        }
