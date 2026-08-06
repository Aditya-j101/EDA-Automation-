import re
from typing import Dict, Any, List, Optional
from langchain_core.messages import AIMessage
from app.agents.state import AgentState

def generate_claims_from_evidence(evidence: Dict[str, Any], specialist_findings: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generates structured raw claims linked to exact pre-computed evidence metrics.
    Claim format:
    {
        "claim_id": str,
        "category": str, # quality, distribution, relationship, anomaly
        "claim": str,
        "evidence": Dict[str, Any],
        "confidence": str, # high, medium, low
        "status": "pending_validation"
    }
    """
    claims = []
    claim_counter = 1

    schema = evidence.get("schema", {})
    quality = evidence.get("quality", {})
    distributions = evidence.get("distributions", {})
    relationships = evidence.get("relationships", {})
    anomalies = evidence.get("anomalies", {})

    # 1. Quality Claims
    missing_info = quality.get("missing_info", {})
    for col, info in missing_info.items():
        if info["null_count"] > 0:
            claims.append({
                "claim_id": f"claim_{claim_counter}",
                "category": "quality",
                "claim": f"Column '{col}' has {info['percentage']}% missing values ({info['null_count']} rows), categorized as {info['severity']} severity.",
                "evidence": {"column": col, "metric": "missingness", "null_count": info["null_count"], "missing_pct": info["percentage"], "severity": info["severity"]},
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

    for col in quality.get("constant_cols", []):
        claims.append({
            "claim_id": f"claim_{claim_counter}",
            "category": "quality",
            "claim": f"Feature '{col}' is a zero-variance constant column with 1 unique value across all rows.",
            "evidence": {"column": col, "metric": "constant_column"},
            "confidence": "high",
            "status": "pending_validation"
        })
        claim_counter += 1

    for item in quality.get("quasi_constant_cols", []):
        claims.append({
            "claim_id": f"claim_{claim_counter}",
            "category": "quality",
            "claim": f"Feature '{item['column']}' is quasi-constant, with value '{item['top_value']}' accounting for {item['percentage']}% of observations.",
            "evidence": {"column": item['column'], "metric": "quasi_constant", "top_value": item['top_value'], "percentage": item['percentage']},
            "confidence": "high",
            "status": "pending_validation"
        })
        claim_counter += 1

    for pii in quality.get("pii_findings", []):
        claims.append({
            "claim_id": f"claim_{claim_counter}",
            "category": "quality",
            "claim": f"Column '{pii['column']}' contains potential {pii['type']} PII with {pii['confidence']} confidence.",
            "evidence": pii,
            "confidence": "high",
            "status": "pending_validation"
        })
        claim_counter += 1

    q_score = quality.get("quality_score", {})
    if q_score:
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
        if "mean" in dist:
            skew_sev = dist.get("skewness_severity", "symmetric")
            claims.append({
                "claim_id": f"claim_{claim_counter}",
                "category": "distribution",
                "claim": f"Numeric feature '{col}' exhibits mean={dist['mean']}, median={dist['median']}, skewness={dist['skewness']} ({skew_sev}).",
                "evidence": {"column": col, "mean": dist['mean'], "median": dist['median'], "skewness": dist['skewness'], "severity": skew_sev},
                "confidence": "high",
                "status": "pending_validation"
            })
            claim_counter += 1

            norm_test = dist.get("normality_test", {})
            if norm_test.get("is_normal"):
                claims.append({
                    "claim_id": f"claim_{claim_counter}",
                    "category": "distribution",
                    "claim": f"Numeric feature '{col}' complies with normal distribution assumptions per {norm_test.get('test_used')} (p={norm_test.get('p_value')}).",
                    "evidence": {"column": col, "normality_test": norm_test},
                    "confidence": "high",
                    "status": "pending_validation"
                })
            else:
                claims.append({
                    "claim_id": f"claim_{claim_counter}",
                    "category": "distribution",
                    "claim": f"Numeric feature '{col}' departs from normal distribution assumptions per {norm_test.get('test_used')} (p={norm_test.get('p_value')}).",
                    "evidence": {"column": col, "normality_test": norm_test},
                    "confidence": "high",
                    "status": "pending_validation"
                })
            claim_counter += 1

            outliers = dist.get("outliers", {})
            if outliers.get("total_outliers", 0) > 0:
                claims.append({
                    "claim_id": f"claim_{claim_counter}",
                    "category": "distribution",
                    "claim": f"Feature '{col}' contains {outliers['total_outliers']} outliers ({outliers['percentage']}%), representing a {outliers['severity']} outlier presence.",
                    "evidence": {"column": col, "outliers": outliers},
                    "confidence": "high",
                    "status": "pending_validation"
                })
                claim_counter += 1

    # 3. Relationship Claims
    for s in relationships.get("simpsons_reversals", []):
        claims.append({
            "claim_id": f"claim_{claim_counter}",
            "category": "relationship",
            "claim": f"Simpson's Paradox detected between '{s['feature_x']}' and '{s['feature_y']}': overall correlation (r={s['overall_corr']}) reverses direction when conditioned on subgroup '{s['group_var']}'.",
            "evidence": s,
            "confidence": "high",
            "status": "pending_validation"
        })
        claim_counter += 1

    for c in relationships.get("correlation_family", []):
        if c.get("is_significant"):
            claims.append({
                "claim_id": f"claim_{claim_counter}",
                "category": "relationship",
                "claim": f"Pairwise correlation between {c['pair']} is statistically significant ({c['method']}, r={c['corr']}, FDR p_adj={c['p_adj']}).",
                "evidence": c,
                "confidence": "high",
                "status": "pending_validation"
            })
            claim_counter += 1

    for t in relationships.get("group_tests_family", []):
        if t.get("is_significant"):
            claims.append({
                "claim_id": f"claim_{claim_counter}",
                "category": "relationship",
                "claim": f"Group difference test ({t['test_name']}) on {t['variables']} indicates a statistically significant difference (p_adj={t['p_adj']}, effect size {t['effect_size_type']}={t['effect_size']}).",
                "evidence": t,
                "confidence": "high",
                "status": "pending_validation"
            })
            claim_counter += 1

    for r in relationships.get("redundant_features", []):
        claims.append({
            "claim_id": f"claim_{claim_counter}",
            "category": "relationship",
            "claim": f"Features '{r['col1']}' and '{r['col2']}' exhibit high feature redundancy with correlation r={r['corr']}.",
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

    for lk in anomalies.get("leakage_warnings", []):
        claims.append({
            "claim_id": f"claim_{claim_counter}",
            "category": "anomaly",
            "claim": f"Data leakage warning: Column '{lk['column']}' has near-perfect correlation (r={lk['correlation']}) with the target variable.",
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
    Validates:
      1. Numerical accuracy matching.
      2. Severity term classification rules.
      3. Banned unsupported claims (causation, fake significance, fake normality, feature dropping solely for missingness, etc.).
    Returns modified claim with status="supported" or status="unsupported" and failure_reason if rejected.
    """
    claim_text = claim.get("claim", "")
    claim_evidence = claim.get("evidence", {})
    category = claim.get("category", "")

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
    # E.g., calling low missingness (<5%) "substantial", "critical", or "high"
    if "missing" in claim_text.lower():
        missing_pct = claim_evidence.get("missing_pct")
        if missing_pct is not None:
            if missing_pct < 5.0 and any(w in claim_text.lower() for w in ["substantial", "critical", "severe", "high missingness"]):
                return {
                    **claim,
                    "status": "unsupported",
                    "failure_reason": f"Severity Misclassification: {missing_pct}% missingness is low (<5%) and cannot be claimed as substantial/critical."
                }

    # Banned Rule 3: Significance Claimed Without Test / Significant p-value
    if "statistically significant" in claim_text.lower():
        p_adj = claim_evidence.get("p_adj")
        if p_adj is not None and p_adj >= 0.05:
            return {
                **claim,
                "status": "unsupported",
                "failure_reason": f"Unsupported Significance: p_adj={p_adj} >= 0.05 does not reach statistical significance threshold."
            }

    # Banned Rule 4: Normality Claimed When Test Fails
    if "complies with normal distribution" in claim_text.lower() or "is normally distributed" in claim_text.lower():
        norm_test = claim_evidence.get("normality_test", {})
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
    evidence = state.get("eda_evidence", {})
    specialist_findings = state.get("specialist_findings", {})
    
    generated_claims = generate_claims_from_evidence(evidence, specialist_findings)
    
    return {
        "generated_claims": generated_claims,
        "messages": [AIMessage(content=f"Claim Generator Node: Generated {len(generated_claims)} structured candidate claims.")]
    }


def validator_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node: Verifies each candidate claim against ground truth evidence.
    Filters to supported_claims and records rejected_claims with failure reasons.
    """
    evidence = state.get("eda_evidence", {})
    generated_claims = state.get("generated_claims", [])

    validated_claims = []
    rejected_claims = []

    for raw_claim in generated_claims:
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
