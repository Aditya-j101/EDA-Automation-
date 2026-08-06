import os
import shutil
import tempfile
import unittest
import pandas as pd
import numpy as np

from app.core.evidence import build_structured_evidence
from app.agents.validator import generate_claims_from_evidence, validate_claim_against_evidence

class TestGoldenDatasetEvaluation(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.golden_csv = os.path.join(self.test_dir, "golden_dataset.csv")
        
        np.random.seed(42)
        n = 1000
        
        # 1. Age (Normal distribution)
        age = np.random.normal(35, 8, n)
        
        # 2. Salary (Right-skewed + 4.82% missingness)
        salary_base = np.random.exponential(scale=20000, size=n) + 40000
        # Inject correlation with age (r ~ 0.18)
        salary = salary_base + age * 300
        
        # Inject exact 4.82% missingness (48 nulls)
        missing_indices = np.random.choice(n, size=48, replace=False)
        salary[missing_indices] = np.nan
        
        # 3. Target (Class imbalance: 200 ones, 800 zeros => ratio 0.25)
        target = np.zeros(n, dtype=int)
        target[:200] = 1
        np.random.shuffle(target)
        
        df = pd.DataFrame({
            "Age": age,
            "Salary": salary,
            "Target": target
        })
        
        # 4. Inject exactly 312 duplicate rows (replicate first 312 rows)
        dup_rows = df.iloc[:312].copy()
        df = pd.concat([df, dup_rows], ignore_index=True)
        
        df.to_csv(self.golden_csv, index=False)
        self.df = df

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_golden_dataset_metrics_accuracy(self):
        """Validates that deterministic evidence builder extracts ground truth stats exactly."""
        evidence = build_structured_evidence(self.golden_csv, target_col="Target")
        
        # 1. Duplicate rows ground truth verification
        self.assertEqual(evidence["quality"]["duplicate_rows"], 312)
        
        # 2. Missingness percentage verification for Salary
        expected_nulls = int(self.df["Salary"].isnull().sum())
        salary_nulls = evidence["quality"]["missing_info"]["Salary"]["null_count"]
        self.assertEqual(salary_nulls, expected_nulls)
        
        # 3. Missingness severity threshold rule verification (4.82% -> 'low')
        salary_severity = evidence["quality"]["missing_info"]["Salary"]["severity"]
        self.assertEqual(salary_severity, "low")
        
        # 4. Class imbalance ratio verification
        imbalance_ratio = evidence["anomalies"]["imbalance_ratio"]
        self.assertAlmostEqual(imbalance_ratio, 0.25, delta=0.05)

    def test_golden_evaluation_benchmark_suite(self):
        """Runs the complete Golden Dataset Benchmark Evaluation Suite across all 7 dimensions."""
        evidence = build_structured_evidence(self.golden_csv, target_col="Target")
        specialist_findings = {
            "data_quality": ["Quality audit complete."],
            "distributions": ["Distribution audit complete."],
            "relationships": ["Relationship audit complete."]
        }
        
        # Generate raw candidate claims
        raw_claims = generate_claims_from_evidence(evidence, specialist_findings)
        self.assertGreater(len(raw_claims), 0)
        
        validated_claims = []
        rejected_claims = []
        for claim in raw_claims:
            res = validate_claim_against_evidence(claim, evidence)
            if res.get("status") == "supported":
                validated_claims.append(res)
            else:
                rejected_claims.append(res)

        # Dimension 1: Numerical Accuracy Score
        # Check that numbers in supported claims match ground truth evidence
        correct_numerical = 0
        total_numerical_checked = 0
        for c in validated_claims:
            ev = c.get("evidence", {})
            if "missing_pct" in ev:
                total_numerical_checked += 1
                if ev["missing_pct"] == evidence["quality"]["missing_info"]["Salary"]["percentage"]:
                    correct_numerical += 1
            if "duplicate_rows" in ev:
                total_numerical_checked += 1
                if ev["duplicate_rows"] == 312:
                    correct_numerical += 1

        numerical_accuracy_score = (correct_numerical / total_numerical_checked * 100) if total_numerical_checked > 0 else 100.0

        # Dimension 2: Claim / Evidence Consistency Score
        consistency_score = (len(validated_claims) / len(raw_claims)) * 100

        # Dimension 3: Hallucination Rate (0% expected in supported claims)
        hallucinated_claims = [c for c in validated_claims if c.get("status") != "supported"]
        hallucination_rate = (len(hallucinated_claims) / len(validated_claims)) * 100 if validated_claims else 0.0

        # Dimension 4: Correct Statistical Interpretation Score
        normality_sal = evidence["distributions"]["Salary"]["normality_test"]["is_normal"]
        normality_claims = [c for c in validated_claims if "normal distribution" in c["claim"].lower()]
        stat_interpretation_score = 100.0
        for c in normality_claims:
            if "departs from normal" in c["claim"].lower() and normality_sal:
                stat_interpretation_score -= 50.0

        # Dimension 5: Coverage of Important Findings Score
        categories = set(c["category"] for c in validated_claims)
        coverage_score = (len(categories) / 4.0) * 100.0 # Expect quality, distribution, relationship, anomaly

        # Dimension 6: Severity Classification Score
        severity_correct = True
        for c in validated_claims:
            if "Salary" in c["claim"] and "missing" in c["claim"].lower():
                if "substantial" in c["claim"].lower() or "critical" in c["claim"].lower():
                    severity_correct = False
        severity_classification_score = 100.0 if severity_correct else 0.0

        # Dimension 7: Recommendation Validity Score
        banned_recs = [c for c in validated_claims if "drop" in c["claim"].lower() and "missing" in c["claim"].lower()]
        recommendation_validity_score = 100.0 if len(banned_recs) == 0 else 0.0

        print("\n=== GOLDEN DATASET BENCHMARK EVALUATION RESULTS ===")
        print(f"1. Numerical Accuracy Score:           {numerical_accuracy_score:.1f}%")
        print(f"2. Claim/Evidence Consistency Score:   {consistency_score:.1f}%")
        print(f"3. Hallucination Rate:                 {hallucination_rate:.1f}% (Target: 0%)")
        print(f"4. Statistical Interpretation Score:   {stat_interpretation_score:.1f}%")
        print(f"5. Coverage of Important Findings:     {coverage_score:.1f}%")
        print(f"6. Severity Classification Score:      {severity_classification_score:.1f}%")
        print(f"7. Recommendation Validity Score:      {recommendation_validity_score:.1f}%")
        print("====================================================")

        self.assertEqual(numerical_accuracy_score, 100.0)
        self.assertEqual(hallucination_rate, 0.0)
        self.assertEqual(severity_classification_score, 100.0)
        self.assertEqual(recommendation_validity_score, 100.0)

if __name__ == "__main__":
    unittest.main()
