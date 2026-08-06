import os
import shutil
import tempfile
import unittest
import pandas as pd
import numpy as np

from app.core.profiler import run_profiling, validate_luhn, detect_pii_shield, compute_quality_score
from app.core.cleaner import run_cleaning
from app.core.feature_engineer import run_feature_engineering
from app.core.analyst import run_statistical_analysis, detect_simpsons_paradox, rank_prioritized_insights
from app.core.advanced_analyst import run_ml_readiness

class TestAnalystValueFeatures(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.sample_csv = os.path.join(self.test_dir, "sample.csv")
        
        np.random.seed(42)
        n = 100
        dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
        age = np.random.normal(35, 10, size=n)
        income = np.random.normal(50000, 15000, size=n)
        
        # Add PII sample columns (Email, Credit Card, Phone)
        emails = [f"user{i}@example.com" for i in range(n)]
        cards = ["4532015112830366" if i % 2 == 0 else "4532015112830367" for i in range(n)] # Luhn valid cards
        phones = [f"+15550100{i:02d}" for i in range(n)]
        
        gender = ["Male" if i % 2 == 0 else "male" for i in range(n)] # Category variant typo
        sentinels = [-999 if i == 0 else i for i in range(n)] # Sentinel value
        purchased = np.random.choice([0, 1], size=n)
        
        df = pd.DataFrame({
            "Date": dates,
            "Age": age,
            "Income": income,
            "Email": emails,
            "CreditCard": cards,
            "Phone": phones,
            "Gender": gender,
            "SentinelCol": sentinels,
            "Target": purchased
        })
        df.to_csv(self.sample_csv, index=False)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_luhn_credit_card_validation(self):
        self.assertTrue(validate_luhn("4532015112830366"))
        self.assertFalse(validate_luhn("1234567890123456"))

    def test_pii_detection_shield_and_confidence(self):
        df = pd.read_csv(self.sample_csv)
        pii = detect_pii_shield(df)
        detected_types = [p["type"] for p in pii]
        self.assertIn("Email Address", detected_types)
        self.assertIn("Credit Card Number", detected_types)
        self.assertIn("Phone Number", detected_types)
        
        # Check HIGH confidence assignment
        email_pii = next(p for p in pii if p["type"] == "Email Address")
        self.assertEqual(email_pii["confidence"], "HIGH")

    def test_profiling_discoveries_and_quality_score(self):
        res = run_profiling(self.sample_csv)
        self.assertIn("quality_score", res)
        qs = res["quality_score"]
        self.assertGreaterEqual(qs["overall_score"], 0.0)
        self.assertLessEqual(qs["overall_score"], 100.0)
        self.assertIn(qs["grade"], ["A+", "A", "B", "C", "D", "F"])
        
        self.assertIn("Gender", res["category_variants"])
        self.assertIn("SentinelCol", res["sentinel_findings"])

    def test_simpsons_paradox_detection(self):
        # Construct synthetic dataset with Simpson's Paradox reversal
        # Overall positive correlation, but negative correlation within subgroups
        sub1 = pd.DataFrame({"X": np.linspace(1, 10, 20), "Y": np.linspace(10, 1, 20), "Z": "GroupA"})
        sub2 = pd.DataFrame({"X": np.linspace(20, 30, 20), "Y": np.linspace(30, 20, 20), "Z": "GroupB"})
        simp_df = pd.concat([sub1, sub2], ignore_index=True)
        
        reversals = detect_simpsons_paradox(simp_df, ["X", "Y"], ["Z"])
        self.assertGreater(len(reversals), 0)
        self.assertIn("SIMPSON'S PARADOX DETECTED", reversals[0]["headline"])

    def test_prioritized_insight_ranking_ci_lower_bound(self):
        res = run_statistical_analysis(self.sample_csv, target_col="Target")
        self.assertIn("top_insights", res)
        insights = res["top_insights"]
        if len(insights) >= 2:
            # Check descending L_CI order
            self.assertGreaterEqual(insights[0]["l_ci"], insights[1]["l_ci"])

if __name__ == "__main__":
    unittest.main()
