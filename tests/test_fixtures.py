import os
import uuid
import shutil
import tempfile
import unittest
import pandas as pd
import numpy as np

from app.core.profiler import run_profiling
from app.core.cleaner import run_cleaning
from app.core.feature_engineer import run_feature_engineering
from app.core.analyst import run_statistical_analysis
from app.core.advanced_analyst import run_ml_readiness

class TestFixturesAndMultiTenancy(unittest.TestCase):

    def setUp(self):
        self.fixture_path = os.path.join("tests", "fixtures", "dirty_dataset.csv")
        self.run_id = uuid.uuid4().hex
        self.workspace_dir = os.path.join("workspaces", self.run_id)
        os.makedirs(os.path.join(self.workspace_dir, "data"), exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.workspace_dir):
            shutil.rmtree(self.workspace_dir, ignore_errors=True)

    def test_fixture_profiling_and_target(self):
        res = run_profiling(self.fixture_path)
        self.assertEqual(res["target_col"], "Target")
        self.assertEqual(res["shape"], (10, 7))
        self.assertIn("HighNullCol", res["missing_info"])
        self.assertIn(">40%", res["missing_info"]["HighNullCol"]["level"])

    def test_fixture_cleaning_and_multi_tenant_isolation(self):
        res = run_cleaning(
            self.fixture_path,
            target_col="Target",
            workspace_dir=self.workspace_dir,
            run_id=self.run_id
        )
        
        # Verify multi-tenant path isolation
        expected_output = os.path.join(self.workspace_dir, "data", "cleaned_data.csv")
        self.assertEqual(res["output_path"], expected_output)
        self.assertTrue(os.path.exists(expected_output))
        
        # Verify missingness contract
        self.assertIn("HighNullCol", res["high_missing_cols"]) # >40% nulls preserved
        self.assertIn("Income_was_missing", res["missing_indicators_added"]) # 5% null indicator added
        
        cleaned_df = pd.read_csv(expected_output)
        self.assertEqual(cleaned_df["Income_was_missing"].iloc[4], 1)
        self.assertEqual(cleaned_df["Income_was_missing"].iloc[0], 0)
        self.assertTrue(cleaned_df["HighNullCol"].isnull().any())

    def test_fixture_feature_engineering(self):
        run_cleaning(self.fixture_path, target_col="Target", workspace_dir=self.workspace_dir, run_id=self.run_id)
        cleaned_path = os.path.join(self.workspace_dir, "data", "cleaned_data.csv")
        
        res = run_feature_engineering(cleaned_path, target_col="Target", workspace_dir=self.workspace_dir, run_id=self.run_id)
        expected_output = os.path.join(self.workspace_dir, "data", "engineered_data.csv")
        self.assertEqual(res["output_path"], expected_output)
        self.assertTrue(os.path.exists(expected_output))

    def test_fixture_structural_ml_preparation(self):
        df = pd.read_csv(self.fixture_path)
        res = run_ml_readiness(self.fixture_path, target_col="Target", workspace_dir=self.workspace_dir, run_id=self.run_id)
        self.assertIn("ml_prep_res", res)
        ml_res = res["ml_prep_res"]
        self.assertTrue(os.path.exists(ml_res["train_path"]))
        self.assertTrue(os.path.exists(ml_res["test_path"]))
        self.assertIn(self.run_id, ml_res["train_path"])

if __name__ == "__main__":
    unittest.main()
