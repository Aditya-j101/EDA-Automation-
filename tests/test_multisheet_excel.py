import os
import shutil
import tempfile
import unittest
import pandas as pd
import numpy as np

from app.tools.ingester import load_dataset, ingest_data
from app.core.evidence import build_structured_evidence
from app.core.profiler import run_profiling

class TestMultiSheetExcel(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.excel_path = os.path.join(self.test_dir, "multisheet_data.xlsx")
        
        # Create Excel file with multiple sheets:
        # Sheet 1: Cover/Metadata sheet (small/empty)
        # Sheet 2: Main Data sheet (100 rows x 4 cols)
        
        df_cover = pd.DataFrame({"Title": ["Report Overview"], "Date": ["2026-08-04"]})
        
        np.random.seed(42)
        df_data = pd.DataFrame({
            "Age": np.random.normal(30, 5, 100),
            "Income": np.random.normal(50000, 10000, 100),
            "Category": np.random.choice(["A", "B"], 100),
            "Target": np.random.choice([0, 1], 100)
        })
        
        with pd.ExcelWriter(self.excel_path, engine='openpyxl') as writer:
            df_cover.to_excel(writer, sheet_name="Cover", index=False)
            df_data.to_excel(writer, sheet_name="Data", index=False)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_multisheet_load_dataset_selects_primary_sheet(self):
        """Verifies load_dataset automatically selects the primary sheet with most data."""
        df = load_dataset(self.excel_path)
        self.assertEqual(df.shape, (100, 4))
        self.assertIn("Income", df.columns)

    def test_multisheet_ingest_data(self):
        """Verifies ingest_data successfully ingests multi-sheet Excel files."""
        out_csv = ingest_data({"type": "excel", "path": self.excel_path})
        self.assertTrue(os.path.exists(out_csv))
        df_ingested = pd.read_csv(out_csv)
        self.assertEqual(df_ingested.shape, (100, 4))

    def test_multisheet_build_structured_evidence(self):
        """Verifies build_structured_evidence runs without failure on multi-sheet Excel files."""
        evidence = build_structured_evidence(self.excel_path)
        self.assertEqual(evidence["schema"]["total_rows"], 100)
        self.assertEqual(evidence["schema"]["total_cols"], 4)

    def test_multisheet_run_profiling(self):
        """Verifies run_profiling runs without failure on multi-sheet Excel files."""
        result = run_profiling(self.excel_path)
        self.assertEqual(result["shape"], (100, 4))

if __name__ == "__main__":
    unittest.main()
