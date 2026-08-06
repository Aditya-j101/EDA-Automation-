import os
import shutil
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from langchain_core.messages import AIMessage

from app.graph.orchestrator import create_eda_graph

class TestDeterministicPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.sample_csv = os.path.join(self.test_dir, "test_data.csv")
        
        np.random.seed(42)
        n = 50
        dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
        sales = np.random.normal(100, 15, size=n)
        sales[2] = np.nan
        region = np.random.choice(["North", "South"], size=n)
        
        df = pd.DataFrame({"Date": dates, "Sales": sales, "Region": region})
        os.makedirs("data", exist_ok=True)
        df.to_csv(self.sample_csv, index=False)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("app.agents.reporter.reporter_node", return_value={"messages": [AIMessage(content="Final report generated.")]})
    def test_full_pipeline_stream(self, mock_reporter):
        app = create_eda_graph()
        
        initial_state = {
            "source_config": {"type": "csv", "path": self.sample_csv},
            "dataset_path": "",
            "messages": [],
            "errors": [],
            "current_step": 0,
            "retries": 0,
        }
        
        executed_nodes = []
        for event in app.stream(initial_state):
            for node_name, node_state in event.items():
                executed_nodes.append(node_name)
                
        expected_nodes = [
            "ingestion", "profiler", "data_quality_agent", "distribution_agent",
            "relationship_agent", "cleaner", "feature_engineer", "analyst",
            "advanced_analyst", "timeseries_analyst", "visualizer", "claim_generator",
            "validator", "reporter"
        ]
        self.assertEqual(executed_nodes, expected_nodes)
        
        # Verify output files generated
        self.assertTrue(os.path.exists("data/cleaned_data.csv"))
        self.assertTrue(os.path.exists("data/engineered_data.csv"))

if __name__ == "__main__":
    unittest.main()
