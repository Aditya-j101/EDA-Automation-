import pandas as pd
import os
import json
from typing import Dict, Any
from sqlalchemy import create_engine, text

def ingest_data(source_config: Dict[str, Any]) -> str:
    """
    Reads the data according to the source_config and normalizes it 
    into a standard CSV for the EDA pipeline to process.
    Returns the path to the normalized CSV.
    """
    source_type = source_config.get("type", "csv").lower()
    
    # Ensure our data directory exists
    os.makedirs("data", exist_ok=True)
    output_path = "data/ingested_data.csv"
    
    try:
        if source_type == "csv":
            path = source_config.get("path")
            df = pd.read_csv(path)
            
        elif source_type == "excel":
            path = source_config.get("path")
            sheet = source_config.get("sheet_name", 0) # Defaults to first sheet
            df = pd.read_excel(path, sheet_name=sheet)
            
        elif source_type == "json":
            path = source_config.get("path")
            # We use json_normalize to automatically flatten nested JSON objects
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # If the JSON is a single dict with a list inside, or a direct list
            if isinstance(data, dict):
                # Attempt to find the first list in the dict
                for key, val in data.items():
                    if isinstance(val, list):
                        data = val
                        break
            df = pd.json_normalize(data)
            
        elif source_type == "mysql":
            # Config expects: username, password, host, port, database, table
            user = source_config.get("username")
            password = source_config.get("password")
            host = source_config.get("host", "localhost")
            port = source_config.get("port", "3306")
            database = source_config.get("database")
            table = source_config.get("table")
            
            # Secure connection string (using pymysql)
            connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
            engine = create_engine(connection_string)
            
            # Read-only query execution
            query = f"SELECT * FROM {table}"
            df = pd.read_sql(query, engine)
            
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

        # Basic Structural Validation
        if df.empty:
            raise ValueError("The ingested dataset is empty.")
        
        # Save to the normalized path
        df.to_csv(output_path, index=False)
        return output_path
        
    except Exception as e:
        raise Exception(f"Ingestion Error ({source_type}): {str(e)}")

