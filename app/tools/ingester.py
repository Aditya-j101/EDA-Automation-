import pandas as pd
import os
import json
import logging
from typing import Dict, Any, Optional
from sqlalchemy import create_engine, text

def load_dataset(dataset_path: str, sheet_name: Optional[Any] = None) -> pd.DataFrame:
    """
    Robustly loads a CSV or Excel dataset into a pandas DataFrame.
    For multi-sheet Excel files:
    1. If sheet_name is explicitly specified, loads that sheet.
    2. If multiple non-empty sheets exist:
       a. If schemas (column sets) are identical across sheets, concatenates them into one DataFrame.
       b. Otherwise, automatically selects the primary sheet with the largest data area (most non-null values).
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at path: {dataset_path}")

    ext = os.path.splitext(dataset_path)[1].lower()
    
    if ext in ['.xlsx', '.xls']:
        try:
            excel_file = pd.ExcelFile(dataset_path)
            sheet_names = excel_file.sheet_names
            
            if not sheet_names:
                raise ValueError("Excel file contains no readable sheets.")
                
            if sheet_name is not None and (sheet_name in sheet_names or isinstance(sheet_name, int)):
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
            elif len(sheet_names) == 1:
                df = pd.read_excel(excel_file, sheet_name=0)
            else:
                # Multi-sheet Excel file handling
                sheets_dict = {}
                for name in sheet_names:
                    try:
                        sheet_df = pd.read_excel(excel_file, sheet_name=name)
                        if not sheet_df.empty:
                            sheets_dict[name] = sheet_df
                    except Exception as s_err:
                        logging.warning(f"Could not read sheet '{name}': {s_err}")
                        
                if not sheets_dict:
                    df = pd.read_excel(excel_file, sheet_name=0)
                elif len(sheets_dict) == 1:
                    df = list(sheets_dict.values())[0]
                else:
                    first_cols = set(list(sheets_dict.values())[0].columns)
                    all_same = all(set(s_df.columns) == first_cols for s_df in sheets_dict.values())
                    
                    if all_same:
                        logging.info(f"Multi-sheet Excel file with matching columns detected across {len(sheets_dict)} sheets ({list(sheets_dict.keys())}). Concatenating sheets...")
                        df = pd.concat(list(sheets_dict.values()), ignore_index=True)
                    else:
                        best_sheet = max(sheets_dict.keys(), key=lambda k: sheets_dict[k].notnull().sum().sum())
                        logging.info(f"Multi-sheet Excel file detected ({sheet_names}). Automatically selected primary data sheet '{best_sheet}' ({sheets_dict[best_sheet].shape[0]} rows x {sheets_dict[best_sheet].shape[1]} cols).")
                        df = sheets_dict[best_sheet]
                        
        except Exception as e:
            df = pd.read_excel(dataset_path)
    else:
        try:
            df = pd.read_csv(dataset_path)
        except Exception:
            df = pd.read_csv(dataset_path, encoding='latin1')

    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    return df


def ingest_data(source_config: Dict[str, Any], workspace_dir: Optional[str] = None) -> str:
    """
    Reads the data according to the source_config and normalizes it 
    into a standard CSV inside the workspace or data directory.
    Returns the path to the normalized CSV.
    """
    source_type = source_config.get("type", "csv").lower()
    
    if workspace_dir:
        data_dir = os.path.join(workspace_dir, "data")
        output_path = os.path.join(data_dir, "ingested_data.csv")
    else:
        output_path = "data/ingested_data.csv"
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        if source_type == "csv":
            path = source_config.get("path")
            df = load_dataset(path)
            
        elif source_type in ["excel", "xlsx", "xls"]:
            path = source_config.get("path")
            sheet = source_config.get("sheet_name")
            df = load_dataset(path, sheet_name=sheet)
            
        elif source_type == "json":
            path = source_config.get("path")
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                for key, val in data.items():
                    if isinstance(val, list):
                        data = val
                        break
            df = pd.json_normalize(data)
            
        elif source_type == "mysql":
            user = source_config.get("username")
            password = source_config.get("password")
            host = source_config.get("host", "localhost")
            port = source_config.get("port", "3306")
            database = source_config.get("database")
            table = source_config.get("table")
            
            connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
            engine = create_engine(connection_string)
            query = f"SELECT * FROM {table}"
            df = pd.read_sql(query, engine)
            
        else:
            path = source_config.get("path")
            if path:
                df = load_dataset(path)
            else:
                raise ValueError(f"Unsupported source type: {source_type}")

        if df.empty:
            raise ValueError("The ingested dataset is empty.")
        
        df.to_csv(output_path, index=False)
        return output_path
        
    except Exception as e:
        raise Exception(f"Ingestion Error ({source_type}): {str(e)}")
