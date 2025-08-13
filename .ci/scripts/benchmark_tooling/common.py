import json
import os
from typing import Any, Dict, List

import pandas as pd


def read_excel_with_json_header(path: str) -> List[Dict[str, Any]]:
    # Read all sheets into a dict of DataFrames, without altering
    all_sheets = pd.read_excel(path, sheet_name=None, header=None, engine="openpyxl")

    results = []
    for sheet, df in all_sheets.items():
        # Extract JSON string from A1 (row 0, col 0)
        json_str = df.iat[0, 0]
        meta = json.loads(json_str) if isinstance(json_str, str) else {}

        # The actual data starts from the next row; treat row 1 as header
        df_data = pd.read_excel(path, sheet_name=sheet, skiprows=1, engine="openpyxl")
        results.append({"groupInfo": meta, "df": df_data, "sheetName": sheet})
    print(f"successfully fetched {len(results)} sheets from {path}")
    return results


def read_all_csv_with_metadata(folder_path: str) -> List[Dict[str, Any]]:
    results = []  # {filename: {"meta": dict, "df": DataFrame}}
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(".csv"):
            continue
        path = os.path.join(folder_path, fname)
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        try:
            meta = json.loads(first_line)
        except json.JSONDecodeError:
            meta = {}
        df = pd.read_csv(path, skiprows=1)
        results.append({"groupInfo": meta, "df": df, "sheetName": fname})
    print(f"successfully fetched {len(results)} sheets from {folder_path}")
    return results
