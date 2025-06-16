
import requests
import pandas as pd
import json
from datetime import datetime
from read_benchmark_data import build_query_params

def fetch_llm_data(payload: dict) -> list:
    url = "https://hud.pytorch.org/api/clickhouse/oss_ci_benchmark_llms"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    return response.json()

def flatten_record(record):
    flat = {
        "timestamp": record.get("metadata_info", {}).get("timestamp"),
        "workflow_id": record.get("workflow_id"),
        "job_id": record.get("job_id"),
        "model": record.get("model"),
        "backend": record.get("backend"),
        "mode": record.get("mode"),
        "dtype": record.get("dtype"),
        "device": record.get("device"),
        "arch": record.get("arch"),
        "granularity_bucket": record.get("granularity_bucket"),
    }
    if "extra" in record:
        for k, v in record["extra"].items():
            flat[f"extra_{k}"] = v
    metric_name = record["metric"]
    flat[metric_name] = record["actual"]
    return flat

def process_records(data: list) -> pd.DataFrame:
    flattened = [flatten_record(entry) for entry in data]
    df = pd.DataFrame(flattened)
    id_cols = [
        "timestamp", "workflow_id", "job_id", "model", "backend", "mode", "dtype",
        "device", "arch", "granularity_bucket",
        "extra_use_torch_compile", "extra_is_dynamic",
        "extra_request_rate", "extra_tensor_parallel_size"
    ]
    df = df.groupby(id_cols, dropna=False).first().reset_index()
    return df

def main():
    props = {
        "archName": "All Platforms",
        "deviceName": "Samsung Galaxy S22 5G (private) (Android 13)",
        "modeName": "inference",
        "modelName": "mv3",
        "backendName": "qnn_q8",
        "benchmarkName": "",  # fallback to repo default
        "repoName": "pytorch/executorch",
        "granularity": "hour",
        "startTime": datetime(2025, 5, 23, 1, 1, 22),
        "stopTime": datetime(2025, 6, 6, 1, 1, 22),
        "branch": "main",
        "commit": "098c58e1adc082ad98ffd6efb41151736fbc1a12"
    }
    dtypes = [""]

    payload = build_query_params(props, dtypes)
    print("Query payload:")
    print(json.dumps(payload, indent=2))

    data = fetch_llm_data(payload)
    df = process_records(data)

    df.to_csv("llm_benchmark_result.csv", index=False)
    print(df.head())

if __name__ == "__main__":
    main()
