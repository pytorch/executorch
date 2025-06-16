import pandas as pd

DEFAULT_ARCH_NAME = "All Platforms"
DEFAULT_DEVICE_NAME = "All Devices"
DEFAULT_MODE_NAME = "All Modes"
DEFAULT_MODEL_NAME = "All Models"
DEFAULT_BACKEND_NAME = "All Backends"

EXCLUDED_METRICS = [
    "load_status", "mean_itl_ms", "mean_tpot_ms", "mean_ttft_ms",
    "std_itl_ms", "std_tpot_ms", "std_ttft_ms",
    "cold_compile_time(s)", "warm_compile_time(s)",
    "speedup_pct", "generate_time(ms)"
]

REPO_TO_BENCHMARKS = {
    "pytorch/executorch": ["ExecuTorch"],
    # 其他 repo 可继续加
}

from datetime import datetime

def format_time(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

from typing import Dict, Any

def build_query_params(props: Dict[str, Any], dtypes: list) -> dict:
    return {
        "arch": "" if props["archName"] == DEFAULT_ARCH_NAME else props["archName"],
        "device": "" if props["deviceName"] == DEFAULT_DEVICE_NAME else props["deviceName"],
        "mode": "" if props["modeName"] == DEFAULT_MODE_NAME else props["modeName"],
        "dtypes": dtypes,
        "excludedMetrics": EXCLUDED_METRICS,
        "benchmarks": [props["benchmarkName"]] if props.get("benchmarkName") else REPO_TO_BENCHMARKS.get(props["repoName"], []),
        "granularity": props["granularity"],
        "models": [] if props["modelName"] == DEFAULT_MODEL_NAME else [props["modelName"]],
        "backends": [] if props["backendName"] == DEFAULT_BACKEND_NAME else [props["backendName"]],
        "repo": props["repoName"],
        "startTime": format_time(props["startTime"]),
        "stopTime": format_time(props["stopTime"]),
    }

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

    # Flatten extra
    if "extra" in record:
        for k, v in record["extra"].items():
            flat[f"extra_{k}"] = v

    # Add metric-specific value
    metric_name = record["metric"]
    flat[metric_name] = record["actual"]

    return flat

def process_records(data: list) -> pd.DataFrame:
    flattened = [flatten_record(entry) for entry in data]
    df = pd.DataFrame(flattened)

    # Group by workflow_id, job_id, timestamp
    id_cols = [
        "timestamp", "workflow_id", "job_id", "model", "backend", "mode", "dtype",
        "device", "arch", "granularity_bucket",
        "extra_use_torch_compile", "extra_is_dynamic",
        "extra_request_rate", "extra_tensor_parallel_size"
    ]

    df = df.groupby(id_cols, dropna=False).first().reset_index()

    return df
