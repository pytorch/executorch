# Executorch Benchmark Tooling

A  library providing tools for fetching, processing, and analyzing ExecutorchBenchmark data from the HUD Open API. This tooling helps compare performance metrics between private and public devices with identical settings.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Tools](#tools)
  - [get_benchmark_analysis_data.py](#get_benchmark_analysis_datapy)
    - [Quick Start](#quick-start)
    - [Command Line Options](#command-line-options)
    - [Example Usage](#example-usage)
    - [Working with Output Files](#working-with-output-files-csv-and-excel)
    - [Python API Usage](#python-api-usage)
- [Running Unit Tests](#running-unit-tests)

## Overview

The Executorch Benchmark Tooling provides a suite of utilities designed to:

- Fetch benchmark data from HUD Open API for specified time ranges
- Clean and process data by filtering out failures
- Compare metrics between private and public devices with matching configurations
- Generate analysis reports in various formats (CSV, Excel, JSON)
- Support filtering by device pools, backends, and models

This tooling is particularly useful for performance analysis, regression testing, and cross-device comparisons.

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Tools

### get_benchmark_analysis_data.py

This script is mainly used to generate analysis data comparing private devices with public devices using the same settings.

It fetches benchmark data from HUD Open API for a specified time range, cleans the data by removing entries with FAILURE indicators, and retrieves all private device metrics along with equivalent public device metrics based on matching [model, backend, device_pool_names, arch] configurations. Users can filter the data by specifying private device_pool_names, backends, and models.

#### Quick Start

```bash
# generate excel sheets for all private devices with public devices using the same settings
python3 .ci/scripts/benchmark_tooling/get_benchmark_analysis_data.py \
  --startTime "2025-06-11T00:00:00" \
  --endTime "2025-06-17T18:00:00" \
  --outputType "excel"

# generate the benchmark stability analysis
python3 .ci/scripts/benchmark_tooling/analyze_benchmark_stability.py \
--primary-file private.xlsx \
--reference-file public.xlsx
```

#### Command Line Options

##### Basic Options:
- `--startTime`: Start time in ISO format (e.g., "2025-06-11T00:00:00") (required)
- `--endTime`: End time in ISO format (e.g., "2025-06-17T18:00:00") (required)
- `--env`: Choose environment ("local" or "prod", default: "prod")
- `--no-silent`: Show processing logs (default: only show results & minimum logging)

##### Output Options:
- `--outputType`: Choose output format (default: "print")
  - `print`: Display results in console
  - `json`: Generate JSON file
  - `df`: Display results in DataFrame format: `{'private': List[{'groupInfo':Dict,'df': DF},...],'public':List[{'groupInfo':Dict,'df': DF}]`
  - `excel`: Generate Excel files with multiple sheets, the field in first row and first column contains the JSON string of the raw metadata
  - `csv`: Generate CSV files in separate folders, the field in first row and first column contains the JSON string of the raw metadata
- `--outputDir`: Directory to save output files (default: current directory)

##### Filtering Options:

- `--device-pools`: Filter by device pool names (e.g., "apple_iphone_15_private", "samsung_s22_private")
- `--backends`: Filter by specific backend names (e.g.,"xnnpack_q8")
- `--models`: Filter by specific model names (e.g., "mv3", "meta-llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8")

#### Example Usage

Filter by multiple private device pools and models:
```bash
# This fetches all private table data for models 'llama-3.2-1B' and 'mv3'
python3 .ci/scripts/benchmark_tooling/get_benchmark_analysis_data.py \
  --startTime "2025-06-01T00:00:00" \
  --endTime "2025-06-11T00:00:00" \
  --device-pools 'apple_iphone_15_private' 'samsung_s22_private' \
  --models 'meta-llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8' 'mv3'
```

Filter by specific device pool and models:
```bash
# This fetches all private iPhone table data for models 'llama-3.2-1B' and 'mv3',
# and associated public iPhone data
python3 .ci/scripts/benchmark_tooling/get_benchmark_analysis_data.py \
  --startTime "2025-06-01T00:00:00" \
  --endTime "2025-06-11T00:00:00" \
  --device-pools 'apple_iphone_15_private' \
  --models 'meta-llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8' 'mv3'
```

#### Working with Output Files CSV and Excel

You can use methods in `common.py` to convert the file data back to DataFrame format. These methods read the first row in CSV/Excel files and return results with the format `list of {"groupInfo":DICT, "df":df.Dataframe{}}`.

```python
import logging
logging.basicConfig(level=logging.INFO)
from .ci.scripts.benchmark_tooling.common import read_all_csv_with_metadata, read_excel_with_json_header

# For CSV files (assuming the 'private' folder is in the current directory)
folder_path = './private'
res = read_all_csv_with_metadata(folder_path)
logging.info(res)

# For Excel files (assuming the Excel file is in the current directory)
file_path = "./private.xlsx"
res = read_excel_with_json_header(file_path)
logging.info(res)
```

#### Python API Usage

To use the benchmark fetcher in your own scripts:

```python
from .ci.scripts.benchmark_tooling.get_benchmark_analysis_data import ExecutorchBenchmarkFetcher

# Initialize the fetcher
fetcher = ExecutorchBenchmarkFetcher(env="prod", disable_logging=False)

# Fetch data for a specific time range
fetcher.run(
    start_time="2025-06-11T00:00:00",
    end_time="2025-06-17T18:00:00"
)

# Use the output_data method for flexible output
results = fetcher.output_data(output_type="excel", output_dir="./results")
```

## Running Unit Tests

The benchmark tooling includes unit tests to ensure functionality.

### Using pytest for unit tests

```bash
# From the executorch root directory
pytest -c /dev/null .ci/scripts/tests/test_get_benchmark_analysis_data.py
```
