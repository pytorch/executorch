# Benchmark Tooling

A library providing tools for fetching, processing, and analyzing ExecutorchBenchmark data from the HUD Open API.

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## Tools

### get_benchmark_analysis_data.py

This script fetches benchmark data from HUD Open API, cleans data that only contains FAILURE_REPORT columns, and retrieves all private device metrics and associated public device metrics based on [model, backend, device, arch].

#### Quick Start

```bash
python3 .ci/scripts/benchmark_tooling/get_benchmark_analysis_data.py \
  --startTime "2025-06-11T00:00:00" \
  --endTime "2025-06-17T18:00:00" \
  --outputType "csv"
```

#### Command Line Options

##### Basic Options:
- `--startTime`: Start time in ISO format (e.g., "2025-06-11T00:00:00") (required)
- `--endTime`: End time in ISO format (e.g., "2025-06-17T18:00:00") (required)
- `--env`: Choose environment ("local" or "prod", default: "prod")
- `--not-silent`: Show processing logs (default: only show results & minimum logging)

##### Output Options:
- `--outputType`: Choose output format (default: "print")
  - `print`: Display results in console
  - `json`: Generate JSON file
  - `df`: Display results in DataFrame format
  - `excel`: Generate Excel files with multiple sheets
  - `csv`: Generate CSV files in separate folders
- `--outputDir`: Directory to save output files (default: current directory)

##### Filtering Options:
- `--devices`: Filter by specific device names (e.g., "samsung-galaxy-s22-5g", "samsung-galaxy-s22plus-5g")
- `--backends`: Filter by specific backend names
- `--models`: Filter by specific model names

#### Working with Output Files

You can use methods in `common.py` to convert the file data back to DataFrame format:

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

# Get results in different formats
# As DataFrames
df_results = fetcher.to_df()

# Export to Excel
fetcher.to_excel(output_dir="./results")

# Export to CSV
fetcher.to_csv(output_dir="./results")

# Export to JSON
json_path = fetcher.to_json(output_dir="./results")

# Get raw dictionary results
dict_results = fetcher.to_dict()

# Use the output_data method for flexible output
results = fetcher.output_data(output_type="excel", output_dir="./results")
```

### analyze_benchmark_stability.py

This script analyzes the stability of benchmark data, comparing the results of private and public devices.

#### Quick Start

```bash
python .ci/scripts/benchmark_tooling/analyze_benchmark_stability.py \
    "Benchmark Dataset with Private AWS Devices.xlsx" \
    --reference_file "Benchmark Dataset with Public AWS Devices.xlsx"
```

## Running Unit Tests

The benchmark tooling includes comprehensive unit tests to ensure functionality.

### Using pytest

```bash
# From the executorch root directory
pytest -c /dev/null .ci/scripts/tests/test_get_benchmark_analysis_data.py
```
