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
- `--no-silent`: Show processing logs (default: only show results & minimum logging)

##### Output Options:
- `--outputType`: Choose output format (default: "print")
  - `print`: Display results in console
  - `json`: Generate JSON file
  - `df`: Display results in DataFrame format: `{'private': List[{'groupInfo':Dict,'df': DF},...],'public':List[{'groupInfo':Dict,'df': DF}]`
  - `excel`: Generate Excel files with multiple sheets, the field in first row and first column contains the json string of the raw metadata
  - `csv`: Generate CSV files in separate folders, the field in first row and first column contains the json string of the raw metadata
- `--outputDir`: Directory to save output files (default: current directory)

##### Filtering Options:

- `--private-device-pools`: Filter by private device pool names (e.g., "samsung-galaxy-s22-5g", "samsung-galaxy-s22plus-5g")
- `--backends`: Filter by specific backend names (e.g.,  "qnn-q8" , ""llama3-spinquan)
- `--models`: Filter by specific model names (e.g "mv3" "meta-llama-llama-3.2-1b-instruct-qlora-int4-eo8")

#### Example Usage
call multiple private device pools and models:
this fetches all the private table data that has model `llama-3.2-1B` and `mv3`
```bash
python3 get_benchmark_analysis_data.py \
--startTime "2025-06-01T00:00:00" \
--endTime "2025-06-11T00:00:00" \
--private-device-pools 'apple_iphone_15_private' 'samsung_s22_private' \
--models 'meta-llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8' 'mv3'
```

this fetches all the private iphone table data that has model `llama-3.2-1B` and `mv3`, and associated public iphone
```bash
python3 get_benchmark_analysis_data.py \
--startTime "2025-06-01T00:00:00" \
--endTime "2025-06-11T00:00:00" \
--private-device-pools 'apple_iphone_15_private' \
--models 'meta-llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8' 'mv3'
```
#### Working with Output Files CSV and Excel

You can use methods in `common.py` to convert the file data back to DataFrame format, those methods read the first row in csv/excel file, and return result with format list of {"groupInfo":DICT, "df":df.Dataframe{}} format.

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

## Running Unit Tests

The benchmark tooling includes unit tests to ensure functionality.

### Using pytest

```bash
# From the executorch root directory
pytest -c /dev/null .ci/scripts/tests/test_get_benchmark_analysis_data.py
```
