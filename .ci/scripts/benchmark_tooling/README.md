# Benchmark Tooling

A library providing tools for benchmarking ExecutorchBenchmark data.

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## Tools

### get_benchmark_analysis_data.py

This script fetches benchmark data from HUD Open API and processes it, grouping metrics by private and public devices.
## Quick start

generates the matching_list json:
```
python get_benchmark_analysis_data.py get_matching_list \
  --startTime 2025-06-11T00:00:00 \
  --endTime 2025-06-17T00:00:00 \
  --category private_mv3_iphone15 \
  --filter "include=private,mv3;"\
  --outputType json
```

if everything looks good, generate the private csv output:
```
python3 get_benchmark_analysis_data.py generate_data \
--startTime "2025-06-11T00:00:00" \
--endTime "2025-06-17T18:00:00" \
--private-matching-json-path "./private_mv3_iphone15.json" --outputType csv \
--includePublic false
```


#### Generate Benchmark Data

```bash
python get_benchmark_analysis_data.py generate_data \
  --startTime 2025-06-11T00:00:00 \
  --endTime 2025-06-17T18:00:00
```

##### Options:
- `--silent`: Hide processing logs, show only results
- `--outputType df`: Display results in DataFrame format
- `--outputType print`: Display results in dictionary format
- `--outputType json --outputDir "/path/to/dir"`: Generate JSON file 'benchmark_results.json'
- `--outputType csv --outputDir "/path/to/dir"`: Generate CSV files in folders (`private` and `public`)

#### Get Matching Lists

The `get_matching_list` command allows you to filter benchmark data based on specific criteria.

##### Get All Matching Lists
```bash
python get_benchmark_analysis_data.py get_matching_list \
  --startTime 2025-06-11T00:00:00 \
  --endTime 2025-06-17T00:00:00 \
  --category all \
  --outputType json
```

##### Get Private Device Matching Lists
```bash
python get_benchmark_analysis_data.py get_matching_list \
  --startTime 2025-06-11T00:00:00 \
  --endTime 2025-06-17T00:00:00 \
  --category private \
  --filter "include=private;"
```

##### Get Public Device Matching Lists
```bash
python get_benchmark_analysis_data.py get_matching_list \
  --startTime 2025-06-11T00:00:00 \
  --endTime 2025-06-17T00:00:00 \
  --category public \
  --filter "exclude=private;"
```

##### Advanced Filtering Examples
Filter for specific models and devices:
```bash
# Get all mv3 models on iPhone 15 except apple_iphone_15_plus
python get_benchmark_analysis_data.py get_matching_list \
  --startTime 2025-06-11T00:00:00 \
  --endTime 2025-06-17T00:00:00 \
  --category private_mv3_iphone5 \
  --filter "include=private,mv3,iphone_15;exclude=apple_iphone_15_plus"
```

Multiple filters (using union logic):
```bash
# Get both mv3 and resnet50 models on iPhone 15 except apple_iphone_15_plus
python get_benchmark_analysis_data.py get_matching_list \
  --startTime 2025-06-11T00:00:00 \
  --endTime 2025-06-17T00:00:00 \
  --category private_models_iphone15 \
  --filter "include=private,mv3,iphone_15;exclude=apple_iphone_15_plus" \
  --filter "include=private,resnet50,iphone_15;exclude=apple_iphone_15_plus"
```

##### Output Options
- `--outputType json --outputDir "/path/to/dir"`: Generate JSON file '{category}.json'

#### Python API Usage

To use the benchmark fetcher in your own scripts:

```python
from benchmark_tooling.get_benchmark_analysis_data import ExecutorchBenchmarkFetcher

# Initialize the fetcher
fetcher = ExecutorchBenchmarkFetcher()

# Fetch data for a specific time range
fetcher.run(
    "2025-06-11T00:00:00",
    "2025-06-17T00:00:00",
    private_device_matching_list,
    public_device_matching_list
)

# Get results as DataFrames
private_dfs, public_dfs = fetcher.toDataFrame()

# Export results to Excel
fetcher.output_data(OutputType.CSV, (output_dir="./results")
```

### analyze_benchmark_stability.py

This script analyzes the stability of benchmark data, comparing the results of private and public devices.

```bash
python analyze_benchmark_stability.py \
    "Benchmark Dataset with Private AWS Devices.xlsx" \
    --reference_file "Benchmark Dataset with Public AWS Devices.xlsx"
```
