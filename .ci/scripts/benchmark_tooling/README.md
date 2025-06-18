# Benchmark Tooling

A library providing tools for benchmarking ExecutorchBenchmark data.

## Read Benchmark Data
`get_benchmark_analysis_data.py` fetches benchmark data from HUD Open API, clean the data that only contains FAILURE_REPORT column,
and get all private device metrics and associated public device metrics if any based on [model,backend,device,ios]

### Quick Start

Install dependencies:
```bash
pip install -r requirements.txt
```

Run with default output (CLI):
```bash
python3 .ci/scripts/benchmark_tooling/get_benchmark_analysis_data.py --startTime "2025-06-11T00:00:00" --endTime "2025-06-17T18:00:00"
```

Additional options:
- `--silent`: Hide processing logs, show only results
- `--outputType df`: Display results in DataFrame format
- `--outputType excel --outputDir "{YOUR_LOCAL_DIRECTORY}"`: Generate Excel file with multiple sheets (`res_private.xlsx` and `res_public.xlsx`)
- `--outputType csv --outputDir "{YOUR_LOCAL_DIRECTORY}"`: Generate CSV files in folders (`private` and `public`)

### Python API Usage

To use the benchmark fetcher in your own scripts:

```python
import ExecutorchBenchmarkFetcher from benchmark_tooling.get_benchmark_analysis_data
fetcher = ExecutorchBenchmarkFetcher()
# Must call run first
fetcher.run()
private, public = fetcher.to_df()
```

## analyze_benchmark_stability.py
`analyze_benchmark_stability.py` analyzes the stability of benchmark data, comparing the results of private and public devices.

### Quick Start
Install dependencies:
```bash
pip install -r requirements.txt
```

```
python .ci/scripts/benchmark_tooling/analyze_benchmark_stability.py \
    Benchmark\ Dataset\ with\ Private\ AWS\ Devices.xlsx \
    --reference_file Benchmark\ Dataset\ with\ Public\ AWS\ Devices.xlsx
```
