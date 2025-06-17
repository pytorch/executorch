# Benchmark tooling
a lib providing tools for benchmarking
# read_benchmark_data.py
read benchmar data from HUD open api, the api returns grouped benchmark data based on execuTorch group values
## How to use it
install requirement packages
```
pip install benchamrk_tooling/requirements.txt
```

### run script manually
the script has fixed list of group settings.
Notice startTime and endTime must be UTC datetime string: "yyyy-mm-ddThh:mm:ss"

To run and display json format
```bash
python3 read_benchamark_data.py  --startTime "2025-06-11T00:00:00" --endTime "2025-06-17T18:00:00"
```

To run and display df format
```bash
python3 read_benchamark_data.py  --startTime "2025-06-11T00:00:00" --endTime "2025-06-17T18:00:00" --outputType 'df'
```

To run and generate execel sheets (this generated two excel file, one for private devices, and one for pulic devices):
```
python3 read_benchamark_data.py  --startTime "2025-06-11T00:00:00" --endTime "2025-06-17T18:00:00" --outputType 'excel' --excelDir "."

```

To use the class as part your script
```
fetcher = ExecutorchBenchmarkFetcher()
# must call run first
fetch.run()
private,public = fetcher.toDataFrame()

```
