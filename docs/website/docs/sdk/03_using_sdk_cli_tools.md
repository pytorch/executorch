# Library and CLI

There are 2 main ways of accessing the SDK: **SDK Library** and **Buck CLI**

> Note: If using AIBench, TensorBoard generation (via SDK) is already integrated. Read more [here](./06_aibench_sdk.md)

## SDK Library

The SDK Library provides functions for interacting with ETRecord, TensorBoard Visualization, and ETDB.

For a full walk through, please refer to this [notebook](https://www.internalfb.com/intern/anp/view/?id=3799219).

---

### Visualizer
```python
from executorch.sdk.fb import visualize_etrecord, visualize_etrecord_path

async def visualize_etrecord(
    etrecord: ETRecord,
    et_dump_path: Optional[str] = None,
    run_name: Optional[str] = None,
) -> str

async def visualize_etrecord_path(
    etrecord_path: str,
    et_dump_path: Optional[str] = None,
    run_name: Optional[str] = None,
) -> str
```

Generates an internal link to a TensorBoard instance visualizing the model graphs in the ETRecord.

If an `et_dump_path` is provided, associate the profiling information from the et_dump to the model_graphs. If a `run_name` is provided, use it to label the generated TB instance.

#### Parameters
- `etrecord/etrecord_path`: Instance/Path of the ETRecord to be visualized
- `et_dump_path`: Optional path to the ETDump to be associated with the ETRecord
- `run_name`: Optional name to associate with the TB instance. One will be genereated if not provided

#### Returns

- Path to an internal TB link

---

### ETDB

```python
from executorch.sdk.fb import debug_etrecord, debug_etrecord_path

async def debug_etrecord(
    etrecord: ETRecord, et_dump_path: Optional[str] = None, verbose: bool = False
)

async def debug_etrecord_path(
    etrecord_path: str, et_dump_path: Optional[str] = None, verbose: bool = False
)
```
Kicks off an interactive ETDB terminal instance for the model graphs in the ETRecord.

If an `et_dump_path` is provided, associate the profiling information from the et_dump to the model_graphs. If a `verbose` flag is provided, the terminal instance will be ran in verbose mode.

#### Parameters
- `etrecord/etrecord_path`: Instance/Path of the ETRecord to be visualized
- `et_dump_path`: Optional path to the ETDump to be associated with the ETRecord
- `verbose`: Optional flag to enable verbose printing.

#### Returns

- None

---

### ETRecord Helper
```python
from executorch.sdk.fb import ETRecord, generate_etrecord, parse_etrecord
```
See [ETRecord](./01_generating_etrecord.md) for more information

---
## Buck CLI

The Buck CLI is perfect for ad hoc visualization and debugging saved ETRecord's (and associated ETDump's). The CLI operates as a convenient wrapper for calling `visualize_etrecord_path` and `debug_etrecord_path`

``` bash
buck run //executorch/sdk/fb:cli ...

          <et_record>: Mandatory Path to ETRecord

          [--et_dump ET_DUMP]: Optional Path to ETDump
          [--run_name RUN_NAME]: Optional Name for TB Run
          [--terminal_mode]: Toggle for using ETDB instead of generating a TB Link
          [--verbose]: Toggle for verbose format when using ETDB
```

Example: Generate TB Visualization via CLI:
``` bash
buck run //executorch/sdk/fb:cli et_record.bin --et_dump et_dump.etdp
```

Example: Kickoff ETDB in Verbose mode via CLI:
``` bash
buck run //executorch/sdk/fb:cli et_record.bin --et_dump et_dump.etdp --terminal_mode --verbose
```
