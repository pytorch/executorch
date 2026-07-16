# VGF neural accelerator statistics in ETDump

The VGF backend emits neural accelerator profiling data through ETDump delegate
metadata when runtime profiling is enabled and a supported `VK_ARM_data_graph`
driver is available.

By default, VGF runtime profiling emits timing events only. Neural accelerator
statistics collection is opt-in because the payload can include binary blobs and
may increase ETDump size and profiling overhead, especially across repeated
inference runs.

To enable VGF neural accelerator statistics collection, set:

```text
EXECUTORCH_VGF_ENABLE_NEURAL_STATISTICS=1
```

When this option is not set, the backend does not emit the
VGF_NEURAL_STATISTICS delegate metadata event, even if general ETDump/runtime
profiling is enabled.

The emitted delegate profiling event name is:

```text
VGF_NEURAL_STATISTICS
```

The delegate metadata payload is a UTF-8 JSON wrapper. The wrapper schema is:

```json
{
  "schema": "executorch.vgf.neural_statistics",
  "schema_version": 1,
  "backend": "VgfBackend",
  "api": "VK_ARM_data_graph",
  "event_name": "VGF_NEURAL_STATISTICS",
  "api_available": true,
  "data_available": true,
  "available": true,
  "reason": "",
  "segments": []
}
```

Each segment can contain:
```json
{
  "segment_id": 0,
  "is_data_graph_pipeline": true,
  "statistics_bind_point_available": true,
  "statistics_memory_host_visible": true,
  "statistics_memory_host_coherent": true,
  "statistics_bind_point_reason": "",
  "debug_database": {
    "available": true,
    "is_text": false,
    "vulkan_result": 0,
    "size": 0,
    "encoding": "base64",
    "reason": "",
    "data": ""
  },
  "statistics_info": {
    "available": true,
    "is_text": true,
    "vulkan_result": 0,
    "size": 0,
    "encoding": "base64",
    "reason": "",
    "data": ""
  },
  "statistics_memory": {
    "available": true,
    "is_text": false,
    "vulkan_result": 0,
    "size": 0,
    "encoding": "base64",
    "reason": "",
    "data": ""
  }
}
```

We don't parse the neural accelerator blobs. Consumers should treat
debug_database, statistics_info, and statistics_memory as opaque bytes.

## Reading from Inspector

```py
from executorch.devtools.inspector import Inspector

inspector = Inspector(etdump_path="run.etdump")
records = inspector.get_vgf_neural_statistics()

for record in records:
    print(record["schema_version"], record["data_available"])
    for segment in record["segments"]:
        stats_bytes = segment["statistics_memory"]["raw_data"]
        debug_db_bytes = segment["debug_database"]["raw_data"]
```

If the Vulkan API, driver, or hardware support is unavailable, normal execution
continues and the JSON wrapper is still emitted with:
```json
{
  "api_available": false,
  "data_available": false,
  "available": false,
  "reason": "..."
}
```