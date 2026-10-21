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

The statistics mode defaults to mode 1. Select mode 0 or 1 explicitly with:

```text
EXECUTORCH_VGF_NEURAL_STATISTICS_MODE=0
# or
EXECUTORCH_VGF_NEURAL_STATISTICS_MODE=1
```

The public values `0` and `1` map to Vulkan
`VK_NEURAL_ACCELERATOR_STATISTICS_MODE_STATISTICS0_ARM` and
`VK_NEURAL_ACCELERATOR_STATISTICS_MODE_STATISTICS1_ARM`, respectively. Invalid
mode values produce a warning and fall back to mode 1. Set these environment
variables before the VGF backend initializes its Vulkan device.

When statistics are requested, the backend enables
`VK_ARM_data_graph_neural_accelerator_statistics` only if both the device
extension and `dataGraphNeuralAcceleratorStatistics` feature are supported. If
support is missing, inference still proceeds and the emitted metadata reports
statistics as unavailable.

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
  "schema_version": 2,
  "backend": "VgfBackend",
  "api": "VK_ARM_data_graph",
  "event_name": "VGF_NEURAL_STATISTICS",
  "api_available": true,
  "data_available": true,
  "available": true,
  "reason": "",
  "statistics_mode": {
    "executorch_index": 1,
    "vulkan_value": 2,
    "vulkan_name": "VK_NEURAL_ACCELERATOR_STATISTICS_MODE_STATISTICS1_ARM"
  },
  "target": {
    "available": true,
    "device_name": "Mali-G2-Pro-NX MC1",
    "vendor_id": 5045,
    "device_id": 0,
    "driver_version": 0
  },
  "counter_layout": {
    "available": true,
    "schema_id": "arm.mali-g2.neural-statistics.mode1",
    "schema_version": 1,
    "endianness": "little",
    "word_type": "uint32",
    "words_per_task": 64,
    "leading_block_count": 1,
    "reason": ""
  },
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

ExecuTorch keeps `debug_database`, `statistics_info`, and
`statistics_memory` as raw/opaque blobs. Schema v2 adds normalized
`statistics_mode`, `target`, and `counter_layout` fields next to those blobs so
offline tooling does not have to infer the mode from counter-value patterns.

The registered counter layout currently covers Arm Mali-G2 targets. For an
unrecognized target, `counter_layout.available` is false rather than claiming
that the G2 schema applies. Inspector continues to accept legacy schema-v1
ETDumps, but v1 records do not contain enough information to identify the mode
without external knowledge.

## Reading from Inspector

```py
from executorch.devtools.inspector import Inspector

inspector = Inspector(etdump_path="run.etdump")
records = inspector.get_vgf_neural_statistics()

for record in records:
    print(record["schema_version"], record["data_available"])
    print(record.get("statistics_mode"))
    print(record.get("target"))
    print(record.get("counter_layout"))
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