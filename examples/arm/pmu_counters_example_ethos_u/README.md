# Ethos-U PMU collection

This bare-metal example shows how to collect PMU counters using the Ethos-U backend.
PMU counters are used to profile events on the NPU, such as total NPU cycles, MAC cycles, and memory traffic events.

The example shows:
- How to link `ethosu_execution_stats` in application cmake to enable collection.
- How to configure what events to collect in application code.
- How to capture the PMU measurements in application code:
  - In an `EthosUStastBuffer` that can be accessed at runtime.
  - In an `etdump` that can be read after running the application.
  - Note that these modes can be enabled independently.
- How to parse the etdump using python.

## Running the example

We will compile for the default configuration: Ethos-U55-128 with `Shared_Sram`.

Install the export environment and target tools as described in
[the Arm README](../README.md), then run from the repository root:

```bash
source examples/arm/arm-scratch/setup_path.sh

python3 -m backends.arm.scripts.aot_arm_compiler \
  --model_name=mv2 \
  --target=ethos-u55-128 \
  --delegate \
  --quantize \
  --output=cmake-out-arm-pmu/mv2.pte

cmake -S examples/arm/pmu_counters_example_ethos_u -B cmake-out-arm-pmu \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/examples/arm/ethos-u-setup/arm-none-eabi-gcc.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DET_PTE_FILE_PATH="$PWD/cmake-out-arm-pmu/mv2.pte"
cmake --build cmake-out-arm-pmu --target arm_pmu_runner -j

# Will output etdump.bin to cwd.
backends/arm/scripts/run_fvp.sh \
  --elf=cmake-out-arm-pmu/arm_pmu_runner \
  --target=ethos-u55-128
```

The runner prints each delegate's PMU cycles and event values.

`cycles` is the total NPU cycle count. Event `97` is `ETHOSU_PMU_MAC_ACTIVE`
(MAC-active cycles), and event `112` is `ETHOSU_PMU_NPU_IDLE` (NPU-idle cycles).
`overflow` indicates whether the counter overflowed while counting.

`run_fvp.sh` automatically decodes the base64 UART payload into `etdump.bin`
in the current directory and runs the Inspector CLI to print its contents.

Decode the PMU metadata using [`parse_etdump.py`](parse_etdump.py), which uses
the programmatic Inspector API:

```bash
# Assumes etdump.bin is available in cwd.
python3 examples/arm/pmu_counters_example_ethos_u/parse_etdump.py
```

The decoded records include cycles, symbolic event IDs, counts, and overflow
flags. Counts from an FVP reflect its model; use hardware to measure device
performance.

## Implementation

This part describes how PMU collection was implemented in the example more in detail.

To enable PMU collection, link the runner against ethosu_execution_stats using whole-archive linking.
For etdump collection, additionally enable `EXECUTORCH_BUILD_ARM_ETDUMP` in cmake before adding ExecuTorch and link
the runner with etdump normally.
```cmake
executorch_target_whole_archive(arm_pmu_runner ethosu_execution_stats)
target_link_libraries(arm_pmu_runner PRIVATE etdump)
```


In the application, configure PMU events before inference using
`set_ethosu55_65_pmu_events()` or `set_ethosu85_pmu_events()` for the target. The NPU cycle counter is always collected separately.
Find all events in `ethosu_pmu_types.h` or online documentation.
U55/U65 support four slots and U85 supports eight; short lists disable unused
slots.
The example explicitly selects MAC-active and NPU-idle counters for both product
families:

```cpp
constexpr ethosu_pmu_event_type events[] = {
    ETHOSU_PMU_MAC_ACTIVE, ETHOSU_PMU_NPU_IDLE};
ET_CHECK_MSG(
    set_ethosu55_65_pmu_events(events) == Error::Ok,
    "Invalid U55/U65 events");
ET_CHECK_MSG(
    set_ethosu85_pmu_events(events) == Error::Ok, "Invalid U85 events");
```

For runtime access to PMU data, register caller-owned records before execution. Each record contains data from one Ethos-U invocation:

```cpp
EthosUExecutionRecord records[16];
EthosUStatsBuffer stats_buffer(records, 16);
set_ethosu_stats_buffer(&stats_buffer);

// Execute model...

set_ethosu_stats_buffer(nullptr);
printf("success: %u\n", unsigned(records[0].success));
printf("cycles: %" PRIu64 "\n", records[0].stats.cycle_count);
printf("cycle overflow: %u\n", unsigned(records[0].stats.cycle_overflow));
```

To collect PMU counters in etdump, pass `ETDumpGen` to `load_method()`. After execution, serialize it with `get_etdump_data()` and extract the artifact in a suitable manner.

To parse PMU counters in python, load the ETDump with `Inspector` and register
`parse_ethosu_delegate_pmu_metadata`. Read the decoded metadata from events
named `ethosu.pmu.counters`:

```python
inspector = Inspector(
    etdump_path="etdump.bin",
    source_time_scale=TimeScale.CYCLES,
    target_time_scale=TimeScale.CYCLES,
    delegate_metadata_parser=parse_ethosu_delegate_pmu_metadata,
)
for block in inspector.event_blocks:
    for event in block.events:
        if event.name == "ethosu.pmu.counters":
            print(event.delegate_debug_metadatas)
```

Alternatively, parse the raw metadata directly to an `EthosUPmuProfileEvent`
dataclass using `parse_pmu_profile_event`:

```python
record = parse_pmu_profile_event(event.raw_delegate_debug_metadatas[0])
print(record.cycle_count)
print(record.counters)
```
