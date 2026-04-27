# ExecuTorch Executor Runner for Espressif ESP32/ESP32-S3

> **Warning:** This example is not tested in CI. Use at your own risk.

This example demonstrates how to run an ExecuTorch model on Espressif ESP32 and
ESP32-S3 microcontrollers. It is based on the
[Arm Cortex-M executor runner](../arm/executor_runner/) and adapted for the
ESP-IDF build system and ESP32 memory architecture.

## Supported Targets

| Chip     | CPU           | Internal SRAM | PSRAM (optional) |
|----------|---------------|---------------|------------------|
| ESP32    | Xtensa LX6 (dual-core, 240MHz) | ~520KB | 4-8MB |
| ESP32-S3 | Xtensa LX7 (dual-core, 240MHz) | ~512KB | 2-32MB (Octal) |

## Prerequisites

1. **ESP-IDF v5.1+**: Install the ESP-IDF toolchain following the
   [official guide](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/).

2. **ExecuTorch**: Clone and set up ExecuTorch:
   ```bash
   git clone https://github.com/pytorch/executorch.git
   cd executorch
   pip install -e .
   ```

3. **Cross-compiled ExecuTorch libraries**: Build ExecuTorch for the ESP32
   target. See the [Cross-Compilation](#cross-compiling-executorch) section.

4. **A .pte model file**: Export a PyTorch model to the ExecuTorch `.pte`
   format. For small models suitable for ESP32, consider:
   - A simple add/multiply model
   - MobileNet V2 (quantized, with PSRAM)
   - Custom small models

## Project Structure

```
examples/espressif/
├── README.md                    # This file
├── build.sh                     # Build helper script
├── executor_runner/
│   ├── CMakeLists.txt           # Component/standalone CMake build
│   ├── esp_executor_runner.cpp  # Main executor runner
│   ├── esp_memory_allocator.h   # Custom memory allocator
│   ├── esp_memory_allocator.cpp
│   ├── esp_perf_monitor.h       # Performance monitoring
│   ├── esp_perf_monitor.cpp
│   └── pte_to_header.py         # Convert .pte to C header
└── project/
    ├── CMakeLists.txt           # ESP-IDF project file
    ├── sdkconfig.defaults       # Default ESP-IDF configuration
    ├── sdkconfig.defaults.esp32s3  # ESP32-S3 specific config
    ├── partitions.csv  # Example partition table; adjust app partition size for your board and model
    └── main/
        ├── CMakeLists.txt       # Main component
        └── main.cpp             # Entry point
```

## Quick Start

The following example has been tested only on an ESP32-S3 dev board with 8 MB of Octal PSRAM. You may need to adjust the `sdkconfig` file for your specific board.

### 1. Export a simple model

```python
import torch
from executorch.exir import to_edge

class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return x + x

model = SimpleModel()
example_input = (torch.randn(1, 8),)

# Export to ExecuTorch
exported = torch.export.export(model, example_input)
edge = to_edge(exported)
et_program = edge.to_executorch()

with open("simple_add.pte", "wb") as f:
    f.write(et_program.buffer)
```

### 2. Convert the model to a C header

```bash
python3 examples/espressif/executor_runner/pte_to_header.py \
    --pte simple_add.pte \
    --outdir examples/espressif/project/
```

### 3. Build with ESP-IDF

```bash
# Source ESP-IDF environment
. $IDF_PATH/export.sh

# Using the build script:
./examples/espressif/build.sh --target esp32s3 --pte simple_add.pte

# Or manually:
cd examples/espressif/project
idf.py set-target esp32s3
idf.py build
```

### 4. Flash and Monitor

```bash
cd examples/espressif/project
idf.py -p /dev/ttyUSB0 flash monitor
```

You should see output like:
```
Starting executorch runner !
I [executorch:esp_executor_runner.cpp:237 et_pal_init()] ESP32 ExecuTorch runner initialized. Free heap: 6097812 bytes.
I [executorch:esp_executor_runner.cpp:242 et_pal_init()] PSRAM available. Free PSRAM: 5764716 bytes.
I [executorch:esp_executor_runner.cpp:1047 executor_runner_main()] PTE @ 0x3c05f9f0 [----ET12]
I [executorch:esp_executor_runner.cpp:568 runner_init()] PTE Model data loaded. Size: 952 bytes.
I [executorch:esp_executor_runner.cpp:583 runner_init()] Model buffer loaded, has 1 methods
I [executorch:esp_executor_runner.cpp:593 runner_init()] Running method forward
I [executorch:esp_executor_runner.cpp:604 runner_init()] Setup Method allocator pool. Size: 2097152 bytes.
I [executorch:esp_executor_runner.cpp:620 runner_init()] Setting up planned buffer 0, size 64.
I [executorch:esp_executor_runner.cpp:716 runner_init()] Method 'forward' loaded.
I [executorch:esp_executor_runner.cpp:718 runner_init()] Preparing inputs...
I [executorch:esp_executor_runner.cpp:780 runner_init()] Input prepared.
I [executorch:esp_executor_runner.cpp:979 run_model()] Starting running 1 inferences...
I [executorch:esp_perf_monitor.cpp:41 StopMeasurements()] Profiler report:
I [executorch:esp_perf_monitor.cpp:42 StopMeasurements()] Number of inferences: 1
I [executorch:esp_perf_monitor.cpp:43 StopMeasurements()] Total CPU cycles: 49545 (49545.00 per inference)
I [executorch:esp_perf_monitor.cpp:48 StopMeasurements()] Total wall time: 205 us (205.00 us per inference)
I [executorch:esp_perf_monitor.cpp:53 StopMeasurements()] Average inference time: 0.205 ms
I [executorch:esp_perf_monitor.cpp:59 StopMeasurements()] Free heap: 6097576 bytes
I [executorch:esp_perf_monitor.cpp:63 StopMeasurements()] Min free heap ever: 6097576 bytes
I [executorch:esp_executor_runner.cpp:999 run_model()] 1 inferences finished
I [executorch:esp_executor_runner.cpp:867 print_outputs()] 1 outputs: 
Output[0][0]: (float) 2.000000
Output[0][1]: (float) 2.000000
Output[0][2]: (float) 2.000000
Output[0][3]: (float) 2.000000
Output[0][4]: (float) 2.000000
Output[0][5]: (float) 2.000000
Output[0][6]: (float) 2.000000
Output[0][7]: (float) 2.000000

```

## Cross-Compiling ExecuTorch

ExecuTorch needs to be cross-compiled for the ESP32 target (Xtensa architecture).

### Using the ESP-IDF toolchain

```bash
# Set up the cross-compilation toolchain
export IDF_TARGET=esp32s3  # or esp32

# Configure ExecuTorch build for ESP32
#Make sure to adjust the list of ops for your model or alter to use one of the selective build methods
cmake --preset esp-baremetal -B cmake-out-esp \
    -DCMAKE_TOOLCHAIN_FILE=$IDF_PATH/tools/cmake/toolchain-${IDF_TARGET}.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_DEVTOOLS=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=OFF \
    -DEXECUTORCH_SELECT_OPS_LIST="aten::add.out," \
    .

cmake --build cmake-out-esp -j$(nproc)
cmake --build cmake-out-esp --target install
```

## Memory Considerations

### ESP32 (no PSRAM)
- Total available SRAM: ~520KB (shared between code and data)
- Recommended method allocator pool: 128-256KB
- Recommended scratch pool: 64-128KB
- **Only very small models will fit!**

### ESP32 / ESP32-S3 with PSRAM
- Internal SRAM: ~512KB (used for code and fast data)
- PSRAM: 2-32MB (used for model data and large buffers)
- Recommended method allocator pool: 1-4MB
- Recommended scratch pool: 256KB-1MB

### Configuring Memory Pools

Memory pool sizes auto-adjust based on PSRAM availability. Override with:

```cmake
# In your project CMakeLists.txt or via idf.py menuconfig
set(ET_ESP_METHOD_ALLOCATOR_POOL_SIZE "1048576")    # 1MB
set(ET_ESP_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE "524288") # 512KB
```

Or as compile definitions:
```bash
idf.py build -DET_ESP_METHOD_ALLOCATOR_POOL_SIZE=1048576
```

## Loading Models

### Compiled-in (default)
The model `.pte` file is converted to a C array and compiled into the firmware.
This is the simplest approach but increases firmware size.

### Filesystem (SPIFFS/LittleFS)
For larger models, load from the filesystem at runtime:

1. Add `-DFILESYSTEM_LOAD=ON` to your build
2. Create a SPIFFS partition with your model:
   ```bash
   # Add to partitions.csv:
   # storage, data, spiffs, , 0x200000
   
   # Create and flash SPIFFS image:
   $IDF_PATH/components/spiffs/spiffsgen.py 0x200000 model_dir spiffs.bin
   esptool.py write_flash 0x210000 spiffs.bin
   ```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `ET_NUM_INFERENCES` | 1 | Number of inference runs |
| `ET_LOG_DUMP_INPUT` | OFF | Log input tensor values |
| `ET_LOG_DUMP_OUTPUT` | ON | Log output tensor values |
| `ET_BUNDLE_IO` | OFF | Enable BundleIO test support |
| `ET_EVENT_TRACER_ENABLED` | OFF | Enable ETDump profiling |
| `FILESYSTEM_LOAD` | OFF | Load model from filesystem |
| `ET_ESP_METHOD_ALLOCATOR_POOL_SIZE` | Auto | Method allocator size |
| `ET_ESP_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE` | Auto | Scratch allocator size |

## Differences from the Arm Example

| Feature | Arm (Cortex-M) | ESP32/ESP32-S3 |
|---------|----------------|----------------|
| Build system | Bare-metal CMake + Arm toolchain | ESP-IDF (FreeRTOS-based) |
| NPU | Ethos-U55/U65/U85 | None (CPU only) |
| Memory | ITCM/DTCM/SRAM/DDR via linker script | IRAM/DRAM/PSRAM via ESP-IDF |
| Performance monitor | ARM PMU + Ethos-U PMU | CPU cycle counter + esp_timer |
| Semihosting | FVP simulator filesystem access | SPIFFS/LittleFS/SD filesystem |
| Entry point | `main()` bare-metal | `app_main()` via FreeRTOS |
| Timing | ARM_PMU_Get_CCNTR() | esp_cpu_get_cycle_count() |

## Troubleshooting

### Model too large for flash
- Use filesystem loading (`FILESYSTEM_LOAD=ON`) with SPIFFS or SD card
- Quantize the model to reduce size
- Use a simpler/smaller model architecture

### Out of memory during inference
- Enable PSRAM if your board has it (`CONFIG_SPIRAM=y`)
- Increase memory pool sizes
- Use a smaller model
- Check `log_mem_status()` output for memory usage details

### Build errors with ExecuTorch libraries
- Ensure ExecuTorch was cross-compiled with the same ESP-IDF toolchain
- Check that `ET_BUILD_DIR_PATH` points to the correct build directory
- Verify the target architecture matches (Xtensa LX6 for ESP32, LX7 for ESP32-S3)

### Watchdog timer resets
- Long inference times may trigger the task watchdog
- Disable with `CONFIG_ESP_TASK_WDT_EN=n` in sdkconfig
- Or increase the timeout: `CONFIG_ESP_TASK_WDT_TIMEOUT_S=30`

