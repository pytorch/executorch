# ExecuTorch on Espressif ESP32 and ESP32-S3

This directory runs ExecuTorch models on Espressif microcontrollers with an
executor runner built by ESP-IDF.

To add ExecuTorch to your own ESP-IDF project, use the
[ESP-IDF component](#esp-idf-component) instead of the runner.

## Supported Targets

| Chip     | CPU           | Internal SRAM | PSRAM (optional) |
|----------|---------------|---------------|------------------|
| ESP32    | Xtensa LX6 (dual-core, 240MHz) | ~520KB | 4-8MB |
| ESP32-S3 | Xtensa LX7 (dual-core, 240MHz) | ~512KB | 2-32MB (Octal) |

## Project Structure

```
examples/espressif/
├── README.md                    # This file
├── setup.sh                     # Install ESP-IDF and esp-emulator
├── build.sh                     # Build helper script
├── export_smoke_model.py        # Export bundled add/multiply test data
├── export_espnn_test_programs.py  # Export ESP-NN delegate test programs
├── quantize_and_lower.py        # Quantize a torchvision model for ESP-NN
├── test_emulator.py             # Check a smoke-test emulator log
├── executor_runner/             # The executor runner and .pte-to-header tool
├── project/                     # ESP-IDF project that builds the runner
├── platform/esp_pal.cpp         # ESP-IDF platform layer (logging, timing)
├── camera_stream/               # Live camera classification on ESP32-S3
├── build_idf_component.sh       # Generate the ESP-IDF component
├── idf_component_template/      # The component's sources and README
├── idf_component_smoke/         # Minimal component consumer for CI
└── idf_component_consumer/      # Component consumer for the CI matrix
```

## Prerequisites

1. **ESP-IDF 6.0 or newer**: `setup.sh` installs 6.1 (see
   [Emulator Smoke Test](#emulator-smoke-test)); for another release, follow the
   [official guide](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/).
2. **ExecuTorch**:
   ```bash
   git clone https://github.com/pytorch/executorch.git
   cd executorch
   ./install_executorch.sh
   ```
3. **Cross-compiled ExecuTorch libraries**: see
   [Cross-Compiling ExecuTorch](#cross-compiling-executorch).
4. **A `.pte` model**: small models such as an add/multiply model, or a
   quantized MobileNet V2 with PSRAM.

## Quick Start

Tested on an ESP32-S3 dev board with 8 MB of octal PSRAM. Other boards may need
`sdkconfig` changes.

### 1. Export a simple model

```python
import torch
from executorch.exir import to_edge

class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return x + x

exported = torch.export.export(SimpleModel(), (torch.randn(1, 8),))
et_program = to_edge(exported).to_executorch()

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
. $IDF_PATH/export.sh

# With the build script:
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

The output ends with the profiler report and the outputs:

```
I [executorch:esp_perf_monitor.cpp:53 StopMeasurements()] Average inference time: 0.205 ms
I [executorch:esp_executor_runner.cpp:999 run_model()] 1 inferences finished
I [executorch:esp_executor_runner.cpp:867 print_outputs()] 1 outputs:
Output[0][0]: (float) 2.000000
...
Output[0][7]: (float) 2.000000
```

The profiler's wall time covers the whole inference loop, including input
preparation and tracing setup.

## Cross-Compiling ExecuTorch

The `esp-baremetal` preset compiles CMake's header and type checks as static
libraries, since they cannot link without the ESP-IDF project's linker
configuration.

```bash
export IDF_TARGET=esp32s3  # or esp32

# Adjust the op list for your model, or use another selective build method.
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

## Emulator Smoke Test

[esp-emulator](https://github.com/espressif/esp-emulator) runs the ESP32-S3
firmware without a board; it does not support the original ESP32. These steps
export a model with two inputs and two outputs (`x + y` and `x * y`) and its
PyTorch reference outputs, then run the runner with 8 MB of octal PSRAM. Run
them from the repository root with your ExecuTorch environment active. Each
step can be rerun on its own.

### 1. Install tools

```bash
bash examples/espressif/setup.sh
source ~/.cache/executorch-espressif/setup_path.sh
```

[setup.sh](setup.sh) installs ESP-IDF 6.1 and esp-emulator at pinned versions,
and installs ESP-IDF's Python dependencies into your active environment. In a
new shell, activate the same environment and source `setup_path.sh` again.
Options:

- `--minimal`: only the Xtensa compiler, the SDK's Python packages and the
  emulator, as CI uses. The full install also needs your platform's USB
  libraries (`libusb-1.0-0` on Ubuntu) for OpenOCD.
- `--skip-idf`, `--skip-emulator`: install only one of them.
- `--targets=<chip>,...`: compilers for other targets.
- `--idf-version=<tag>`: another ESP-IDF release, such as `v6.0.1`; use a separate
  `ESPRESSIF_TOOLS_DIR` for it.
- `ESPRESSIF_TOOLS_DIR` picks the install directory; `IDF_PATH` and
  `IDF_TOOLS_PATH` reuse an existing ESP-IDF.

### 2. Export the test model

```bash
python examples/espressif/export_smoke_model.py --output cmake-out-esp-emulator/smoke.bpte
```

The `.bpte` holds the program, test inputs and expected outputs.

### 3. Build the runtime

```bash
cmake --preset esp-baremetal -G Ninja -B cmake-out-esp-emulator/runtime \
    -DCMAKE_TOOLCHAIN_FILE="$IDF_PATH/tools/cmake/toolchain-esp32s3.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_DEVTOOLS=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=OFF \
    -DEXECUTORCH_SELECT_OPS_LIST=aten::add.out,aten::mul.out
cmake --build cmake-out-esp-emulator/runtime
cmake --install cmake-out-esp-emulator/runtime
```

Set `CMAKE_BUILD_PARALLEL_LEVEL` to limit build parallelism.

### 4. Build the firmware

```bash
idf.py -C examples/espressif/project -B "$PWD/cmake-out-esp-emulator/firmware" \
    -DIDF_TARGET=esp32s3 \
    -DSDKCONFIG="$PWD/cmake-out-esp-emulator/sdkconfig" \
    -DET_BUILD_DIR_PATH="$PWD/cmake-out-esp-emulator/runtime" \
    -DET_PTE_FILE_PATH="$PWD/cmake-out-esp-emulator/smoke.bpte" \
    -DET_BUNDLE_IO=ON -DET_NUM_INFERENCES=3 \
    -DET_ATOL=0.000001 -DET_RTOL=0.000001 reconfigure
cmake --build cmake-out-esp-emulator/firmware
idf.py -C examples/espressif/project -B "$PWD/cmake-out-esp-emulator/firmware" merge-bin
```

For incremental builds, rerun the build and merge commands; reconfigure only
when changing CMake options.

### 5. Run the emulator

```bash
esp-emu --chip esp32s3 \
    --firmware cmake-out-esp-emulator/firmware/merged-binary.bin \
    --elf cmake-out-esp-emulator/firmware/executorch_esp_runner.elf \
    --net user --psram-size 8M --exit-on 'Program complete.'
```

### 6. Check the result

Capture the emulator's output with `> emulator.log 2>&1`, then:

```bash
python examples/espressif/test_emulator.py emulator.log
```

It checks platform initialization, PSRAM, that every requested inference
completed, the bundled outputs, and firmware completion, so it also catches
runs the emulator's `--timeout` ends early.

[test_espressif.sh](../../.ci/scripts/test_espressif.sh) runs these steps in CI,
plus the ESP-NN delegate's fixtures with optimized and ANSI C reference
kernels. It covers execution and ESP-IDF integration on ESP32-S3, not
performance or peripherals on real boards.

## Memory Considerations

The runner requires memory-planned inputs and outputs: export with
`MemoryPlanningPass(alloc_graph_input=True, alloc_graph_output=True)`, the
default.

| Board | Method allocator pool | Scratch pool |
|-------|-----------------------|--------------|
| ESP32, no PSRAM (~520KB SRAM for code and data; only very small models fit) | 128-256KB | 64-128KB |
| ESP32 / ESP32-S3 with PSRAM (model data and large buffers in PSRAM) | 1-4MB | 256KB-1MB |

Pool sizes adjust to PSRAM availability. Override them in CMake:

```cmake
set(ET_ESP_METHOD_ALLOCATOR_POOL_SIZE "1048576")    # 1MB
set(ET_ESP_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE "524288") # 512KB
```

or on the command line: `idf.py build -DET_ESP_METHOD_ALLOCATOR_POOL_SIZE=1048576`.

## Loading Models

By default the `.pte` is compiled into the firmware as a C array. For larger
models, load it from a SPIFFS partition:

1. Build with `-DFILESYSTEM_LOAD=ON`.
2. Add a partition and flash the image:
   ```bash
   # partitions.csv: storage, data, spiffs, , 0x200000
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

## Troubleshooting

- **Model too large for flash**: load it from the filesystem
  (`FILESYSTEM_LOAD=ON`), quantize it, or use a smaller model.
- **Out of memory during inference**: enable PSRAM (`CONFIG_SPIRAM=y`), raise
  the pool sizes, or use a smaller model; `log_mem_status()` reports usage.
- **Build errors with the ExecuTorch libraries**: cross-compile them with the
  same ESP-IDF toolchain and target (LX6 for ESP32, LX7 for ESP32-S3), and point
  `ET_BUILD_DIR_PATH` at that build.
- **Watchdog resets**: long inferences can trip the task watchdog; raise
  `CONFIG_ESP_TASK_WDT_TIMEOUT_S` or disable it with `CONFIG_ESP_TASK_WDT_EN=n`.

## ESP-IDF Component

[`pytorch/executorch`](https://components.espressif.com/components/pytorch/executorch)
on the ESP Component Registry packages the runtime, its kernel libraries, the
ESP-NN delegate and a C inference API as source that ESP-IDF builds with your
project. Try its example, with a version from the registry page:

```bash
idf.py create-project-from-example "pytorch/executorch==<version>:hello_executorch"
cd hello_executorch
idf.py set-target esp32s3
idf.py build flash monitor
```

To use it in your project, depend on one exact version:

```yaml
dependencies:
  pytorch/executorch: "==<version>"
```

The project then makes a kernel bundle for its models with the component's
`tools/gen_kernels.py`. The component's
[README](idf_component_template/README.md), also on the registry page, covers
exporting a model, making the bundle and running the model.

To generate the component from this checkout instead:

```bash
./examples/espressif/build_idf_component.sh --allow-dev-stamp \
    --out /path/to/your_project/components/executorch
```

Run it with ExecuTorch installed from the same commit; otherwise the component
leaves out its `hello_executorch` example.

On pull requests that can change the component, CI builds
`idf_component_smoke/` and `idf_component_consumer/` with it and runs them on
the emulator. Nightly, `idf_component_consumer/` is built in every
`sdkconfig.ci.*` configuration on ESP-IDF 6.0.1, 6.1 and master and run on each
target's emulator
([nightly-espressif-idf.yml](../../.github/workflows/nightly-espressif-idf.yml));
the `ciflow/nightly` label runs it on a pull request.

### Publishing

[publish-espressif-idf.yml](../../.github/workflows/publish-espressif-idf.yml)
builds and emulates the component in every consumer configuration, uploads it
to the staging registry, then to production, and builds a consumer from each.
Each upload waits for approval from a reviewer of the `esp-component-registry`
environment other than whoever started the run. Anyone with write access can
push a `v*` tag, so approvers check the commit, not just the tag's name.

**Releases and release candidates.** ExecuTorch's release tags publish the
component:

1. Pushing `vX.Y.Z-rcN` publishes `X.Y.Z-rc.N` to staging only. Before
   approving, run the nightly matrix on the tag and wait for its macOS and
   Windows configure jobs; ESP-IDF master failures are advisory:
   ```bash
   gh workflow run nightly-espressif-idf.yml --ref vX.Y.Z-rcN
   ```
2. Pushing `vX.Y.Z` publishes `X.Y.Z` to staging, then production. Check the
   nightly matrix as for an RC, and approve the production upload once
   `executorch` X.Y.Z is on PyPI and a project builds against the staging
   version.

**Respins.** A fix to a released component publishes `X.Y.Z~N`, with N
counting up from 1:

1. Commit the fix to `release/X.Y`. Since `vX.Y.Z`, the tagged commit may change
   only `examples/espressif/`, `.ci/`, `.github/` and `docs/`; if `release/X.Y`
   has other changes, commit the fix on top of `vX.Y.Z` and merge it into
   `release/X.Y`.
2. Tag the fix's commit, which publishes to staging, then production:
   ```bash
   git tag vX.Y.Z-esp-idf.N <commit>
   git push origin vX.Y.Z-esp-idf.N
   ```
3. The approver checks that the commit is on `release/X.Y` and that its
   changes since `vX.Y.Z` were reviewed.

**Dev versions.** `X.Y.Z-dev.YYYYMMDD` publishes the latest nightly, at most
once a day:

1. Once the nightly's ExecuTorch wheels are on the nightly index, run:
   ```bash
   gh workflow run publish-espressif-idf.yml --ref nightly \
       -f registry=production -f dry_run=false
   ```
   The run fails if any wheel is missing or was built from another commit.
2. The approver reviews what changed since the last approved dev run in the
   files the upload jobs run: `publish-espressif-idf.yml`,
   `.ci/scripts/upload_espressif_idf_component.sh` and its requirements file,
   and `.ci/scripts/test_espressif_idf_nightly.sh`.

To retry a failed upload, use **Re-run failed jobs** on the run.
