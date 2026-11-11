# ExecuTorch Executor Runner for Espressif ESP32/ESP32-S3

The ESP32-S3 runner has an [emulator smoke test](#emulator-smoke-test).
The original ESP32 target is not covered by this test.

This example demonstrates how to run an ExecuTorch model on Espressif ESP32 and
ESP32-S3 microcontrollers. It is based on the
[Arm Cortex-M executor runner](../arm/executor_runner/) and adapted for the
ESP-IDF build system and ESP32 memory architecture.

## ESP-IDF component (generated, source-vendored)

For integrating ExecuTorch into an existing ESP-IDF project without the
cross-compile-then-link flow below, generate a self-contained ESP-IDF
component from sources in this repository:

```bash
./examples/espressif/build_idf_component.sh --allow-dev-stamp \
    --out /path/to/your_project/components/executorch
```

Run it from a Python environment with ExecuTorch installed from this exact
commit (reinstall after committing or pulling), or for a release or respin
stamp, that release's package: the generator exports the component's
example model with it, and a development build without that package
leaves the example out. `--allow-dev-stamp` versions the component
`X.Y.Z-dev.<commit date in UTC>`; without it the generator stamps a release
version and requires the checkout to be at that release's tag (see the
script's header for respin and RC stamps).

The output is a registry-shaped component (`idf_component.yml`,
`CMakeLists.txt`, `Kconfig`, `VERSION`, `sbom*.yml`,
`THIRD_PARTY_LICENSES/`) containing the runtime, the
portable, quantized, Cadence generic and `esp_nn::` kernel libraries,
the ESP-NN delegate, the shared
ESP-IDF PAL from `platform/esp_pal.cpp` (ESP_LOG + `esp_timer` timing),
a C inference API (`et_model.h`, flash-XIP model bytes with
internal/PSRAM arena placement), and `examples/hello_executorch`, a
complete project that runs a small model. Only the kernels a
project's models use are registered: the component's `tools/gen_kernels.py` makes a
kernel bundle from the project's `.pte` files, and the project declares
both with `executorch_add_model` and `executorch_add_kernels` (see the
component's `README.md`). Application pipelines built on the runtime
(for example wake word front ends and detectors) stay with the
application.
Consumers need ESP-IDF, plus the `executorch` Python package the
component was built from to make kernel bundles (see the component's
`README.md`) — no host CMake build, no flatc.
`idf_component_smoke/` is a minimal consumer that CI generates, builds
and runs on the emulator (`.ci/scripts/test_espressif_idf_component.sh`).
`idf_component_consumer/` builds the same app from the packed component
through the component manager, with ESP-IDF's own Python, for every
target an emulator runs on ESP-IDF 6.1, and for ESP32-S3 and ESP32-C6
on ESP-IDF 6.0.1 (`.ci/scripts/test_espressif_idf_consumer.sh`). Nightly,
it is built for every target with each `sdkconfig.ci.*` configuration that
idf-build-apps lists, on ESP-IDF 6.0.1, 6.1 and master, and each build except
the one for ESP32-P4 before rev3 runs on its emulator
(`.ci/scripts/test_espressif_idf_nightly.sh`). The debug, performance and
no-logging configurations run all four test models on every target; the
single-model configurations, and one that registers every op of every kernel
library, run on ESP32-S3 and ESP32-C6. macOS and Windows jobs configure
`idf_component_smoke` with the packed component
(`.github/workflows/nightly-espressif-idf.yml`). Pull requests run this
matrix only when they change it, that is the workflow, its scripts or
`idf_component_consumer/`, and leave out ESP-IDF master; the
`ciflow/nightly` label runs it on any pull request.

### Publishing the component

`.github/workflows/publish-espressif-idf.yml` publishes the component to the
ESP Component Registry as `pytorch/executorch`. It packs the component
once, builds that archive in every `idf_component_consumer` configuration
on ESP-IDF 6.0.1 and 6.1, runs each build except the build-only one for
ESP32-P4 before rev3 on its emulator, and validates the archive with both
registries before uploading it. After each upload, a consumer build
resolves the new version from that registry, checks that the registry
serves the tested files, and runs it on the emulator.

- An RC tag `vX.Y.Z-rcN` publishes `X.Y.Z-rc.N` to the staging registry,
  https://components-staging.espressif.com.
- A release tag `vX.Y.Z` publishes `X.Y.Z` to staging, then to
  https://components.espressif.com. The first release follows an RC that
  this workflow published to staging, so the upload and the registry
  checks have run before a release depends on them.
- A fix to the component after a release is a respin, `X.Y.Z~N`, with N
  counting up from 1. Commit it to `release/X.Y`, changing nothing since
  the release tag outside `examples/espressif/`, `.ci/`, `.github/` and
  `docs/`, and push the tag `vX.Y.Z-esp-idf.N` at that commit (git refs
  cannot hold `~`); the tag publishes it to staging, then production. That
  works while `release/X.Y` has changed nothing else since the release
  tag, `version.txt` included. Once it has, commit the fix on top of
  `vX.Y.Z`, merge that commit into `release/X.Y`, and tag the fix's
  commit. A release whose publishing failed in CI is published as a
  respin once the CI is fixed.
- A dev version, `X.Y.Z-dev.YYYYMMDD`, lets projects use the component
  before a release. Run the workflow on the `nightly` branch with
  `registry` `production` and `dry_run` cleared, between 00:00 UTC and the
  next nightly cut, so that it publishes the last nightly of a finished
  day. Projects resolve from production, where a version range picks up a
  pre-release only if it opts in. The run publishes only if that day's
  ExecuTorch nightly wheels, for every platform and CPython, were built
  from the same commit, because projects install them to make their kernel
  bundles. Before the first dev upload, run it once with `dry_run` left
  set, so its packing has run.
- A manual run, on the release, RC or respin tag, publishes the tag's
  version, and on the `nightly` branch, its dev version; it only validates
  unless `dry_run` is cleared. With `registry` `production`, it uploads to
  staging first. A run that packs but uploads nothing rehearses the staging
  upload with `--dry-run`.
- A tag runs the workflow as of its commit, so a `release/X.Y` branch cut
  before the workflow landed publishes nothing until the workflow is
  cherry-picked to it.
- A run keeps its tested archive in two places. The jobs after an approval
  read a copy kept 14 days, so within them **Re-run failed jobs** retries
  an upload or registry check with that archive; uploads skip versions the
  registry already has. The jobs before the approvals read a copy GitHub
  keeps 2 days. After those days, and for a run more than 30 days old,
  which GitHub does not re-run, the way to retry is **Re-run all jobs** or
  a new run on the tag: it packs and tests a new archive, whose registry
  check passes only if its files match those the registry already serves.
- Runs for one tag or branch go one at a time: a run waiting for an
  approval holds back the next run on its ref, and GitHub cancels that run
  if a newer one arrives. Reject a stale deployment before dispatching
  another run on the same ref.

Each upload waits for a reviewer of the `esp-component-registry`
environment: on each registry, `pytorch/executorch`'s trusted uploader
names this repository, this workflow and that environment, so no other job
can upload with the repository's GitHub identity. Before approving an upload,
check that the run's ref and commit are the release's announced tag and
commit, or for a respin, that its tag's commit is on `release/X.Y` and
its changes since `vX.Y.Z` were reviewed. A dev version's run comes from
the `nightly` branch, whose copy of this workflow is main's and enforces
its own limits, so before approving one, also review what changed since
the last approved dev run, or for the first one their whole content, in
every file the environment's jobs run: `publish-espressif-idf.yml`,
`.ci/scripts/upload_espressif_idf_component.sh` and the
`.ci/scripts/espressif_idf_upload_requirements.txt` it installs, and
`.ci/scripts/test_espressif_idf_nightly.sh`, whose fetch mode those jobs
run first. Before approving the staging upload of a release or RC,
dispatch the nightly matrix (`nightly-espressif-idf.yml`) on the tag and
wait for its macOS and Windows configure jobs to pass; a failure on
ESP-IDF master, which breaks on its own schedule, is advisory. A dev
version has only its own run's checks. Approve the production upload of a
release once `executorch` X.Y.Z is on PyPI, since a project needs it to
make kernel bundles, and the release owner has built a project against the
version on staging; a dev version's run has already checked its nightly
wheels.

The one-time setup goes in this order, because GitHub creates a missing
environment, unprotected, as soon as a workflow names it:

1. Add rulesets that let only release managers create, update or delete
   `v*` tags, since the environment trusts them, and `release/*` branches,
   which respin tags must be on, and only the account that pushes the
   nightly branch update or delete `nightly`, which the environment trusts
   for dev versions.
2. Create the `esp-component-registry` environment with required
   reviewers, self-review prevented, no admin bypass, and deployments
   limited to `v*` tags, plus `nightly` before the first dev version.
3. Create the `pytorch` namespace on the staging registry, which takes
   namespace requests of its own, apart from the production registry's.
4. On each registry, add a trusted uploader to `pytorch/executorch` (in
   Permissions, under the component) with this repository, the workflow
   `publish-espressif-idf.yml` and the environment
   `esp-component-registry`, and no branch, since a tag push carries none.

The executor runner below remains the reference for the prebuilt-library
flow and for benchmarking.

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
├── setup.sh                     # Install ESP-IDF and esp-emulator
├── build.sh                     # Build helper script
├── export_smoke_model.py        # Export bundled add/multiply test data
├── test_emulator.py             # Check a smoke-test emulator log
├── executor_runner/
│   ├── CMakeLists.txt           # Component/standalone CMake build
│   ├── esp_executor_runner.cpp  # Main executor runner
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

The profiler reports wall time for the whole inference loop, including input
preparation and tracing setup.

## Cross-Compiling ExecuTorch

ExecuTorch needs to be cross-compiled for the ESP32 target (Xtensa architecture).

The `esp-baremetal` preset sets `CMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY`
so CMake's header and type checks compile without linking executables. These
standalone probes lack the ESP-IDF project's linker configuration; requiring
an executable link can incorrectly report valid headers and types as missing.

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

## Emulator Smoke Test

[Espressif's esp-emulator](https://github.com/espressif/esp-emulator) can run
the ESP32-S3 firmware locally without a board. The following steps export a model
with two inputs and two outputs (`x + y` and `x * y`), bundle PyTorch reference
outputs, and run the existing runner with 8 MB of octal PSRAM. Each step can be
run independently: rebuild after code changes, rerun existing firmware, or check
a saved log without installing or building anything.

Run these Bash commands from the ExecuTorch repository root with your ExecuTorch
venv or conda environment active and pip installed.

### 1. Install tools

```bash
bash examples/espressif/setup.sh
source ~/.cache/executorch-espressif/setup_path.sh
```

The ESP-IDF and esp-emulator versions are pinned in [setup.sh](setup.sh). The
installer verifies the emulator installer's pinned SHA-256 checksum before
running it. ESP-IDF's Python dependencies are installed into your active
environment, so the SDK and ExecuTorch share the same Python. The generated
`setup_path.sh` adds the installed tools to `PATH` and selects the ESP32-S3 target.
In a new shell, activate the same Python environment and source this file.
Installation only needs to be done once.

The default install includes ESP-IDF's standard ESP32-S3 development tools,
including GDB, OpenOCD, and the ULP toolchains. OpenOCD requires your platform's
USB libraries (`libusb-1.0-0` on Ubuntu).

For the minimal toolset used by CI, run:

```bash
bash examples/espressif/setup.sh --minimal
```

This installs only the Xtensa compiler, SDK Python packages, and emulator. It
does not require libusb. Source the generated `setup_path.sh` as above to activate
the tools. Rerun setup without `--minimal` to add the full development toolset.

Set `ESPRESSIF_TOOLS_DIR` to choose another installation directory. The installer
prints the corresponding activation command. It also accepts `IDF_PATH` and
`IDF_TOOLS_PATH` for an existing ESP-IDF installation. The setup script checks
the SDK version before installing dependencies into the active environment.
Use `--skip-idf` or `--skip-emulator` to install only the tools you need.
`--targets=<chip>,...` installs the compilers for other targets (and, for
`esp32` without `--skip-emulator`, ESP-IDF's QEMU), and
`--idf-version=v6.0.1` installs that ESP-IDF
release instead of 6.1; use a separate `ESPRESSIF_TOOLS_DIR` for it.

### 2. Export the test model

```bash
python examples/espressif/export_smoke_model.py --output cmake-out-esp-emulator/smoke.bpte
```

The `.bpte` contains the program, test inputs, and expected outputs. Exporting is
only needed when the model or its test data changes.

### 3. Build the runtime

Ninja chooses the default build parallelism. Set `CMAKE_BUILD_PARALLEL_LEVEL`
to limit the number of concurrent build jobs; the CI wrapper honors it too.

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

The build embeds the bundled model and enables reference-output comparisons.
The project's default ESP32-S3 configuration enables 8 MB of octal PSRAM.
For incremental builds, rerun the build and merge commands; reconfigure when
changing CMake options.

### 5. Run the emulator

```bash
esp-emu --chip esp32s3 \
    --firmware cmake-out-esp-emulator/firmware/merged-binary.bin \
    --elf cmake-out-esp-emulator/firmware/executorch_esp_runner.elf \
    --net user --psram-size 8M --exit-on 'Program complete.'
```

This boots the existing firmware and prints its output in the terminal. Change
the firmware paths to run another build.

### 6. Check the test result

To validate a run automatically, append `> emulator.log 2>&1` to the emulator
command above to capture its output, then run:

```bash
python examples/espressif/test_emulator.py emulator.log
```

This check uses only the Python standard library. It requires the ESP platform
hooks, PSRAM initialization, completion of all requested inferences, matching
bundled outputs, and firmware completion. The completed inference count must
match the count announced by the firmware, so changing `ET_NUM_INFERENCES` does
not require changing this checker. A missing success marker or failed comparison
fails the test. This also detects incomplete runs when the emulator's optional
`--timeout` returns exit status zero. The check can be rerun on a saved log
without rebuilding or starting the emulator.

The [CI wrapper](../../.ci/scripts/test_espressif.sh) composes these same steps,
adds a host timeout, and collects logs and firmware. It also runs the ESP-NN
delegate's BundleIO fixtures with optimized S3 kernels and ANSI C reference
kernels in separate jobs. The
[GitHub Actions workflow](../../.github/workflows/test-espressif.yml) runs it on
pull requests changing the ESP integration, pushes to `main` and `release/*`,
weekly, and on manual dispatch. Branch and scheduled runs cover changes to
shared runtime and export code.

This covers portable and ESP-NN delegate execution and ESP-IDF integration on S3.
It does not establish performance or peripheral behavior on real boards, and the
emulator does not support the original ESP32.

## Memory Considerations

The runner requires memory-planned tensor inputs and outputs. Export with
`MemoryPlanningPass(alloc_graph_input=True, alloc_graph_output=True)` (the default).
Initialization reports an error for unplanned I/O.

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
| Performance monitor | ARM PMU + Ethos-U PMU | esp_timer (64-bit wall time) |
| Semihosting | FVP simulator filesystem access | SPIFFS/LittleFS/SD filesystem |
| Entry point | `main()` bare-metal | `app_main()` via FreeRTOS |
| Timing | ARM_PMU_Get_CCNTR() | esp_timer_get_time() |

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
