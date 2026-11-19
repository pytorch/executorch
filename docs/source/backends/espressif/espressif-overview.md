# Espressif ESP-NN Backend

The Espressif backend runs int8-quantized models on Espressif ESP32 microcontrollers with [ESP-NN](https://github.com/espressif/esp-nn), Espressif's neural-network kernel library. ESP-NN has SIMD kernels for the ESP32-S3 and portable C kernels for the other chips. Operators that ESP-NN cannot run stay in the program and execute on ExecuTorch's portable and quantized CPU kernels.

```{note}
This backend is a prototype under active development. Its APIs, the delegate payload format and the supported operator set may change.
```

## Features

- **ESP-NN delegate.** The default path. Quantized subgraphs are delegated to `EspNNBackend` as compact int8 graphs. It supports:
  - convolution, including depthwise, grouped, dilated and transposed;
  - linear and pooling;
  - elementwise add and mul;
  - softmax, resize and data movement;
  - common activations, which either fuse into the preceding layer or run as 256-entry lookup tables.

  See {doc}`/backends/espressif/espressif-partitioner` for the full operator list.
- **ESP-NN operator library.** An experimental path in which supported patterns become direct `esp_nn::` operator calls instead of delegates. It covers quantized `mul` so far; see [Operator library](#operator-library).
- **CPU fallback.** Unsupported operators run on the portable kernels. A program can contain several delegates with CPU regions between them.
- **Quantizer.** `EspNNQuantizer`, a PT2E quantizer, produces exactly the int8 scheme the ESP-NN kernels execute, and supports quantization-aware training.
- **ESP-IDF component.** The runtime, the kernel libraries and the ESP-NN delegate are packaged as `pytorch/executorch` on the [Espressif component registry](https://components.espressif.com/components/pytorch/executorch).
- **Host reference.** A bit-exact host model of the device arithmetic, used by the tests and for checking programs before they reach a board.
- **Memory and cores.** Weights can be copied into PSRAM at load time, and large convolutions can be split across both cores of the ESP32-S3.

## Target Requirements

- **ESP32-S3.** This is the primary target, with ESP-NN's SIMD kernels. The backend has been verified on an ESP32-S3 with 16 MB flash and 8 MB octal PSRAM (ESP32-S3-DevKitC-1 N16R8). CI runs on an emulated ESP32-S3 with both ESP-NN's SIMD kernels and its portable C kernels.
- **Other ESP32 chips.** These use ESP-NN's portable C kernels, which are roughly an order of magnitude slower than the S3's SIMD kernels.
- **Memory.** PSRAM is recommended for image models, whose activations rarely fit in internal RAM. Without PSRAM, a model's activations must fit in about 280 KB of internal RAM on the original ESP32.
- **Software.** ESP-IDF 6.0 or newer, and the `espressif/esp-nn` component, version 1.4.0 or newer. The component manager fetches ESP-NN from the Espressif component registry. The experimental operator library (`-DET_ESPRESSIF_OPS=ON`) pins `espressif/esp-nn` to exactly 1.4.1, so a project that requires another ESP-NN version gets a component manager conflict.

## Development Requirements

Exporting a program needs the ExecuTorch Python package with the backend's quantizer and partitioner. The backend was merged after the 1.6 release branch was cut, so it isn't in ExecuTorch 1.6.0 or earlier releases. It is in the nightly builds, and in releases from 1.7 on. To use it now, do one of the following:

- Install the nightly that matches the ESP-IDF component version you use. Each published component's README gives the exact `torch` and `executorch` versions to install.
- [Install ExecuTorch from source](https://docs.pytorch.org/executorch/main/using-executorch-building-from-source.html).

Building firmware needs ESP-IDF. The setup script installs a matching ESP-IDF (v6.1 by default) and the `esp-emu` emulator into `~/.cache/executorch-espressif`:

```bash
examples/espressif/setup.sh                        # ESP-IDF tools for the ESP32-S3, plus esp-emu
examples/espressif/setup.sh --minimal              # only the compiler, the SDK's Python packages and esp-emu
examples/espressif/setup.sh --targets=esp32s3,esp32 --idf-version=v6.1
source ~/.cache/executorch-espressif/setup_path.sh
```

Run the script with your ExecuTorch virtual environment or conda environment active. ESP-IDF installs its Python dependencies there. With `esp32` in `--targets`, the script also installs ESP-IDF's QEMU for the original ESP32, which needs the system's `libSDL2` and `libslirp`.

----

## Using the Espressif Backend

Quantize the model with `EspNNQuantizer`, then lower it with `to_espnn_edge`. This wraps `to_edge_transform_and_lower` with an `EspNNPartitioner` and returns the edge program together with a report of what was delegated. The example below uses MobileNetV2 from torchvision. Calibrate on a few hundred real samples from your dataset; random data produces poor quantization scales.

```python
import torch
import torchvision
from executorch.backends.espressif import EspNNQuantizer, to_espnn_edge
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

model = torchvision.models.mobilenet_v2(weights="DEFAULT").eval()
example_inputs = (torch.randn(1, 3, 224, 224),)

# 1. Capture the model and insert observers for the ESP-NN int8 scheme.
prepared = prepare_pt2e(
    torch.export.export(model, example_inputs).module(), EspNNQuantizer()
)

# 2. Calibrate with representative data.
with torch.no_grad():
    for images in calibration_batches:
        prepared(images)
quantized = convert_pt2e(prepared)

# 3. Lower: int8 regions become ESP-NN delegates, the rest stays on the CPU.
edge, report = to_espnn_edge(torch.export.export(quantized, example_inputs))
print(report["partitions"], report["delegated_ops"], len(report["cpu_ops"]))

# 4. Serialize the ExecuTorch program.
with open("mobilenet_v2_espnn.pte", "wb") as file:
    file.write(edge.to_executorch().buffer)
```

MobileNetV3-Small, for example, lowers to a single delegate of 122 operators with none left on the CPU. On an ESP32-S3 its outputs are bit-identical to the host emulator's, and an inference takes 649 ms with the cache settings under [Flash and cache settings](#flash-and-cache-settings). [`examples/espressif/quantize_and_lower.py`](https://github.com/pytorch/executorch/blob/main/examples/espressif/quantize_and_lower.py) contains this flow as a script.

See {doc}`/backends/espressif/espressif-partitioner` for the partitioner options and the lowering report, and {doc}`/backends/espressif/espressif-quantization` for the quantization scheme and quantization-aware training.

### Operator library

```{warning}
The operator library is experimental and opt-in. It currently implements one operator.
```

With `use_operator_library=True`, `to_espnn_edge` doesn't partition the program. It runs `EspPassManager`, which rewrites supported quantize/dequantize patterns into direct calls to `esp_nn::` operators, and leaves everything else on the portable and quantized kernels:

```python
edge, report = to_espnn_edge(quantized_program, use_operator_library=True)
print(report["esp_nn_ops"], report["cpu_ops"])
```

The report counts the `esp_nn::` operator calls (`esp_nn_ops`). It also lists the remaining CPU operators by method (`cpu_ops`), leaving out quantize/dequantize and structural nodes. `allow_cpu_fallback=False` raises if any operator is left on the CPU. The library currently provides `esp_nn::quantized_mul_int8`, for quantized `mul`.

----

## Runtime Integration

There are two ways to run a program on a device:

- **The ESP-IDF component** from the Espressif registry. This is the way to add ExecuTorch to an application.
- **A source checkout**, with the example runner. Use this to develop the backend, or to test against unreleased changes.

### Use the ExecuTorch ESP-IDF component

[`pytorch/executorch`](https://components.espressif.com/components/pytorch/executorch) packages the runtime, the portable and quantized kernels, and the ESP-NN delegate as source that builds with your project, for ESP-IDF 6.0 or newer. It also provides:

- `et_model.h`, a small C API that loads and runs a `.pte` from flash;
- `tools/gen_kernels.py`, which generates the kernel bundle that registers the kernels your models use;
- `examples/hello_executorch`, a complete project.

To use the backend through the component:

1. Depend on one exact version of the component in `main/idf_component.yml`, and commit `dependencies.lock`:

   ```yaml
   dependencies:
     pytorch/executorch: "==<version>"
   ```

2. Export the model with the `executorch` Python package that matches the component version, as shown above.
3. Generate the kernel bundle from the `.pte`, and regenerate it after every export:

   ```bash
   python <component>/tools/gen_kernels.py --pte main/model.pte --out main/et_kernels
   ```

4. Declare the model and the bundle in the component's `CMakeLists.txt`:

   ```cmake
   idf_component_register(SRCS "main.c" PRIV_REQUIRES executorch)
   executorch_add_model(PTE "${CMAKE_CURRENT_SOURCE_DIR}/model.pte")
   executorch_add_kernels(DIR "${CMAKE_CURRENT_SOURCE_DIR}/et_kernels")
   ```

5. Load and run the model with `et_model.h`. The `.pte` data must be 16-byte aligned.

The ESP-NN delegate is linked when the kernel bundle records `EspNNBackend`, which `gen_kernels.py` does for a program that uses it, or when the project calls `executorch_add_backend(EspNNBackend)`. The component's [README](https://github.com/pytorch/executorch/blob/main/examples/espressif/idf_component_template/README.md) covers these steps in full, including version pinning, models delivered over the air, and the size and logging options.

### Build from a source checkout

#### Cross-compile the ExecuTorch runtime

Build the runtime with the `esp-baremetal` CMake preset and the ESP-IDF toolchain file. Select the CPU operators the program needs, and enable the quantized kernels, which provide the quantize and dequantize operators at the delegate boundaries:

```bash
cmake --preset esp-baremetal -G Ninja -B cmake-out-esp \
    -DCMAKE_TOOLCHAIN_FILE="${IDF_PATH}/tools/cmake/toolchain-esp32s3.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_DEVTOOLS=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_SELECT_OPS_LIST="aten::add.out,<the program's CPU operators>" \
    -DMAX_KERNEL_NUM=256 \
    -DEXECUTORCH_BUILD_PTHREADPOOL=OFF -DEXECUTORCH_BUILD_CPUINFO=OFF
cmake --build cmake-out-esp --target install
```

- **Operator selection.** The program lists the operators it needs. Select the portable ones; `quantized_ops_lib` provides the `quantized_decomposed::` ones:

  ```python
  plan = edge.to_executorch().executorch_program.execution_plan[0]
  print(",".join(sorted(
      f"{op.name}.{op.overload}" for op in plan.operators
      if not op.name.startswith("quantized_decomposed::")
  )))
  ```

  The example runner links `executorch_selected_kernels`, which exists only when a selection is given. A fully delegated program therefore still needs at least one selected operator, such as `aten::add.out`.
- **`MAX_KERNEL_NUM`.** With a selection, the build sizes the kernel registry from the selected operators only. `quantized_ops_lib` registers more kernels than that, so set the registry size explicitly.
- **pthreadpool and cpuinfo.** pthreadpool needs POSIX calls that ESP-IDF doesn't provide. With both off, the quantized kernels run single-threaded.

#### Run a program with the example runner

The ESP-IDF runner in [`examples/espressif/project`](https://github.com/pytorch/executorch/tree/main/examples/espressif/project) includes the ESP-NN delegate when built with `-DET_ESPNN_BACKEND=ON`, and the operator library with `-DET_ESPRESSIF_OPS=ON`:

```bash
idf.py -C examples/espressif/project -B build-espnn \
    -DIDF_TARGET=esp32s3 \
    -DET_ESPNN_BACKEND=ON \
    -DET_BUILD_DIR_PATH=$PWD/cmake-out-esp \
    -DET_PTE_FILE_PATH=$PWD/mobilenet_v2_espnn.pte \
    build flash monitor
```

For a BundleIO program (`.bpte`), add `-DET_BUNDLE_IO=ON` with tolerances such as `-DET_ATOL=1e-5 -DET_RTOL=1e-5`. The runner then compares the outputs with the bundled reference outputs and logs `Test_result: PASS` or `FAIL`.

#### Flash and cache settings

The example project's defaults suit small programs. Image models need two changes:

- **App partition.** The runner embeds the program in the firmware image, and the project's `partitions.csv` reserves 2 MB for the app. A MobileNetV3-Small program is 2.7 MB, so the build fails with `Part 'factory' 0/0 @ 0x10000 size 0x200000 (overflow ...)`. Use a larger app partition, for example:

  ```
  # Name,   Type, SubType, Offset,  Size
  nvs,      data, nvs,     0x9000,  24K,
  phy_init, data, phy,     0xf000,  4K,
  factory,  app,  factory, 0x10000, 6M,
  ```

  To select it, set `CONFIG_PARTITION_TABLE_CUSTOM_FILENAME` to its path, through an extra sdkconfig defaults file (`-DSDKCONFIG_DEFAULTS="sdkconfig.defaults;<your defaults file>"`).
- **Caches.** ESP-NN reads the weights from PSRAM through the data cache, so its size matters. ESP-IDF's ESP32-S3 defaults are a 32 KB data cache with 32-byte lines and a 16 KB instruction cache. These settings are faster:

  ```
  CONFIG_ESP32S3_DATA_CACHE_64KB=y
  CONFIG_ESP32S3_DATA_CACHE_LINE_64B=y
  CONFIG_ESP32S3_INSTRUCTION_CACHE_32KB=y
  ```

  For MobileNetV3-Small at 224×224 on an ESP32-S3 N16R8 at 240 MHz, they reduce the inference time from 866 ms to 649 ms. The outputs stay bit-identical.

#### Use the delegate component in your own source build

1. Add the delegate's component to your project's `EXTRA_COMPONENT_DIRS`:

   ```cmake
   list(APPEND EXTRA_COMPONENT_DIRS "${EXECUTORCH_ROOT}/backends/espressif/espnn_runtime")
   ```

2. Import the cross-compiled ExecuTorch targets with `find_package(executorch)`, with `CMAKE_FIND_PACKAGE_TARGETS_GLOBAL` set to `TRUE` so that the component can link `executorch_core`. Link `executorch`, `executorch_selected_kernels` and `quantized_ops_lib` into your application.
3. Provide the ExecuTorch platform layer (`et_pal_*`). The delegate component defines none. [`examples/espressif/platform/esp_pal.cpp`](https://github.com/pytorch/executorch/blob/main/examples/espressif/platform/esp_pal.cpp) implements it on ESP-IDF; link it whole-archive so that it replaces the runtime's weak defaults.
4. Load and run the program with the usual `Program` and `Method` APIs, with memory-planned buffers in PSRAM when they don't fit in internal RAM.

[`examples/espressif/camera_stream`](https://github.com/pytorch/executorch/tree/main/examples/espressif/camera_stream) is a complete application built this way. It runs live camera classification on an ESP32-S3, with a browser viewer.

### Configuration

The delegate has the following options in menuconfig, under *ExecuTorch ESP-NN backend* (*ExecuTorch → ExecuTorch ESP-NN backend* in the ESP-IDF component). In `sdkconfig` and `sdkconfig.defaults` files, they take the `CONFIG_` prefix, for example `CONFIG_ESPNN_DELEGATE_PROFILE=y`:

| Kconfig option | Default | Effect |
|---|---|---|
| `ESPNN_DELEGATE_CONSTANTS_TO_PSRAM` | y | Copies weights from memory-mapped flash into PSRAM at load time when they fit, avoiding flash cache misses on large models. Requires `CONFIG_SPIRAM`; without PSRAM the option is hidden and off. |
| `ESPNN_DELEGATE_DUAL_CORE` | n | Splits large convolutions across both cores. ESP-NN's worker is process-wide. Supported on the ESP32-S3 and ESP32-P4 with ESP-NN's optimized kernels and both cores enabled; elsewhere the runtime prints a warning and runs on one core. |
| `ESPNN_DELEGATE_PROFILE` | n | Prints a per-operator time table after each delegate's first run. |

ESP-NN selects its kernels from ESP-IDF's target. `CONFIG_NN_OPTIMIZED=y`, the default, uses the SIMD kernels on the ESP32-S3. `CONFIG_NN_ANSI_C=y` selects the portable C reference kernels, for verification and debugging.

`executorch::backends::espressif::delegate_stats()`, declared in `espnn/espnn_backend.h`, returns:
- the number of delegate calls;
- the total time spent inside the delegates;
- the size of their activation arenas.

----

## Reference

**→{doc}`/backends/espressif/espressif-partitioner` — Partitioner options and supported operators.**

**→{doc}`/backends/espressif/espressif-quantization` — Quantization scheme and quantization-aware training.**

**→{doc}`/backends/espressif/espressif-troubleshooting` — Debug common issues.**

**→{doc}`/backends/espressif/espressif-arch-internals` — Backend internals.**

**→{doc}`/backends/espressif/tutorials/espressif-tutorials` — Tutorials.**

```{toctree}
:maxdepth: 2
:hidden:
:caption: Espressif Backend

/backends/espressif/espressif-partitioner
/backends/espressif/espressif-quantization
/backends/espressif/espressif-troubleshooting
/backends/espressif/espressif-arch-internals
/backends/espressif/tutorials/espressif-tutorials
```
