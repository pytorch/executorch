# ESP-NN backend for Espressif chips

This backend runs int8-quantized PyTorch models on Espressif microcontrollers.
It is tuned for the **ESP32-S3** and also runs on the original ESP32. It
delegates quantized subgraphs to [ESP-NN](https://github.com/espressif/esp-nn),
Espressif's kernel library. ESP-NN has SIMD kernels for the ESP32-S3 and
portable C kernels for the other chips.

It has two ways to run a model:

| Path | What runs on the chip | When to use it |
|---|---|---|
| **ExecuTorch delegate** | A normal `.pte` program. `EspNNBackend` runs the int8 partitions with ESP-NN, and ExecuTorch's portable kernels run everything else (float ops, unsupported layers, pre- and post-processing). | Any model that `torch.export` can capture, including models that only partly map to ESP-NN. |
| **Standalone int8 graph** | The same delegate payload (an XTS3 graph), run directly by the small `espnn_runtime` interpreter with no ExecuTorch runtime linked. | Models that are fully int8 end to end, when firmware size matters (the runner is about 330 KB). |

The two paths share one lowering flow: the PT2E quantizer, the partitioner and
the XTS3 serializer. The host reference interpreter reproduces the device
arithmetic, so every test and example checks the board's output bit-exactly.

## Contents

```
backends/espressif/
  espnn_preprocess.py        EspNNBackend: BackendDetails.preprocess -> XTS3 graph
  partition/                 EspNNPartitioner: int8 regions -> delegate, rest -> CPU
  quantizer/                 EspNNQuantizer: PT2E int8 annotations ESP-NN can execute
  lowering.py                to_espnn_edge(): export -> edge -> lowered program + report
  serialization/             XTS3 format writer and parser, op lowering rules
  reference/                 bit-exact host model of the ESP-NN kernels and the CPU fallback
  espnn_runtime/             ESP-IDF component: XTS3 interpreter + ExecuTorch BackendInterface
  test/                      pytest suite (lowering, operators, CPU fallback)
examples/espressif/
  quantize_and_lower.py      minimal quantize -> lower -> .pte example
  aot_espnn_compiler.py      full compiler CLI: calibration, QAT, accuracy, device test vectors
  espnn_runner/              ESP-IDF app that runs and verifies bundles on a board
  camera_stream/             live camera classification demo (ESP32-S3)
  scripts/                   cross-build, board runner, benchmarks, license headers
  benchmarks/                benchmark results per chip
```

## Supported operators

All delegated operators are int8 with per-tensor activations and per-channel
symmetric weights, as annotated by `EspNNQuantizer`.

| Group | Operators |
|---|---|
| Convolution | `conv2d` (any stride and padding, groups 1 or depthwise, dilation through phase decomposition), `conv_transpose2d`, `linear` |
| Activations | `relu`, `relu6`, `hardtanh`, plus table lookups for `hardswish`, `hardsigmoid`, `sigmoid`, `silu`, `tanh`, `gelu`, `elu`, `leaky_relu` |
| Elementwise | `add`, `mul` (with broadcasting), `softmax` (last dim) |
| Pooling | `max_pool2d`, `avg_pool2d`, `adaptive_avg_pool2d`, `mean` over H and W |
| Resize | `upsample_nearest2d`, `upsample_bilinear2d` (`align_corners` either way) |
| Data movement | `reshape`, `view`, `flatten`, `unflatten`, `permute`, `transpose`, `squeeze`, `unsqueeze`, `slice`, `select`, `expand`, `pad`, `cat`, `stack`, `split`, `chunk`, `unbind` |

Batch norm that follows a convolution is folded into it by PT2E
(`prepare_pt2e` / `prepare_qat_pt2e`); dropout in an exported eval-mode
model is a no-op.
Partitions can have several inputs and outputs. Anything else stays in the
program and runs on the CPU, including float ops, LSTM, layer norm and
attention matmuls. The partitioner reports what it delegated and what it left
behind. Pass `allow_cpu_fallback=False` to make leftover operators an error.

## Quantize and lower a model

[`examples/espressif/quantize_and_lower.py`](../../examples/espressif/quantize_and_lower.py)
contains the whole flow:

```python
import torch
import torchvision
from executorch.backends.espressif import EspNNQuantizer, to_espnn_edge
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

model = torchvision.models.mobilenet_v2(weights="DEFAULT").eval()
example_inputs = (torch.randn(1, 3, 224, 224),)

# 1. Capture the model and insert observers for the ESP-NN int8 scheme
exported = torch.export.export(model, example_inputs).module()
prepared = prepare_pt2e(exported, EspNNQuantizer())

# 2. Calibrate (use real samples from your dataset here)
with torch.no_grad():
    for images in calibration_batches:
        prepared(images)
quantized = convert_pt2e(prepared)

# 3. Lower: int8 regions -> EspNNBackend delegates, the rest -> portable CPU kernels
edge, report = to_espnn_edge(torch.export.export(quantized, example_inputs))
print(report["partitions"], report["delegated_ops"], report["cpu_ops"])

# 4. Serialize the ExecuTorch program
with open("mobilenet_v2_espnn.pte", "wb") as f:
    f.write(edge.to_executorch().buffer)
```

```
$ python examples/espressif/quantize_and_lower.py
1 EspNN partition(s), 101 delegated operators, 0 operators left for the CPU
wrote mobilenet_v2_espnn.pte
```

### The compiler CLI

`aot_espnn_compiler.py` builds on the same steps for production use. It
calibrates on an image folder, can run quantization-aware training, and
reports float and int8 accuracy. It also writes a bundle that contains test
vectors, so the board can check its own output:

```bash
# ExecuTorch program (.pte inside an .xtp bundle) with one epoch of QAT
python examples/espressif/aot_espnn_compiler.py \
    --model mobilenet_v3_small --mode executorch \
    --calib-dir imagenette2-160/train --eval-dir imagenette2-160/val \
    --calib-samples 512 --qat-epochs 1 \
    --output mnv3s.xtp --pte-output mnv3s.pte

# Standalone int8 graph (.xtb bundle), no ExecuTorch runtime on the device
python examples/espressif/aot_espnn_compiler.py \
    --model mobilenet_v3_small --mode standalone \
    --calib-dir imagenette2-160/train --output mnv3s.xtb

# Your own model: path/to/file.py:factory, one --input-shape per input
python examples/espressif/aot_espnn_compiler.py \
    --model my_models.py:build_kws --weights none --input-shape 1,1,49,10 \
    --calib-npy kws_calib.npy --output kws.xtp
```

Tips:

- Calibrate on a few hundred real samples. Random data gives useless scales.
- Use per-channel PTQ for MobileNetV2-style networks. Networks with hardswish
  and squeeze-excite blocks (MobileNetV3, EfficientNet) lose more accuracy
  with PTQ alone, and `--qat-epochs` recovers most of it.
- `--crc-outputs` makes the board compare a CRC of large outputs (such as
  segmentation maps) instead of storing them in the bundle.

## Running on the ESP32-S3

You need ESP-IDF v5.1 or newer (tested with v5.5) and an ESP32-S3 board. The
examples assume 16 MB flash and 8 MB octal PSRAM, such as the
ESP32-S3-DevKitC-1 N16R8. Every command below runs in an activated ESP-IDF
shell (`. $IDF_PATH/export.sh`).

### 1. Cross-compile ExecuTorch

Skip this step for the standalone path.

```bash
examples/espressif/scripts/build_executorch_esp.sh esp32s3
# -> cmake-out-esp-esp32s3/{lib,include}
```

The script uses the IDF toolchain file. It builds the ExecuTorch runtime, the
portable kernels and the quantized kernels as static libraries, with exceptions
and RTTI off and with `-mlongcalls`. Set `ET_SELECT_OPS_MODEL=model.pte` to
build only the kernels that a given program needs.

### 2. Add the component to your application

`backends/espressif/espnn_runtime` is a regular ESP-IDF component, and it
pulls in `espressif/esp-nn` from the component registry. Add it to your
project with `EXTRA_COMPONENT_DIRS`:

```cmake
set(EXTRA_COMPONENT_DIRS "${EXECUTORCH_ROOT}/backends/espressif/espnn_runtime")
```

Then enable the delegate with `CONFIG_ESPNN_EXECUTORCH_DELEGATE=y`, using
menuconfig: *ExecuTorch ESP-NN backend*. The component links
`libexecutorch.a`, the kernel registries and the delegate from
`ESPNN_EXECUTORCH_BUILD_DIR`, which defaults to
`cmake-out-esp-${IDF_TARGET}`. It also registers `EspNNBackend` with the
runtime at startup.

The component does not define the ExecuTorch platform layer (`et_pal_*`).
Applications either compile the shared
[`examples/espressif/executor_runner/esp_pal.cpp`](../../examples/espressif/executor_runner/esp_pal.cpp),
as `espnn_runner` does, or provide their own.

Other options:

| Kconfig | Default | Effect |
|---|---|---|
| `ESPNN_DELEGATE_CONSTANTS_TO_PSRAM` | y | Copies weights from memory-mapped flash into PSRAM when they fit (faster than flash cache misses on large models) |
| `ESPNN_DELEGATE_DUAL_CORE` | n | Splits large convolutions across both cores |
| `ESPNN_DELEGATE_PROFILE` | n | Prints a per-operator time table after a delegate's first run |

Load and run a program with the usual ExecuTorch API. `pte_data` points to
the program, for example embedded in flash with `EMBED_FILES` or read from a
partition:

```cpp
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/executor/program.h>
#include <espnn/espnn_backend.h>

using namespace executorch::runtime;

runtime_init();
executorch::extension::BufferDataLoader loader(pte_data, pte_size);
auto program = Program::load(&loader).get();
auto meta = program.method_meta("forward").get();
/* MemoryManager with the method's planned buffers (PSRAM on the S3), then: */
auto method = program.load_method("forward", &memory_manager).get();
method.set_input(input_tensor, 0);
method.execute();
auto stats = executorch::backends::espressif::delegate_stats();  /* time spent inside ESP-NN delegates */
```

[`examples/espressif/espnn_runner/main/pte_suite.cpp`](../../examples/espressif/espnn_runner/main/pte_suite.cpp)
is a complete working example, including memory planning and output checks.

### 3. Build, flash and verify on the board

The runner example builds and flashes a bundle, then compares the board's
output against the host reference:

```bash
python examples/espressif/scripts/run_bundle.py --port /dev/ttyACM0 \
    --executorch-build-dir cmake-out-esp-esp32s3 mnv3s.xtp
```

```
ESP32-S3 (QFN56) (revision v0.2): 16 MB flash; building executorch runner for esp32s3
CHIP target=esp32s3 cores=2 revision=2 cpu_mhz=240 flash_mb=16 psram_kb=8192 ...
ET CASE 1/1 name=mobilenet_v3_small ... delegates=1 delegate_us=... cpu_us=... exact=1 status=PASS
ESPNN RUNNER: PASS
RESULT: PASS (1 cases)
```

You can also work in the example directly with `idf.py`. See
[`examples/espressif/espnn_runner/README.md`](../../examples/espressif/espnn_runner/README.md).
[`examples/espressif/camera_stream`](../../examples/espressif/camera_stream/README.md)
runs a model live on camera frames and streams the results to a browser.

### The original ESP32

The same flow works on the ESP32 (`idf.py set-target esp32`, or let the
scripts detect the chip). ESP-NN has no SIMD kernels for it and uses portable
C, about 13 million int8 multiply-accumulates per second at 240 MHz, roughly
an order of magnitude slower than the S3. Most ESP32 modules have no PSRAM, so
activations must fit in about 280 KB of internal RAM. Small models fit, such
as keyword spotting, tiny CNNs, and MobileNets at low resolution. Models that
need more RAM fail with a clear allocation error.

## Benchmarks

`examples/espressif/scripts/benchmark.py` compiles a fixed set of models. It
groups as many as fit into each flash image, runs them on the connected board
and writes `examples/espressif/benchmarks/<target>.{md,json}`. Every row is
verified against the host reference before it is timed, so a row without
`PASS` is not reported as a result.

```bash
python examples/espressif/scripts/benchmark.py --port /dev/ttyACM0 \
    --images imagenette2-160 --executorch-build-dir cmake-out-esp-esp32s3 --dashboard
```

The results below are at 240 MHz, measured over 10 runs after a warm-up run,
with batch 1. The S3 board was an ESP32-S3 N16R8 (octal PSRAM at 80 MHz, 64 KB
data cache). The ESP32 board was an ESP32-D0WDQ6 (revision 1.0) with 4 MB flash and no PSRAM.

### ESP32-S3

| Model | Input | Path | Ops | Weights | Arena | Latency | On-board check |
|---|---|---|---|---|---|---|---|
| Keyword spotting DS-CNN | 1×1×49×10 | standalone | 12 | 0.03 MB | 16 KB | 16.4 ms | bit-exact |
| Small residual CNN | 1×3×64×64 | standalone | 16 | 0.24 MB | 36 KB | 40.3 ms | bit-exact |
| MobileNetV3-Small | 1×3×96×96 | standalone | 108 | 2.68 MB | 72 KB | 116 ms | bit-exact |
| MobileNetV3-Small | 1×3×224×224 | standalone | 108 | 2.68 MB | 392 KB | 582 ms | bit-exact |
| MobileNetV2 | 1×3×224×224 | standalone | 65 | 3.70 MB | 1.4 MB | 1.61 s | bit-exact |
| MobileNetV3-Large | 1×3×224×224 | standalone | 121 | 5.71 MB | 980 KB | 1.58 s | bit-exact |
| EfficientNet-B0 | 1×3×224×224 | standalone | 190 | 5.65 MB | 2.3 MB | 2.50 s | bit-exact |
| LR-ASPP MobileNetV3 segmentation | 1×3×224×224 | ExecuTorch, 146 delegated / 0 CPU | | 3.43 MB `.pte` | 1.0 MB | 3.58 s | bit-exact (CRC) |
| SSDLite320 MobileNetV3 detection heads | 1×3×320×320 | ExecuTorch, 243 delegated / 0 CPU | | 3.72 MB `.pte` | 2.0 MB | 3.83 s | bit-exact (CRC) |
| Small vision transformer | 1×3×64×64 | ExecuTorch, 11 delegated / 30 CPU | | 0.04 MB `.pte` | 38 KB | 85 ms (delegate 22, CPU 63) | 39.3 dB vs host |

Arenas that fit internal RAM (the small models) stay there; every 224×224
model's arena is in PSRAM. The transformer is checked by SQNR because its float CPU
operators (LayerNorm, attention, softmax) round differently from the host.

### ESP32 (no PSRAM, 4 MB flash)

Standalone path only. The ExecuTorch path needs ExecuTorch cross-built for
this chip (`build_executorch_esp.sh esp32`), which was not part of this run.

| Model | Input | Ops | Weights | Arena | Latency | On-board check |
|---|---|---|---|---|---|---|
| Keyword spotting DS-CNN (2.7 MMAC) | 1×1×49×10 | 12 | 0.03 MB | 16 KB | 203.4 ms | bit-exact |
| Small residual CNN (9.3 MMAC) | 1×3×64×64 | 16 | 0.24 MB | 36 KB | 705.0 ms | bit-exact |
| MobileNetV3-Small | 1×3×96×96 | 108 | 2.68 MB | 72 KB | 1440.7 ms | bit-exact |
| MobileNetV3-Small | 1×3×224×224 | 108 | 2.68 MB | 392 KB | does not fit 280 KB RAM | – |
| MobileNetV2 / V3-Large / EfficientNet-B0 at 224 | | | 3.7–5.7 MB | | do not fit 4 MB flash | – |

Operator suite (`op_suite.xtb`, 20 runs each), all 11 cases bit-exact against
the host reference and ESP-NN's ANSI C kernels:

| Case | Latency |
|---|---|
| SE block: hardswish LUT, mul, softmax | 6.8 ms |
| Detection heads (reshape/permute/cat) | 9.0 ms |
| Segmentation: dilated conv + transposed conv | 12.0 ms |
| Upsample nearest ×2 | 2.6 ms |
| Upsample bilinear ×2 / align_corners | 2.9 / 2.2 ms |
| Broadcast, pad, split, cat | 2.9 ms |
| Two inputs, two outputs | 1.1 ms |
| Transpose, select, stack | 0.4 ms |

Full tables with arena sizes and host accuracy are in
[`examples/espressif/benchmarks/`](../../examples/espressif/benchmarks/).

## Testing

```bash
pytest backends/espressif/test
```

The tests export small models for every operator group, lower them, run the
XTS3 graphs through the reference interpreter and compare the result against
the quantized PyTorch model. CPU fallback tests run the lowered program in the
ExecuTorch Python runtime. On hardware,
`examples/espressif/scripts/make_test_bundles.py` packs the same models into
`op_suite.xtb` and `pte_suite.xtp` for the runner, and every case must be
bit-exact on the board.

The code follows the ExecuTorch coding style and passes the repository's
`lintrunner` checks (flake8, ufmt, clang-format, cppcheck, cmake-format and
the license and include linters):

```bash
lintrunner --paths-cmd "git ls-files backends/espressif examples/espressif"
```

## Limitations

- Only int8 is delegated. Float32 operators run on ExecuTorch's portable
  kernels, which are slow on these chips.
- Activations are per-tensor and weights are per-channel symmetric. Dynamic
  shapes are not supported, so each program has a fixed input shape.
- Attention matmuls, LSTM/GRU and normalization layers other than folded batch
  norm fall back to the CPU.
- Dual-core execution only helps large convolutions, and it is off by default.
- The delegate handle comes from ExecuTorch's runtime allocator, but each
  delegate allocates its activation arena, scratch buffers and optional PSRAM
  weight copy with ESP-IDF's `heap_caps` allocator at `init()`, so it can
  choose between internal RAM and PSRAM. Nothing is allocated during
  `execute()`.

## License

Every source file in this backend and its examples is released under the
BSD 3-Clause license (`SPDX-License-Identifier: BSD-3-Clause`), the same
license as ExecuTorch. See the `LICENSE` file at the repository root. ESP-NN
is Apache-2.0 and is fetched from the Espressif component registry, not
vendored.
