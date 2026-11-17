# ESP-NN backend for Espressif chips

This backend is a **prototype under active development**.

This backend runs int8-quantized PyTorch models on Espressif microcontrollers.
It runs on the **ESP32-S3**, **ESP32-P4** and **ESP32-S31**, and on the
original ESP32. It delegates quantized subgraphs to
[ESP-NN](https://github.com/espressif/esp-nn), Espressif's kernel library.
ESP-NN has SIMD kernels for the ESP32-S3, PIE SIMD kernels shared by the
ESP32-P4 and ESP32-S31, and portable C kernels for the other chips.

A lowered model is a normal `.pte` program. `EspNNBackend` runs the int8
partitions with ESP-NN, and ExecuTorch's portable and quantized kernels run
everything else (float ops, unsupported layers, pre- and post-processing).
Each delegate payload is a compact int8 graph (XTS3) that the delegate's
interpreter executes. A host reference interpreter reproduces the device
arithmetic, so fully delegated programs are checked bit-exactly.

The full documentation, including the operator constraints, quantization,
troubleshooting and a tutorial, is in
[`docs/source/backends/espressif`](../../docs/source/backends/espressif/espressif-overview.md).

The Fused Quant **operator-library flow is under development** alongside the
delegate. Its experimental recipe is a standalone factory, following the shared
Fused Quant frontend convention. The two flows have distinct quantizers:

| Quantizer | Export flow |
|---|---|
| `EspNNQuantizer` | Existing delegate: PT2E Q/DQ graphs partitioned by `EspNNPartitioner` into `EspNNBackend` delegates |
| `get_esp_fused_quant_quantizer` | Operator library: shared Fused Quant annotation and fusion, followed by `EspPassManager` lowering to direct ESP-NN operators or portable fallback |

Both flows use ESP-NN 1.5.0 kernels. See the operator-library
[lowering code](passes/lower_to_esp.py) and [operator tests](test/ops/)
for supported configurations. Shared quantization rules also cover
operators that retain portable execution. The delegate's supported
operators and export example are described below.

Spatial operators use contiguous NHWC tensors; model inputs and outputs
retain their original layout. Layout copies remain visible in the
operator-library report.

Set the recipe's `EspTarget` and `EspKernelMode` to match the device's ESP-NN
configuration when planning scratch:

```python
recipe = get_esp_fused_quant_recipe(
    target=EspTarget.ESP32P4, kernel_mode=EspKernelMode.OPTIMIZED
)
```

The defaults are `EspTarget.ESP32S3` and `EspKernelMode.OPTIMIZED`.
Accelerated Conv2d export requires the optional [scratch-size bindings](#scratch-size-queries).
If the selected binding or query is unavailable, convolution retains portable
execution; strict export rejects that fallback. Other operator exports do not
require the bindings.

Scratch storage uses the planned arena. Applications must serialize
operator-library inference, delegate inference, and other ESP-NN calls.

## Scratch-size queries

Optional Python bindings expose ESP-NN 1.5.0's convolution, depthwise, softmax,
mean, hard-swish, batched fully connected, and logistic scratch-size queries:

```bash
CMAKE_ARGS="-DEXECUTORCH_BUILD_ESP_NN_PYBINDS=ON" pip install -e . --no-build-isolation
```

The host libraries retain only sizing code. CMake fetches the pinned ESP-NN
sources; `ESP_NN_LOCAL_PATH` can reuse an existing 1.5.0 checkout or ESP-IDF
managed component. The host build uses GNU-style compiler section flags and
linker garbage collection (`--gc-sections`, or `-dead_strip` on Apple platforms);
MSVC-style toolchains fail at configuration. Downloaded sources are checked
against the pinned SHA-256, and a local source tree must declare version 1.5.0.

Select the device's `EspTarget` and `EspKernelMode` to match its ESP-NN build:

```python
from executorch.backends.espressif import EspKernelMode, EspTarget
from executorch.backends.espressif.scratch import get_scratch_queries

queries = get_scratch_queries(EspTarget.ESP32P4, EspKernelMode.OPTIMIZED)
if queries is not None:
    scratch_bytes = queries.conv2d_scratch_size(
        input_dims=(8, 8, 16),
        filter_dims=(3, 3, 16),
        output_dims=(8, 8, 32),
        stride=(1, 1),
        padding=(1, 1),
    )
```

Convolution queries use `(width, height, channels)` for dimensions,
`(width, height)` for stride and padding, and unit dilation. Dimensions,
strides, and depthwise channel multipliers must be in `[1, 65535]`; padding must be in
`[0, 65535]`. Conservative bounds on padded tensors, aligned filters, staging,
and coordinates reject shapes that could overflow the vendor's int32 arithmetic
before calling the sizing query. These bounds also apply to the fully connected
query's `row_len` and `out_ch`. Softmax and mean dimensions, and fully connected
`batches`, must be positive int32 values; softmax width must also fit an int32
byte count for four bytes per element. Invalid bounds raise `ValueError`.
Sizes are in bytes, including zero for kernels that need no scratch. The vendor
fully connected query also returns zero for batches above 65535.
`get_scratch_queries` returns `None` only when the selected module is absent;
errors loading an installed binding propagate to the caller.

Optimized mode selects S3 queries for ESP32-S3, shared PIE queries for
ESP32-P4/ESP32-S31, and generic queries for other `EspTarget` values. ANSI mode
selects ANSI queries for every chip. Query results are checked on the host for
all configurations; device execution tests currently cover ESP32-S3.
CI sets `EXECUTORCH_REQUIRE_ESP_NN_SCRATCH=1` when running `test_scratch.py`, so
all four profiles and all seven query families must be available and pass.

## Contents

```
backends/espressif/
  espnn_preprocess.py        EspNNBackend: BackendDetails.preprocess -> XTS3 graph
  partition/                 EspNNPartitioner: int8 regions -> delegate, rest -> CPU
  quantizer/                 EspNNQuantizer (delegate), get_esp_fused_quant_quantizer (operator library)
  lowering.py                delegate lowering, Fused Quant operator-library recipe + report
  serialization/             XTS3 format writer and parser, op lowering rules
  reference/                 bit-exact host model of the ESP-NN kernels and the CPU fallback
  espnn_runtime/             ESP-IDF component: XTS3 interpreter + ExecuTorch BackendInterface
  test/                      pytest suite (lowering, operators, CPU fallback, parser)
examples/espressif/
  quantize_and_lower.py           minimal quantize -> lower -> .pte example
  export_streaming_model.py       streaming model with delegate-owned state
  export_espnn_test_programs.py   the test models as BundleIO programs for the runner
  camera_stream/                  live camera classification on the ExecuTorch runtime
```

Delegate programs run on the board with the existing
[`examples/espressif/project`](../../examples/espressif/project) runner, built
with `-DET_ESPNN_BACKEND=ON`.

## Supported operators

All delegated operators are int8 with per-tensor activations and per-channel
symmetric weights, as annotated by `EspNNQuantizer`.

| Group | Operators |
|---|---|
| Convolution | `conv2d` (stride, padding and dilation at most 255, groups 1, grouped or depthwise, dilation through phase decomposition), `conv_transpose2d` (groups 1), `linear` |
| Activations | `relu`, `relu6`, `hardtanh`, plus table lookups for `hardswish`, `hardsigmoid`, `sigmoid`, `silu`, `tanh`, `gelu`, `elu`, `leaky_relu` |
| Elementwise | `add`, `mul` (with broadcasting), `softmax` (dim 1 of batch-1 rank-2 or rank-4 tensors) |
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

## Quantize and lower with the delegate

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

# 3. Lower: int8 regions -> EspNNBackend delegates, the rest -> CPU kernels
edge, report = to_espnn_edge(torch.export.export(quantized, example_inputs))
print(report["partitions"], report["delegated_ops"], report["cpu_ops"])

# 4. Serialize the ExecuTorch program
program = edge.to_executorch()
with open("mobilenet_v2_espnn.pte", "wb") as f:
    f.write(program.buffer)
```

```
$ python examples/espressif/quantize_and_lower.py
1 EspNN partition(s), 101 delegated operators, 0 operators left for the CPU
wrote mobilenet_v2_espnn.pte
```

Calibrate on a few hundred real samples; random data gives useless scales.

### Inputs and outputs

The delegate keeps rank-4 tensors in NHWC order. Inputs are always copied
into slots in the delegate's activation arena, quantized or transposed if
needed, because ESP-NN kernels may load whole vectors past the end of a
tensor they read. For the same reason, an output that an operator reads keeps
its arena slot and is copied out at the end. The delegate writes any other
int8 output directly when its buffer is 16-byte aligned and already in this
layout: the tensor is not rank 4, has one channel or one pixel, or has a
channels-last dim order. Each output is handled independently, whether its
buffer is memory-planned by ExecuTorch or supplied by the caller.

Float outputs and NCHW outputs with several channels and pixels need dtype
or layout conversion. The exporter records which outputs need conversion,
so their int8 buffers can share the arena with intermediate tensors. Older
exports without this metadata use the arena for all outputs. An output
outside the arena whose buffer turns out misaligned is written to a staging
buffer and copied out. The delegate allocates that buffer at `init()`, sized
for every such output, in PSRAM when there is any.

For the smallest and fastest program, give the delegate int8, channels-last
inputs and outputs and leave them out of ExecuTorch's memory plan, so that
the application's own buffers are used without conversion:

```python
from executorch.exir import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.quantize_io_pass import quantize_input, quantize_output

# Example inputs (and calibration data) in channels-last memory format. An
# output that PyTorch would make contiguous, such as a slice of a cat, needs
# .clone(memory_format=torch.channels_last) at the end of forward().
example_inputs = (torch.randn(1, 3, 96, 96).to(memory_format=torch.channels_last),)
# ... quantize and calibrate as above ...
program = torch.export.export(quantized, example_inputs)
quantize_input(program, 0)
quantize_output(program, 0)
edge, report = to_espnn_edge(program)
config = ExecutorchBackendConfig(
    memory_planning_pass=MemoryPlanningPass(
        alloc_graph_input=False, alloc_graph_output=False
    )
)
pte = edge.to_executorch(config).buffer
```

The application then binds its buffers once, with `Method::set_input()` and
`Method::set_output_data_ptr()`, and each step is a call to
`Method::execute()`. An output may share memory with an input (for example,
to update a state tensor in place), since the inputs are copied before any
operator runs.

### Streaming state

A streaming model keeps a few frames of history between calls in a mutable
buffer. By default the buffer stays in the program, which passes it to the
delegate and back on every call. With `delegate_mutable_buffers=True`, the
delegate that reads and updates a buffer owns it instead: the program loses
that input and output, and the delegate keeps the buffer as int8 in its arena,
written in place when no operator reads the old value after the new one is
produced.

```python
class Streaming(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("frames", torch.zeros(1, 40, 2, 1))
        self.conv = torch.nn.Conv2d(40, 32, (3, 1))

    def forward(self, x):  # x: (1, 40, 1, 1), one new frame
        window = torch.cat([self.frames, x], dim=2)
        self.frames.copy_(window[:, :, 1:])
        return self.conv(window)

# ... prepare_pt2e, calibrate, convert_pt2e as above, then reset the buffer
# that calibration updated before exporting.
edge, report = to_espnn_edge(
    torch.export.export(quantized, (torch.zeros(1, 40, 1, 1),)),
    delegate_mutable_buffers=True,
)
print(report["delegated_buffers"])  # ['frames']
```

The delegate takes a buffer only when one partition holds every access to it
and the buffer is read and written with the same int8 quantization, as a
`cat` and slice shift register is with `EspNNQuantizer`. It takes a method's
buffers only when they all end up in one partition that keeps another input
and output, and at most 16 of them. Other buffers stay in the program, with a
warning.

On the device, each loaded method has its own state. It starts from the
buffer's initial value when the method is loaded and persists across
`execute()` calls. `reset_delegate_states()` (from `espnn/espnn_backend.h`)
or the `reset_states` backend option restores the initial state of every
EspNNBackend delegate before its next `execute()`; to reset one method alone,
load it again. After an `execute()` that fails, the state is undefined until
it is reset. The reset may be requested from any task.

```cpp
executorch::backends::espressif::reset_delegate_states();
// or, through the generic backend-option API:
executorch::runtime::BackendOptions<1> options;
options.set_option("reset_states", true);
executorch::runtime::set_option("EspNNBackend", options.view());
```

### Streaming keyword spotting

[`examples/espressif/export_streaming_model.py`](../../examples/espressif/export_streaming_model.py)
puts the pieces above together for a streaming keyword-spotting model: one
frame of features in, one score out, the frame history in delegate-owned
buffers, int8 input and output in the application's memory. The resulting
program has one input and one output, no CPU operators, and needs no
planned memory.

Quantization choices that matter for such models:

- `EspNNQuantizer` observes activations with histograms. For small streaming
  models, min/max ranges (`minmax_quantizer()` in the example) often track a
  float model, or a TensorFlow Lite reference, more closely.
- To reproduce an existing int8 model exactly, give each activation the
  reference model's scale and zero point: after `prepare_pt2e`, replace its
  observers with `FixedQParamsObserver(scale, zero_point, ...)`.
- Weight scales are `max|w| / 127` per output channel; check that a layer with
  very small weights is not clamped by the weight observer's `eps`.
- Calibrate on consecutive frames, so the histories see realistic values, and
  zero the buffers before exporting, since calibration updated them.

On the device, the application binds its buffers once and calls `execute()`
for each frame:

```cpp
#include <espnn/espnn_backend.h>
#include <executorch/runtime/executor/method.h>

// After Program::load() and load_method() with a method allocator sized by
// method_meta(), and no planned buffers (the program has none):
method.set_input(EValue(Tensor(&features_impl)), 0);  // int8[1, 40, 1, 1]
method.set_output_data_ptr(&score, sizeof(score), 0);  // int8[1, 1]
for (;;) {
  compute_features(features);  // writes the bound buffer in place
  method.execute();
  if (score > threshold) {
    on_detection();
    executorch::backends::espressif::reset_delegate_states();
  }
}
```

## Export with the Fused Quant operator library

Use the shared export API with `get_esp_fused_quant_recipe()`. The recipe
configures `get_esp_fused_quant_quantizer`, calibrates the model, fuses Q/DQ into Fused
Quant operators, and runs `EspPassManager`. Unsupported operators decompose
to portable kernels before serialization. No delegate partitions are created.

```python
from executorch.backends.espressif import (
    get_esp_fused_quant_recipe,
    get_operator_library_report,
)
from executorch.export import export

session = export(
    model.eval(),
    example_inputs=[example_inputs],
    export_recipe=get_esp_fused_quant_recipe(
        calibration_inputs_fn=lambda: calibration_samples,
    ),
)
edge = session.get_edge_program_manager()
print(get_operator_library_report(edge))
program = session.get_executorch_program_manager()
with open("model_esp_ops.pte", "wb") as f:
    f.write(program.buffer)
```

Each calibration sample is a tuple matching the model's inputs. Pass
`allow_cpu_fallback=False` to the recipe to reject remaining portable compute;
Q/DQ boundary operators are allowed. The report lists portable compute and
counts accelerated operators across all methods.
Reports also accept the graph after `to_executorch()`: memory views appear as
layout operations, allocation nodes are omitted, and operator targets include
their lowered `.out` overloads.

`EspPassManager` expects Fused Quant Edge programs. Legacy PT2E Q/DQ graphs
passed directly to it retain portable execution; export float models with
`get_esp_fused_quant_recipe()` to use the operator library. The recipe uses
explicit layout operations rather than dim-order operators, matching the
ESP-IDF runtime and the operator implementation tests.
When checking supported int8 operators, lowering rejects malformed per-tensor
qparams, such as non-positive scales or out-of-range zero points. Valid
unsupported configurations remain portable.
ESP-NN pooling requires matching input and output qparams. A shared optimization
that absorbs a following requantization can change the output qparams and leave
that pool on the portable path.

For this flow, build the existing runner with `-DET_ESPRESSIF_OPS=ON` and
`-DET_ESPNN_BACKEND=OFF`, using `model_esp_ops.pte` as the model. The runtime
still needs the portable fallback operators and Q/DQ kernels described below.
`to_espnn_edge()` is the separate delegate entry point; its former
`use_operator_library` option is replaced by this recipe.

## Running on the ESP32-S3, ESP32-P4 and ESP32-S31

You need ESP-IDF and an activated ESP-IDF shell.
[`examples/espressif/setup.sh`](../../examples/espressif/setup.sh) installs a
matching ESP-IDF and the emulator.

### 1. Cross-compile ExecuTorch

Build the runtime with the `esp-baremetal` preset. Select the portable CPU
operators your program needs. The quantized kernels provide the
quantize/dequantize operators, and the ESP-IDF component provides the ESP-NN
operators. Both export examples above retain the serialized program manager
as `program`; use it to collect portable operators across all methods:

```python
print(",".join(sorted({
    f"{op.name}.{op.overload}"
    for plan in program.executorch_program.execution_plan
    for op in plan.operators
    if not op.name.startswith(("quantized_decomposed::", "esp_nn::"))
})))
```

```bash
cmake --preset esp-baremetal -G Ninja -B cmake-out-esp \
    -DCMAKE_TOOLCHAIN_FILE="${IDF_PATH}/tools/cmake/toolchain-esp32s3.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_DEVTOOLS=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_SELECT_OPS_LIST=<operators printed above> \
    -DMAX_KERNEL_NUM=256 \
    -DEXECUTORCH_BUILD_PTHREADPOOL=OFF -DEXECUTORCH_BUILD_CPUINFO=OFF
cmake --build cmake-out-esp --target install
```

For the ESP32-P4 and ESP32-S31, use the toolchain file that ESP-IDF writes
when a project configures for the target, as described in
[Cross-Compiling ExecuTorch](../../examples/espressif/README.md#cross-compiling-executorch).

- `quantized_ops_lib` registers kernels beyond the selected list, so
  `MAX_KERNEL_NUM` is set explicitly instead of being sized from the
  selection.
- pthreadpool needs POSIX calls that ESP-IDF doesn't provide. With it off,
  the quantized kernels run single-threaded.
- The runner links `executorch_selected_kernels`, which only exists when a
  selection is given. A program with no portable CPU operators
  still needs at least one selected operator, such as `aten::add.out`.

### 2. Build the runner with the ESP-NN backend

```bash
idf.py -C examples/espressif/project -B build-espnn \
    -DIDF_TARGET=esp32s3 \
    -DET_ESPNN_BACKEND=ON \
    -DET_BUILD_DIR_PATH=$PWD/cmake-out-esp \
    -DET_PTE_FILE_PATH=$PWD/mobilenet_v2_espnn.pte \
    build flash monitor
```

`ET_ESPNN_BACKEND` adds
[`backends/espressif/espnn_runtime`](espnn_runtime) to the project. The
component pulls in `espressif/esp-nn` from the component registry, links
against the ExecuTorch targets that `executor_runner` imports, and registers
`EspNNBackend` at startup. It defines no platform layer; the runner's
`esp_pal.cpp` provides it.

ESP-NN defaults to `CONFIG_NN_OPTIMIZED=y` and selects its implementation from
ESP-IDF's `IDF_TARGET`. For `esp32s3`, this enables the S3 assembly kernels;
for `esp32p4` and `esp32s31`, the PIE kernels.
`CONFIG_NN_ANSI_C=y` selects reference C kernels for verification and debugging.

For a BundleIO program (`.bpte`), add `-DET_BUNDLE_IO=ON` and the tolerances
`-DET_ATOL=... -DET_RTOL=...`. The runner then checks the outputs against the
bundled reference and logs `Test_result: PASS` or `FAIL`.

Other options, in menuconfig under *ExecuTorch ESP-NN backend*:

| Kconfig | Default | Effect |
|---|---|---|
| `ESPNN_DELEGATE_CONSTANTS_TO_PSRAM` | y | Copies weights from memory-mapped flash into PSRAM when they fit (faster than flash cache misses on large models) |
| `ESPNN_DELEGATE_DUAL_CORE` | n | Splits large convolutions across both cores |
| `ESPNN_DELEGATE_SCRATCH_IN_ARENA` | n | Plans each operator's ESP-NN scratch in the activation arena instead of a separate buffer sized for the largest operator. Saves internal RAM when the arena fits there; for a large model it moves the scratch to PSRAM with the arena |
| `ESPNN_DELEGATE_PROFILE` | n | Times every operator and prints a per-operator time table after a delegate's first run. The timer reads cost about 1 µs per operator on the ESP32-S3, so leave it off in production |

### The original ESP32

The delegate also runs on the ESP32. ESP-NN has no SIMD kernels for it and
uses portable C, about 13 million int8 multiply-accumulates per second at
240 MHz, roughly an order of magnitude slower than the S3. Most ESP32 modules
have no PSRAM, so activations must fit in about 280 KB of internal RAM.

## Testing

```bash
pytest backends/espressif/test
```

The tests export small models for every operator group, lower them, run the
XTS3 graphs through the reference interpreter and compare the result against
the quantized PyTorch model. `test_xts3_parser_host.py` builds the device
parser for the host and feeds it corrupted graphs.

End to end, `.ci/scripts/test_espressif.sh` runs the same models through the
ExecuTorch runtime on an emulated ESP32-S3, ESP32-P4 or ESP32-S31.
`examples/espressif/export_espnn_test_programs.py` exports each one as a
BundleIO program with the host reference's outputs. CI builds the runner with
`-DET_ESPNN_BACKEND=ON` for each program and runs it in `esp-emu`. Fully
delegated programs must match to float precision, which means bit-exact int8
results. Programs with CPU regions get a tolerance for the portable kernels'
float rounding. Separate CI jobs run ESP-NN's optimized kernels on each
chip, and its ANSI C reference kernels on ESP32-S3, against the same bundled
outputs. Logs include the kernel configuration in their filenames; artifacts
retain the bundled fixtures, SDK configuration, and resolved component versions.

### On the host

The delegate also builds outside ESP-IDF, for Linux and macOS, over ESP-NN's
portable C kernels. `host/include` stands in for the few ESP-IDF APIs it uses,
and ESP-NN is fetched from GitHub (or taken from `-DESPNN_LOCAL_PATH=...`).
`test_host_runtime.py` runs programs through the ExecuTorch runtime, this
build and the portable kernels, and checks every output against the host
reference (bit-exactly for fully delegated programs), with no board or
emulator:

```bash
cmake -S backends/espressif/test/host -B cmake-out-espnn-host -G Ninja
cmake --build cmake-out-espnn-host
ESPNN_HOST_RUNNER=cmake-out-espnn-host/espnn_host_runner \
    pytest backends/espressif/test/test_host_runtime.py
```

`-DESPNN_HOST_SANITIZE=ON` builds with AddressSanitizer and
UndefinedBehaviorSanitizer, `-DESPNN_HOST_OPTIMIZED=ON` uses ESP-NN's optimized
C kernels instead of the reference ones, `-DESPNN_HOST_PSRAM=ON` makes the
heap stand-in serve PSRAM allocations, and `-DESPNN_HOST_SCRATCH=ON` makes
the convolution and mean kernels request scratch and overwrite all of it, as
a check on where the delegate puts it. `ESPNN_DELEGATE_CONSTANTS_TO_PSRAM`,
`ESPNN_DELEGATE_PROFILE` and `ESPNN_DELEGATE_SCRATCH_IN_ARENA` are CMake
options here, off by default. `.ci/scripts/test_espressif_host.sh`
runs a plain build and three sanitized ones that cover these options. The host
build checks the delegate's logic, not ESP-NN's S3 assembly kernels, which only
the emulator and the board run.

The code follows the ExecuTorch coding style and passes the repository's
`lintrunner` checks:

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
  delegate allocates its activation arena, scratch and staging buffers and
  optional PSRAM weight copy with ESP-IDF's `heap_caps` allocator at
  `init()`, so it can choose between internal RAM and PSRAM. Nothing is
  allocated during `execute()`, except that with `ESPNN_DELEGATE_DUAL_CORE`
  ESP-NN's worker task grows its own scratch buffer the first time a large
  convolution needs it.

## License

Every source file in this backend and its examples is released under the
BSD 3-Clause license (`SPDX-License-Identifier: BSD-3-Clause`), the same
license as ExecuTorch. See the `LICENSE` file at the repository root. ESP-NN
is Apache-2.0 and is fetched from the Espressif component registry (or, for
the host build, from GitHub), not vendored.
