# OpenVINO Backend

In this tutorial we will walk you through the process of setting up the prerequisites, building OpenVINO backend library, exporting `.pte` models with OpenVINO optimizations, and executing the exported models on Intel hardware.

<!----This will show a grid card on the page----->
::::{grid} 2
:::{grid-item-card}  What you will learn in this tutorial:
:class-card: card-prerequisites
* In this tutorial you will learn how to lower and deploy a model with OpenVINO.
:::
:::{grid-item-card}  Tutorials we recommend you complete before this:
:class-card: card-prerequisites
* [Introduction to ExecuTorch](intro-how-it-works.md)
* [Setting up ExecuTorch](getting-started.md)
* [Building ExecuTorch with CMake](using-executorch-building-from-source.md)
:::
::::

## Introduction to OpenVINO

[OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) is an open-source toolkit designed to enhance AI inference on Intel hardware by reducing latency and increasing throughput while preserving accuracy. It optimizes hardware utilization and simplifies AI development and deep learning integration across domains such as computer vision, large language models (LLMs), and generative AI.

OpenVINO is integrated as an Executorch delegate to accelerate AI applications deployed with Executorch APIs.

## Supported Hardware

OpenVINO backend supports the following hardware:

- Intel CPUs
- Intel integrated GPUs
- Intel discrete GPUs
- Intel NPUs

For more information on the supported hardware, please refer to [OpenVINO System Requirements](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html) page.

## Instructions for Building OpenVINO Backend

### Quick Start (pip wheel, Linux and Windows)

The easiest way to use the OpenVINO backend is via the prebuilt pip wheel:

```bash
pip install executorch[openvino]
```

For now the extra installs no runtime on Python 3.15, where OpenVINO publishes no build.
On Linux aarch64, OpenVINO needs glibc 2.35 or newer, as on Ubuntu 22.04 or Raspberry Pi OS
Bookworm and later. On an older aarch64 system, `pip install executorch[openvino]` does not
fail: pip finds no OpenVINO 2025.4 or newer for that system, so it installs an older
ExecuTorch release instead. On such a system, install `executorch` without the extra, then
build OpenVINO from source as described below.

The backend automatically discovers the OpenVINO C library from the pip-installed
package — no environment variable setup is needed.

See the [OpenVINO backend README](../../backends/openvino/README.md) for verification steps and troubleshooting.

### Building from Source

If you need a custom build (different platform, custom configuration, etc.), follow the instructions below.

### Prerequisites

Before you begin, ensure you have the OpenVINO runtime installed and configured on your system. For most users, we recommend installing it via pip:

```bash
pip install "openvino>=2025.4.0,<2027.0.0"
```

Alternatively, you can build OpenVINO from source or follow the official [OpenVINO installation guide](https://docs.openvino.ai/2025/get-started/install-openvino.html) for platform-specific options:

```bash
git clone https://github.com/openvinotoolkit/openvino.git
cd openvino && git checkout releases/2026/4
git submodule update --init --recursive
sudo ./install_build_dependencies.sh
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=ON
make -j<N>

cd ..
cmake --install build --prefix <your_preferred_install_location>
cd <your_preferred_install_location>
source setupvars.sh
```
For more information about building OpenVINO from source, refer to the [OpenVINO Build Instructions](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build_linux.md).

### Setup

Follow the steps below to setup your build environment:

1. **Setup ExecuTorch Environment**: Refer to the [Environment Setup](using-executorch-building-from-source.md#environment-setup) guide for detailed instructions on setting up the ExecuTorch environment.

2. **Setup OpenVINO Backend Environment**
- Install the dependent libs. Ensure that you are inside `executorch/backends/openvino/` directory
   ```bash
   pip install -r requirements.txt
   ```

3. Navigate to `scripts/` directory.

4. **Build OpenVINO Backend**: Once the prerequisites are in place, run the `openvino_build.sh` script to start the build process, OpenVINO backend will be built under `cmake-out/backends/openvino/` as `libopenvino_backend.a`

   ```bash
   ./openvino_build.sh
   ```

## Precision

When the device is `CPU`, a program computes in float32 by default. Without this default,
OpenVINO would compute in bfloat16 on x86 CPUs with bfloat16 units, which keeps only two to
three significant digits, and in float16 on ARM CPUs, which keeps about three.
Both are faster. To trade accuracy for speed, ask for the faster precision when
lowering:

```python
from executorch.backends.openvino.partitioner import OpenvinoPartitioner
from executorch.exir.backend.compile_spec_schema import CompileSpec

OpenvinoPartitioner(
    [CompileSpec("device", b"CPU"), CompileSpec("inference_precision", b"bf16")]
)
```

On an ARM CPU, ask for `f16` instead: there float32 is about 1.8x slower than
float16, as measured on an Orin Nano.

The value is OpenVINO's `INFERENCE_PRECISION_HINT`: `f32`, `f16` or `bf16`. An
empty value means the default, and any other value fails at lowering. A CPU
without float16 support computes an `f16` program in bfloat16 if it has bfloat16
units, as some x86 CPUs do, and in float32 otherwise. OpenVINO has no
bfloat16 on ARM CPUs, so there `bf16` keeps the ARM default, float16 where the
CPU supports it. Which x86 CPUs compute in bfloat16 is up to OpenVINO; see its
[Precision Control](https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference/precision-control.html)
page. Other devices, such as GPUs, keep OpenVINO's default unless the option is
set.

The default also applies to a program exported in float16: on a CPU it computes
in float32 unless you lower it with `inference_precision`.

On x86 CPUs, OpenVINO applies the precision when the runtime loads a program.
So a program lowered without `inference_precision` computes in float32, even
one exported with an ExecuTorch release that did not have this default. On an
x86 CPU with bfloat16 units, float32 is slower and can use more memory; export it
again with `inference_precision` set to `bf16` to get the earlier speed. Other
x86 CPUs already computed such a program in float32. On ARM CPUs, OpenVINO applies the
precision when the program is lowered, so a program exported with such an
earlier release keeps computing in float16 until it is exported again.

OpenVINO's CPU plugin computes 64-bit integers in 32 bits. An integer result
outside the int32 range wraps around without an error. This applies to every
int64 tensor, including the int64 result of `cumsum` and `prod` over smaller
integer types. In OpenVINO 2026.4.1, the CPU plugin also computes some int32 and
int64 ops in float32. On x86 CPUs these are `sum`, `amax`, `amin`, `max` and
`min`. On ARM CPUs they are element-wise ops such as `add`, `maximum` and
`clamp`, and the same reductions along any dimension but the last. Once a value
or a running total is above 16,777,216 (2^24) or below -16,777,216, such a
result can be off, even inside the int32 range. For example, on x86 `sum` over a
tensor that holds 16,777,216 and 1 gives 16,777,216, and so does `amax` over one
that holds 16,777,217.

In OpenVINO 2026.4.1, the CPU plugin's `min(dim)` and `max(dim)` can return a
later index on ties than PyTorch, which returns the first one. This happens with
float32 as well as integer dtypes such as int32 and uint8.

## Copies

OpenVINO turns a copy, such as `x.clone()`, into nothing, so a partition that returns
a copy would return its input. The backend handles only these returned copies: a
`clone()` or a full slice such as `x[:, :]` of a partition input or a constant, in
the default dim order.

- A copy of a partition input (a real input, a buffer copied before the program
  changes it, or a value computed outside OpenVINO) runs on the CPU, outside
  OpenVINO.
- A copy of a constant (a parameter, a buffer the program does not change, or a
  tensor attribute) stays in the partition, so the program stores the constant
  once. It stays only when:
  - nothing outside the partition reads the constant,
  - no other output of the partition is a copy of the same constant,
  - its type is bool, uint8, int8, int16, int32, float16, bfloat16 or float32,
  - for a float copy, the precision is `f32`, the CPU default. On a GPU this
    needs `inference_precision` set to `f32`. The NPU does not accept `f32`, so
    float copies do not stay there. On an ARM CPU with float16 support, at `f16`
    or `bf16`, OpenVINO would round a float32 copy.

Any other clone or full slice of a constant runs on the CPU. If the partition
also reads the constant, the program then stores it twice, and a constant marked
`no_copy` fails to export. With dim order off
(`EdgeCompileConfig(_skip_dim_order=True)`, the default for LLM exports), a
clone that runs on the CPU works only without a `memory_format` or with
`torch.contiguous_format`. With `torch.preserve_format` it fails at run time,
and with `torch.channels_last` it fails at export.

Any other returned copy is not handled. It can come back with wrong values, or
make ops in the partition that read it return wrong values, without an error.
Examples are a copy made with `.to()`, a copy in another dim order such as
`channels_last`, and a copy of a value computed inside the partition. A copy of
a value that the partition also returns, such as `self.buf.clone()` after the
program changes `self.buf`, fails to lower.

When a float32 constant holds a subnormal value, the CPU plugin turns its
subnormal values and its negative zeros into zero. So a float32 copy that stays
in the partition returns `[1e-40, -0.0, 1.0]` as `[0.0, 0.0, 1.0]`, and `x * b`
reads such a constant `b` the same way.

A whole program lowered with `to_backend` has no CPU part, so it gets none of
the handling above. When it returns an input unchanged, in the default dim
order, the backend copies it with OpenVINO's `Identity` on the CPU device. On an
ARM CPU with float16 support, at `f16` or `bf16`, that copy of a float32 input
is rounded, like the other float32 outputs. On every CPU, the copy of an int64
input is computed in 32 bits, like every int64 tensor.

## Build Instructions for Examples

### AOT step:
Refer to the [README.md](../../examples/openvino/README.md) in the `executorch/examples/openvino` folder for detailed instructions on exporting deep learning models from various model suites (TIMM, Torchvision, Hugging Face) to openvino backend using Executorch. Users can dynamically specify the model, input shape, and target device.

Below is an example to export a ResNet50 model from Torchvision model suite for CPU device with an input shape of `[1, 3, 256, 256]`

```bash
cd executorch/examples/openvino
python aot_optimize_and_infer.py --export --suite torchvision --model resnet50 --input_shape "(1, 3, 256, 256)" --device CPU
```
The exported model will be saved as 'resnet50.pte' in the current directory.

### Build C++ OpenVINO Examples

After building the OpenVINO backend following the [instructions](#setup) above, the executable will be saved in `<executorch_root>/cmake-out/`.

The executable requires a model file (`.pte` file generated in the aot step) and the number of inference executions.

#### Example Usage

Run inference with a given model for 10 executions:

```
./executor_runner \
    --model_path=model.pte \
    --num_executions=10
```



## Support

If you encounter any issues while reproducing the tutorial, please file a github
issue on ExecuTorch repo and tag use `#openvino` tag
