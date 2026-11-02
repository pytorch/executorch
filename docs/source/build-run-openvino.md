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

For now the extra installs no runtime on Python 3.15, or on Linux aarch64 with Python
3.14. OpenVINO publishes no build for Python 3.15, and its aarch64 builds for Python 3.14
need glibc 2.35 or newer.

The backend automatically discovers the OpenVINO C library from the pip-installed
package — no environment variable setup is needed.

See the [OpenVINO backend README](../../backends/openvino/README.md) for verification steps and troubleshooting.

### Building from Source

If you need a custom build (different platform, custom configuration, etc.), follow the instructions below.

### Prerequisites

Before you begin, ensure you have the OpenVINO runtime installed and configured on your system. For most users, we recommend installing it via pip:

```bash
pip install "openvino>=2025.1.0,<2027.0.0"
```

Alternatively, you can build OpenVINO from source or follow the official [OpenVINO installation guide](https://docs.openvino.ai/2025/get-started/install-openvino.html) for platform-specific options:

```bash
git clone https://github.com/openvinotoolkit/openvino.git
cd openvino && git checkout releases/2025/1
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
without float16 support computes an `f16` program in float32. OpenVINO has no
bfloat16 on ARM CPUs, so there `bf16` keeps the ARM default, float16 where the
CPU supports it. Which x86 CPUs compute in bfloat16 is up to OpenVINO; see its
[Precision Control](https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference/precision-control.html)
page. Other devices, such as GPUs, keep OpenVINO's default unless the option is
set.

The default also applies to a program exported in float16: on a CPU it computes
in float32 unless you lower it with `inference_precision`.

On x86 CPUs, OpenVINO applies the precision when the runtime loads a program.
So a program lowered without `inference_precision` computes in float32, even
one exported with an ExecuTorch release that did not have this default. That
is slower and can use more memory. Export it again with `inference_precision`
set to `bf16` to get the earlier speed. On ARM CPUs, OpenVINO applies the
precision when the program is lowered, so a program exported with such an
earlier release keeps computing in float16 until it is exported again.

OpenVINO's CPU plugin computes 64-bit integers in 32 bits. An integer result
outside the int32 range wraps around without an error. This applies to every
int64 tensor, including the int64 result of `cumsum` and `prod` over smaller
integer types.

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
