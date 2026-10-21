# ExecuTorch Native Python runtime and SDK wheel

This directory contains the standalone wheel build recipe. Python sources live
alongside the runtime in [`../pybindings`](../pybindings); CMake installs them as
`executorch_native`. The wheel does not depend on the `executorch` distribution.

`executorch-native` packages the Native runtime, Vulkan engine (including compiled
shaders), and GGML CPU engine. It installs into `executorch_native`, independently
of the existing `executorch` distribution. Python execution uses NumPy and the
same shared runtime as C++ consumers, with no PyTorch runtime dependency.
Export `.ptn` files using a compatible version of `executorch` on the development
machine; exporters are not bundled in this wheel.

The initial wheel supports Linux. Build a wheel for each supported CPython version
and architecture; the Python extension uses the CPython ABI. Vulkan execution
requires a system Vulkan loader and a compatible driver. GGML runs on the CPU
without a Vulkan installation.

## Install and run Python

The wheel is not yet published to PyPI. Install a built, repaired wheel and run
the bundled pretrained MobileNetV2 example:

```sh
python -m pip install /path/to/executorch_native-*.whl
python -m executorch_native.vision
```

Once published, install with `python -m pip install executorch-native`.
The demo runs entirely offline with Native and NumPy, from any working directory.
It uses the bundled model, preprocessed cat image tensor, ImageNet labels, and
reference logits; no PyTorch, torchvision, Pillow, exporter, or compiler is needed.
It prints the top five predictions (`tiger cat` first) and checks all logits against
the reference at `rtol=1e-3, atol=1e-3`. GGML is the default engine; use
`--engine vulkan` to select Vulkan. Asset provenance and checksums are in
[`../examples/vision/data/README.md`](../examples/vision/data/README.md).

For your own exported model, load its `.ptn` and preprocessed input:

```python
import numpy as np
from executorch_native import Model

model = Model.load("native-demo/mv2/mv2.ptn")
spec = model.method_info().inputs[0]
image = np.fromfile("native-demo/mv2/mv2_input.bin", dtype=spec.dtype).reshape(spec.shape)
session = model.create_session()  # GGML CPU; use engine="vulkan" for Vulkan.
(logits,) = session.run([image])
print(logits.argmax())
```

Models expose `method_names` and `method_info(name)` for input/output shapes and
NumPy dtypes. `session.run(inputs, method="forward")` returns a list of owned NumPy
arrays. Input shapes and dtypes must match exactly; noncontiguous inputs are copied
into the model's layout. Bfloat16 I/O is not supported. Dynamic inputs currently
use their serialized upper-bound shapes, matching the C++ Model API. Sessions own
their model and engine lifetimes, serialize concurrent calls, and release the GIL
during execution. Returned arrays remain valid after another call or session deletion.

The Python equivalent of the C++ vision runner accepts the same `.ptn`, raw FP32
input/reference, and optional categories files:

```sh
python -m executorch_native.vision \
  --package native-demo/mv2/mv2.ptn \
  --input native-demo/mv2/mv2_input.bin --expected native-demo/mv2/mv2_expected.bin \
  --categories native-demo/mv2/categories.txt
```

`--package` and `--input` must be supplied together. Custom models use only the
labels and reference explicitly supplied with `--categories` and `--expected`.
Those flags can also override the bundled demo's labels and reference. The vision
runner also accepts artifacts from `executorch.examples.native.vision.export`.
Loading uses a local path; a future Hugging Face downloader can pass its cached `.ptn` path to
the same `Model.load` API without changing the execution code.

## Regenerate the demo or export another image

Use the existing [Native vision exporter](../../../examples/native/vision/export.py)
from the ExecuTorch source root. Exporting runs on a development machine with
PyTorch, torchvision, Pillow, and a compatible ExecuTorch exporter containing the
shared BatchNorm fusion pass. It does not require the Native wheel.

```sh
python -m executorch.examples.native.vision.export \
  --model mv2 --image examples/native/vision/cat.jpg --out-dir native-demo
python -m executorch_native.vision \
  --package native-demo/mv2/mv2.ptn \
  --input native-demo/mv2/mv2_input.bin \
  --expected native-demo/mv2/mv2_expected.bin \
  --categories native-demo/mv2/categories.txt
```

The exporter downloads MobileNetV2's ImageNet-1K V2 weights on first use (cached
by torchvision), preprocesses the image with the weights' standard transforms,
and writes the model, FP32 input and reference logits, and labels under
`native-demo/mv2/`. Native's default AOT pipeline folds evaluation BatchNorm into
convolutions. Use `--image /path/to/photo.jpg` to export another image, or
`--engine vulkan` on the runner to select Vulkan.

The source image is PyTorch/Caffe2's
`caffe2/python/trt/data/tabby_tiger_cat.jpg` test image. Image preprocessing and
downloads happen during export; the installed demo uses only bundled artifacts.
The Python runner lives beside the C++ runner in `backends/native/examples/vision`;
the wheel installs it as `executorch_native.vision` and puts the shared artifacts
under `executorch_native/share/examples/vision/data`.

## Use the installed SDK

The SDK requires the libzip runtime. If your wheel does not bundle it, install
the system package before running an SDK application, for example:

```sh
sudo apt install libzip5
```

The package name varies by distribution and release. Wheels repaired with
`auditwheel` normally bundle libzip, as described below.

After obtaining a built wheel:

```sh
python -m pip install executorch_native-*.whl
cmake -S "$(python -m executorch_native --example-path)" -B vision-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(python -m executorch_native --cmake-prefix-path)"
cmake --build vision-build --parallel
vision-build/run_classification --package model.ptn --input input.bin \
  --expected expected.bin
```

This compiles only the runner and its small I/O utilities. The example also
requires the gflags development package (for example `libgflags-dev` on Ubuntu).
It needs neither an ExecuTorch checkout nor a shader compiler, GGML, or libzip
development package. The wheel is a prototype; it is not yet published to PyPI.

Other applications can consume the same SDK:

```cmake
find_package(executorch_native CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE executorch_native::runtime)
```

Both `ptn::VulkanEngineHost` and `ptn::GgmlEngineHost` are available through this
target. Native headers retain their `executorch/backends/native/...` include
paths. The wheel owns no files under the `executorch` Python package and exports
only the `ptn` API from its shared library, keeping bundled implementation symbols
private when an application also loads ExecuTorch.

## Build a wheel

Build from a complete ExecuTorch source checkout with its pinned submodules
initialized. Build dependencies are the same as the Native CMake build: a C++20
compiler, libzip development files, and `glslc` from Vulkan SDK 1.4.341.1 or newer.

```sh
python -m pip install build auditwheel patchelf
python -m build --wheel backends/native/python --outdir dist/native \
  -Ccmake.define.EXECUTORCH_NATIVE_EXTRA_LICENSES=/path/to/libzip/LICENSE
auditwheel repair dist/native/*.whl --wheel-dir dist/native/repaired
```

Supply the license files for external libraries bundled by the build or repair
step through `EXECUTORCH_NATIVE_EXTRA_LICENSES` (a semicolon-separated list).
Distribute the repaired wheel: repair bundles shared dependencies such as libzip
and assigns a compatible manylinux tag. For release wheels, build in the chosen
manylinux image for each supported architecture; building on a newer Linux host
does not establish compatibility with older systems. GGML uses portable CPU
settings, with native tuning and optional x86 instruction sets disabled, rather
than assuming the build machine's instruction set is available on every consumer.
Keep the existing ExecuTorch release version; the wheel reads `version.txt`.

The build reuses existing CMake targets and installs only the Native wheel
component. Building source distributions from this subdirectory is not supported;
build wheels directly from the complete checkout.

## Validate the installed wheel

Generate the small test model on the build machine, then configure the tests
against an installed, repaired wheel in a fresh environment:

```sh
python backends/native/python/tests/make_fixture.py --flatc /path/to/flatc \
  --output /tmp/native-fixture
cmake -S backends/native/python/tests -B sdk-test \
  -DCMAKE_PREFIX_PATH="$(python -m executorch_native --cmake-prefix-path)" \
  -DNATIVE_TEST_PACKAGE=/tmp/native-fixture/add.ptn
cmake --build sdk-test
ctest --test-dir sdk-test --output-on-failure
```

Add `-DNATIVE_TEST_VULKAN=ON` to test both engines when a Vulkan device is
available. The installed vision example also accepts this fixture with
`--input /tmp/native-fixture/input.bin --expected /tmp/native-fixture/expected.bin`.
The fixture generator needs `flatc`; consumers of the wheel do not. Run the Python
binding tests against the installed wheel:

```sh
NATIVE_TEST_PACKAGE=/tmp/native-fixture/add.ptn \
  python -m unittest discover -s backends/native/pybindings/test -p test_bindings.py
```

Set `NATIVE_TEST_ENGINE=vulkan` to exercise the same tests on Vulkan.

The tests include a bounded subprocess run of the bundled demo from a temporary
working directory, plus custom inputs, labels, reference overrides, and argument
pairing. To verify the offline installation, run the tests in a fresh environment
containing only the repaired Native wheel and its NumPy dependency, with network
access disabled.
