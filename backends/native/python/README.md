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

The wheel is not yet published to PyPI. Install a built, repaired wheel; once
published, install with `python -m pip install executorch-native`.

```sh
python -m pip install /path/to/executorch_native-*.whl
```

Export a `.ptn` and its preprocessed input with the existing
[Native vision exporter](../../../examples/native/vision/export.py) from the
ExecuTorch source root. Exporting runs on a development machine with PyTorch,
torchvision, Pillow, and a compatible ExecuTorch exporter; it does not require the
Native wheel.

```sh
python -m executorch.examples.native.vision.export \
  --model mv2 --image /path/to/photo.jpg --out-dir native-demo
```

The exporter writes the model, FP32 input and reference logits, and labels under
`native-demo/mv2/`. Load the `.ptn` and run it with Native and NumPy:

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
input/reference, and optional categories files. It prints the top five predictions
and, with `--expected`, checks all logits at `rtol=1e-3, atol=1e-3`:

```sh
python -m executorch_native.vision \
  --package native-demo/mv2/mv2.ptn \
  --input native-demo/mv2/mv2_input.bin \
  --expected native-demo/mv2/mv2_expected.bin \
  --categories native-demo/mv2/categories.txt
```

The Python runner lives beside the C++ runner in `backends/native/examples/vision`;
the wheel installs it as `executorch_native.vision`.

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
The wheel reads `version.txt` for local builds. `BUILD_VERSION` overrides it in
release and nightly builds, just as it does for the `executorch` wheel.

The build reuses existing CMake targets and installs only the Native wheel
component. Building source distributions from this subdirectory is not supported;
build wheels directly from the complete checkout.

## Release and nightly CI

Dedicated Linux x86_64 and aarch64 wheel workflows build `executorch-native`
with the same triggers, Python version matrix, source revision, and reusable
PyTorch wheel workflow as `executorch`. Native has its own workflows because
the reusable workflow keys its concurrency group and artifact names by
repository, so two packages from one repository cannot share a workflow run.
Native includes both GGML and Vulkan in each wheel; it does not have separate
CUDA variants. The initial platform coverage is Linux.

The existing wheel builder invokes `setup.py bdist_wheel`. The small `setup.py` in this directory
forwards that command to the PEP 517 recipe above. The Native hooks build libzip
and FlatBuffers in the manylinux environment, install the shader compiler, repair
the wheel, and test it.

The wheels are not published yet. The workflows pass a non-`push` trigger event
for `executorch-native`, so the reusable workflow keeps the repaired wheels as
workflow artifacts and does not upload them. Before publishing, the package must
be registered with the release infrastructure: the PyPI project, the
download.pytorch.org nightly index, and release promotion. Then pass
`github.event_name` through as for `executorch`.

The post-build check exports add and Conv/BatchNorm/ReLU models using the source
revision being released. It then creates a fresh virtual environment and installs
only the repaired Native wheel with its declared dependencies. It checks GGML
results against PyTorch, runs the Python binding suite, and rejects an
environment containing Torch or ExecuTorch. Vulkan hardware is not required
for this CPU check.

To reproduce the check from the ExecuTorch source root, first export in your
development environment, then let the test create the clean runtime environment:

```sh
python .ci/scripts/wheel/export_native.py --output /tmp/native-fixtures
python .ci/scripts/wheel/test_native.py \
  --wheel /path/to/repaired/executorch_native-*.whl \
  --source . --fixtures /tmp/native-fixtures
```

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

Set `NATIVE_TEST_ENGINE=vulkan` to exercise the same tests on Vulkan. The tests
use the untrained fixture, including a subprocess run of `executorch_native.vision`
with custom inputs, labels, and reference.
