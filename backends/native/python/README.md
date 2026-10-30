# ExecuTorch Native SDK wheel

This directory contains the standalone wheel build recipe. Python sources live
alongside the runtime in [`../pybindings`](../pybindings); CMake installs them as
`executorch_native`. The wheel does not depend on the `executorch` distribution.

`executorch-native` packages the Native runtime, Vulkan engine (including compiled
shaders), and GGML CPU engine. It installs into `executorch_native`, independently
of the existing `executorch` distribution. It has no PyTorch dependency and does
not install exporters or Python execution bindings. Export `.ptn` files using a
matching version of `executorch` on the development machine.

The initial wheel supports Linux. It contains C++20 headers and a shared library;
it is independent of the CPython ABI, but platform and C++ ABI compatibility still
apply. Vulkan execution requires a system Vulkan loader and a compatible driver.

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
The fixture generator needs `flatc`; consumers of the wheel do not.
