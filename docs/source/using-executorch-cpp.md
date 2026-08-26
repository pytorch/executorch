# Using ExecuTorch with C++

In order to support a wide variety of devices, from high-end mobile phones down to tiny embedded systems, ExecuTorch provides an API surface with a high degree of customizability. The C++ APIs expose advanced configuration options, such as controlling memory allocation, placement, and data loading. To meet the needs of both application and embedded programming, ExecuTorch provides a low-level, highly-customizable core set of APIs, and set of high-level extensions, which abstract away many of the low-level details that are not relevant for mobile application programming.

## High-Level APIs

The C++ `Module` class provides the high-level interface to load and execute a model from C++. It is responsible for loading the .pte file, configuring memory allocation and placement, and running the model. The Module constructor takes a file path and provides a simplified `forward()` method to run the model.

In addition the Module class, the tensor extension provides an encapsulated interface to define and manage tensor memory. It provides the `TensorPtr` class, which is a "fat" smart pointer. It provides ownership over the tensor  data and metadata, such as size and strides. The `make_tensor_ptr` and `from_blob` methods, defined in `tensor.h`, provide owning and non-owning tensor creation APIs, respectively.

```cpp
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

using namespace ::executorch::extension;

// Load the model.
Module module("/path/to/model.pte");

// Create an input tensor.
float input[1 * 3 * 256 * 256];
auto tensor = from_blob(input, {1, 3, 256, 256});

// Perform an inference.
const auto result = module.forward(tensor);

if (result.ok()) {
  // Retrieve the output data.
  const auto output = result->at(0).toTensor().const_data_ptr<float>();
}
```

For more information on the Module class, see [Running an ExecuTorch Model Using the Module Extension in C++](extension-module.md). For information on high-level tensor APIs, see [Managing Tensor Memory in C++](extension-tensor.md).

For complete examples of building and running a C++ application using the Module API, refer to our [examples GitHub repository](https://github.com/meta-pytorch/executorch-examples/tree/main/mv2/cpp).

## Low-Level APIs

Running a model using the low-level runtime APIs allows for a high-degree of control over memory allocation, placement, and loading. This allows for advanced use cases, such as placing allocations in specific memory banks or loading a model without a file system. For an end to end example using the low-level runtime APIs, see [Detailed C++ Runtime APIs Tutorial](running-a-model-cpp-tutorial.md).

## Building with CMake

There are two ways to get the C++ runtime. Linking the prebuilt libraries from the pip
package needs no source checkout and is the quicker option. Building from source gives
you every option the project has, and is what you need for a platform the wheel does not
cover.

### Using the prebuilt libraries from the pip package

On Linux and macOS, `pip install executorch` ships the runtime as prebuilt shared libraries together
with the headers and a CMake package. So a C++ program can use ExecuTorch without building it from
source, and without knowing much CMake.

#### Run your first model in four steps

Copy these three files into an empty folder and follow along. No prior CMake knowledge needed.

**1. Install, and make a model file.**

```
pip install executorch torch --extra-index-url https://download.pytorch.org/whl/cpu
```

`torch` is named explicitly because the wheel does not depend on it, so you bring your own torch and
keep the version under your control. You need it only to create a model file in step 1, not to run
the C++ program.

A C++ program loads a `.pte` file, which is a model that has already been exported. C++ cannot
create one, so make it in Python first:

```python
# export.py
import torch
from executorch.exir import to_edge_transform_and_lower

class Add(torch.nn.Module):
    def forward(self, x, y):
        return x + y

example = (torch.ones(2, 2), torch.ones(2, 2))
program = to_edge_transform_and_lower(
    torch.export.export(Add(), example)
).to_executorch()
open("model.pte", "wb").write(program.buffer)
```

```
python export.py
```

**2. Write the program.**

```cpp
// main.cpp
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <cstdio>

using namespace executorch::extension;

int main() {
  Module module("model.pte");

  std::array<float, 4> a{1, 2, 3, 4};
  std::array<float, 4> b{10, 20, 30, 40};

  const auto result = module.forward({make_tensor_ptr({2, 2}, a.data()),
                                      make_tensor_ptr({2, 2}, b.data())});
  if (!result.ok()) {
    std::printf("forward failed: 0x%x\n", (unsigned)result.error());
    return 1;
  }

  const auto out = result->at(0).toTensor();
  for (int i = 0; i < out.numel(); ++i) {
    std::printf("%g ", out.const_data_ptr<float>()[i]);
  }
  std::printf("\n");
  return 0;
}
```

**3. Write six lines of CMake.**

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.28)
project(app CXX)

find_package(executorch REQUIRED COMPONENTS kernels_optimized)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE executorch::runtime
                                  executorch::kernels_optimized)
```

Two lines matter. `find_package` finds the installed ExecuTorch, and `target_link_libraries` says
which parts you want. Every model needs at least these two: `runtime` is the engine that executes a
program, and a kernel component such as `kernels_optimized` provides the maths the model computes
with. With only the engine, a model loads and then fails with a missing operator.

**4. Build and run.**

```
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$(python -c 'import executorch, pathlib; print(pathlib.Path(executorch.__path__[0]) / "share" / "cmake")')"
cmake --build build
./build/app
```

```
11 22 33 44
```

That is the two input arrays added together. The long `python -c` part just prints where pip put the
CMake package, so CMake can find it. Run `./build/app` from the folder holding `model.pte`, because
the path in `main.cpp` is relative.

#### Adding kernels and backends

Add a component to both lines to get more. Nothing else in the program changes.

```cmake
find_package(executorch REQUIRED COMPONENTS kernels_optimized backend_xnnpack)

target_link_libraries(app PRIVATE executorch::runtime
                                  executorch::kernels_optimized
                                  executorch::backend_xnnpack)
```

These are the components the package provides:

| Component | What it gives you | Where |
| --- | --- | --- |
| `runtime` | The engine. Always needed. | Linux, macOS |
| `kernels_optimized` | Fast CPU operators. The usual choice. | Linux, macOS |
| `backend_xnnpack` | The XNNPACK backend, for models exported with it. | Linux, macOS |
| `threadpool` | Multi-threaded execution. | Linux, macOS |
| `etdump` | Profiling, to record what ran and how long it took. | Linux, macOS |
| `kernels_quantized` | The quantized operator kernels | Linux, macOS |
| `backend_cuda` | The CUDA delegate | Linux |
| `extension_cuda` | The CUDA stream extension | Linux |
| `backend_openvino` | The OpenVINO delegate | Linux |
| `backend_mlx` | The MLX delegate, for Apple GPU execution | macOS, Apple Silicon |

To see what your own install offers, ask CMake:

```cmake
find_package(executorch REQUIRED)
foreach(_component
        runtime kernels_optimized kernels_quantized backend_xnnpack
        backend_mlx backend_cuda extension_cuda backend_openvino threadpool etdump)
  if(TARGET executorch::${_component})
    message(STATUS "have ${_component}")
  endif()
endforeach()
```

On macOS the Core ML delegate is registered inside the Python extension rather than shipped as a
separate C++ library, so a C++ application there cannot link it as a component; use it from
Python, or build from source if you need it in C++.

Profiling a Core ML model records `DELEGATE_CALL`, which tells you how long the delegate ran in
total. It does not record the individual operators inside the delegate, because that detail comes
from the Core ML developer tools sources, and the wheel does not build them. An XNNPACK model
records both.

A backend is only needed if the model was exported for it. Linking XNNPACK does not make a plain
model faster, and a model exported for XNNPACK will fail to load without it. If you are not sure
what a model needs, start with `runtime` and `kernels_optimized` and add what the error asks for.

If you would rather not choose, one variable links the common set:

```cmake
find_package(executorch REQUIRED)
target_link_libraries(app PRIVATE ${EXECUTORCH_LIBRARIES})
```

The quantized kernels are deliberately left out of that variable, because loading
`executorch.kernels.quantized` in Python registers the same operators and a duplicate registration
stops the runtime.

The OpenVINO delegate needs one more step. The wheel ships the adapter, not the OpenVINO runtime
itself, and the adapter opens `libopenvino_c.so` by name when the model is loaded. Python callers
get that path set for them on import; a standalone C++ program does not, so install the runtime and
point the program at it:

```bash
pip install "executorch[openvino]"
export OPENVINO_LIB_PATH="$(python -c 'import glob, openvino, os; print(sorted(glob.glob(os.path.join(os.path.dirname(openvino.__file__), "libs", "libopenvino_c.so*")))[0])')"
```

Without it the delegate still registers and the program still links, and the failure arrives later,
when the model is loaded.

The MLX delegate has a similar requirement. Its Metal kernels live in a separate `mlx.metallib`
file, which MLX looks for next to whichever library holds MLX code. The wheel ships it beside the
delegate, so a program that links the delegate where it sits needs nothing extra. A program that
copies the delegate next to its own binary has to copy that file too, and `find_package` reports
where it is:

```cmake
find_package(executorch REQUIRED COMPONENTS backend_mlx)
message(STATUS "Metal kernels: ${MLX_METALLIB_PATH}")
```

#### When something does not work

- `find_package` could not find executorch: the `-DCMAKE_PREFIX_PATH=...` argument is missing or
  points somewhere else. Run the `python -c` line on its own and check the folder exists.
- The program builds but fails to load the model: the path is relative, so run it from the folder
  containing the `.pte` file.
- A missing operator at run time: add a kernel component, usually
  `executorch::kernels_optimized`.
- The model fails to load complaining about a backend: link the backend it was exported for.
- `executorch::runtime` is not a target: imported targets need CMake 3.28 or newer. On an older
  CMake the package still works, but you name variables instead of targets, and you have to pass on
  the definitions and the C++ standard requirement yourself:

  ```cmake
  cmake_minimum_required(VERSION 3.19)
  project(app CXX)

  find_package(executorch REQUIRED)

  add_executable(app main.cpp)
  target_include_directories(app PRIVATE ${EXECUTORCH_INCLUDE_DIRS})
  target_compile_definitions(app PRIVATE ${EXECUTORCH_COMPILE_DEFINITIONS})
  target_compile_features(app PRIVATE cxx_std_${EXECUTORCH_CXX_STANDARD})
  target_link_libraries(app PRIVATE ${EXECUTORCH_LIBRARIES})
  set_target_properties(
    app PROPERTIES INSTALL_RPATH "${EXECUTORCH_RUNTIME_LIBRARY_DIR}"
  )
  ```

  Leaving out `EXECUTORCH_COMPILE_DEFINITIONS` fails with a missing
  `torch/headeronly/macros/cmake_macros.h`, because the vendored headers look for a file that only
  exists inside a PyTorch build.

  `INSTALL_RPATH` matters once you run `cmake --install`. CMake gives your program a search path while
  it sits in the build directory and removes that path when installing, so an installed program cannot
  find the libraries unless you record where they live.

  Quantized kernels are not part of `EXECUTORCH_LIBRARIES`, so add them when your model needs them:

  ```cmake
  target_link_libraries(app PRIVATE ${EXECUTORCH_QUANTIZED_KERNELS_LIBRARY})
  ```

You should not need `LD_LIBRARY_PATH`. The shipped libraries record where their neighbours live, so
they find each other once the program links against the installed package.

On Linux, linking the runtime asks the linker for `DT_RUNPATH` rather than the older `DT_RPATH`.
That is deliberate: `DT_RPATH` is searched before `LD_LIBRARY_PATH` and applies to a dependency's
own dependencies, so it would stop you pointing an instrumented or locally built library at your
application. `DT_RUNPATH` leaves you that control.

The setting is a property of the whole link rather than of one library, so it applies to the search
paths your own project adds as well. If your application relies on `DT_RPATH` being searched
transitively, ask for it after the runtime:

```cmake
add_library(prefer_rpath INTERFACE)
target_link_options(prefer_rpath INTERFACE "LINKER:--disable-new-dtags")
target_link_libraries(app PRIVATE executorch::runtime prefer_rpath)
```

The order matters. A target's own link options are emitted before those of its dependencies, and the
last of the two settings decides the tag for every entry in the link.

### Running on a GPU with the CUDA package

The CUDA build is a separate package. Releases cover CUDA 12.6, 13.0 and 13.2, so pick the index
matching the CUDA version you have (`cu126`, `cu130` or `cu132`). For CUDA 12.6:

```
pip install executorch torch \
  --index-url https://download.pytorch.org/whl/cu126 \
  --extra-index-url https://pypi.org/simple
```

CUDA wheels are built for Python 3.10 through 3.14, which is every Python this project supports.

The second index is required: a bare `--index-url` replaces PyPI rather than adding to it, and some
dependencies are only on PyPI. The torch you install has to come from the same CUDA index, because
exporting a model for CUDA runs through torch.

Everything above stays the same. Add the CUDA backend to both CMake lines:

```cmake
find_package(executorch REQUIRED COMPONENTS kernels_optimized backend_cuda)

target_link_libraries(app PRIVATE executorch::runtime
                                  executorch::kernels_optimized
                                  executorch::backend_cuda)
```

The model has to be exported for CUDA as well, on a machine with a GPU. That step also needs the
CUDA compiler (`nvcc`) on your `PATH`, because the backend compiles the model into GPU code ahead
of time. `pip install` does not provide it, so install the CUDA Toolkit for this step and check it
with `nvcc --version`.

```python
# export_cuda.py, the same model as before with one line added
import torch
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.extension.export_util.utils import save_pte_program

class Add(torch.nn.Module):
    def forward(self, x, y):
        return x + y

example = (torch.ones(2, 2), torch.ones(2, 2))
program = to_edge_transform_and_lower(
    torch.export.export(Add(), example), partitioner=[CudaPartitioner([])]
).to_executorch()
save_pte_program(program, "model", ".")
```

`save_pte_program` is used instead of writing the buffer by hand because the CUDA backend puts its
model weights in a **separate data file** next to `model.pte`. The compiled GPU code stays
inside `model.pte`. Writing only the program file loses the weights, so the model then fails when it
runs.

The backend chooses that file's name, so check what was written:

```
$ ls
aoti_cuda_blob.ptd  model.pte
```

Load both from C++, passing the data file as the second argument:

```cpp
Module module("model.pte", "aoti_cuda_blob.ptd");
```

The CUDA backend is still experimental, so exporting prints a warning saying so.

By default the runtime copies inputs to the GPU and results back, so your program keeps passing
ordinary CPU tensors and nothing else changes.

One thing to check first, and the numbers matter. If a model fails with a message about no kernel
image being available for the device, your GPU is not one these packages were built for. The floor
is **compute capability 8.0**, which means an NVIDIA Ampere generation card or newer.

Check your own GPU:

```
python -c 'import torch; print(torch.cuda.get_device_capability())'
```

A result below `(8, 0)` is not covered. Above the floor the answer depends on the package: each one
covers what the PyTorch build for the same platform and CUDA version covers, and the ARM packages
reach fewer cards in that range than the x86_64 ones, so a card at or above `(8, 0)` can still be
outside an ARM package. Note that `torch.cuda.get_arch_list()` is not the right check either:
PyTorch builds for a wider set at the bottom than these packages do, so a GPU can appear in that
list and still not be supported.

### Building from source


ExecuTorch uses CMake as the primary build system. Inclusion of the module and tensor APIs are controlled by the `EXECUTORCH_BUILD_EXTENSION_MODULE` and `EXECUTORCH_BUILD_EXTENSION_TENSOR` CMake options. As these APIs may not be supported on embedded systems, they are disabled by default when building from source. The low-level API surface is always included. To link, add the `executorch` target as a CMake dependency, along with `executorch_backends`, `executorch_extensions`, and `extension_kernels`, to link all configured backends, extensions, and kernels.

```
# CMakeLists.txt
add_subdirectory("executorch")
...
target_link_libraries(
    my_target
    PRIVATE executorch
    executorch::backends
    executorch::extensions
    executorch::kernels)
```

See [Building from Source](using-executorch-building-from-source.md) for more information on the CMake build process.

## Reference Runners

The ExecuTorch repository includes several reference runners, which are simple programs that load and execute a .pte file, typically with random inputs. These can be used to sanity check model execution on a development platform and as a code reference for runtime integration.

The `executor_runner` target is built by default when building with CMake. It can be invoked as follows:
```
./cmake-out/executor_runner --model_path path/to/model.pte
```

The runner source code can be found in the ExecuTorch repo under [examples/portable/executor_runner.cpp](https://github.com/pytorch/executorch/blob/main/examples/portable/executor_runner/executor_runner.cpp). Some backends, such as CoreML, have dedicated runners to showcase backend and platform-specific functionality. See [examples/apple/coreml](https://github.com/pytorch/executorch/tree/main/examples/apple/coreml) and the [examples](https://github.com/pytorch/executorch/tree/main/examples) directory for more information.

## Next Steps

- [Runtime API Reference](executorch-runtime-api-reference.rst) for documentation on the available C++ runtime APIs.
- [Running an ExecuTorch Model Using the Module Extension in C++](extension-module.md) for information on the high-level Module API.
- [Managing Tensor Memory in C++](extension-tensor.md) for information on high-level tensor APIs.
- [Running an ExecuTorch Model in C++ Tutorial](running-a-model-cpp-tutorial.md) for information on the low-level runtime APIs.
- [Building from Source](using-executorch-building-from-source.md) for information on CMake build integration.
