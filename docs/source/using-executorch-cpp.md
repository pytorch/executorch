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

On Linux, `pip install executorch` includes prebuilt shared libraries, the public
headers, and a CMake package, so a C++ application can link the runtime without building
ExecuTorch itself:

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.28)
project(my_app CXX)

find_package(executorch REQUIRED COMPONENTS kernels_optimized)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE executorch::runtime
                                     executorch::kernels_optimized)
```

Point CMake at the installed package when you configure:

```
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$(python -c 'import executorch, pathlib; print(pathlib.Path(executorch.__path__[0]) / "share" / "cmake")')"
cmake --build build
```

The application uses the same `Module` and `TensorPtr` APIs described above:

```cpp
// main.cpp
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <cstdio>
#include <vector>

using namespace executorch::extension;

int main() {
  Module module("model.pte");

  std::vector<float> data(2 * 8, 1.0f);
  auto input = make_tensor_ptr({2, 8}, data.data());

  const auto result = module.forward(input);
  if (!result.ok()) {
    std::printf("forward failed: 0x%x\n", (unsigned)result.error());
    return 1;
  }
  std::printf("ok, %zu outputs\n", result->size());
  return 0;
}
```

#### What each component provides

Ask for the components your model needs. A component the wheel was not built with is
reported while CMake configures, rather than failing later at link time.

| Component | What it provides |
| --- | --- |
| `executorch::runtime` | the program loader and executor. Always present. |
| `executorch::kernels_optimized` | CPU operator kernels. Needed for any operator a delegate does not claim. |
| `executorch::kernels_quantized` | quantized operator kernels, for a quantized model. Link it only when you need it: see the note below. |
| `executorch::backend_xnnpack` | the XNNPACK delegate. |
| `executorch::threadpool` | the shared thread pool. |
| `executorch::etdump` | the profiler. |

The runtime on its own loads a program but registers only primitive operators, not the
kernels a model computes with, so a model that is not fully delegated needs a kernel
component too. Linking a delegate is what registers it: a program delegated to XNNPACK
fails to load in an application that did not link `executorch::backend_xnnpack`.

#### The quantized kernels are opt in

`executorch::kernels_quantized` is the one component that `${EXECUTORCH_LIBRARIES}` does
not include, so you have to name it. The reason is a conflict with the Python side:
`executorch.kernels.quantized` loads a plugin that carries its own copy of the same
kernels, and the runtime stops when the same operator is registered twice:

```
Re-registering quantized_decomposed::add.out
```

That only affects a process holding both, for example an application that embeds a
Python interpreter. A plain C++ application can link this component freely. It is kept
out of the default set so that linking whatever the package offers cannot put you in that
position by accident.

To require a minimum version, pass it to `find_package`:

```cmake
find_package(executorch 1.0 REQUIRED)
```

#### On CMake older than 3.28

The example above needs CMake 3.28. Older versions write the `$ORIGIN` marker (the
"look next to me" token in a library search path) incorrectly, which would leave you with
a target that runs where it was built and fails once the application is copied
elsewhere. Rather than hand you a target that behaves that way, the package defines no
imported targets below 3.28 and exports plain variables instead.

An imported target carries more than a library path, so on this route you have to apply
the rest yourself. Linking the libraries alone does not compile:

```cmake
cmake_minimum_required(VERSION 3.19)
project(my_app CXX)

find_package(executorch REQUIRED)

add_executable(my_app main.cpp)
target_include_directories(my_app PRIVATE ${EXECUTORCH_INCLUDE_DIRS})
target_compile_definitions(my_app PRIVATE ${EXECUTORCH_COMPILE_DEFINITIONS})
target_link_libraries(my_app PRIVATE ${EXECUTORCH_LIBRARIES})
set_property(TARGET my_app PROPERTY CXX_STANDARD ${EXECUTORCH_CXX_STANDARD})
set_property(TARGET my_app PROPERTY CXX_STANDARD_REQUIRED ON)
```

On this route the application also has to record where the libraries live, or it runs
from its build directory and then fails to start once installed with a message like
`libexecutorch.so: cannot open shared object file`. CMake records the wheel's library
directory while building, because the libraries are named by absolute path, but it removes
that entry on install. Ask for it to be kept:

```cmake
set_property(TARGET my_app PROPERTY INSTALL_RPATH "${EXECUTORCH_RUNTIME_LIBRARY_DIR}")
```

The imported target route above does not need this: the package attaches its own search
paths to `executorch::runtime`, and those survive installation.

`EXECUTORCH_LIBRARIES` names the runtime and every component the wheel shipped, so you
cannot choose components on this route. The quantized kernels are the exception described
above, offered as `EXECUTORCH_QUANTIZED_KERNELS_LIBRARY` for a consumer that wants them:

```cmake
target_link_libraries(my_app PRIVATE ${EXECUTORCH_QUANTIZED_KERNELS_LIBRARY})
```

Upgrade to CMake 3.28 and link the specific targets you need instead.

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
