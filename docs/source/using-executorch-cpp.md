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

On Linux the pip package carries everything a C++ application needs, so nothing has to be built
from source. This walkthrough goes from an empty directory to a program that prints real numbers.
Four steps, and each one can be checked before moving on.

#### Quickstart: from nothing to a running program

**Step 0. Install the package.** Any Python from 3.10 to 3.14 works:

```
pip install executorch
```

Check that the C++ pieces are really there. This prints the directory holding the CMake package,
and it is the same path used in step 2:

```
python -c 'import executorch, pathlib; print(pathlib.Path(executorch.__path__[0]) / "share" / "cmake")'
```

**Step 1. Make a model file.** A C++ application loads a `.pte` file, which is a model that has
already been exported. Nothing in C++ creates one, so make one in Python first:

```python
# export_add.py
import torch
from executorch.exir import to_edge_transform_and_lower

class Add(torch.nn.Module):
    def forward(self, x, y):
        return x + y

example = (torch.ones(2, 2), torch.ones(2, 2))
program = to_edge_transform_and_lower(
    torch.export.export(Add(), example)
).to_executorch()
with open("add.pte", "wb") as handle:
    handle.write(program.buffer)
```

```
python export_add.py
```

That writes `add.pte`, about a kilobyte for this model.

**Step 2. Write the application.** `Module` loads the file and `make_tensor_ptr` wraps plain arrays
as inputs, so no ExecuTorch specific memory handling is needed:

```cpp
// main.cpp
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <iostream>

using namespace executorch::extension;

int main() {
  Module module("add.pte");
  std::array<float, 4> a{1, 2, 3, 4};
  std::array<float, 4> b{10, 20, 30, 40};
  auto x = make_tensor_ptr({2, 2}, a.data());
  auto y = make_tensor_ptr({2, 2}, b.data());
  const auto result = module.forward({x, y});
  if (!result.ok()) {
    std::cerr << "forward failed\n";
    return 1;
  }
  const auto out = result->at(0).toTensor();
  for (int i = 0; i < out.numel(); ++i) {
    std::cout << out.const_data_ptr<float>()[i] << " ";
  }
  std::cout << "\n";
  return 0;
}
```

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.24)
project(my_app CXX)

find_package(executorch REQUIRED COMPONENTS kernels_optimized)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE executorch::runtime
                                     executorch::kernels_optimized)
```

`executorch::runtime` is the engine, and `executorch::kernels_optimized` provides the operator
implementations the model computes with. A model needs both: without a kernel component it loads
and then fails to find an operator.

**Step 3. Build it.** Point CMake at the directory printed in step 0:

```
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$(python -c 'import executorch, pathlib; print(pathlib.Path(executorch.__path__[0]) / "share" / "cmake")')"
cmake --build build
```

**Step 4. Run it.** Run from the directory holding `add.pte`, since the path in the source is
relative:

```
./build/my_app
```

```
11 22 33 44
```

Those are the two input arrays added together, which confirms the whole path works: the model file,
the runtime, the kernels, and the tensor wrappers.

Two things worth knowing when this does not work the first time:

- `find_package(executorch)` needs CMake 3.28 or newer for imported targets such as
  `executorch::runtime`. On an older CMake the package still works, but link against
  `${EXECUTORCH_LIBRARIES}` instead and add `${EXECUTORCH_INCLUDE_DIRS}` yourself.
- The shipped libraries already record where their neighbours live, so no `LD_LIBRARY_PATH` is
  needed. If a library cannot be found at run time, check that the application was linked against
  the installed package rather than a separate source build.

#### The details


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
| `executorch::backend_xnnpack` | the XNNPACK delegate. |
| `executorch::threadpool` | the shared thread pool. |
| `executorch::etdump` | the profiler. |

The runtime on its own loads a program but registers only primitive operators, not the
kernels a model computes with, so a model that is not fully delegated needs a kernel
component too. Linking a delegate is what registers it: a program delegated to XNNPACK
fails to load in an application that did not link `executorch::backend_xnnpack`.

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
target_link_options(my_app PRIVATE "LINKER:--enable-new-dtags")
```

The second line matters on Linux. Without it this linker records the older `DT_RPATH` tag, which is
searched before `LD_LIBRARY_PATH` and also applies to your dependencies' own dependencies, so you
could not point the application at a different build of the runtime. With it you get `DT_RUNPATH`,
which only affects your application and stays overridable.

The imported target route does not need this on Linux: the package sets its search paths as
explicit link options, and those survive installation.

On macOS it does need one line. CMake removes an entry that points at a directory holding a
library the application linked, so the entry naming the wheel's own directory is deleted from
the installed binary and it stops finding the runtime:

```cmake
set_property(TARGET my_app PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
```

An application deployed beside the libraries is unaffected either way, because the
`@loader_path` and `$ORIGIN` entries are kept.

`EXECUTORCH_LIBRARIES` names the runtime and every component the wheel shipped, so you
cannot choose components on this route. Upgrade to CMake 3.28 and link the specific
targets you need instead.

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
