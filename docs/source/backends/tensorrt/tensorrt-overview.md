# TensorRT Backend

The TensorRT backend runs ExecuTorch models on NVIDIA GPUs with
[TensorRT](https://developer.nvidia.com/tensorrt), NVIDIA's inference library. At export
time the model graph is compiled into TensorRT engines. The engines are stored inside the
`.pte` file, and at run time the delegate hands each one back to TensorRT.

The delegate is built and shipped by the [Torch-TensorRT](https://github.com/pytorch/TensorRT)
project, so it comes from its wheel and not from the ExecuTorch wheel. Everything after
installation is the normal ExecuTorch API: the same `.pte` file, the same `Module` class, the
same runtime.

## Features

- **TensorRT engines inside a `.pte`.** One file holds the program and its engines, so the
  runtime loads a model the usual way.
- **Works with the CUDA backend in one program.** Operators TensorRT cannot convert can be
  compiled by the ExecuTorch [CUDA backend](../cuda/cuda-overview.md) instead of falling back
  to the CPU. Both delegates live in the same `.pte` and run in the same method.
- **GPU resident inputs and outputs.** The copies around the method boundary can be turned
  off, so a pipeline that already holds its data on the GPU does not pay for them.
- **CUDA graph replay.** An engine can be recorded once and replayed with a single launch,
  which helps models that launch many short kernels. This needs a Torch-TensorRT build newer
  than the 2026-10-06 nightly, see [Feature availability](#feature-availability).
- **Shared activation scratch.** Many engines in one program can share one scratch buffer
  instead of each keeping its own.
- **Shared engines.** Loading the same program twice keeps one copy of the engine weights in
  GPU memory. This also needs a build newer than the 2026-10-06 nightly.
- **Caller-owned CUDA stream.** The application can give every delegate the same stream,
  including a CUDA green context stream, so the model stays inside one part of the GPU.

## Target Requirements

- **Hardware**: NVIDIA GPU.
- **Operating system**: Linux, on x86_64 and on aarch64.
- **CUDA**: 13.x. The delegate links CUDA 13, so a CUDA 12 build is not supported.
- **Drivers**: an NVIDIA driver that matches the CUDA version.

TensorRT itself and the CUDA runtime arrive as dependencies of the wheels, so a system
install of either is not required.

## Development Requirements

One command installs everything needed to export a model. The `executorch` extra pulls in a
CUDA build of ExecuTorch, the delegate wheel, Torch-TensorRT, and PyTorch:

```bash
pip install --pre "torch-tensorrt[executorch]" \
  --index-url https://download.pytorch.org/whl/nightly/cu132 \
  --extra-index-url https://pypi.org/simple \
  --extra-index-url https://pypi.nvidia.com
```

Swap `cu132` for the channel that matches your CUDA, for example `cu134` for CUDA 13.4. Keep
PyTorch, ExecuTorch, and Torch-TensorRT on the same channel.

All three indexes matter. Without `--pre` pip takes the latest stable Torch-TensorRT from the
public index, which pairs with an older ExecuTorch release than this page describes. Without
NVIDIA's index the TensorRT package resolves to a source build that takes about twenty
minutes and then fails.

A CUDA build of ExecuTorch is required, not only to build against. The delegate needs a
library that only the CUDA wheels carry, so a CPU-only build installs and then fails on
import.

If you also want the CUDA backend to pick up the operators TensorRT rejects, install a CUDA
toolkit as well. That backend compiles those operators at export time with `nvcc`.

### Feature availability

Two features on this page are newer than the 2026-10-06 Linux nightly of Torch-TensorRT and
its delegate package, which was the newest one when this page was written:

- [CUDA graph replay](#cuda-graph-replay). With the 2026-10-06 build, `torch_tensorrt.save`
  rejects `use_cuda_graphs` with a `TypeError`.
- [Shared engines](#shared-engines). The 2026-10-06 delegate keeps every engine private, so
  loading a program twice still holds two copies of its weights.

Both are on the Torch-TensorRT main branch and in its 2.15 release branch. To use them,
install a wheel newer than 2026-10-06 once one is published, or build Torch-TensorRT from
source. Everything else on this page works with the 2026-10-06 build.

## Using the TensorRT Backend

### Exporting a model

Compile the model with Torch-TensorRT, then save it in ExecuTorch format:

```python
import torch
import torch_tensorrt


class MyModel(torch.nn.Module):
    def forward(self, x):
        return x + 1


with torch.no_grad():
    model = MyModel().eval().cuda()
    example_input = (torch.randn(2, 3, 4, 4).cuda(),)

    exported_program = torch.export.export(model, example_input)
    trt_gm = torch_tensorrt.dynamo.compile(
        exported_program,
        arg_inputs=example_input,
        min_block_size=1,
    )

    torch_tensorrt.save(
        trt_gm,
        "model.pte",
        output_format="executorch",
        arg_inputs=example_input,
        retrace=False,
    )
```

`retrace=False` is required. Re-exporting the compiled graph drops the engine, and the error
only shows up much later when the program is loaded.

### Letting the CUDA backend take what TensorRT rejects

TensorRT has no converter for every ATen operator. By default the leftover operators run on
the CPU, which costs a copy in each direction. Pass a `CudaPartitioner` and they are compiled
by the ExecuTorch CUDA backend instead, so the whole model stays on the GPU:

```python
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner

torch_tensorrt.save(
    trt_gm,
    "coalesced.pte",
    output_format="executorch",
    arg_inputs=example_input,
    retrace=False,
    partitioners=[
        CudaPartitioner([CudaBackend.generate_method_name_compile_spec("forward")])
    ],
)
```

The TensorRT partitioner always runs first, and the `CudaPartitioner` picks up the rest. For
a model such as `cos(erfinv(tanh(x)))`, where TensorRT cannot take `erfinv`, the result is one
`.pte` whose delegate list reads `['TensorRTBackend', 'CudaBackend', 'TensorRTBackend']`.

The CUDA backend writes its external weights to a `.ptd` file beside the `.pte` under a fixed
name. Save each coalesced model into its own directory. A second export into the same
directory overwrites the first model's weights, and the first model then still loads and
returns a wrong answer with no error.

### Keeping inputs and outputs on the GPU

By default ExecuTorch inserts a host to device copy before the first delegate and a device to
host copy after the last one, so a method is safe to call with CPU tensors. If your data is
already on the GPU, turn those copies off:

```python
from executorch.exir import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.propagate_device_config import PropagateDeviceConfig

torch_tensorrt.save(
    trt_gm,
    "device_resident.pte",
    output_format="executorch",
    arg_inputs=example_input,
    retrace=False,
    backend_config=ExecutorchBackendConfig(
        propagate_device_config=PropagateDeviceConfig(
            skip_h2d_for_method_inputs=True,
            skip_d2h_for_method_outputs=True,
        ),
        enable_non_cpu_memory_planning=True,
        memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
    ),
)
```

Three points are easy to miss:

- **Both skip flags need `enable_non_cpu_memory_planning=True`.** Copy insertion happens
  during device aware memory planning. Setting it to `False` raises a `ValueError`.
- **Inputs must be unplanned**, with `MemoryPlanningPass(alloc_graph_input=False)`. Without
  it the program reserves its own input buffer and the runtime copies the caller's memory
  into it, which puts the copy straight back.
- **Leave the outputs planned when Python runs the program.** The program's own device arena
  then holds the output, and Python returns a copy of it as a CUDA tensor. With
  `alloc_graph_output=False` the caller provides the output memory instead. A C++ caller does
  that with `Module::set_output`. The Python bindings do it correctly starting with ExecuTorch
  1.6.0.dev20261008, but the 2026-10-06 delegate wheel pins ExecuTorch 1.6.0.dev20260925, so
  with that wheel keep the outputs planned.

The choice is baked into the `.pte`. A program exported this way wants CUDA tensors, so pass
CUDA tensors. A host tensor works only when TensorRT is the first to read that input. On most
GPUs it then stages a copy on every call, which is the cost the export existed to remove. When
the CUDA backend reads it first, as it can in a coalesced program, it refuses the host memory
and the call fails.

## Runtime Integration

### Python

Import the delegate package once, anywhere before a program is loaded. The import is what
registers the backend. Nothing else about your code changes:

```python
from pathlib import Path

import torch
import torch_tensorrt_executorch_runtime  # noqa: F401
from executorch.runtime import Runtime

program = Runtime.get().load_program(Path("model.pte"))
forward = program.load_method("forward")
outputs = forward.execute((torch.ones(2, 3, 4, 4),))
```

If the delegate cannot be loaded, that import raises straight away instead of leaving the
failure to show up later as a program that will not load.

A coalesced program needs no extra registration. Both backends are registered, and the
program records which parts go where. Its weights are a different matter. If the part the
CUDA backend compiled has weights, they are in the `aoti_cuda_blob.ptd` file saved beside the
`.pte`, and the runtime does not look for that file on its own. Pass it when you load:

```python
program = Runtime.get().load_program(
    Path("coalesced.pte"), data_path=Path("aoti_cuda_blob.ptd")
)
```

Without it, loading a program whose CUDA part has weights fails.

### C++ against the installed wheels

The wheels ship a prebuilt delegate and a CMake package, so a C++ application can link them
without building anything from source:

```cmake
find_package(executorch REQUIRED COMPONENTS backend_cuda kernels_optimized)
find_package(executorch_backend_tensorrt REQUIRED)

target_link_libraries(my_app PRIVATE
  executorch::runtime
  executorch::backend_cuda
  executorch::backend_tensorrt
  executorch::kernels_optimized
)
```

`kernels_optimized` supplies the `et_copy` operators that move data across the method
boundary. `backend_cuda` is needed by any coalesced program, and also by a TensorRT only
program that still has those boundary copies, because it registers the device allocator they
use.

The two packages live in two distributions, so point CMake at both. ExecuTorch is a namespace
package, so its path has to come from its distribution metadata:

```bash
cmake -DCMAKE_PREFIX_PATH="$(TORCH_TENSORRT_SKIP_DELEGATE_REGISTRATION=1 python -c 'import importlib.metadata as m, torch_tensorrt_executorch_runtime as r, pathlib; print(str(pathlib.Path(str(m.distribution("executorch").locate_file("executorch"))) / "share" / "cmake") + ";" + str(pathlib.Path(r.__file__).parent))')" ...
```

CMake 3.28 or newer is needed for the form above, because the `backend_cuda` component
rejects older versions. CMake is not part of the wheels, so install it yourself.

There is no header to include. The delegate registers itself with the backend registry from a
static initializer inside the shared library, and everything after that is the ordinary
ExecuTorch C++ API:

```cpp
#include <cstdio>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

using namespace executorch::extension;

int main() {
  Module module("model.pte");

  std::vector<float> data(2 * 3 * 4 * 4, 1.0f);
  auto input = make_tensor_ptr({2, 3, 4, 4}, std::move(data));

  const auto outputs = module.forward(input);
  if (!outputs.ok()) {
    printf("forward failed\n");
    return 1;
  }
  printf("first output value: %f\n",
         outputs->at(0).toTensor().const_data_ptr<float>()[0]);
  return 0;
}
```

For a coalesced program whose CUDA part has weights, pass the data file to the constructor
as well, for the same reason as in Python:

```cpp
Module module("coalesced.pte", "aoti_cuda_blob.ptd");
```

Linking the target also records the wheel's own library directory in your binary, which is
right for an application built against an installed wheel and wrong for anything you plan to
redistribute. Turn it off with `EXECUTORCH_BACKEND_TENSORRT_EMBED_RUNPATH` and
`CMAKE_SKIP_BUILD_RPATH`, both set before `find_package`.

### Building from source

The delegate source ships inside `libtorchtrt.tar.gz` as
`torch_tensorrt/src/torch_tensorrt/executorch/`. Turn on ExecuTorch's CUDA backend and the
extensions a `Module` app uses, add the delegate next to ExecuTorch, and link the target it
provides:

```cmake
set(EXECUTORCH_BUILD_CUDA ON)
set(EXECUTORCH_BUILD_EXTENSION_TENSOR ON)
set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON)
set(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR ON)
set(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP ON)
set(EXECUTORCH_BUILD_EXTENSION_MODULE ON)

add_subdirectory("executorch")
add_subdirectory("torch_tensorrt/src/torch_tensorrt/executorch")

target_link_libraries(my_runner PRIVATE
  executorch
  executorch::backends
  executorch::extensions
  executorch::kernels
  executorch::backend_tensorrt
)
```

Turn on the CUDA backend even for a program that only uses TensorRT. It builds the shared
`extension_cuda` library the delegate links, and it registers the device allocator the
boundary copies use. Without it the delegate stops at configure time. It also needs PyTorch
installed in the Python environment that CMake finds. The other options are the extensions
the CUDA backend and the `Module` class depend on, and CMake stops with an error if one of
them is missing.

Building the delegate needs CUDA Toolkit 12.5 or newer. `libextension_cuda` stays a shared
library on purpose, so that every CUDA-capable delegate in the process reads the same caller
stream.

## Performance Options

### Choosing the CUDA stream

With no stream chosen, both delegates run on `cudaStreamPerThread`, the default stream of the
calling thread. A coalesced program run from one thread is therefore ordered correctly as it
is, with nothing to add.

To run every delegate on a stream of your own, for example a green context stream, scope a
guard over the whole execution:

```cpp
#include <executorch/extension/cuda/caller_stream.h>

using namespace executorch::extension;

cuda::CallerStreamGuard guard(stream);
module.forward(input);
```

The guard lives in the `extension_cuda` library, so add it to the link recipe above:

```cmake
find_package(executorch REQUIRED COMPONENTS backend_cuda extension_cuda kernels_optimized)

target_link_libraries(my_app PRIVATE executorch::extension_cuda)
```

One guard reaches every CUDA-capable delegate, because they all resolve the same shared
`libextension_cuda`. The stream must be on the same device as the engine. The CUDA backend
refuses a caller stream for a method that uses its own CUDA graphs.

### Green contexts

Because both delegates honor the caller's stream, that stream can be a CUDA green context
stream. A green context holds a fixed number of streaming multiprocessors, so the whole model
is confined to that part of the GPU and the rest stays free for other work. Create the stream
with `cuGreenCtxStreamCreate` and scope the same guard over it.

This has been exercised on a 108 SM card with a green context holding 8 of them, running a
program split across the TensorRT delegate and the CUDA backend.

### CUDA graph replay

This feature needs a Torch-TensorRT build newer than the 2026-10-06 nightly, see
[Feature availability](#feature-availability).

Replay is off by default. When it is on, the delegate records an engine's kernel launches
once as a CUDA graph and then replays the whole engine with one launch. It helps engines with
fixed shapes that launch many short kernels, where the CPU launch work is a large part of
each call.

Turn it on at export time:

```python
torch_tensorrt.save(
    trt_gm,
    "model.pte",
    output_format="executorch",
    arg_inputs=example_input,
    retrace=False,
    use_cuda_graphs=True,
)
```

It is not free. Each engine keeps one stable device buffer per input and output, and every
call copies each input in and each output out. For an engine with few kernels, or with large
inputs and outputs, those copies can cost more than the launches they save. Shapes that
change on every call never replay and still pay the copies.

Some engines never replay, even with replay on. An engine on the shared activation scratch
below always takes the ordinary path, so turning on both options gives no replay. Engines
with aliased outputs, GPUs without stream-ordered memory, and drivers older than CUDA 12.5
also keep the ordinary path, as does a call on a green context stream. The delegate logs the
reason once.

Recording is also not safe next to every kind of work. While a recording is running, creating
or destroying a TensorRT engine elsewhere in the process is unsafe, and so is a whole device
sync such as `cudaDeviceSynchronize` or `torch.cuda.synchronize()`. Leave replay off when
other threads may do any of that. A C++ host can refuse a program's saved request for one
load:

```cpp
#include <executorch/extension/module/module.h>

using namespace executorch::extension;
using namespace executorch::runtime;

BackendOptions<1> options;
options.set_option("use_cuda_graphs", false);
LoadBackendOptionsMap by_backend;
by_backend.set_options("TensorRTBackend", options.view());
module.load(by_backend);
```

### Shared activation scratch

A TensorRT execution context allocates its own activation scratch and holds it for as long as
the context lives. A model lowered to many small engines pays that cost many times over, and
can run out of GPU memory on the engine count alone. This option backs all of a device's
contexts from one buffer instead, grown to the largest size any call has asked for:

```cpp
#include <executorch/runtime/backend/interface.h>

using namespace executorch::runtime;

BackendOptions<1> options;
options.set_option("use_shared_activation_scratch", true);
set_option("TensorRTBackend", options.view());
```

Set it before loading the methods whose contexts should use the pool. The memory saved is the
sum of the separate scratch sizes less the largest of them.

The cost is parallelism. Contexts that share one buffer do not run at the same time on the
device. The backend holds a per device lock from the claim on the buffer through the enqueue,
so two calls on one device are serialized at submission. The pool also never shrinks, and a
device that has run a pooled engine must not be reset with `cudaDeviceReset()`.

### Shared engines

This feature needs a Torch-TensorRT build newer than the 2026-10-06 nightly, see
[Feature availability](#feature-availability).

When the same program is loaded twice, for example once per robot arm, handles loaded from
the same engine bytes, for the same device and the same weight streaming request, share one
engine. The engine weights are in GPU memory once. Each handle still gets its own execution
context and buffers.

This is on by default. Measured on an 8 GB Jetson Orin Nano, loading one robot policy a second
time grew GPU memory by 806 MiB without sharing and by 122 MiB with it. Those numbers are the
cost of the second load, not the total. Finding a match means hashing the engine bytes on
every load, which costs roughly 0.15 s per GB on a Jetson AGX Thor. Pass the
`use_shared_engines` load option as `false` to keep one module's engines private.

## Examples

- [Export a static shape model](https://github.com/pytorch/TensorRT/blob/main/examples/torchtrt_executorch_example/export_static_shape.py)
- [Export a coalesced TensorRT and CUDA model](https://github.com/pytorch/TensorRT/blob/main/examples/torchtrt_executorch_example/export_coalesced.py)
- [Export a model that keeps its data on the GPU](https://github.com/pytorch/TensorRT/blob/main/examples/torchtrt_executorch_example/export_device_resident.py)
- [C++ reference runner, including the green context option](https://github.com/pytorch/TensorRT/tree/main/examples/executorch_reference_runner)

The Torch-TensorRT documentation covers the export options in more detail, including dynamic
shapes, programs with more than one method, and a zero copy KV cache for decoder models.
