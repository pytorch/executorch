# TensorRT Backend

The TensorRT backend runs ExecuTorch models on NVIDIA GPUs with
[TensorRT](https://developer.nvidia.com/tensorrt), NVIDIA's inference library. At export time,
TensorRT compiles the parts of the model it supports into engines (programs optimized for your
GPU), and by default the engines are stored in the `.pte` file. At run time, the TensorRT delegate
hands each engine back to TensorRT.

The delegate is built and shipped by the [Torch-TensorRT](https://github.com/pytorch/TensorRT)
project, not by ExecuTorch. You export with Torch-TensorRT, and you run with the normal ExecuTorch
runtime and its usual APIs. The
[Torch-TensorRT ExecuTorch guide](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html)
covers every option in detail.

This page describes Torch-TensorRT 2.15 and newer.

## Features

- **TensorRT engines inside a `.pte`.** By default, one file holds the program and its engines.
- **Works with the CUDA backend in one program.** Operators that TensorRT cannot compile can go to
  the ExecuTorch [CUDA backend](../cuda/cuda-overview.md) instead of the CPU, so the whole model
  stays on the GPU.
- **Inputs and outputs that stay on the GPU.** The copies at the edges of the program can be
  turned off, for a pipeline that already has its data on the GPU.
- **Your own CUDA stream.** The stream can also belong to a CUDA green context, which keeps the
  model inside a fixed part of the GPU. TensorRT warns that this setup can crash, as
  [Performance Options](#performance-options) explains.
- **CUDA graph replay.** The GPU work of an engine is recorded once and then replayed with a single
  launch, which saves CPU time on every call.
- **Shared activation scratch.** Many engines can share one GPU buffer for their temporary results
  (activations), which saves memory.
- **Shared engines.** Loading the same program twice keeps one copy of the engine weights in GPU
  memory.
- **Weight streaming.** An engine can keep part of its weights in CPU memory.
- **Zero-copy KV cache.** The engine of a large language model (LLM) can update its key-value (KV)
  cache in place.
- **Engines in a separate data file.** The engines can be kept in a `.ptd` file next to the `.pte`.
- **Optimization profiles.** One engine can carry tunings for different input sizes, and a C++
  application picks one per call.

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

    Recording happens again after every shape change, so leave replay off when other threads may
    do these things.
- [**Shared activation scratch.**](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html#shared-activation-scratch)
  The `use_shared_activation_scratch` option, set for the whole process before the program loads,
  lets all engines on a device share one buffer for their temporary results. That saves memory for
  a model split into many engines, but two engines on the shared scratch never run at the same
  time. The scratch never shrinks, and a device that used it must not be reset with
  `cudaDeviceReset()`.
- [**Shared engines.**](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html#shared-engines)
  On by default: loading the same program twice, on the same device and with the same weight
  streaming budget, keeps one copy of each engine in GPU memory. Engines are matched by a hash of
  their bytes, which a crafted program could match on purpose, so every program in the process must
  be trusted. To keep one program's engines private, pass the `use_shared_engines` option as
  `false` when you load it.
- [**Weight streaming.**](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html#weight-streaming)
  An engine compiled with `enable_weight_streaming=True` can keep part of its weights in CPU memory.
  Set how many bytes stay on the GPU with `weight_streaming_budget_per_engine` in
  `torch_tensorrt.save`, or with the `weight_streaming_budget` load option. The load option takes
  the number of bytes as a decimal string, or as an integer below 2 GiB.
- [**Engines in a separate data file.**](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html#keep-the-engines-in-a-separate-file)
  Pass `external_engine_data="engines"` to `torch_tensorrt.save` to write the engines to
  `engines.ptd` instead of the `.pte`, and pass that file when you load the program.
  The Python loader takes one data file, so for a program that also uses the CUDA backend, pass
  `external_engine_data="aoti_cuda_blob"` to share its file.
- [**Optimization profiles.**](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html#optimization-profiles)
  One engine can be tuned for several ranges of input shapes, for example one for decoding one
  token at a time and one for a long prompt. A C++ application picks the profile for each call
  with `torch_tensorrt::executorch_backend::OptimizationProfileGuard`, from the
  `torch_tensorrt/executorch/TensorRTBackend.h` header. Without the guard, every engine runs
  profile 0, and Python cannot choose a profile. The Python packages do not carry that header, and
  the guide explains where to get it.

## Examples

- [Export a static shape model](https://github.com/pytorch/TensorRT/blob/main/examples/torchtrt_executorch_example/export_static_shape.py)
- [Export a coalesced TensorRT and CUDA model](https://github.com/pytorch/TensorRT/blob/main/examples/torchtrt_executorch_example/export_coalesced.py)
- [Export a model that keeps its data on the GPU](https://github.com/pytorch/TensorRT/blob/main/examples/torchtrt_executorch_example/export_device_resident.py)
- [Export a decoder step whose KV cache stays in memory you own](https://github.com/pytorch/TensorRT/blob/main/examples/torchtrt_executorch_example/export_kv_cache_decode.py)
- [Export one engine with two optimization profiles](https://github.com/pytorch/TensorRT/blob/main/examples/torchtrt_executorch_example/export_multi_profile.py)
- [C++ and Python reference runners](https://github.com/pytorch/TensorRT/tree/main/examples/executorch_reference_runner)
