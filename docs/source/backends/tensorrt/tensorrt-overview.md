# TensorRT Backend

The TensorRT backend runs ExecuTorch models on NVIDIA GPUs with
[TensorRT](https://developer.nvidia.com/tensorrt), NVIDIA's inference library. At export time,
TensorRT compiles the parts of the model it supports into engines (programs optimized for your
GPU), and the engines are stored in the `.pte` file. At run time, the TensorRT delegate hands each
engine back to TensorRT.

The delegate is built and shipped by the [Torch-TensorRT](https://github.com/pytorch/TensorRT)
project, not by ExecuTorch. You export with Torch-TensorRT, and you run with the normal ExecuTorch
runtime and its usual APIs. The
[Torch-TensorRT ExecuTorch guide](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html)
covers every option in detail.

This page describes Torch-TensorRT 2.15 and newer. Features marked "newer than 2.15" need a later
Torch-TensorRT build, such as a
[nightly build](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html#nightly-builds).

## Features

- **TensorRT engines inside a `.pte`.** One file holds the program and its engines.
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
- **Newer than 2.15:** engines in a separate data file, and optimization profiles (tunings for
  different input sizes) chosen per call.

## Target Requirements

- **Hardware**: an NVIDIA GPU that CUDA 13 supports. The prebuilt ExecuTorch CUDA packages need
  compute capability 8.0 (Ampere) or newer.
- **Operating system**: Linux on x86-64 with glibc (the GNU C library) 2.28 or newer, Linux on
  aarch64 with glibc 2.35 or newer, or Windows on x86-64.
- **CUDA**: 13. The prebuilt delegate links the CUDA 13 runtime.
- **Drivers**: an NVIDIA driver that supports CUDA 13.

On Linux, TensorRT and the CUDA runtime come as dependencies of the Python packages, so you do not
need a system install of either. On Windows, TensorRT comes the same way, but the delegate loads
the CUDA runtime from a CUDA 13 toolkit install. Install the toolkit, and make sure `CUDA_PATH` or
`CUDA_HOME` points at it.

By default, TensorRT builds each engine for the operating system, the GPU and the TensorRT version
it is exported with. So export your programs on the kind of machine that will run them, with the
same packages.

On Windows, ExecuTorch's CUDA backend cannot export a program, and ExecuTorch's Python runtime there
does not include the CUDA backend. So
[sending unsupported operators to the CUDA backend](#sending-unsupported-operators-to-the-cuda-backend)
is a Linux setup. A program that uses only TensorRT still runs from Python on Windows, because there
the delegate registers the CUDA memory allocator itself.

## Development Requirements

One command installs everything, in a fresh virtual environment:

```bash
pip install "torch_tensorrt[executorch]"
```

That gives you Torch-TensorRT, PyTorch, a CUDA build of ExecuTorch, and the TensorRT delegate,
which is the `torch-tensorrt-executorch-runtime` package. On Windows, add the CUDA package index,
because PyPI has only the CPU builds of PyTorch and ExecuTorch for Windows:

```bash
pip install "torch_tensorrt[executorch]" --extra-index-url https://download.pytorch.org/whl/cu132
```

The
[installation section](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html#installation)
of the Torch-TensorRT guide has the other options.

The CUDA build of ExecuTorch is required even to run a program: the delegate needs ExecuTorch's
`extension_cuda` library, which only the CUDA builds ship. To send the operators that TensorRT
rejects to the CUDA backend, you also need a CUDA toolkit, because that backend compiles them at
export time with `nvcc`, the CUDA compiler. PyTorch finds `nvcc` on your `PATH` or under
`CUDA_HOME`.

## Using the TensorRT Backend

The Python examples on this page run in order, and later ones reuse names from earlier ones.

### Exporting a model

Compile the model with Torch-TensorRT, then save it in ExecuTorch format:

```python
import torch
import torch_tensorrt


class MyModel(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x) + 1


with torch.no_grad():
    model = MyModel().eval().cuda()
    example_inputs = (torch.randn(2, 3, 4, 4, device="cuda"),)

    exported_program = torch.export.export(model, example_inputs)
    trt_gm = torch_tensorrt.dynamo.compile(
        exported_program,
        arg_inputs=example_inputs,
        min_block_size=1,
    )
    torch_tensorrt.save(
        trt_gm,
        "model.pte",
        output_format="executorch",
    )
```

The Torch-TensorRT page on
[saving models](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/saving_models.html#executorch-save)
describes the other options of `save`. It also shows how to put several methods in one file, and
how to let the engine update a KV cache in place.

### Sending unsupported operators to the CUDA backend

TensorRT cannot convert every PyTorch operator. By default the ones it rejects run on ExecuTorch's
CPU kernels, with a copy in each direction on every call. If an operator such as `erfinv` has no
CPU kernel, the method fails to load. Pass a `CudaPartitioner`, and the ExecuTorch CUDA backend
compiles those operators instead:

```python
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner


class CoalescedModel(torch.nn.Module):
    def forward(self, x):
        return torch.cos(torch.erfinv(torch.tanh(x)))


with torch.no_grad():
    coalesced_model = CoalescedModel().eval().cuda()
    coalesced_inputs = (torch.randn(64, 64, device="cuda"),)

    coalesced_gm = torch_tensorrt.dynamo.compile(
        torch.export.export(coalesced_model, coalesced_inputs),
        arg_inputs=coalesced_inputs,
        min_block_size=1,
    )
    torch_tensorrt.save(
        coalesced_gm,
        "coalesced.pte",
        output_format="executorch",
        partitioners=[
            CudaPartitioner([CudaBackend.generate_method_name_compile_spec("forward")])
        ],
    )
```

TensorRT takes what it can first, and the `CudaPartitioner` takes the rest. TensorRT cannot take
`erfinv`, so this program's delegates are `['TensorRTBackend', 'CudaBackend', 'TensorRTBackend']`.
The Torch-TensorRT docs call such a program *coalesced*.

The export writes the weights of the CUDA part to a data file, `aoti_cuda_blob.ptd`, next to the
`.pte`. Pass that file when you load the program: if the CUDA part has weights, loading fails
without it. The file has the same name for every model, so a second export into the same directory
overwrites the first one's weights, and the first model can then return wrong answers with no
error. Save each model into its own directory.

### Keeping inputs and outputs on the GPU

By default, the program copies its inputs to the GPU before the first delegate, and its outputs
back after the last one, so you can call it with CPU tensors. If your data is already on the GPU,
turn those copies off. In the flag names, h2d means host (CPU) to device (GPU), and d2h the other
way:

```python
from executorch.exir import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.propagate_device_config import PropagateDeviceConfig

torch_tensorrt.save(
    trt_gm,
    "device_resident.pte",
    output_format="executorch",
    backend_config=ExecutorchBackendConfig(
        propagate_device_config=PropagateDeviceConfig(
            skip_h2d_for_method_inputs=True,
            skip_d2h_for_method_outputs=True,
        ),
        memory_planning_pass=MemoryPlanningPass(
            alloc_graph_input=False, alloc_graph_output=False
        ),
    ),
)
```

- The two skip flags work only with `enable_non_cpu_memory_planning`, which is on by default.
  With either flag set, turning it off makes the export raise a `ValueError`.
- `alloc_graph_input=False` is required. Without it the program keeps its own input buffer, and
  every call fails with error `0x10` (`NotSupported`). The log then says
  `Planned-buffer copy needs host memory on both sides`.
- `alloc_graph_output=False` keeps the outputs out of the program's memory. From Python, the
  runtime allocates GPU memory for each output, and the next call does not overwrite an output you
  still hold, unless you pass `clone_outputs=False`. From C++, you must give the method your own
  GPU buffer once with `Module::set_output`, before the first run, as the
  [CUDA backend page](../cuda/cuda-overview.md#things-worth-knowing) shows. If you keep outputs
  planned instead, the next call overwrites them, so Python's `execute()` copies them on every call
  unless you pass `clone_outputs=False`.

Pass CUDA tensors to a program exported this way. The Python runtime refuses a CPU tensor. From
C++, what happens to a CPU tensor depends on which delegate reads that input first:

- TensorRT: the call works, but copies the tensor on every call.
- The CUDA backend: the call fails.

## Runtime Integration

### Python

Import the delegate package once, before you load a program. The import registers the delegate.
ExecuTorch's own CUDA backend needs no import, and nothing else in your code changes:

```python
from pathlib import Path

import torch
import torch_tensorrt_executorch_runtime  # noqa: F401
from executorch.runtime import Runtime

program = Runtime.get().load_program(Path("model.pte"))
forward = program.load_method("forward")
outputs = forward.execute((torch.ones(2, 3, 4, 4),))
```

If the delegate cannot load, the import raises a `DelegateCompatibilityError` right away. If the
export wrote a data file, such as `aoti_cuda_blob.ptd` from the CUDA backend, pass it when you load:

```python
program = Runtime.get().load_program(
    Path("coalesced.pte"), data_path=Path("aoti_cuda_blob.ptd")
)
outputs = program.load_method("forward").execute((torch.ones(64, 64),))
```

A program that keeps its inputs and outputs on the GPU takes CUDA tensors and returns CUDA tensors:

```python
program = Runtime.get().load_program(Path("device_resident.pte"))
outputs = program.load_method("forward").execute(
    (torch.ones(2, 3, 4, 4, device="cuda"),)
)
```

### C++

The Python packages ship the prebuilt delegate and its CMake package, so a C++ application can link
it without building anything from source. The steps below are for Linux. On Windows, link the
same components and copy the DLLs beside your program, as
[CUDA on Windows](../../using-executorch-cpp.md#cuda-on-windows) shows, but export TensorRT programs
on Windows itself.

**1. Write the program.** Running a program needs no header from the delegate. The delegate
registers itself when its library loads, and the rest is the ordinary ExecuTorch C++ API. Save
this as `main.cpp`:

```cpp
#include <cstdio>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

using namespace executorch::extension;

int main() {
  Module module("model.pte");

  auto input = make_tensor_ptr({2, 3, 4, 4}, std::vector<float>(2 * 3 * 4 * 4, 1.0f));

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

**2. Write `CMakeLists.txt`.**

```cmake
cmake_minimum_required(VERSION 3.28)
project(my_app CXX)

find_package(executorch REQUIRED COMPONENTS backend_cuda kernels_optimized)
find_package(executorch_backend_tensorrt REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE
  executorch::runtime
  executorch::backend_cuda
  executorch::backend_tensorrt
  executorch::kernels_optimized
)
```

- `kernels_optimized` provides the CPU operators: the copies between the CPU and the GPU at the
  edges of the program, and any operator that stays on the CPU.
- `backend_cuda` registers the allocator for the program's GPU memory, so a program that uses only
  TensorRT needs it too.
- You need CMake 3.28 or newer. On older versions `find_package(executorch ...)` fails, because the
  package provides the `executorch::` targets only from 3.28. CMake is not part of the packages, so
  install it yourself.

**3. Build and run.** The two CMake packages are inside two Python packages, so point CMake at
both:

```bash
ET_CMAKE=$(python -c 'import importlib.metadata as m, pathlib; print(pathlib.Path(str(m.distribution("executorch").locate_file("executorch"))) / "share" / "cmake")')
DELEGATE=$(TORCH_TENSORRT_SKIP_DELEGATE_REGISTRATION=1 python -c 'import pathlib, torch_tensorrt_executorch_runtime as r; print(pathlib.Path(r.__file__).parent)')
cmake -S . -B build -DCMAKE_PREFIX_PATH="${ET_CMAKE};${DELEGATE}"
cmake --build build
./build/my_app
```

`TORCH_TENSORRT_SKIP_DELEGATE_REGISTRATION=1` in the second command lets Python import the
delegate package to read its path, without loading the delegate. Run `./build/my_app` from the
folder that holds `model.pte`.

If the export wrote a data file, pass it to the constructor:

```cpp
Module module("coalesced.pte", "aoti_cuda_blob.ptd");
```

The Torch-TensorRT guide explains
[how to ship such an application](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html#run-from-c)
to other machines, and links to the instructions for building the delegate from source.

## Performance Options

Each option below links to its section of the Torch-TensorRT guide. The guide's section
[Setting a runtime option](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html#setting-a-runtime-option)
shows how to set the options that apply at run time, from C++ or from Python.

- [**Choosing the CUDA stream.**](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html#choosing-the-cuda-stream)
  By default, both delegates run on `cudaStreamPerThread`, the default stream of the calling
  thread, so a program split between them runs in the right order on one thread. To use your own
  stream, put one guard around the call. It reaches every delegate:

  ```cpp
  #include <executorch/extension/cuda/caller_stream.h>

  using namespace executorch::extension;

  cuda::CallerStreamGuard guard(stream);
  module.forward(input);
  ```

  The guard is in the `extension_cuda` library. Your own CUDA calls, such as creating the stream,
  also need the CUDA runtime library from a CUDA 13 toolkit. In `CMakeLists.txt`, replace the
  `find_package(executorch ...)` line from step 2 with the first line below, add the second line
  after it, and add the third line after `add_executable`:

  ```cmake
  find_package(executorch REQUIRED COMPONENTS backend_cuda extension_cuda kernels_optimized)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(my_app PRIVATE executorch::extension_cuda CUDA::cudart)
  ```

  - The stream must be on the same device as the engine.
  - The CUDA backend refuses a caller stream for a method that uses its own CUDA graph option.
  - Synchronize the stream before you read GPU outputs, because the CUDA backend can return while
    its work is still queued.
  - The stream can belong to a CUDA green context. A green context owns a fixed number of the GPU's
    streaming multiprocessors (SMs), so the model stays inside that part of the GPU. Creating its
    stream with `cuGreenCtxStreamCreate` also needs `CUDA::cuda_driver`. TensorRT then warns that
    the engine was built without green context constraints, which it says may cause crashes or
    undefined behavior. Torch-TensorRT has no option to build engines with them.
- [**CUDA graph replay.**](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html#cuda-graph-replay)
  Turn it on when you save. It helps engines with fixed shapes that launch many short kernels:

  ```python
  torch_tensorrt.save(
      trt_gm,
      "cuda_graphs.pte",
      output_format="executorch",
      use_cuda_graphs=True,
  )
  ```

  - It costs a copy of every input and output, and an engine whose input shapes change on every
    call never replays. Calls on a green context stream, or on a stream of another CUDA context,
    do not replay either, and the guide lists a few rarer cases.
  - A program saved with replay on can still have it turned off at load, with the
    `use_cuda_graphs` option.
  - While a graph is being recorded, no other thread may:
    - create or destroy a TensorRT engine or execution context (the object that runs an engine)
      outside this delegate,
    - call `cudaDeviceSynchronize()` or `torch.cuda.synchronize()`,
    - or load a program that uses the CUDA backend, which can make such a sync.

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
- [**Engines in a separate data file**](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html#keep-the-engines-in-a-separate-file)
  (newer than 2.15). Pass `external_engine_data="engines"` to `torch_tensorrt.save` to write the
  engines to `engines.ptd` instead of the `.pte`, and pass that file when you load the program.
  The Python loader takes one data file, so for a program that also uses the CUDA backend, pass
  `external_engine_data="aoti_cuda_blob"` to share its file.
- [**Optimization profiles**](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html#optimization-profiles)
  (newer than 2.15). One engine can be tuned for several ranges of input shapes, for example one
  for decoding one token at a time and one for a long prompt. A C++ application picks the profile for
  each call with `torch_tensorrt::executorch_backend::OptimizationProfileGuard`, from the
  `torch_tensorrt/executorch/TensorRTBackend.h` header. Without the guard, every engine runs
  profile 0, and Python cannot choose a profile. The Python packages do not carry that header, and
  the guide explains where to get it.

## Examples

- [Export a static shape model](https://github.com/pytorch/TensorRT/blob/main/examples/torchtrt_executorch_example/export_static_shape.py)
- [Export a model whose input shapes can change](https://github.com/pytorch/TensorRT/blob/main/examples/torchtrt_executorch_example/export_dynamic_shape.py)
- [Export a model split between TensorRT and the CUDA backend](https://github.com/pytorch/TensorRT/blob/main/examples/torchtrt_executorch_example/export_coalesced.py)
- [Export a model that keeps its data on the GPU](https://github.com/pytorch/TensorRT/blob/main/examples/torchtrt_executorch_example/export_device_resident.py)
- [Export a decoder step whose KV cache stays in memory you own](https://github.com/pytorch/TensorRT/blob/main/examples/torchtrt_executorch_example/export_kv_cache_decode.py)
- [Export one engine with two optimization profiles](https://github.com/pytorch/TensorRT/blob/main/examples/torchtrt_executorch_example/export_multi_profile.py)
  (newer than 2.15)
- [C++ and Python reference runners](https://github.com/pytorch/TensorRT/tree/main/examples/executorch_reference_runner)
  (the two multi-profile programs need a build newer than 2.15)
