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

This page describes Torch-TensorRT 2.15 and newer. What is marked "nightly only" is not in
Torch-TensorRT 2.15 or the ExecuTorch build it installs. For those features, use a
[nightly build](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html#nightly-builds),
or a later release once there is one.

## Features

- **TensorRT engines inside a `.pte`.** One file holds the program and its engines.
- **Works with the CUDA backend in one program.** Operators that TensorRT cannot compile can go to
  the ExecuTorch [CUDA backend](../cuda/cuda-overview.md) instead of the CPU, so the whole model
  stays on the GPU.
- **Inputs and outputs that stay on the GPU.** The copies at the edges of the program can be
  turned off, for a pipeline that already has its data on the GPU.
- **Your own CUDA stream.** It can belong to a CUDA green context, which gives the model a fixed
  share of the GPU's compute units.
- **CUDA graph replay.** A whole engine runs with one launch.
- **Shared activation scratch.** Many engines can share one buffer for their temporary results.
- **Shared engines.** Loading the same program twice keeps one copy of the engine weights in GPU
  memory.
- **Weight streaming.** An engine can keep part of its weights in CPU memory.
- **Zero-copy KV cache.** The engine of a large language model (LLM) can update its key-value (KV)
  cache in place.
- **Nightly only:** engines in a separate data file, optimization profiles (tunings for different
  input sizes) chosen per call, CUDA graph replay for engines on the shared activation scratch, and
  setting runtime options from Python.

## Target Requirements

- **Hardware**: an NVIDIA GPU that CUDA 13 supports. ExecuTorch's CUDA backend is built for
  compute capability 8.0 (Ampere) and newer.
- **Operating system**: Linux on x86-64 with glibc (the GNU C library) 2.28 or newer, Linux on
  aarch64 with glibc 2.35 or newer, or Windows on x86-64 (nightly only).
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
[sending operators to the CUDA backend](#sending-the-rest-to-the-cuda-backend) is a Linux setup.

## Development Requirements

Install the packages as the
[installation section](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html#installation)
of the Torch-TensorRT guide shows. On Windows, follow its
[nightly builds](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html#nightly-builds)
part. That gives you Torch-TensorRT, PyTorch, a CUDA build of ExecuTorch, and the TensorRT
delegate, which is the `torch-tensorrt-executorch-runtime` package.

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
        retrace=False,
    )
```

`retrace=False` saves the compiled graph as it is, which nightly builds of Torch-TensorRT do by
default. Torch-TensorRT 2.15 re-exports the graph by default instead, so on 2.15 pass
`retrace=False`.

The Torch-TensorRT page on
[saving models](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/saving_models.html#executorch-save)
describes the other options of `save`. It also shows how to put several methods in one file, and
how to let the engine update a KV cache in place.

### Sending the rest to the CUDA backend

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
        retrace=False,
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
    retrace=False,
    backend_config=ExecutorchBackendConfig(
        propagate_device_config=PropagateDeviceConfig(
            skip_h2d_for_method_inputs=True,
            skip_d2h_for_method_outputs=True,
        ),
        memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
    ),
)
```

- The two skip flags work only with `enable_non_cpu_memory_planning`, which is on by default.
  With either flag set, turning it off makes the export raise a `ValueError`.
- `alloc_graph_input=False` is required. Without it the program keeps its own input buffer, and
  handing it a GPU tensor fails with error `0x10` (`NotSupported`). The log then says
  `Error copying tensor data at input 0`.
- The outputs stay in the program's own GPU memory. To have them outside that memory, also pass
  `alloc_graph_output=False`. Then each output needs a GPU buffer before the call: in C++ you pass
  your own with `Module::set_output`, and the Python runtime allocates one for you.

Pass CUDA tensors to a program exported this way. A CPU tensor costs a copy on every call when the
first delegate to use that input is TensorRT, and fails when it is the CUDA backend. **Nightly
only:** the Python runtime refuses it in both cases.

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
it without building anything from source. The steps below are for Linux.

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

- `kernels_optimized` provides the CPU operators: the copies between the CPU and the GPU at the
  edges of the program, and any operator that stays on the CPU.
- `backend_cuda` registers the allocator for the program's GPU memory, so a program that uses only
  TensorRT needs it too.

The two CMake packages are inside two Python packages, so point CMake at both. ExecuTorch's
location comes from its package metadata, because `executorch.__file__` is `None`:

```bash
ET_CMAKE=$(python -c 'import importlib.metadata as m, pathlib; print(pathlib.Path(str(m.distribution("executorch").locate_file("executorch"))) / "share" / "cmake")')
DELEGATE=$(TORCH_TENSORRT_SKIP_DELEGATE_REGISTRATION=1 python -c 'import pathlib, torch_tensorrt_executorch_runtime as r; print(pathlib.Path(r.__file__).parent)')
cmake -DCMAKE_PREFIX_PATH="${ET_CMAKE};${DELEGATE}" ...
```

`TORCH_TENSORRT_SKIP_DELEGATE_REGISTRATION=1` in the second command lets Python import the
delegate package to read its path, without loading the delegate.

You need CMake 3.28 or newer: on older versions the ExecuTorch package does not provide the
`executorch::` targets used above. CMake is not part of the packages, so install it yourself.

Running a program needs no header from the delegate. It registers itself when its library loads, and
the rest is the ordinary ExecuTorch C++ API:

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
shows how to set the options that apply at run time: from C++, or from Python with a nightly
build.

- [**Choosing the CUDA stream.**](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html#choosing-the-cuda-stream)
  By default, both delegates run on `cudaStreamPerThread`, the default stream of the calling
  thread, so a program split between them runs in the right order on one thread. To use your own
  stream, put one `executorch::extension::cuda::CallerStreamGuard` around the call. It reaches
  every delegate.
  - The stream must be on the same device as the engine, and the CUDA backend refuses a caller
    stream for a method that uses the CUDA backend's own CUDA graph option.
  - The guard comes from `executorch/extension/cuda/caller_stream.h`. Add the `extension_cuda`
    component to `find_package(executorch ...)` and link `executorch::extension_cuda`. Your own CUDA
    calls, such as creating the stream, also need `CUDA::cudart` from `find_package(CUDAToolkit)`,
    with a CUDA 13 toolkit.
  - Synchronize the stream before you read GPU outputs, because the CUDA backend can return while
    its work is still queued.
  - The stream can belong to a CUDA green context. A green context owns a fixed number of the GPU's
    streaming multiprocessors (SMs), so the model stays inside that part of the GPU. Creating its
    stream with `cuGreenCtxStreamCreate` also needs `CUDA::cuda_driver`. TensorRT then warns that
    the engine was built without green context constraints, which it says may cause crashes or
    undefined behavior. Torch-TensorRT has no option to build engines with them.
- [**CUDA graph replay.**](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html#cuda-graph-replay)
  Pass `use_cuda_graphs=True` to `torch_tensorrt.save`. It helps engines with fixed shapes that
  launch many short kernels.
  - It costs a copy of every input and output, and an engine whose input shapes change on every
    call never replays. Calls on a green context stream, or on a stream of another CUDA context,
    do not replay either, and the guide lists a few rarer cases.
  - A program saved with replay on can still have it turned off at load, with the
    `use_cuda_graphs` option.
  - While a graph is being recorded, no other thread may create or destroy a TensorRT engine or
    execution context (the object that runs an engine) anywhere outside this delegate, or call
    `cudaDeviceSynchronize()` or `torch.cuda.synchronize()`. Recording happens again after every
    shape change, so leave replay off when other threads may do these things.
- [**Shared activation scratch.**](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html#shared-activation-scratch)
  The `use_shared_activation_scratch` option, set for the whole process before the program loads,
  lets all engines on a device share one buffer for their temporary results. That saves memory for
  a model split into many engines, but two engines on the shared scratch never run at the same
  time. The scratch never shrinks, and a device that used it must not be reset with
  `cudaDeviceReset()`. Engines on the shared scratch replay CUDA graphs only in nightly builds.
- [**Shared engines.**](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html#shared-engines)
  On by default: loading the same program twice, on the same device and with the same weight
  streaming budget, keeps one copy of each engine in GPU memory. Engines are matched by a hash of
  their bytes, which a crafted program could match on purpose, so every program in the process must
  be trusted. To keep one program's engines private, pass the `use_shared_engines` option as
  `false` when you load it (from Python, nightly only).
- [**Weight streaming.**](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html#weight-streaming)
  An engine compiled with `enable_weight_streaming=True` can keep part of its weights in CPU memory.
  Set how many bytes stay on the GPU with `weight_streaming_budget_per_engine` in
  `torch_tensorrt.save`, or with the `weight_streaming_budget` load option.
- [**Engines in a separate data file**](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html#keep-the-engines-in-a-separate-file)
  (nightly only). Pass `external_engine_data="engines"` to `torch_tensorrt.save` to write the
  engines to `engines.ptd` instead of the `.pte`, and pass that file when you load the program.
  The Python loader takes one data file, so for a program that also uses the CUDA backend, pass
  `external_engine_data="aoti_cuda_blob"` to share its file.
- [**Optimization profiles**](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html#optimization-profiles)
  (nightly only). One engine can be tuned for several ranges of input shapes, for example one for
  decoding one token at a time and one for a long prompt. A C++ application picks the profile for
  each call with `torch_tensorrt::executorch_backend::OptimizationProfileGuard`, from the
  `torch_tensorrt/executorch/TensorRTBackend.h` header. Without the guard, every engine runs
  profile 0, and Python cannot choose a profile.
  - The Python packages do not carry the delegate's headers. Compile against `cpp/include` in the
    Torch-TensorRT source, or `torch_tensorrt/include` in `libtorchtrt.tar.gz`, at the same version
    as your delegate, and against the headers of a TensorRT SDK (the C++ development package).
  - On Windows, the packaged delegate does not export the guard.

## Examples

- [Export a static shape model](https://github.com/pytorch/TensorRT/blob/main/examples/torchtrt_executorch_example/export_static_shape.py)
- [Export a model whose input shapes can change](https://github.com/pytorch/TensorRT/blob/main/examples/torchtrt_executorch_example/export_dynamic_shape.py)
- [Export a model split between TensorRT and the CUDA backend](https://github.com/pytorch/TensorRT/blob/main/examples/torchtrt_executorch_example/export_coalesced.py)
- [Export a model that keeps its data on the GPU](https://github.com/pytorch/TensorRT/blob/main/examples/torchtrt_executorch_example/export_device_resident.py)
- [Export a decoder step whose KV cache stays in memory you own](https://github.com/pytorch/TensorRT/blob/main/examples/torchtrt_executorch_example/export_kv_cache_decode.py)
- [Export one engine with two optimization profiles](https://github.com/pytorch/TensorRT/blob/main/examples/torchtrt_executorch_example/export_multi_profile.py)
  (nightly only)
- [C++ and Python reference runners](https://github.com/pytorch/TensorRT/tree/main/examples/executorch_reference_runner)
  (the two multi-profile programs need a nightly build)
