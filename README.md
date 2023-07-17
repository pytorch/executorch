# executorch
A unified ML software stack within the PyTorch platform for edge devices. It defines new compiler entry points as well as a state-of-art runtime.

https://fburl.com/executorch

## Why Executorch?
Compared to the legacy Lite Interpreter, there are some major benefits:
* Performance wins compared to Lite Interpreter
  * Faster (orders of magnitude lower framework tax in both [DSP] (https://fb.workplace.com/notes/156263446923296) and [CPU](https://fb.workplace.com/notes/821255839309664))
  * Much smaller binary size, [~1.5 MB vs. ~30 KB without operators](https://docs.google.com/document/d/11_QzIO1TEaRtLIcX4ubVzx-sUl2RPUm9Iwbn2Kt1ce4/edit#heading=h.7xrtrf77n4w5)
  * Smaller memory footprint because we do ahead of time memory planning in ExecuTorch and also have clear granular control over where the runtime allocations are done.
* Long term alignment with the direction of PyTorch infrastructure
  * Lite Interpreter relies on TorchScript, which is being phased out; ExecuTorch is the planned replacement for Lite Interpreter.
* Model Authoring & Productivity gains
  * More and better defined entry points to perform model, device, and/or use-case specific optimizations (e.g. better backend delegation, user-defined compiler transformations, default or user-defined memory planning, etc)
  * Ability to lower constructs like dynamic control flow to run on device.

## Meta Internal Users
See the [Using PyTorch > Executorch](https://www.internalfb.com/intern/wiki/PyTorch/Using_PyTorch/Executorch/)
wiki for pointers to internal workplace groups, how-tos, and other resources.

## Docs
* [Executorch stack diagram](https://docs.google.com/drawings/d/1bBIbG6YDIjdx8emS_6K23YM6WyRkKVpPt-26nznxroU/edit)
* [High-level design doc](https://docs.google.com/document/d/1Z12w6-KtwoFDh781LQAbfwUdEZw9cwTCmz5BmKuS6U8/edit#)
* Planning docs
  * H22022 [roadmap](https://fburl.com/executorch-plan)
  * H12022 [roadmap](https://fburl.com/executorch-plan-h12022), [summary runtime](https://fb.workplace.com/notes/1023022781732209), [summary EXIR](https://fb.workplace.com/notes/1094704288071438)

## Better Engineering
* [Coding guidelines](https://docs.google.com/document/d/1RERjvvUSNNQ_gysD-kkHhvbWyfCAaXk7pes9ZdZ1kqM/edit)
* [BE Tasks](https://www.internalfb.com/intern/taskgraph/?q=5567456559966061) -- Please add "[executorch][BE]" in the task title

## Model Migration
* [Model inventory onboarding guide](https://docs.google.com/document/d/1ofoKUvufDFZdZdEYQ1jgTCsuNbSLspDsahgiHhWncCY/edit)
* [End-to-end model testing](https://docs.google.com/document/d/1AeLlSgwhe9Gnj-44kIYv9iyLWdpb6ASF_epr0q4ey5s/edit)
## Design goals
* Minimal binary size (< 50KB not including kernels)
* Minimal framework tax: loading program, initializing executor, kernel and
  backend-delegate dispatch, runtime memory utilization
* Portable (cross-compile across many toolchains)
* Executes ATen kernels (or ATen custom kernels)
* Executes custom op kernels
* Supports inter op asynchronous execution
* Supports static memory allocation (heapless)
* Supports custom allocation across memory hierarchies
* Supports control flow needed by models
* Allows selective build of kernels
* Allows backend delegation with lightweight interface

## Terminology
### ATen mode
ATen mode uses the ATen (pytorch core) implementation of Tensor (`at::Tensor`)
along with related types (ScalarType, etc.)
* `at::Tensor` is big and complex, and often allocates memory with new/malloc
* The ATen kernels, which rely on the full `at::Tensor` API, are usable in this
  configuration
* Those kernels also tend to do dynamic memory allocation, and often have extra
  flexibility (and thus overhead) to handle things not needed by mobile/embedded
  clients: e.g., CUDA support, sparse tensor support, dtype promotion
### Lean mode
Lean mode uses Executorch's smaller `torch::executor::Tensor` (aka ETensor)
implementation, along with related types (`torch::executor::ScalarType`, etc.)
* ETensor's API is a source-compatible subset of `at::Tensor`. Code that is
  written against ETensor can also build against `at::Tensor`.
* "lean mode kernels" are any operator implementations that are written to be
  compatible with ETensor. But that means that the can also build against
  `at::Tensor` if desired, and used in the same model as ATen kernels.
* ETensor does not own or allocate memory on its own
  * (TODO(T133200526): NOTE: Dynamic shapes are not yet supported. Remove this
    warning when they are.) To support dynamic shapes, kernels can allocate
    Tensor data using the MemoryAllocator provided by the client.
### Portable kernels
See [//executorch/kernels/portable/README.md](portable/README.md) for technical details.

Portable kernels, which live under `//executorch/kernels/portable`, are:
* Lean mode kernels
* Compatible with ATen operator signatures
* Written in portable C++ so that they can build for any target
* Written as reference implementations, prioritizing clarity and simplicity
  over optimization
* Generally much smaller in code size than ATen kernels
* Written to avoid dynamically allocating memory using new/malloc
  * (TODO(T133200526): NOTE: Dynamic shapes are not yet supported. Remove this
    warning when they are.) To support dynamic shapes, some kernels may allocate
    Tensor data using the MemoryAllocator provided by the client.

## Local tests
### General tests
```
buck2 test fbcode//executorch/...
```
### Run a model in lean mode
* Uses the lean Executorch `Tensor` class and related types
* Uses the kernels under `//executorch/kernels/portable` instead of the ATen kernels
```
buck2 run fbcode//executorch/sdk/runners:executor_runner -- \
    --model_path=fbcode/executorch/test/models/linear_out.ff
```
### Run a model in ATen mode
* Instead of the lean Executorch `Tensor`, using ATen tensor so that all ATen kernels can be leveraged
* Note there can be significant size regression in ATen mode
```
buck2 run fbcode//executorch/sdk/runners:executor_runner_aten -- \
    --model_path=fbcode/executorch/test/models/linear_out.ff
```

## Special build modes
### Android/mobile builds
In xplat:
```
buck2 build @fbandroid/mode/opt @fbandroid/mode/ndk_libcxx \
    -c user.ndk_cxxflags="-frtti -fexceptions" \
    fbsource//xplat/executorch/sdk/runners:executor_runner
```
## ARVR builds
In xplat:
```
buck2 build @arvr/mode/android/linux/opt-stripped \
    -c ndk.custom_libcxx=false \
    fbsource//xplat/executorch/sdk/runners:executor_runner
```
