@nocommit test edit 1
@nocommit test edit 2

# ExecuTorch
A unified ML software stack within the PyTorch platform for edge devices. It defines new compiler entry points as well as a state-of-art runtime.

## Why ExecuTorch?
Compared to the legacy Lite Interpreter, there are some major benefits:
* Performance wins compared to Lite Interpreter
  * Faster (orders of magnitude lower framework tax in both DSP and CPU)
  * Much smaller binary size, 1.5 MB vs 30 KB without operators.
  * Smaller memory footprint because we do ahead of time memory planning in ExecuTorch and also have clear granular control over where the runtime allocations are done.
* Long term alignment with the direction of PyTorch infrastructure
  * Lite Interpreter relies on TorchScript, which is being phased out; ExecuTorch is the planned replacement for Lite Interpreter.
* Model Authoring & Productivity gains
  * More and better defined entry points to perform model, device, and/or use-case specific optimizations (e.g. better backend delegation, user-defined compiler transformations, default or user-defined memory planning, etc)
  * Ability to lower constructs like dynamic control flow to run on device.


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

## Quick Links

- [Basics: Terminology](/docs/website/docs/basics/terminology.md)
- [Wiki (internal-only)](https://www.internalfb.com/intern/wiki/PyTorch/Using_PyTorch/Executorch/)
- [Static docs website (internal-only)](https://www.internalfb.com/intern/staticdocs/executorch/)
- [Testing (internal-only)](https://www.internalfb.com/intern/staticdocs/executorch/docs/fb/poc/)

## Quick Links for Partners

- [Setting up ExecuTorch from GitHub](/docs/website/docs/tutorials/00_setting_up_executorch.md)
    - (Optional) [Building with CMake](/docs/website/docs/tutorials/cmake_build_system.md)
- [Exporting to Executorch](/docs/website/docs/tutorials/exporting_to_executorch.md)
    - [EXIR Spec](/docs/website/docs/ir_spec/00_exir.md)
    - [Exporting manual](/docs/website/docs/export/00_export_manual.md)
    - [Quantization](/docs/website/docs/tutorials/quantization_flow.md)
    - [Delegate to a backend](/docs/website/docs/tutorials/backend_delegate.md)
    - [Profiling](/docs/website/docs/tutorials/profiling.md)
- [Executorch Google Colab](https://colab.research.google.com/drive/1m8iU4y7CRVelnnolK3ThS2l2gBo7QnAP#scrollTo=1o2t3LlYJQY5)

## Directory Structure [WIP]

```
executorch
├── backends                        #  1st party backend implementations.
|   ├── xnnpack
|   ├── vulkan
|   ├── backend_api.py              # TODO move to exir/backend
|   ├── backend_details.py          # TODO move to exir/backend
|   ├── partioner.py                # TODO move to exir/backend
├── build                           #  Utilities for managing the build system.
├── bundled_program                 #  Utilities for attaching reference inputs and outputs to models. TODO move to extension
├── codegen                         #  Tooling to autogenerate bindings between kernels and the runtime. TODO move to tool
├── configurations                  #  TODO delete this
├── docs                            #  Static docs tooling
├── examples                        #  Examples of various user flows, such as model export, delegates, and runtime execution.
|   ├── executor_runner
|   ├── export
|   ├── models
├── exir                            #  Ahead of time library, model capture and lowering apis.
|   ├── capture                     #  Program capture.
|   ├── dialects                    #  Op sets for various dialects in the export process.
|   ├── emit                        #  Conversion from ExportedProgram to Executorch execution instructions.
|   ├── program                     #  Export artifacts.
|   ├── serialize                   #  Serialize final export artifact.
├── extension                       #  Extensions built on top of the runtime.
|   ├── aten_util
|   ├── data_loader                 # 1st party data loader implementations.
|   ├── memory_allocator            # 1st party memory allocator implementations.
|   ├── pybindings                  # Python api for executorch runtime.
|   ├── pytree                      # C++ and Python flattening and unflattening lib for pytrees.
|   ├── testing_util
├── kernels                         #  1st party kernel implementations.
|   ├── aten
|   ├── optimized
|   ├── portable                    #  Reference implementations of ATen operators.
|   ├── prim_ops                    #  Special ops used in executorch runtime for control flow and symbolic primitives.
|   ├── quantized
├── profiler                        #  Utilities for profiling. TODO delete in favor of ETDump in sdk/
├── runtime                         #  core cpp runtime of executorch
|   ├── backend                     #  Backend definition and registration.
|   ├── core                        #  Core structures used across all levels of the runtime
|   ├── executor                    #  Model loading, initalization, and execution.
|   ├── kernel                      #  Kernel registration and management.
|   ├── platform                    #  Layer between architecture specific code and user calls.
├── schema                          #  Executorch program definition, TODO move under serialization/
├── scripts                         #  Utility scripts for size management, dependency management, etc.
├── sdk                             #  Model profiling, debugging, and introspection: NOT READY YET FOR OSS USE
├── shim                            #  Compatibility layer between OSS and Internal builds
├── test                            #  Broad scoped end2end tests
├── third-party                     #  third-party dependencies
├── util                            #  TODO delete this
```

## License
ExecuTorch is BSD licensed, as found in the LICENSE file.
