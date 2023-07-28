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
- [Exporting to Executorch](/docs/website/docs/tutorials/exporting_to_executorch.md)
    - [EXIR Spec](/docs/website/docs/ir_spec/00_exir.md)
    - [Exporting manual](/docs/website/docs/export/00_export_manual.md)
    - [Delegate to a backend](/docs/website/docs/tutorials/backend_delegate.md)
- [Executorch Google Colab](https://colab.research.google.com/drive/1oJBt3fj_Tr3FE7L9RdUgSKK9XzJfUv4F#scrollTo=fC4CB3kFhHPJ)

## License
ExecuTorch is BSD licensed, as found in the LICENSE file.
