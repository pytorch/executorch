# Examples


The series of demos featured in this directory exemplify a broad spectrum of workflows for deploying ML models on edge devices using ExecuTorch. These demos offer practical insights into key processes such as model exporting, quantization, backend delegation, module composition, memory planning, program saving and  loading for inference on ExecuTorch runtime.

ExecuTorch's extensive support spans from simple modules like "Add" to comprehensive models like `MobileNet V3`, `Wav2Letter`, `Llama 2`, and more, showcasing its versatility in enabling the deployment of a wide spectrum of models across various edge AI applications.


## Directory structure
```bash
examples
├── models                            # Contains a set of popular and representative PyTorch models
├── portable                          # Contains end-to-end demos for ExecuTorch in portable mode
├── xnnpack                           # Contains end-to-end ExecuTorch demos with first-party optimization using XNNPack
├── selective_build                   # Contains demos of selective build for optimizing the binary size of the ExecuTorch runtime
|── arm                               # Contains demos of the Arm TOSA and Ethos-U NPU flows
|── qualcomm                          # Contains demos of Qualcomm QNN backend
├── demo-apps                         # Contains demo apps for Android and iOS
├── third-party                       # Third-party libraries required for working on the demos
└── README.md                         # This file
```


## Using the examples

A user's journey may commence by exploring the demos located in the [`portable/`](./portable) directory. Here, you will gain insights into the fundamental end-to-end workflow to generate a binary file from a ML model in [portable mode](/docs/website/docs/basics/terminology.md) and run it on the ExecuTorch runtime.


## Demo of XNNPACK delegation

The demos in the [`xnnpack/`](./xnnpack) directory provide valuable insights into the process of lowering and executing an ExecuTorch model with built-in performance enhancements. These demos specifically showcase the workflow involving [XNNPACK backend](https://github.com/pytorch/executorch/tree/main/backends/xnnpack) delegation and quantization.


## Demo of Selective Build

To understand how to deploy the ExecuTorch runtime with optimization for binary size, explore the demos available in the [`selective_build/`](./selective_build) directory. These demos are specifically designed to illustrate the [Selective Build](/docs/website/docs/tutorials/selective_build.md), offering insights into reducing the binary size while maintaining efficiency.


## Demo Apps

Explore mobile apps with ExecuTorch models integrated and deployable on Android and iOS in the [`demo-apps/android/`](./demo-apps/android) and [`demo-apps/apple_ios/`](./demo-apps/apple_ios) directories, respectively.


## Demo of ARM

TBD


## Demo of QNN Backend

Demos of [ExecuTorch QNN Backend](./qualcomm) by Qualcomm


## Dependencies

Various models and workflows listed in this directory have dependencies on some other packages. You need to follow the setup guide in [Setting up ExecuTorch from GitHub](/docs/website/docs/tutorials/00_setting_up_executorch.md) to have appropriate packages installed.
