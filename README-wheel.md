**ExecuTorch** is a [PyTorch](https://pytorch.org/) platform that provides
infrastructure to run PyTorch programs everywhere from AR/VR wearables to
standard on-device iOS and Android mobile deployments. One of the main goals for
ExecuTorch is to enable wider customization and deployment capabilities of the
PyTorch programs.

The `executorch` pip package is in alpha.
* Supported python versions: 3.10, 3.11
* Compatible systems: Linux x86_64, macOS aarch64

The prebuilt `executorch.extension.pybindings.portable_lib` module included in
this package provides a way to run ExecuTorch `.pte` files, with some
restrictions:
* Only [core ATen
  operators](https://pytorch.org/executorch/stable/ir-ops-set-definition.html)
  are linked into the prebuilt module
* Only the [XNNPACK backend
  delegate](https://pytorch.org/executorch/main/native-delegates-executorch-xnnpack-delegate.html)
  is linked into the prebuilt module
* [macOS only] [Core ML](https://pytorch.org/executorch/main/build-run-coreml.html) and [MPS](https://pytorch.org/executorch/main/build-run-mps.html) backend delegates are linked into the prebuilt module.

Please visit the [ExecuTorch website](https://pytorch.org/executorch/) for
tutorials and documentation. Here are some starting points:
* [Getting
  Started](https://pytorch.org/executorch/stable/getting-started-setup.html)
  * Set up the ExecuTorch environment and run PyTorch models locally.
* [Working with
  local LLMs](https://pytorch.org/executorch/stable/llm/getting-started.html)
  * Learn how to use ExecuTorch to export and accelerate a large-language model
    from scratch.
* [Exporting to
  ExecuTorch](https://pytorch.org/executorch/main/tutorials/export-to-executorch-tutorial.html)
  * Learn the fundamentals of exporting a PyTorch `nn.Module` to ExecuTorch, and
    optimizing its performance using quantization and hardware delegation.
* Running LLaMA on
  [iOS](https://pytorch.org/executorch/stable/llm/llama-demo-ios.html) and
  [Android](https://pytorch.org/executorch/stable/llm/llama-demo-android.html)
  devices.
  * Build and run LLaMA in a demo mobile app, and learn how to integrate models
    with your own apps.
