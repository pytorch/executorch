**ExecuTorch** is a [PyTorch](https://pytorch.org/) platform that provides
infrastructure to run PyTorch programs everywhere from AR/VR wearables to
standard on-device iOS and Android mobile deployments. One of the main goals for
ExecuTorch is to enable wider customization and deployment capabilities of the
PyTorch programs.

The `executorch` pip package is in beta.
* Supported python versions: 3.10, 3.11, 3.12
* Compatible systems: Linux x86_64, macOS aarch64

The prebuilt `executorch.runtime` module included in this package provides a way
to run ExecuTorch `.pte` files, with some restrictions:
* Only [core ATen operators](docs/source/ir-ops-set-definition.md) are linked into the prebuilt module
* Only the [XNNPACK backend delegate](docs/source/backends/xnnpack/xnnpack-overview.md) is linked into the prebuilt module.
* \[macOS only] [Core ML](docs/source/backends/coreml/coreml-overview.md) and [MPS](docs/source/backends/mps/mps-overview.md) backend
  are also linked into the prebuilt module.

Please visit the [ExecuTorch website](https://pytorch.org/executorch) for
tutorials and documentation. Here are some starting points:
* [Getting Started](https://pytorch.org/executorch/main/getting-started-setup)
  * Set up the ExecuTorch environment and run PyTorch models locally.
* [Working with local LLMs](docs/source/llm/getting-started.md)
  * Learn how to use ExecuTorch to export and accelerate a large-language model
    from scratch.
* [Exporting to ExecuTorch](https://pytorch.org/executorch/main/tutorials/export-to-executorch-tutorial)
  * Learn the fundamentals of exporting a PyTorch `nn.Module` to ExecuTorch, and
    optimizing its performance using quantization and hardware delegation.
* Running etLLM on [iOS](https://github.com/meta-pytorch/executorch-examples/tree/main/llm/apple) and [Android](https://github.com/meta-pytorch/executorch-examples/tree/main/llm/android) devices.
  * Build and run LLaMA in a demo mobile app, and learn how to integrate models
    with your own apps.
