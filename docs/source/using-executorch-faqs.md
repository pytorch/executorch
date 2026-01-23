# Frequently Asked Questions

This page summarizes frequently asked questions and provides guidance on issues that commonly occur when adopting ExecuTorch.

If a specific issue is not covered here, consider searching for or creating an issue on GitHub under [Issues](https://github.com/pytorch/executorch/issues) or [Discussions](https://github.com/pytorch/executorch/discussions).

## Installation

### Missing /usr/include/python3.x

Most likely `python-dev` library needs to be installed. Please run
```
sudo apt install python<version>-dev
```
if you are using Ubuntu, or use an equivalent install command.

### ModuleNotFoundError: No module named 'pytorch_tokenizers'

The `pytorch_tokenizers` package is required for LLM export functionality. Install it from the ExecuTorch source code:
```
pip install -e ./extension/llm/tokenizers/
```

## Export

### Missing out variants: { _ }

The model likely contains torch custom operators. Custom ops need an Executorch implementation and need to be loaded at export time. See the [ExecuTorch Custom Ops Documentation](kernel-library-custom-aten-kernel.md#apis) for details on how to do this.

### RuntimeError: PyTorch convert function for op _ not implemented

The model likely contains an operator that is not yet supported on ExecuTorch. In this case, consider searching for or creating an issue on [GitHub](https://github.com/pytorch/executorch/issues).

## Runtime

ExecuTorch error codes are defined in [executorch/core/runtime/error.h](https://github.com/pytorch/executorch/blob/main/runtime/core/error.h).

### Inference is Slow / Performance Troubleshooting

If building the runtime from source, ensure that the build is done in release mode. For CMake builds, this can be done by passing `-DCMAKE_BUILD_TYPE=Release`.

Ensure the model is delegated. If not targeting a specific accelerator, use the XNNPACK delegate for CPU performance. Undelegated operators will typically fall back to the ExecuTorch portable library, which is designed as a fallback, and is not intended for performance sensitive operators. To target XNNPACK, pass an `XnnpackPartitioner` to `to_edge_transform_and_lower`. See [Building and Running ExecuTorch with XNNPACK Backend](tutorial-xnnpack-delegate-lowering.md) for more information.

Thread count can have a significant impact on CPU performance. The optimal thread count may depend on the model and application. By default, ExecuTorch will currently use as many threads as there are cores. Consider setting the thread count to cores / 2, or just set to 4 on mobile CPUs.

Thread count can be set with the following function. Ensure this is done prior to loading or running a model.
```
::executorch::extension::threadpool::get_threadpool()->_unsafe_reset_threadpool(num_threads);
```

For a deeper investigation into model performance, ExecuTorch supports operator-level performance profiling. See [Using the ExecuTorch Developer Tools to Profile a Model](tutorials/devtools-integration-tutorial) <!-- @lint-ignore --> for more information.

### Numerical Accuracy Issues

If you encounter numerical accuracy issues or unexpected model outputs, ExecuTorch provides debugging tools to identify numerical discrepancies. See [Using the ExecuTorch Developer Tools to Debug a Model](tutorials/devtools-debugging-tutorial) <!-- @lint-ignore --> for a step-by-step guide on debugging numerical issues in delegated models.

### Missing Logs

ExecuTorch provides hooks to route runtime logs. By default, logs are sent to stdout/stderr, but users can override `et_pal_emit_log_message` to route logs to a custom destination. The Android and iOS extensions also provide out-of-box log routing to the appropriate platform logs. See [Runtime Platform Abstraction Layer (PAL)](runtime-platform-abstraction-layer.md) for more information.

### Error setting input: 0x10 / Attempted to resize a bounded tensor...

This usually means the inputs provided do not match the shape of the example inputs used during model export. If the model is expected to handle varying size inputs (dynamic shapes), make sure the model export specifies the appropriate bounds. See [Expressing Dynamism](https://pytorch.org/docs/stable/export.html#expressing-dynamism) for more information on specifying dynamic shapes.

### Error 0x14 (Operator Missing)

This usually means that the selective build configuration is incorrect. Ensure that the operator library is generated from the current version of the model and the corresponding `et_operator_library` is a dependency of the app-level `executorch_generated_lib` and the generated lib is linked into the application.

This can also occur if the ExecuTorch portable library does not yet have an implementation of the given ATen operator. In this case, consider search for or creating an issue on [GitHub](https://github.com/pytorch/executorch/issues).

### Error 0x20 (Not Found)

This error can occur for a few reasons, but the most common is a missing backend target. Ensure the appropriate backend target is linked. For XNNPACK, this is `xnnpack_backend`. If the backend is linked but is still not available, try linking with --whole-archive: `-Wl,--whole-archive libxnnpack_backend.a -Wl,--no-whole-archive`.

### Duplicate Kernel Registration Abort

This manifests as a crash call stack including ExecuTorch kernel registration and failing with an `et_pal_abort`. This typically means there are multiple `gen_operators_lib` targets linked into the applications. There must be only one generated operator library per target, though each model can have its own `gen_selected_ops/generate_bindings_for_kernels` call.
