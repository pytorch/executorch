# ExecuTorch Core ML Delegate

This subtree contains the Core ML Delegate implementation for ExecuTorch.
Core ML is an optimized framework for running machine learning models on Apple devices. The delegate is the mechanism for leveraging the Core ML framework to accelerate operators when running on Apple devices.  To learn how to use the CoreML delegate, see the [documentation](https://github.com/pytorch/executorch/blob/main/docs/source/backends/coreml/coreml-overview.md).

## Layout
- `compiler/` : Lowers a module to Core ML backend.
- `partition/`: Partitions a module fully or partially to Core ML backend.
- `quantizer/`: Quantizes a module in Core ML favored scheme.
- `scripts/` : Scripts for installing dependencies and running tests.
- `runtime/`: Core ML delegate runtime implementation.
    - `inmemoryfs`: InMemory filesystem implementation used to serialize/de-serialize AOT blob.
    - `kvstore`: Persistent Key-Value store implementation.
    - `delegate`: Runtime implementation.
    - `include` : Public headers.
    - `sdk` : SDK implementation.
    - `tests` :  Unit tests.
    - `workspace` : Xcode workspace for the runtime.
- `third-party/`: External dependencies.

## Help & Improvements
If you have problems or questions or have suggestions for ways to make
implementation and testing better, please create an issue on [github](https://www.github.com/pytorch/executorch/issues).
