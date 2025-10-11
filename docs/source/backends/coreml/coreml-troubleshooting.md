# Troubleshooting

This page describes common issues that you may encounter when using the Core ML backend and how to debug and resolve them.

### Issues during lowering
1. "ValueError: In op, of type [X], named [Y], the named input [Z] must have the same data type as the named input x. However, [Z] has dtype fp32 whereas x has dtype fp16."

This happens because the model is in FP16, but Core ML interprets some of the arguments as FP32, which leads to a type mismatch.  The solution is to keep the PyTorch model in FP32.  Note that the model will be still be converted to FP16 during lowering to Core ML unless specified otherwise in the compute_precision [Core ML `CompileSpec`](coreml-partitioner.md#coreml-compilespec).  Also see the [related issue in coremltools](https://github.com/apple/coremltools/issues/2480).

2. coremltools/converters/mil/backend/mil/load.py", line 499, in export
    raise RuntimeError("BlobWriter not loaded")

If you're using Python 3.13, try reducing your python version to Python 3.12.  coremltools does not support Python 3.13 per [coremltools issue #2487](https://github.com/apple/coremltools/issues/2487).

### Issues during runtime
1. [ETCoreMLModelCompiler.mm:55] [Core ML]  Failed to compile model, error = Error Domain=com.apple.mlassetio Code=1 "Failed to parse the model specification. Error: Unable to parse ML Program: at unknown location: Unknown opset 'CoreML7'." UserInfo={NSLocalizedDescription=Failed to par$

This means the model requires the Core ML opset 'CoreML7', which requires running the model on iOS >= 17 or macOS >= 14.

## Extracting the mlpackage for profiling and debugging

[Core ML *.mlpackage files](https://apple.github.io/coremltools/docs-guides/source/convert-to-ml-program.html#save-ml-programs-as-model-packages) can be extracted from a Core ML-delegated *.pte file.  This can help with debugging and profiling for users who are more familiar with *.mlpackage files:
```bash
python examples/apple/coreml/scripts/extract_coreml_models.py -m /path/to/model.pte
```

Note that if the ExecuTorch model has graph breaks, there may be multiple extracted *.mlpackage files.
