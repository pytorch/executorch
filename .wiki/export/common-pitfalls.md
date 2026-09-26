---
title: "Export Common Pitfalls"
category: EXPORT_PATTERN
backends: []
last_validated: 2026-04-05
source_issues: [10451, 10179, 10297, 10226, 11523, 10014, 10066, 1020, 10151, 11128, 10065, 10009, 2910, 1350]
---

# Export Common Pitfalls

## torch.export Errors

### Dynamic Shapes Not Supported

`torch.export` requires static graph tracing. Control flow (if/else on tensor values), dynamic shapes, and data-dependent operations cause failures.

**Symptom:** `torch._dynamo.exc.Unimplemented` or `GuardOnDataDependentSymNode`

**Workarounds:**
- Use `--disable_dynamic_shape` for LLM exports when dynamic shapes are not needed [Source: #10226]
- For models with conditional branches (e.g., GEMM vs GEMV paths), restructure to use a single static path [Source: #10297]
- When dynamic shapes fail, consider separate static models for prefill and decode [Source: #10226]

### Operator Decomposition Issues

Some operators are not directly supported by backends and require decomposition. Warnings like "ET ignoring certain decomposition requests" are usually benign. [Source: #10179]

**Common pattern:**
```
UserWarning: Decomposition for <op> requested by backend but not available
```

These warnings mean ExecuTorch tried to decompose an op for a backend but no decomposition exists. They typically don't affect correctness. [Source: #10179]

## to_edge / EdgeCompileConfig Gotchas

### dim_order_ops Breaking Backend Delegation

When dim order is enabled (default since v0.6), `_to_dim_order_copy` ops appear in the graph. Some backends (CoreML, XNNPACK) don't recognize this op, causing delegation failures. [Source: #10451]

**Error:**
```
RuntimeError: XNNPACK backend only supports contiguous memory format for inputs.
Expecting dim_order: (0, 1, 2), but got (2, 0, 1) for a placeholder node
```
[Source: #11523]

**Workaround:** Disable dim order in EdgeCompileConfig:
```python
from executorch.exir import EdgeCompileConfig
edge_config = EdgeCompileConfig(_check_ir_validity=False)
# Or pass _skip_dim_order=True depending on version
```
[Source: #10451]

### to_backend() vs to_edge_transform_and_lower()

Always prefer `to_edge_transform_and_lower()` over the older `to_backend()` API. The newer API applies necessary graph transforms before lowering. [Source: #10297]

```python
# PREFERRED
from executorch.exir import to_edge_transform_and_lower
executorch_program = to_edge_transform_and_lower(
    exported_program,
    partitioner=[XnnpackPartitioner()],
).to_executorch()

# AVOID (older API, fewer transforms applied)
edge = to_edge(exported_graph)
edge_delegated = edge.to_backend(XnnpackPartitioner())
```
[Source: #10297]

## Delegation Failures and Debugging

### Ops Not Getting Delegated

When operators aren't delegated to a backend, they run on the portable CPU kernels which are significantly slower. Common reasons:

1. **torch.mm with two dynamic inputs:** XNNPACK only delegates mm when one input is a constant weight tensor. If both inputs are dynamic, mm falls through to CPU. [Source: #10297]
2. **Non-float dtypes:** Some backends (XNNPACK) only delegate float operations. Integer or bool operations fall through. [Source: #10297]
3. **Unsupported ops:** Check the backend's op support list. Ops like `native_layer_norm` may not be delegated by XNNPACK. [Source: #10297]

**How to diagnose:**
```python
# After lowering, check delegation statistics
edge_program = to_edge_transform_and_lower(exported_program, partitioner=[...])
# Print the graph to see which ops are delegated vs non-delegated
print(edge_program.exported_program().graph_module)
```

Use the ExecuTorch profiler to identify which non-delegated ops consume the most time:
```python
from executorch.devtools import Inspector
inspector = Inspector(etdump_path="./etdump.etdp")
inspector.print_data_tabular()
```
[Source: #10297]

### CoreML Delegation: ImageType Not Supported

The ExecuTorch CoreML delegate uses the `torch.export.export` path (not `torch.jit.trace`). It only supports `ct.TensorType` and `ct.StateType`. `ct.ImageType` is not supported. If your model uses image preprocessing (scale/bias), you must bake that into the model wrapper. [Source: #10179]

**Key difference from direct coremltools:**
- ET CoreML: `torch.export.export` path only
- Direct coremltools: supports both `torch.jit.trace` and `torch.export.export`
- ET CoreML: no `ct.ImageType` support, must handle normalization in the model [Source: #10179]

### Extracting Backend Artifacts for Debugging

For CoreML, you can extract the `.mlpackage` from the `.pte` file for inspection. Search for "extracting the mlpackage" in the ExecuTorch CoreML backend documentation. [Source: #10179]

## PTE File Size Issues

### Quantized PTE Larger Than Float Model

If a 4-bit quantized PTE file is larger than the original float model, you are likely hitting a known bug in `examples/models/llama/export_llama.py`. Use `examples/qualcomm/oss_scripts/llama` instead for QNN backend exports. [Source: #10226]

**The bug:** `export_llama.py` duplicated weight data in certain quantization configurations, fixed in PR #12167.

### Two Llama Export Codebases

There are two separate Llama export paths:
1. `examples/models/llama/` - Generic, multi-backend, but has known bugs with QNN quantization
2. `examples/qualcomm/oss_scripts/llama/` - QNN-optimized, actively maintained for Qualcomm HTP

For QNN/Qualcomm deployments, always use `examples/qualcomm/oss_scripts/llama/`. [Source: #10226]

## torch.export Tracing Issues

### "Attempted to call function marked as skipped"

**Error:** `torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped`

Models with complex Python logic (C/C++ extensions, unicodedata, custom tokenizers) may fail strict tracing. [Source: #11128]

**Fix:** Use non-strict export mode:
```python
exported = torch.export.export(model, inputs, strict=False)
```

### Untraceable Models (Stable Diffusion, HuggingFace Pipelines)

Models using C/C++-based tokenizers (e.g., `tokenizers.AddedToken`, `unicodedata.category`) cannot be traced with `torch.export`. These are upstream PyTorch limitations, not ExecuTorch-specific. [Source: #10065]

## PYTHONPATH / Package Path Conflicts

### program.fbs Not Found (pip install + local repo)

**Error:** `FileNotFoundError: .../exir/_serialize/program.fbs`

**Root cause:** Having the ExecuTorch repo in `PYTHONPATH` or working directory shadows the pip-installed package, causing it to look for flatbuffer schemas in the source tree instead of the pip package. [Source: #10009, #2910]

**Fix:**
```bash
unset PYTHONPATH
# Or ensure the ET repo directory is not in your Python path
```

## Static Memory Planning Gotchas

### "Attempted to resize a static tensor"

**Error:** `Attempted to resize a static tensor to a new shape at dimension 0`

ExecuTorch plans memory at export time based on example input shapes. If runtime inputs have different shapes, this error occurs. [Source: #1350]

**Fix:** Ensure runtime inputs match the shapes used during export, or use dynamic shapes if the model supports them.

## Installation Shortcuts

As of v0.6, `pip install executorch` includes CoreML and XNNPACK export support out of the box on macOS. You don't need to build from source for basic exports:
```bash
pip install executorch torch torchvision torchaudio
```
[Source: #10066]

## See Also

- [Model-Specific Export Patterns](model-specific.md) — LLM, vision, audio export recipes
- [Quantization Recipes](../quantization/recipes.md) — Quantization before export
- [Build Failures](../troubleshooting/build-failures.md) — Build issues during export
- [Runtime Errors](../troubleshooting/runtime-errors.md) — Missing ops after export
