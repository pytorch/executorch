---
title: "Quantization Debugging"
category: DEBUGGING
backends: []
last_validated: 2026-04-05
source_issues: [10226, 10179, 11034, 10297, 1141, 10960, 11355, 13842]
---

# Quantization Debugging

## Accuracy Debugging After Quantization

### Symptom: Gibberish Output from Quantized LLM

Quantized LLMs (especially 1B parameter models with 4-bit weights) frequently produce gibberish output. This is a known issue with the basic PTQ flow in `examples/models/llama/export_llama.py`. [Source: #11034]

**Root causes:**
1. Basic PTQ quantization algorithm is not sophisticated enough for small LLMs [Source: #10226]
2. The generic `export_llama.py` path has bugs (oversized PTE, wrong quantization) [Source: #10226]

**Fix:** Use the improved quantization flow:
```bash
# Use the QNN-specific script with better quantization
python examples/qualcomm/oss_scripts/llama/llama.py \
  --compile_only -m SM8750 \
  --model_mode hybrid \
  --decoder_model <model> \
  ...
```
[Source: #11034]

### Symptom: Large Accuracy Drop After CoreML Export

If accuracy drops significantly after exporting to CoreML (e.g., mIoU from 0.57 to 0.30):

1. **Check input preprocessing:** ExecuTorch CoreML does NOT support `ct.ImageType`. If your direct coremltools conversion used scale/bias via ImageType, you must bake normalization into the model. [Source: #10179]
2. **Check export path:** ET CoreML uses `torch.export.export`, not `torch.jit.trace`. If your coremltools comparison used `torch.jit.trace`, the discrepancy may be from different export paths. [Source: #10179]
3. **Compare outputs:** Extract the `.mlpackage` from the `.pte` and run it directly to isolate whether the issue is in export or runtime. [Source: #10179]

### Symptom: Quantized PTE Larger Than Float Model

A 4-bit quantized PTE being larger than the float model indicates a bug, not expected behavior. Known to occur with `export_llama.py` + QNN quantization. Fixed in PR #12167. [Source: #10226]

**Sanity check:**
| Quant | Expected PTE Size (Llama 3.2 1B) |
|-------|----------------------------------|
| float | ~2.4 GB |
| 8a8w | ~1.2 GB |
| 16a4w | ~0.8-1.1 GB |

If your 16a4w PTE is 2.9 GB, you are hitting the bug. Switch to `oss_scripts/llama/`. [Source: #10226]

## Common Quantization Errors

### "XNNPACK backend only supports contiguous memory format"

```
RuntimeError: XNNPACK backend only supports contiguous memory format for inputs.
Expecting dim_order: (0, 1, 2), but got (2, 0, 1) for a placeholder node
```

This occurs when quantization introduces `_to_dim_order_copy` ops that XNNPACK cannot handle. [Source: #11523]

**Workaround:** Disable dim order in edge compilation. This is being tracked for a proper fix in coremltools and XNNPACK. [Source: #10451]

### XNNPACK Int8 Out-of-Bounds Write

Passing raw `int8` tensors directly to XNNPACK (without going through the quantization flow) causes out-of-bounds memory writes and invalid outputs. The `int8` dtype is reserved for XNNPACK's internal quantized representation. [Source: #10960]

**Fix:** Always use float inputs and let the quantization flow (prepare_pt2e/convert_pt2e) handle quantization. Do not manually cast inputs to int8.

### Dynamic Quantization Missing Calibration

PT2E dynamic quantization with XNNPACK fails at runtime if the model is not calibrated before `convert_pt2e`. Even dynamic quantization needs calibration to determine quantization parameters. [Source: #11355]

```python
# Required: run calibration data through prepared model
prepared = prepare_pt2e(model, quantizer)
for batch in calibration_data:
    prepared(batch)  # Don't skip this!
converted = convert_pt2e(prepared)
```

### SharedQuantizationSpec RecursionError

Using `SharedQuantizationSpec` causes `RecursionError` if the reference node creates a circular dependency. Ensure the `SharedQuantizationSpec` references a different node, not the node being annotated. [Source: #13842]

### Non-Delegated Ops Dominating Inference Time

After quantization, check which ops are delegated vs non-delegated:

```python
from executorch.devtools import Inspector
inspector = Inspector(etdump_path="./etdump.etdp")
inspector.print_data_tabular()
```

Look at the `occurrences_in_non_delegated_graphs` column. Ops like `aten.mm.default` running on CPU instead of the backend can be 10x slower. [Source: #10297]

## How to Compare Quantized vs Float Outputs

### Step 1: Export Both Versions

```python
# Float model
float_program = to_edge_transform_and_lower(
    torch.export.export(model, inputs),
    partitioner=[partitioner],
).to_executorch()

# Quantized model
quantized_model = prepare_pt2e(torch.export.export(model, inputs), quantizer)
# ... calibrate ...
quantized_model = convert_pt2e(quantized_model)
quant_program = to_edge_transform_and_lower(
    quantized_model,
    partitioner=[partitioner],
).to_executorch()
```

### Step 2: Run Both and Compare

```python
from executorch.extension.pybindings import portable_lib

# Run float
float_module = portable_lib._load_for_executorch("float.pte")
float_out = float_module.forward([input_tensor])

# Run quantized
quant_module = portable_lib._load_for_executorch("quant.pte")
quant_out = quant_module.forward([input_tensor])

# Compare
diff = torch.abs(float_out[0] - quant_out[0])
print(f"Max diff: {diff.max()}, Mean diff: {diff.mean()}")
```

### Step 3: For CoreML, Extract and Compare

Extract the `.mlpackage` from the PTE to run directly via coremltools:
```bash
# See docs/source/backends/coreml/coreml-overview.md#extracting-the-mlpackage
```
[Source: #10179]

## Profiling Quantized Model Performance

Use ETDump for operator-level profiling:

```cpp
// C++ runtime
#include <executorch/devtools/etdump/etdump_flatcc.h>

auto etdump_gen = std::make_unique<ETDumpGen>();
Module model(model_path, Module::LoadMode::MmapUseMlockIgnoreErrors,
             std::move(etdump_gen));
// ... run inference ...
ETDumpResult result = etdump_gen->get_etdump_data();
// Write to file for analysis
```

Then analyze in Python:
```python
from executorch.devtools import Inspector
inspector = Inspector(etdump_path="./etdump.etdp")
inspector.print_data_tabular()
```
[Source: #10297]

## See Also

- [Quantization Recipes](recipes.md) — Scheme selection, calibration best practices
- [QNN Quantization Guide](../backends/qnn/quantization.md) — QNN-specific recipes and errors
- [QNN Known Issues](../backends/qnn/known-issues.md) — Gibberish LLM output diagnosis
- [Performance Troubleshooting](../troubleshooting/performance.md) — Profiling quantized models
