---
title: "Quantization Recipes"
category: QUANTIZATION
backends: []
last_validated: 2026-04-05
source_issues: [10226, 11034, 1141, 11523, 10297, 10104, 10188, 10960, 11355, 11689, 11693, 11694, 13099, 1340]
---

# Quantization Recipes

## When to Use Which Quantization Scheme

| Scheme | Activations | Weights | Best For | Notes |
|--------|------------|---------|----------|-------|
| 8a8w | INT8 | INT8 | Vision models, general inference | Good accuracy-performance tradeoff |
| 16a4w | FP16 | INT4 | LLMs on Qualcomm HTP | Smaller model size, good for weight-bound models |
| 16a8w | FP16 | INT8 | LLMs where 4-bit is too aggressive | Better accuracy than 4-bit |

## Model Family to Quantization Mapping

### LLMs (Llama, Phi, Qwen)

**QNN Backend (Qualcomm):**
- Use `qnn_16a4w` for weight compression with acceptable accuracy [Source: #10226]
- Use the `examples/qualcomm/oss_scripts/llama/` path -- the generic `export_llama.py --qnn` path has known accuracy and file size bugs [Source: #10226]
- The basic PTQ quantization via `export_llama.py --pt2e_quantize qnn_*` produces poor results (gibberish output for 1B models). The improved flow in `oss_scripts/llama` has significantly better accuracy. [Source: #11034]

**XNNPACK Backend (CPU):**
- Use `XNNPACKQuantizer` for INT8 symmetric quantization
- Not all quantized ops can be lowered -- check the delegation statistics after export [Source: #11523]

### Vision Models

- XNNPACKQuantizer works well for standard vision models (MobileNet, ResNet)
- For YOLO12, quantization + XNNPACK lowering has known dim_order issues [Source: #11523]
- Vision models with unusual ops (e.g., custom attention) may need per-op quantization config

### RNN Models (GRU, LSTM)

GRU quantization with XNNPACKQuantizer fails with `IndexError: list index out of range` if the initial hidden state is not explicitly passed. [Source: #10104]

**Fix:** Pass initial hidden state and return both outputs:
```python
class SingleLayerGRU(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, x):
        h0 = torch.randn(1, 1, self.hidden_size)
        return self.gru(x, h0)  # Return both outputs
```

### QAT (Quantization-Aware Training)

QAT quantization is NOT supported in the automated ExecuTorch export pipeline. Advanced users should handle QAT export manually outside the standard pipeline. [Source: #13099]

## PT2E Quantization Flow

The standard ExecuTorch quantization flow uses PyTorch's PT2E quantization:

```python
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e

# 1. Export the model
exported_program = torch.export.export(model, example_inputs)

# 2. Prepare for quantization (insert observers)
prepared = prepare_pt2e(exported_program, quantizer)

# 3. Calibrate with representative data
for batch in calibration_data:
    prepared(batch)

# 4. Convert to quantized model
quantized = convert_pt2e(prepared)

# 5. Lower to edge and backend
executorch_program = to_edge_transform_and_lower(
    quantized,
    partitioner=[backend_partitioner],
).to_executorch()
```
[Source: #1141]

## Quantized Graph Representation

After quantization, the graph contains dequant-op-quant patterns:
```
... -> dequant -> opX -> quant -> dequant -> opY -> quant -> ...
```

Backend partitioners pattern-match `dequant -> op -> quant` for lowering into fixed-point primitives. [Source: #1141]

## Calibration Best Practices

1. **Use representative data:** Calibration data should match the inference distribution
2. **Dataset size:** A few hundred samples is typically sufficient for calibration
3. **For LLMs:** Use text generation tasks (e.g., wikitext evaluation) for calibration [Source: #10226]

```bash
# Example: LLM calibration via wikitext
python examples/qualcomm/oss_scripts/llama/llama.py \
  --compile_only \
  --tasks wikitext --limit 1 \
  ...
```

## Passing Pre-Quantized Inputs

It's possible to remove input/output q/dq nodes and pass already-quantized data:

```python
# Before: float input -> quantize -> delegate -> dequantize -> float output
# After:  quantized input -> delegate -> quantized output
```

Caveats:
- You must update the ExportedProgram input dtypes
- Quantization parameters (scale, zero_point) from removed q/dq nodes must be preserved externally
- Use `FixedQParamsQuantizationSpec` for fixed-point quantization (e.g., Q7 format) [Source: #1141]

## Backend-Specific Quantization Notes

### XNNPACK
- Only float operations are delegated; integer ops fall through to CPU
- `torch.mm` with two dynamic inputs is NOT delegated (needs one constant weight) [Source: #10297]
- Enable optimized kernels for non-delegated ops: `EXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON` [Source: #10297]
- BF16 is NOT supported for delegation (only fp16/fp32). BF16 ops fall through to CPU portable/optimized kernels. BF16 dynamic quantization support is in progress. [Source: #10188]
- `batch_norm` is only delegated when it follows convolution (conv+bn fusion). Standalone `batch_norm` is NOT supported. [Source: #1340]
- Passing raw int8 tensors directly to XNNPACK causes out-of-bounds writes. int8 dtype is reserved for XNNPACK's internal quantized representation -- always use float inputs with the quantization flow. [Source: #10960]
- Dynamic quantization (PT2E) still requires calibration: run sample inputs through the prepared graph before `convert_pt2e`. [Source: #11355]

### CoreML
- CoreML may convert to fp16 internally
- Accuracy drops from fp16 conversion are usually minor but can be significant for certain architectures [Source: #10179]
- CoreML does NOT support integer operations: integer ReLU, integer mm, int16 mm are all rejected by the partitioner [Source: #11693]
- CoreML does NOT support tensors with rank > 5 [Source: #11694]
- CoreML fails to lower `addmm` with integer alpha/beta parameters. Workaround: cast alpha/beta to float [Source: #11689]

### QNN (Qualcomm)
- Use the dedicated QNN quantization flow (`oss_scripts/llama`) for LLMs
- Generic PTQ via `export_llama.py` is known to produce poor accuracy [Source: #11034]
- See the QNN backend wiki for SoC-specific quantization constraints

## See Also

- [QNN Quantization Guide](../backends/qnn/quantization.md) — QNN-specific recipes, per-SoC constraints, mixed precision
- [QNN SoC Compatibility](../backends/qnn/soc-compatibility.md) — V68/V69/V73 feature matrix affecting quantization
- [Quantization Debugging](debugging.md) — Accuracy issues after quantization
- [XNNPACK Known Issues](../backends/xnnpack/known-issues.md) — XNNPACK-specific quantization gotchas
- [CoreML Overview](../backends/coreml/overview.md) — CoreML quantization constraints
