# QNN Lowering / Export

## Common Setup

```python
from executorch.backends.qualcomm.serialization.qc_schema import QnnExecuTorchBackendType
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    get_soc_to_chipset_map,
    to_edge_transform_and_lower_to_qnn,
)

soc_model = get_soc_to_chipset_map()["SM8650"]  # adjust SoC as needed
```

---

## FP16 Export

```python
backend_options = generate_htp_compiler_spec(use_fp16=True)
compiler_specs = generate_qnn_executorch_compiler_spec(
    soc_model=soc_model,
    backend_options=backend_options,
)
edge_prog_mgr = to_edge_transform_and_lower_to_qnn(model, example_inputs, compiler_specs)
et_program = edge_prog_mgr.to_executorch()
```

---

## Quantized (PTQ) Export

```python
import torch
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer

# 1. Export to ATen IR
m = torch.export.export(model.eval(), example_inputs, strict=True).module()

# 2. Prepare for quantization
quantizer = QnnQuantizer(
    backend=QnnExecuTorchBackendType.kHtpBackend,
    soc_model=soc_model,
)
m = prepare_pt2e(m, quantizer)

# 3. Calibrate
m(*example_inputs)

# 4. Convert
m = convert_pt2e(m)

# 5. Lower to QNN
backend_options = generate_htp_compiler_spec(use_fp16=False)
compiler_specs = generate_qnn_executorch_compiler_spec(
    soc_model=soc_model,
    backend_options=backend_options,
)
edge_prog_mgr = to_edge_transform_and_lower_to_qnn(m, example_inputs, compiler_specs)
et_program = edge_prog_mgr.to_executorch()
```

---

## Quantized (QAT) Export

Same as PTQ but use `prepare_qat_pt2e` and run a training loop instead of calibration:

```python
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_qat_pt2e

m = prepare_qat_pt2e(m, quantizer)
# training loop
m(*example_inputs)
m = convert_pt2e(m)
# ... same lowering steps as PTQ
```

---

## Quantization Options

| QuantDtype | Activation | Weight |
|---|---|---|
| `use_16a16w` | uint16 | int16 |
| `use_16a8w` | uint16 | int8 |
| `use_16a4w` | uint16 | int4 |
| `use_16a4w_block` | uint16 | int4 (block-wise) |
| `use_8a8w` | uint8 | int8 |
| `use_8a4w` | uint8 | int4 |

**Fine-grained control with QuantRecipe:**

```python
from executorch.backends.qualcomm.quantizer.quant_recipe import QuantRecipe, QuantGranularity

recipe = QuantRecipe(quant_dtype=QuantDtype.use_8a8w, is_qat=False)
recipe.add_node_target(targets={torch.ops.aten.linear.default}, quant_dtype=QuantDtype.use_16a8w)
recipe.add_regex(regex={"layers.[0-3].attention"}, quant_dtype=QuantDtype.use_16a4w)
```

---

## Pass Pipelines (QnnPassManager)

| Pipeline | When Called | Key Passes |
|---|---|---|
| `transform_for_annotation_pipeline` | Before `prepare_pt2e` (called internally by `QnnQuantizer`) | RemoveRedundancy, Decompose*, Recompose*, ReplaceInfValues |
| `transform_for_export_pipeline` | After `torch.export` | Decompose*, CanonicalizeConv, LiftConstantScalarOperands |
| `get_to_edge_transform_passes` | Before `to_edge` | AnnotateQuantAttrs, FoldQDQ, LayoutTransform, TagQuantIO, **ResolveDebugHandle (must be last)** |
| `transform_for_preprocess_pipeline` | Inside `QnnBackend.preprocess` | FoldQDQ(force_fold=True), InsertRequantize, InsertIOQDQ, LayoutTransform(insert_permute=True), FuseConsecutiveCast |

---

## Skipping Ops / Partial Delegation

```python
from executorch.backends.qualcomm.utils.utils import skip_annotation

# Skip specific node targets from being delegated
skip_annotation(model, skipped_ops={torch.ops.aten.add.Tensor})
```

---

## Dumping Context Binary

```python
from executorch.backends.qualcomm.utils.utils import dump_context_from_pte

dump_context_from_pte("model.pte", output_dir="./context_bins/")
```

---

## SoC Reference

See `_soc_info_table` in `backends/qualcomm/serialization/qc_schema.py`.
